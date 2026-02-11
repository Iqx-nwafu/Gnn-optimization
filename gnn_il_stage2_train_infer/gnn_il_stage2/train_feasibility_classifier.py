
# train_feasibility_classifier_v2_fast.py
# Faster FeasibilityClassifier training:
# - cache samples+pairs on GPU: --cache_samples_on_gpu
# - mixed precision: --amp (cuda only)
# - class-balanced sampler: --use_sampler
#
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model_gnn import FeasibilityClassifier
from utils_data import load_graph, load_samples, train_val_split


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--use_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--cache_samples_on_gpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()


def confusion_counts(prob: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Tuple[int, int, int, int]:
    pred = (prob >= thr).to(torch.int64)
    y = y.to(torch.int64)
    tp = int(((pred == 1) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    return tp, tn, fp, fn


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision0 = safe_div(tn, tn + fn)
    recall0 = safe_div(tn, tn + fp)
    f1_0 = safe_div(2 * precision0 * recall0, precision0 + recall0) if (precision0 + recall0) > 0 else 0.0
    precision1 = safe_div(tp, tp + fp)
    recall1 = safe_div(tp, tp + fn)
    f1_1 = safe_div(2 * precision1 * recall1, precision1 + recall1) if (precision1 + recall1) > 0 else 0.0
    bal_acc = 0.5 * (recall0 + recall1)
    return dict(acc=acc, bal_acc=bal_acc,
                precision0=precision0, recall0=recall0, f1_0=f1_0,
                precision1=precision1, recall1=recall1, f1_1=f1_1)


def main():
    args = parse_args()
    ddir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feas_path = ddir / "feas_pairs.pt"
    if not feas_path.exists():
        raise FileNotFoundError(f"{feas_path} not found.")

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    graph = load_graph(args.dataset_dir, device=str(device))
    samples = load_samples(args.dataset_dir, device="cpu")
    meta = json.loads((ddir / "meta.json").read_text(encoding="utf-8"))

    pairs = torch.load(feas_path, map_location="cpu")
    state_idx_cpu = pairs["state_idx"].long()
    cand_pad_cpu = pairs["cand_pad"].long()
    cand_size_cpu = pairs["cand_size"].long()
    y_cpu = pairs["label_ok"].float()

    P = int(state_idx_cpu.size(0))
    tr_idx, va_idx = train_val_split(P, args.val_ratio, args.seed)

    H0_cpu = samples["H0"]
    irrig_cpu = samples["irrigated"]
    reach_cpu = samples["reachable"]

    y_tr = y_cpu[tr_idx]
    pos = float(y_tr.sum().item())
    neg = float((1 - y_tr).sum().item())
    pos_rate = pos / max(1.0, pos + neg)
    pos_w = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    if args.use_sampler:
        w = torch.where(y_tr > 0.5,
                        torch.full_like(y_tr, 0.5 / max(pos_rate, 1e-6)),
                        torch.full_like(y_tr, 0.5 / max(1 - pos_rate, 1e-6)))
        sampler = WeightedRandomSampler(weights=w.double(), num_samples=int(tr_idx.numel()), replacement=True)
        dl_tr = DataLoader(TensorDataset(tr_idx), batch_size=args.batch, sampler=sampler,
                           num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        dl_tr = DataLoader(TensorDataset(tr_idx), batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers, pin_memory=args.pin_memory)

    dl_va = DataLoader(TensorDataset(va_idx), batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, pin_memory=args.pin_memory)

    torch.manual_seed(args.seed)
    model = FeasibilityClassifier(
        node_in_dim=graph.node_x.shape[1],
        edge_attr_dim=graph.edge_attr.shape[1],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 指定设备类型为 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and use_cuda))

    cached = False
    if args.cache_samples_on_gpu and use_cuda:
        try:
            H0 = H0_cpu.to(device, non_blocking=True)
            irrig = irrig_cpu.to(device, non_blocking=True)
            reach = reach_cpu.to(device, non_blocking=True)
            state_idx = state_idx_cpu.to(device, non_blocking=True)
            cand_pad = cand_pad_cpu.to(device, non_blocking=True)
            cand_size = cand_size_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            cached = True
            print("[INFO] cached samples+pairs on GPU")
        except RuntimeError as e:
            print("[WARN] cache_samples_on_gpu failed, fallback to CPU tensors:", repr(e))
            H0, irrig, reach = H0_cpu, irrig_cpu, reach_cpu
            state_idx, cand_pad, cand_size, y = state_idx_cpu, cand_pad_cpu, cand_size_cpu, y_cpu
    else:
        H0, irrig, reach = H0_cpu, irrig_cpu, reach_cpu
        state_idx, cand_pad, cand_size, y = state_idx_cpu, cand_pad_cpu, cand_size_cpu, y_cpu

    log_path = out_dir / "feas_train_log_v2.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["epoch","split","loss","acc","bal_acc","precision0","recall0","f1_0","precision1","recall1","f1_1",
                       "pos_rate_train","pos_weight","cached_on_gpu","amp"])

    def fetch_batch(pidx: torch.Tensor):
        if cached:
            pidx = pidx.to(device, non_blocking=True)
            sel = state_idx[pidx]
            return (H0[sel], irrig[sel], reach[sel], cand_pad[pidx], cand_size[pidx], y[pidx])
        pidx_cpu = pidx.long()
        sel = state_idx[pidx_cpu]
        return (H0[sel].to(device, non_blocking=True),
                irrig[sel].to(device, non_blocking=True),
                reach[sel].to(device, non_blocking=True),
                cand_pad[pidx_cpu].to(device, non_blocking=True),
                cand_size[pidx_cpu].to(device, non_blocking=True),
                y[pidx_cpu].to(device, non_blocking=True))

    def run_eval(dl):
        model.eval()
        tot_loss = 0.0
        tp=tn=fp=fn=0
        n = 0
        with torch.no_grad():
            for (pidx,) in dl:
                bH0, birr, brea, bcand, bsz, by = fetch_batch(pidx)
                # 指定设备类型为 'cuda'
                with torch.amp.autocast('cuda', enabled=(args.amp and use_cuda)):
                    logit = model(graph.node_x, graph.edge_index, graph.edge_attr,
                                  graph.lateral_to_node_idx, graph.side_idx,
                                  bH0, birr, brea, bcand, bsz)
                    loss = loss_fn(logit, by)
                prob = torch.sigmoid(logit)
                _tp,_tn,_fp,_fn = confusion_counts(prob, by, 0.5)
                tp += _tp; tn += _tn; fp += _fp; fn += _fn
                bs = int(by.numel())
                tot_loss += float(loss.item()) * bs
                n += bs
        m = metrics_from_conf(tp, tn, fp, fn)
        m["loss"] = tot_loss / max(n, 1)
        return m

    best_val = 1e18
    best_path = out_dir / "feasibility_v2.pt"

    print(f"[INFO] pairs={P} train_pos_rate={pos_rate:.6f} pos_weight={float(pos_w.item()):.3f} "
          f"use_sampler={args.use_sampler} cached={cached} amp={bool(args.amp and use_cuda)}")

    for ep in range(1, args.epochs + 1):
        model.train()
        for (pidx,) in dl_tr:
            bH0, birr, brea, bcand, bsz, by = fetch_batch(pidx)
            opt.zero_grad(set_to_none=True)
            # 指定设备类型为 'cuda'
            with torch.amp.autocast('cuda', enabled=(args.amp and use_cuda)):
                logit = model(graph.node_x, graph.edge_index, graph.edge_attr,
                              graph.lateral_to_node_idx, graph.side_idx,
                              bH0, birr, brea, bcand, bsz)
                loss = loss_fn(logit, by)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()

        tr = run_eval(dl_tr)
        va = run_eval(dl_va)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([ep,"train",tr["loss"],tr["acc"],tr["bal_acc"],tr["precision0"],tr["recall0"],tr["f1_0"],
                           tr["precision1"],tr["recall1"],tr["f1_1"],pos_rate,float(pos_w.item()),int(cached),int(bool(args.amp and use_cuda))])
            wcsv.writerow([ep,"val",va["loss"],va["acc"],va["bal_acc"],va["precision0"],va["recall0"],va["f1_0"],
                           va["precision1"],va["recall1"],va["f1_1"],pos_rate,float(pos_w.item()),int(cached),int(bool(args.amp and use_cuda))])

        print(f"[E{ep:03d}] train loss={tr['loss']:.4f} acc={tr['acc']:.3f} bal={tr['bal_acc']:.3f} "
              f"rec0={tr['recall0']:.3f} rec1={tr['recall1']:.3f} | "
              f"val loss={va['loss']:.4f} acc={va['acc']:.3f} bal={va['bal_acc']:.3f} rec0={va['recall0']:.3f} rec1={va['recall1']:.3f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "state_dict": model.state_dict(),
                "config": {
                    "node_in_dim": graph.node_x.shape[1],
                    "edge_attr_dim": graph.edge_attr.shape[1],
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout,
                    "pos_weight": float(pos_w.item()),
                    "use_sampler": bool(args.use_sampler),
                    "cached_on_gpu": bool(cached),
                    "amp": bool(args.amp and use_cuda),
                },
                "meta": {"root": meta.get("root","J0"), "node_ids": meta["node_ids"], "lateral_ids": meta["lateral_ids"]},
            }
            torch.save(ckpt, best_path)
            print(f"  [SAVE] {best_path} (best val={best_val:.4f})")

    print("[DONE] wrote:", log_path)


if __name__ == "__main__":
    main()
