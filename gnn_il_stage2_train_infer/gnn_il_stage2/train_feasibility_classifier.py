
# train_feasibility_classifier.py
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_gnn import FeasibilityClassifier
from utils_data import load_graph, load_samples, train_val_split


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    ddir = Path(args.dataset_dir)
    if not (ddir/"feas_pairs.pt").exists():
        raise FileNotFoundError(f"{ddir/'feas_pairs.pt'} not found. Build it using Stage 1 build_feasibility_pairs.py")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    graph = load_graph(args.dataset_dir, device=str(device))
    samples = load_samples(args.dataset_dir, device="cpu")
    meta = json.loads((ddir/"meta.json").read_text(encoding="utf-8"))

    pairs = torch.load(ddir/"feas_pairs.pt", map_location="cpu")
    state_idx = pairs["state_idx"].long()
    cand_pad = pairs["cand_pad"].long()
    cand_size = pairs["cand_size"].long()
    y = pairs["label_ok"].float()

    P = state_idx.size(0)
    tr_idx, va_idx = train_val_split(P, args.val_ratio, args.seed)

    dl_tr = DataLoader(TensorDataset(tr_idx), batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(TensorDataset(va_idx), batch_size=args.batch, shuffle=False, num_workers=0)

    H0 = samples["H0"]
    irrig = samples["irrigated"]
    reach = samples["reachable"]

    torch.manual_seed(args.seed)
    model = FeasibilityClassifier(
        node_in_dim=graph.node_x.shape[1],
        edge_attr_dim=graph.edge_attr.shape[1],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = out_dir/"feas_train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","split","loss","acc"])

    def eval_dl(dl):
        model.eval()
        tot_loss=tot_acc=0.0
        n=0
        with torch.no_grad():
            for (pidx,) in dl:
                pidx=pidx.long()
                sel = state_idx[pidx]
                bH0 = H0[sel].to(device)
                birr= irrig[sel].to(device)
                brea= reach[sel].to(device)
                bcand = cand_pad[pidx].to(device)
                bsz = cand_size[pidx].to(device)
                by = y[pidx].to(device)

                logit = model(graph.node_x, graph.edge_index, graph.edge_attr,
                              graph.lateral_to_node_idx, graph.side_idx,
                              bH0, birr, brea, bcand, bsz)
                loss = loss_fn(logit, by)
                prob = torch.sigmoid(logit)
                acc = ((prob>=0.5).float() == by).float().mean().item()

                bs = pidx.numel()
                tot_loss += float(loss.item())*bs
                tot_acc  += acc*bs
                n += bs
        return dict(loss=tot_loss/n, acc=tot_acc/n)

    best_val = 1e18
    best_path = out_dir/"feasibility.pt"

    for ep in range(1, args.epochs+1):
        model.train()
        for (pidx,) in dl_tr:
            pidx=pidx.long()
            sel = state_idx[pidx]
            bH0 = H0[sel].to(device)
            birr= irrig[sel].to(device)
            brea= reach[sel].to(device)
            bcand = cand_pad[pidx].to(device)
            bsz = cand_size[pidx].to(device)
            by = y[pidx].to(device)

            logit = model(graph.node_x, graph.edge_index, graph.edge_attr,
                          graph.lateral_to_node_idx, graph.side_idx,
                          bH0, birr, brea, bcand, bsz)
            loss = loss_fn(logit, by)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        tr = eval_dl(dl_tr)
        va = eval_dl(dl_va)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([ep,"train",tr["loss"],tr["acc"]])
            w.writerow([ep,"val",va["loss"],va["acc"]])

        print(f"[E{ep:03d}] train loss={tr['loss']:.4f} acc={tr['acc']:.3f} | val loss={va['loss']:.4f} acc={va['acc']:.3f}")

        if va["loss"] < best_val:
            best_val = va["loss"]
            ckpt = {
                "state_dict": model.state_dict(),
                "config": {
                    "node_in_dim": graph.node_x.shape[1],
                    "edge_attr_dim": graph.edge_attr.shape[1],
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout
                },
                "meta": {"root": meta.get("root","J0"), "node_ids": meta["node_ids"], "lateral_ids": meta["lateral_ids"]},
            }
            torch.save(ckpt, best_path)
            print(f"  [SAVE] {best_path} (best val={best_val:.4f})")

    print("[DONE]")


if __name__ == "__main__":
    main()
