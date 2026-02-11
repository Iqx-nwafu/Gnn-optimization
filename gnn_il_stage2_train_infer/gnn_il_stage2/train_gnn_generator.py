
# train_gnn_generator.py
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_gnn import GroupGenerator
from utils_data import load_graph, load_samples, train_val_split


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_size", type=float, default=0.2)
    ap.add_argument("--pos_weight", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    graph = load_graph(args.dataset_dir, device=str(device))
    samples = load_samples(args.dataset_dir, device="cpu")
    meta = json.loads((Path(args.dataset_dir)/"meta.json").read_text(encoding="utf-8"))

    H0 = samples["H0"]
    irrig = samples["irrigated"]
    reach = samples["reachable"]
    tgt = samples["target"]
    tsize = samples["target_size"]

    S, L = tgt.shape
    train_idx, val_idx = train_val_split(S, args.val_ratio, args.seed)

    # auto pos_weight
    if args.pos_weight and args.pos_weight > 0:
        pos_w = torch.tensor([args.pos_weight], dtype=torch.float32)
    else:
        avg_k = float(tgt.sum(dim=1).float().mean().item())
        pos_w = torch.tensor([(L - avg_k) / max(avg_k, 1.0)], dtype=torch.float32)

    ds_tr = TensorDataset(train_idx)
    ds_va = TensorDataset(val_idx)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    model = GroupGenerator(
        node_in_dim=graph.node_x.shape[1],
        edge_attr_dim=graph.edge_attr.shape[1],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = out_dir/"train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","split","loss","bce","ce_size","token_acc@k","size_acc"])

    def eval_dl(dl):
        model.eval()
        tot_loss=tot_bce=tot_ce=0.0
        tot_tok=tot_size=0.0
        n=0
        with torch.no_grad():
            for (bidx,) in dl:
                bidx=bidx.long()
                bH0=H0[bidx].to(device)
                birr=irrig[bidx].to(device)
                brea=reach[bidx].to(device)
                btgt=tgt[bidx].to(device).float()
                bsz=tsize[bidx].to(device).long()

                tok, szlog = model(graph.node_x, graph.edge_index, graph.edge_attr,
                                   graph.lateral_to_node_idx, graph.side_idx,
                                   bH0, birr, brea)

                loss_b = bce(tok, btgt)
                y = (bsz-2).clamp(0,2)
                loss_s = ce(szlog, y)
                loss = loss_b + args.lambda_size*loss_s

                # token acc@k among available
                avail = (brea.bool() & (~birr.bool()))
                masked = tok.masked_fill(~avail, float("-inf"))
                k = int(bsz.float().mean().round().clamp(2,4).item())
                topk = torch.topk(masked, k=k, dim=1).indices
                pred = torch.zeros_like(btgt)
                pred.scatter_(1, topk, 1.0)
                inter = (pred*btgt).sum(dim=1)
                denom = btgt.sum(dim=1).clamp_min(1.0)
                tok_acc = (inter/denom).mean().item()

                size_acc = (szlog.argmax(dim=1)==y).float().mean().item()

                bs = bidx.numel()
                tot_loss += float(loss.item())*bs
                tot_bce  += float(loss_b.item())*bs
                tot_ce   += float(loss_s.item())*bs
                tot_tok  += tok_acc*bs
                tot_size += size_acc*bs
                n += bs
        return dict(loss=tot_loss/n, bce=tot_bce/n, ce=tot_ce/n, tok=tot_tok/n, size=tot_size/n)

    best_val = 1e18
    best_path = out_dir/"generator.pt"

    for ep in range(1, args.epochs+1):
        model.train()
        for (bidx,) in dl_tr:
            bidx=bidx.long()
            bH0=H0[bidx].to(device)
            birr=irrig[bidx].to(device)
            brea=reach[bidx].to(device)
            btgt=tgt[bidx].to(device).float()
            bsz=tsize[bidx].to(device).long()

            tok, szlog = model(graph.node_x, graph.edge_index, graph.edge_attr,
                               graph.lateral_to_node_idx, graph.side_idx,
                               bH0, birr, brea)

            loss_b = bce(tok, btgt)
            y = (bsz-2).clamp(0,2)
            loss_s = ce(szlog, y)
            loss = loss_b + args.lambda_size*loss_s

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        tr = eval_dl(dl_tr)
        va = eval_dl(dl_va)

        with log_path.open("a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([ep,"train",tr["loss"],tr["bce"],tr["ce"],tr["tok"],tr["size"]])
            w.writerow([ep,"val",va["loss"],va["bce"],va["ce"],va["tok"],va["size"]])

        print(f"[E{ep:03d}] train loss={tr['loss']:.4f} tok@k={tr['tok']:.3f} size={tr['size']:.3f} | "
              f"val loss={va['loss']:.4f} tok@k={va['tok']:.3f} size={va['size']:.3f}")

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
                    "lambda_size": args.lambda_size,
                    "pos_weight": float(pos_w.item()),
                },
                "meta": {"root": meta.get("root","J0"), "node_ids": meta["node_ids"], "lateral_ids": meta["lateral_ids"]},
            }
            torch.save(ckpt, best_path)
            print(f"  [SAVE] {best_path} (best val={best_val:.4f})")

    print("[DONE]")


if __name__ == "__main__":
    main()
