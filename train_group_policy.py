# train_group_policy.py
"""
Train a group-generation policy network via imitation learning (supervised learning).

Input:
- dataset.pt produced by build_imitation_dataset.py

Model:
- MLP that maps [H0_norm, step_frac, remaining_frac, mask_selected(L)] -> 
    (a) size logits over {2,3,4}
    (b) membership logits over L laterals for the *next group*

Training losses:
- size: CrossEntropyLoss
- membership: BCEWithLogitsLoss (with pos_weight to counter class imbalance)

Output:
- policy.pt (checkpoint with state_dict + metadata)

Requires:
- torch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class GroupPolicyNet(nn.Module):
    def __init__(self, input_dim: int, n_laterals: int, hidden: int = 512, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for i in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head_size = nn.Linear(hidden, 3)         # 0->2,1->3,2->4
        self.head_mem = nn.Linear(hidden, n_laterals) # logits per lateral

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        size_logits = self.head_size(h)
        mem_logits = self.head_mem(h)
        return size_logits, mem_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset.pt")
    p.add_argument("--out", required=True, help="policy.pt")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--alpha_mem", type=float, default=1.0, help="weight for membership loss")
    p.add_argument("--pos_weight", type=float, default=0.0, help="Override pos_weight for BCE; 0=auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total = 0
    correct_size = 0
    topk_exact = 0  # whether predicted top-k matches ground-truth set exactly (k from GT)
    bce_sum = 0.0
    ce_sum = 0.0

    for x, y_size, y_mem in loader:
        x = x.to(device)
        y_size = y_size.to(device)
        y_mem = y_mem.to(device)

        size_logits, mem_logits = model(x)

        # size acc
        pred_size = size_logits.argmax(dim=1)
        correct_size += (pred_size == y_size).sum().item()

        # membership: exact set match using GT k
        # (decode using mem logits; for evaluation we use GT k to isolate membership quality)
        for i in range(x.size(0)):
            k = [2, 3, 4][int(y_size[i].item())]
            scores = mem_logits[i]
            # exclude already selected laterals: mask_selected is in x[3:]
            selected_mask = x[i, 3:]
            scores = scores.masked_fill(selected_mask > 0.5, float("-inf"))
            topk = torch.topk(scores, k=k).indices
            gt = (y_mem[i] > 0.5).nonzero(as_tuple=False).view(-1)
            if gt.numel() == k and torch.equal(torch.sort(topk).values, torch.sort(gt).values):
                topk_exact += 1

        total += x.size(0)

    return {
        "size_acc": correct_size / max(total, 1),
        "topk_exact": topk_exact / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = torch.load(args.dataset, map_location="cpu")
    X = data["X"]
    y_size = data["y_size"]
    y_mem = data["y_membership"]
    laterals = data["laterals"]
    H0_min = float(data["H0_min"])
    H0_max = float(data["H0_max"])

    N, D = X.shape
    L = len(laterals)
    assert y_mem.shape == (N, L), f"y_membership shape mismatch: {y_mem.shape} vs {(N,L)}"

    # auto pos_weight based on sparsity
    if args.pos_weight > 0:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)
    else:
        p = float(y_mem.mean().item())
        p = max(min(p, 0.99), 1e-6)
        pos_weight = torch.tensor([(1.0 - p) / p], dtype=torch.float32)

    ds = TensorDataset(X, y_size, y_mem)
    n_val = int(round(N * args.val_ratio))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    model = GroupPolicyNet(input_dim=D, n_laterals=L, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_size_fn = nn.CrossEntropyLoss()
    loss_mem_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(args.device))

    best_val = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, ys, ym in train_loader:
            x = x.to(args.device)
            ys = ys.to(args.device)
            ym = ym.to(args.device)

            size_logits, mem_logits = model(x)

            loss_size = loss_size_fn(size_logits, ys)
            loss_mem = loss_mem_fn(mem_logits, ym)
            loss = loss_size + args.alpha_mem * loss_mem

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()

        metrics = evaluate(model, val_loader, args.device)
        score = metrics["topk_exact"] * 0.7 + metrics["size_acc"] * 0.3  # simple composite

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | val size_acc={metrics['size_acc']:.4f} | val topk_exact={metrics['topk_exact']:.4f} | best={best_val:.4f}")

    ckpt = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "model_cfg": {
            "input_dim": D,
            "n_laterals": L,
            "hidden": args.hidden,
            "depth": args.depth,
            "dropout": args.dropout,
        },
        "laterals": laterals,
        "H0_min": H0_min,
        "H0_max": H0_max,
        "meta": {
            "dataset": str(Path(args.dataset).resolve()),
            "seed": args.seed,
            "pos_weight": float(pos_weight.item()),
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "alpha_mem": args.alpha_mem,
            "device_trained": args.device,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved policy to: {out_path}")


if __name__ == "__main__":
    main()
