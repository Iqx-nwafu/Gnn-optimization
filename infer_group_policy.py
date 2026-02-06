# infer_group_policy.py
"""
Generate a rotation schedule for a given H0 using a trained imitation policy network,
with feasibility checking and repair using the hydraulic evaluator.

Input:
- policy.pt (from train_group_policy.py)
- Nodes.xlsx / Pipes.xlsx
- H0 value (can be non-integer, e.g., 17.8)

Output:
- schedule.json / group_metrics.csv / lateral_metrics.csv in --out folder

Note:
- The model predicts group size and member laterals; we always verify feasibility.
- If an infeasible group is predicted, we run a repair search over high-scoring candidates.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError(
        "Cannot import tree_evaluator.py. Put this script in the same folder as tree_evaluator.py, "
        "or add that folder to PYTHONPATH."
    ) from e


class GroupPolicyNet(nn.Module):
    def __init__(self, input_dim: int, n_laterals: int, hidden: int = 512, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.head_size = nn.Linear(hidden, 3)
        self.head_mem = nn.Linear(hidden, n_laterals)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.head_size(h), self.head_mem(h)


@dataclass(frozen=True)
class GroupResult:
    laterals: List[str]
    ok: bool
    min_margin: float
    min_pressure: float
    pressures: List[float]
    surplus: List[float]
    group_mean_surplus: float
    group_std_surplus: float
    repaired: bool
    repair_note: str


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def build_lateral_order(nodes_path: str, pipes_path: str) -> Tuple[List[str], Dict[str, str], Dict[str, float]]:
    nodes = te.load_nodes_xlsx(nodes_path)
    _edges = te.load_pipes_xlsx(pipes_path)
    field_nodes = sorted([nid for nid in nodes.keys() if te.is_field_node_id(nid)])
    laterals, lat2node = te.build_lateral_ids_for_field_nodes(field_nodes)
    laterals = sorted(laterals)
    node_z = {nid: float(nodes[nid][0]) for nid in nodes.keys()}
    return laterals, lat2node, node_z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="policy.pt")
    p.add_argument("--nodes", required=True)
    p.add_argument("--pipes", required=True)
    p.add_argument("--root", default="J0")
    p.add_argument("--H0", type=float, required=True)
    p.add_argument("--Hmin", type=float, default=11.59)
    p.add_argument("--q_lateral", type=float, default=0.012)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top_pool", type=int, default=12, help="candidate pool size for repair search")
    p.add_argument("--top_pool_max", type=int, default=20, help="max pool size for repair search")
    p.add_argument("--repair_samples", type=int, default=2000, help="random samples fallback")
    return p.parse_args()


def fix_group_size(k_pred: int, remaining: int) -> int:
    """Avoid impossible endings (e.g., leaving 1 lateral)."""
    # if near end, just finish
    if remaining in (2, 3, 4):
        return remaining
    # avoid leaving 1: if remaining - k == 1, adjust
    k = k_pred
    k = max(2, min(4, k))
    if remaining - k == 1:
        # try alternative sizes
        for kk in (4, 3, 2):
            if kk <= remaining and remaining - kk != 1:
                return kk
        # fallback
        return 2
    return min(k, remaining)


def eval_group(
    evaluator: te.TreeHydraulicEvaluator,
    lat2node: Dict[str, str],
    Hmin: float,
    q_lateral: float,
    laterals: Sequence[str],
) -> GroupResult:
    res = evaluator.evaluate_group(list(laterals), lateral_to_node=lat2node, q_lateral=q_lateral)
    pressures = []
    surplus = []
    for lat in laterals:
        nid = lat2node[lat]
        p = float(res.pressures[nid])
        pressures.append(p)
        surplus.append(p - Hmin)
    mean_s = sum(surplus) / len(surplus)
    var_s = sum((x - mean_s) ** 2 for x in surplus) / len(surplus)
    return GroupResult(
        laterals=list(laterals),
        ok=bool(res.ok),
        min_margin=float(res.min_margin),
        min_pressure=float(res.min_pressure_head),
        pressures=pressures,
        surplus=surplus,
        group_mean_surplus=mean_s,
        group_std_surplus=math.sqrt(max(var_s, 0.0)),
        repaired=False,
        repair_note="",
    )


@torch.no_grad()
def choose_group_with_repair(
    model: nn.Module,
    x_feat: torch.Tensor,          # [D]
    selected_mask: torch.Tensor,   # [L], 1=selected
    laterals: List[str],
    evaluator: te.TreeHydraulicEvaluator,
    lat2node: Dict[str, str],
    Hmin: float,
    q_lateral: float,
    rng: random.Random,
    top_pool: int,
    top_pool_max: int,
    repair_samples: int,
) -> GroupResult:
    device = next(model.parameters()).device
    x = x_feat.to(device).unsqueeze(0)  # [1,D]
    size_logits, mem_logits = model(x)
    size_logits = size_logits.squeeze(0).cpu()
    mem_logits = mem_logits.squeeze(0).cpu()

    # predicted group size
    k_pred = [2, 3, 4][int(size_logits.argmax().item())]
    remaining_idx = (selected_mask < 0.5).nonzero(as_tuple=False).view(-1)
    remaining = int(remaining_idx.numel())
    k = fix_group_size(k_pred, remaining)

    # score remaining laterals
    scores = mem_logits.clone()
    scores[selected_mask > 0.5] = float("-inf")

    # first attempt: top-k
    topk_idx = torch.topk(scores, k=k).indices.tolist()
    cand_lats = [laterals[i] for i in topk_idx]
    gr = eval_group(evaluator, lat2node, Hmin, q_lateral, cand_lats)
    if gr.ok:
        return gr

    # repair search: enumerate combinations within top-P pool
    pool = top_pool
    tried = 0
    best_feasible: Optional[Tuple[float, List[str], str]] = None

    while pool <= top_pool_max:
        pool_idx = torch.topk(scores, k=min(pool, remaining)).indices.tolist()
        pool_lats = [laterals[i] for i in pool_idx]
        pool_scores = [float(scores[i].item()) for i in pool_idx]

        # try sizes in order: k, then smaller to regain feasibility
        for kk in [k] + [x for x in (4, 3, 2) if x < k]:
            if remaining < kk:
                continue
            # limit enumeration if huge
            combos = itertools.combinations(range(len(pool_lats)), kk)
            # If combos too many, sample a subset
            max_enum = 1200
            all_list = []
            for ci, comb in enumerate(combos):
                if ci >= max_enum:
                    break
                all_list.append(comb)
            if not all_list:
                continue
            # score combos by sum of logits
            all_list.sort(key=lambda comb: sum(pool_scores[i] for i in comb), reverse=True)

            for comb in all_list:
                tried += 1
                lats = [pool_lats[i] for i in comb]
                gr2 = eval_group(evaluator, lat2node, Hmin, q_lateral, lats)
                if gr2.ok:
                    sc = sum(pool_scores[i] for i in comb)
                    best_feasible = (sc, lats, f"repair: pool={pool}, kk={kk}, tried={tried}")
                    break
            if best_feasible is not None:
                break

        if best_feasible is not None:
            sc, lats, note = best_feasible
            gr_ok = eval_group(evaluator, lat2node, Hmin, q_lateral, lats)
            return GroupResult(
                laterals=gr_ok.laterals,
                ok=gr_ok.ok,
                min_margin=gr_ok.min_margin,
                min_pressure=gr_ok.min_pressure,
                pressures=gr_ok.pressures,
                surplus=gr_ok.surplus,
                group_mean_surplus=gr_ok.group_mean_surplus,
                group_std_surplus=gr_ok.group_std_surplus,
                repaired=True,
                repair_note=note,
            )

        pool += 4

    # fallback: random feasible sampling
    rem_lats = [laterals[i] for i in remaining_idx.tolist()]
    for _ in range(repair_samples):
        kk = k
        if len(rem_lats) in (2, 3, 4):
            kk = len(rem_lats)
        else:
            kk = rng.choice([2, 3, 4])
            kk = min(kk, len(rem_lats))
            if len(rem_lats) - kk == 1:
                kk = 2
        lats = rng.sample(rem_lats, kk)
        gr3 = eval_group(evaluator, lat2node, Hmin, q_lateral, lats)
        if gr3.ok:
            return GroupResult(
                laterals=gr3.laterals,
                ok=True,
                min_margin=gr3.min_margin,
                min_pressure=gr3.min_pressure,
                pressures=gr3.pressures,
                surplus=gr3.surplus,
                group_mean_surplus=gr3.group_mean_surplus,
                group_std_surplus=gr3.group_std_surplus,
                repaired=True,
                repair_note=f"fallback-random: kk={kk}",
            )

    # if still failed, return the original infeasible group (caller may abort)
    return GroupResult(
        laterals=cand_lats,
        ok=False,
        min_margin=gr.min_margin,
        min_pressure=gr.min_pressure,
        pressures=gr.pressures,
        surplus=gr.surplus,
        group_mean_surplus=gr.group_mean_surplus,
        group_std_surplus=gr.group_std_surplus,
        repaired=True,
        repair_note="FAILED: no feasible group found",
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Load network + lateral order
    nodes = te.load_nodes_xlsx(args.nodes)
    edges = te.load_pipes_xlsx(args.pipes)
    laterals, lat2node, _node_z = build_lateral_order(args.nodes, args.pipes)
    lat2idx = {lat: i for i, lat in enumerate(laterals)}

    # Load policy
    ckpt = torch.load(args.policy, map_location="cpu")
    if ckpt["laterals"] != laterals:
        raise RuntimeError(
            "Lateral ordering mismatch between policy and current network.\n"
            "Rebuild dataset/train using the same Nodes/Pipes."
        )

    cfg = ckpt["model_cfg"]
    model = GroupPolicyNet(**cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    H0_min, H0_max = float(ckpt["H0_min"]), float(ckpt["H0_max"])
    denom = max(H0_max - H0_min, 1e-9)
    H0_norm = (args.H0 - H0_min) / denom

    evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    L = len(laterals)
    selected = torch.zeros(L, dtype=torch.float32)

    groups: List[GroupResult] = []
    t0 = time.time()

    while True:
        remaining = int((selected < 0.5).sum().item())
        if remaining == 0:
            break

        # step features
        G_so_far = len(groups)
        # we don't know final G; use a proxy based on remaining
        step_frac = 0.0 if G_so_far == 0 else min(1.0, G_so_far / 80.0)
        remaining_frac = remaining / L
        x_feat = torch.cat([torch.tensor([H0_norm, step_frac, remaining_frac], dtype=torch.float32), selected.clone()], dim=0)

        gr = choose_group_with_repair(
            model=model,
            x_feat=x_feat,
            selected_mask=selected,
            laterals=laterals,
            evaluator=evaluator,
            lat2node=lat2node,
            Hmin=args.Hmin,
            q_lateral=args.q_lateral,
            rng=rng,
            top_pool=args.top_pool,
            top_pool_max=args.top_pool_max,
            repair_samples=args.repair_samples,
        )

        if not gr.ok:
            raise RuntimeError(f"Failed to generate a feasible group. Note: {gr.repair_note}")

        for lat in gr.laterals:
            selected[lat2idx[lat]] = 1.0
        groups.append(gr)

    elapsed = time.time() - t0

    # compute group-level objective
    group_means = [g.group_mean_surplus for g in groups]
    mu = sum(group_means) / len(group_means)
    var = sum((x - mu) ** 2 for x in group_means) / len(group_means)
    std = math.sqrt(max(var, 0.0))
    J = 0.5 * mu + 0.5 * std

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    group_rows = []
    lat_rows = []
    for gi, g in enumerate(groups, start=1):
        group_rows.append([
            args.H0, gi, len(g.laterals),
            g.min_pressure, g.min_margin,
            g.group_mean_surplus, g.group_std_surplus,
            int(g.repaired), g.repair_note,
            " ".join(g.laterals),
        ])
        for lat, p, s in zip(g.laterals, g.pressures, g.surplus):
            lat_rows.append([args.H0, gi, lat, lat2node[lat], p, s])

    write_csv(
        out_dir / "group_metrics.csv",
        ["H0", "group_id", "group_size", "min_pressure_head", "min_margin",
         "group_mean_surplus", "group_std_surplus", "repaired", "repair_note", "laterals"],
        group_rows,
    )
    write_csv(
        out_dir / "lateral_metrics.csv",
        ["H0", "group_id", "lateral_id", "node_id", "pressure_head", "surplus_head"],
        lat_rows,
    )

    payload = {
        "H0": args.H0,
        "Hmin": args.Hmin,
        "objective": {
            "definition": "J = 0.5*mu + 0.5*sigma, computed on per-group mean surplus",
            "mu_group_mean": mu,
            "std_group_mean": std,
            "J": J,
            "group_means": group_means,
        },
        "groups": [
            {
                "group_id": gi,
                "laterals": g.laterals,
                "nodes": [lat2node[lat] for lat in g.laterals],
                "pressures": g.pressures,
                "surplus": g.surplus,
                "min_pressure": g.min_pressure,
                "min_margin": g.min_margin,
                "group_mean_surplus": g.group_mean_surplus,
                "group_std_surplus": g.group_std_surplus,
                "repaired": g.repaired,
                "repair_note": g.repair_note,
            }
            for gi, g in enumerate(groups, start=1)
        ],
        "meta": {
            "policy": str(Path(args.policy).resolve()),
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "n_groups": len(groups),
            "n_laterals": len(laterals),
            "device": args.device,
        },
    }
    (out_dir / "schedule.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # quick summary file
    write_csv(
        out_dir / "summary.csv",
        ["H0", "n_groups", "mu_group_mean", "std_group_mean", "J", "elapsed_sec"],
        [[args.H0, len(groups), mu, std, J, elapsed]],
    )

    print(f"Done. Groups={len(groups)}, mu={mu:.46f}, std={std:.6f}, J={J:.6f}, elapsed={elapsed:.2f}s")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
