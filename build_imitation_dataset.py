# build_imitation_dataset.py
"""
Build imitation-learning dataset from near-optimal schedules.

Input:
- Nodes.xlsx / Pipes.xlsx (for consistent lateral ordering)
- A folder produced by rotation_optimize_groupstats.py, containing subfolders:
    H0_15.00/schedule.json, H0_16.00/schedule.json, ...

Output:
- dataset.pt (torch.save) with fields:
    laterals: list[str] (ordering)
    H0_min, H0_max
    X: float32 tensor [N, 3+L]   (H0_norm, step_frac, remaining_frac, mask_selected[L])
    y_size: int64 tensor [N]     (0->2, 1->3, 2->4)
    y_membership: float32 tensor [N, L] (multi-hot for the *next group*)
    meta: dict

Notes:
- We treat each schedule as an ordered sequence of groups (as emitted by the optimizer).
- For each step t, the input mask indicates which laterals have already been irrigated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError(
        "Cannot import tree_evaluator.py. Put this script in the same folder as tree_evaluator.py, "
        "or add that folder to PYTHONPATH."
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True)
    p.add_argument("--pipes", required=True)
    p.add_argument("--opt_root", required=True, help="Root folder containing H0_*/schedule.json")
    p.add_argument("--out", required=True, help="Output dataset path, e.g., dataset.pt")
    p.add_argument("--holdout_H0", nargs="*", type=float, default=[], help="H0 values to hold out (not included)")
    return p.parse_args()


def build_lateral_order(nodes_path: str, pipes_path: str) -> Tuple[List[str], Dict[str, str]]:
    nodes = te.load_nodes_xlsx(nodes_path)
    _edges = te.load_pipes_xlsx(pipes_path)
    field_nodes = sorted([nid for nid in nodes.keys() if te.is_field_node_id(nid)])
    lateral_ids, lateral_to_node = te.build_lateral_ids_for_field_nodes(field_nodes)
    lateral_ids = sorted(lateral_ids)  # stable
    return lateral_ids, lateral_to_node


def main() -> None:
    args = parse_args()

    laterals, _lat2node = build_lateral_order(args.nodes, args.pipes)
    L = len(laterals)
    lat2idx = {lat: i for i, lat in enumerate(laterals)}

    opt_root = Path(args.opt_root)
    schedule_paths = sorted(opt_root.glob("H0_*/*schedule.json"))
    if not schedule_paths:
        # also accept .../H0_xx/schedule.json
        schedule_paths = sorted(opt_root.glob("H0_*/schedule.json"))

    records_X = []
    records_size = []
    records_mem = []
    used_H0 = []

    # collect H0 list first
    H0_values = []
    for sp in schedule_paths:
        data = json.loads(sp.read_text(encoding="utf-8"))
        H0 = float(data["H0"])
        if any(abs(H0 - h) < 1e-9 for h in args.holdout_H0):
            continue
        H0_values.append(H0)

    if not H0_values:
        raise RuntimeError("No schedules found after applying holdout_H0.")

    H0_min, H0_max = min(H0_values), max(H0_values)
    denom = max(H0_max - H0_min, 1e-9)

    for sp in schedule_paths:
        data = json.loads(sp.read_text(encoding="utf-8"))
        H0 = float(data["H0"])
        if any(abs(H0 - h) < 1e-9 for h in args.holdout_H0):
            continue

        groups = data["groups"]
        G = len(groups)
        selected = [0.0] * L

        for t, g in enumerate(groups):
            g_lats: List[str] = g["laterals"]
            k = len(g_lats)
            if k not in (2, 3, 4):
                raise ValueError(f"Unexpected group size {k} in {sp}")

            # input features
            H0_norm = (H0 - H0_min) / denom
            step_frac = 0.0 if G <= 1 else (t / (G - 1))
            remaining_cnt = L - int(sum(selected))
            remaining_frac = remaining_cnt / L

            x = [H0_norm, step_frac, remaining_frac] + selected[:]  # 3+L
            y_size = {2: 0, 3: 1, 4: 2}[k]
            y_mem = [0.0] * L
            for lat in g_lats:
                if lat not in lat2idx:
                    raise KeyError(f"Unknown lateral id {lat} (ordering mismatch).")
                y_mem[lat2idx[lat]] = 1.0

            records_X.append(x)
            records_size.append(y_size)
            records_mem.append(y_mem)
            used_H0.append(H0)

            # update selected
            for lat in g_lats:
                selected[lat2idx[lat]] = 1.0

    X = torch.tensor(records_X, dtype=torch.float32)
    y_size = torch.tensor(records_size, dtype=torch.int64)
    y_mem = torch.tensor(records_mem, dtype=torch.float32)

    payload = {
        "laterals": laterals,
        "H0_min": float(H0_min),
        "H0_max": float(H0_max),
        "X": X,
        "y_size": y_size,
        "y_membership": y_mem,
        "meta": {
            "opt_root": str(opt_root),
            "n_samples": int(X.shape[0]),
            "n_laterals": int(L),
            "holdout_H0": [float(x) for x in args.holdout_H0],
            "used_H0_unique": sorted(set(float(x) for x in used_H0)),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved dataset to: {out_path}")
    print(f"X: {tuple(X.shape)}, y_size: {tuple(y_size.shape)}, y_membership: {tuple(y_mem.shape)}")
    print(f"H0 range: [{H0_min}, {H0_max}]  | unique H0: {len(set(used_H0))}")


if __name__ == "__main__":
    main()
