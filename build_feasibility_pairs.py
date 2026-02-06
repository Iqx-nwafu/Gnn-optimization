
# build_feasibility_pairs.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError("tree_evaluator.py must be importable (place it next to this script or add to PYTHONPATH).") from e


def sample_random_group(
    rng: np.random.Generator,
    available: np.ndarray,
    size: int,
) -> np.ndarray:
    """available: 1 where lateral is selectable (reachable=1 and irrigated=0)."""
    idx = np.flatnonzero(available)
    if idx.size < size:
        return np.array([], dtype=np.int64)
    pick = rng.choice(idx, size=size, replace=False)
    return np.sort(pick).astype(np.int64)


def pad_group(g: np.ndarray, pad_to: int = 4) -> np.ndarray:
    out = -np.ones((pad_to,), dtype=np.int64)
    out[: g.size] = g
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--pipes", required=True)
    ap.add_argument("--root", default="J0")
    ap.add_argument("--Hmin", type=float, default=11.59)
    ap.add_argument("--q_lateral", type=float, default=0.012)

    ap.add_argument("--neg_per_pos", type=int, default=8)
    ap.add_argument("--max_states", type=int, default=0, help="cap #states used (0=all)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    graph = torch.load(ddir / "graph_tensors.pt", map_location="cpu")
    samples = torch.load(ddir / "samples.pt", map_location="cpu")

    lateral_ids: List[str] = []
    # meta.json is optional for this script, but helpful for mapping
    meta_path = ddir / "meta.json"
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lateral_ids = meta["lateral_ids"]

    L = int(graph["lateral_to_node_idx"].shape[0])
    if not lateral_ids:
        lateral_ids = [f"lat_{i}" for i in range(L)]

    lateral_to_node = {}
    # reconstruct lateral_to_node using nodes.xlsx + helper (ensures correct)
    nodes = te.load_nodes_xlsx(args.nodes)
    field_nodes = [nid for nid in nodes.keys() if te.is_field_node_id(nid)]
    lat_ids_true, lat_to_node_true = te.build_lateral_ids_for_field_nodes(field_nodes)
    # align to dataset ordering by name
    lat_to_node_true = {k: v for k, v in lat_to_node_true.items()}
    for lid in lateral_ids:
        if lid not in lat_to_node_true:
            raise KeyError(f"Dataset lateral id {lid} not found in build_lateral_ids_for_field_nodes mapping.")
        lateral_to_node[lid] = lat_to_node_true[lid]

    # group states by H0 to reuse evaluator
    H0 = samples["H0"].numpy().astype(np.float32)
    irrig = samples["irrigated"].numpy().astype(np.uint8)
    reach = samples["reachable"].numpy().astype(np.uint8)
    tgt = samples["target"].numpy().astype(np.uint8)

    S = H0.shape[0]
    max_states = args.max_states if args.max_states and args.max_states > 0 else S

    # deterministic subsample if cap is set
    rng = np.random.default_rng(args.seed)
    state_indices = np.arange(S, dtype=np.int64)
    if max_states < S:
        state_indices = rng.choice(state_indices, size=max_states, replace=False)
        state_indices = np.sort(state_indices)

    # create evaluators per unique H0 in used states
    edges = te.load_pipes_xlsx(args.pipes)

    # prepare outputs
    state_idx_out: List[int] = []
    cand_pad_out: List[np.ndarray] = []
    cand_size_out: List[int] = []
    label_ok_out: List[int] = []
    min_margin_out: List[float] = []

    # simple cache per H0 (key=tuple(lateral_ids))
    cache_per_H0: Dict[float, Dict[Tuple[str, ...], Tuple[bool, float]]] = {}

    def eval_group_cached(ev: te.TreeHydraulicEvaluator, H0v: float, group_lids: List[str]) -> Tuple[bool, float]:
        key = tuple(sorted(group_lids))
        cache = cache_per_H0.setdefault(float(H0v), {})
        if key in cache:
            return cache[key]
        res = ev.evaluate_group(group_lids, lateral_to_node=lateral_to_node, q_lateral=args.q_lateral)
        out = (bool(res.ok), float(res.min_margin))
        cache[key] = out
        return out

    # build evaluators for H0 values on the fly
    evaluator_cache: Dict[float, te.TreeHydraulicEvaluator] = {}

    def get_eval(H0v: float) -> te.TreeHydraulicEvaluator:
        if float(H0v) in evaluator_cache:
            return evaluator_cache[float(H0v)]
        ev = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=float(H0v), Hmin=args.Hmin)
        evaluator_cache[float(H0v)] = ev
        return ev

    sizes = np.array([2, 3, 4], dtype=np.int64)

    for si in state_indices:
        H0v = float(H0[si])
        ev = get_eval(H0v)

        available = (reach[si] == 1) & (irrig[si] == 0)

        # Positive = teacher target
        pos_idx = np.flatnonzero(tgt[si] == 1).astype(np.int64)
        if pos_idx.size < 2 or pos_idx.size > 4:
            continue

        pos_lids = [lateral_ids[i] for i in pos_idx.tolist()]
        ok, mm = eval_group_cached(ev, H0v, pos_lids)

        state_idx_out.append(int(si))
        cand_pad_out.append(pad_group(pos_idx))
        cand_size_out.append(int(pos_idx.size))
        label_ok_out.append(1 if ok else 0)
        min_margin_out.append(float(mm))

        # Negatives: random candidates
        for _ in range(args.neg_per_pos):
            size = int(rng.choice(sizes))
            g_idx = sample_random_group(rng, available.astype(np.uint8), size)
            if g_idx.size != size:
                continue
            g_lids = [lateral_ids[i] for i in g_idx.tolist()]
            ok2, mm2 = eval_group_cached(ev, H0v, g_lids)

            state_idx_out.append(int(si))
            cand_pad_out.append(pad_group(g_idx))
            cand_size_out.append(int(size))
            label_ok_out.append(1 if ok2 else 0)
            min_margin_out.append(float(mm2))

    out = {
        "state_idx": torch.tensor(np.array(state_idx_out, dtype=np.int64)),
        "cand_pad": torch.tensor(np.stack(cand_pad_out, axis=0).astype(np.int64)),
        "cand_size": torch.tensor(np.array(cand_size_out, dtype=np.int64)),
        "label_ok": torch.tensor(np.array(label_ok_out, dtype=np.uint8)),
        "min_margin": torch.tensor(np.array(min_margin_out, dtype=np.float32)),
    }
    torch.save(out, ddir / "feas_pairs.pt")
    print(f"[DONE] states_used={len(state_indices)} pairs={out['state_idx'].shape[0]}")
    print(f"       wrote: {ddir/'feas_pairs.pt'}")


if __name__ == "__main__":
    main()
