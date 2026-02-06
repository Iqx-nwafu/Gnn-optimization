
# build_gnn_il_dataset.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError("tree_evaluator.py must be importable (place it next to this script or add to PYTHONPATH).") from e


def _node_num(nid: str) -> Optional[int]:
    nid = str(nid).strip().upper()
    if not nid.startswith("J"):
        return None
    try:
        return int(nid[1:])
    except ValueError:
        return None


def _material_onehot(material: str, vocab: List[str]) -> np.ndarray:
    material = str(material).strip().upper()
    arr = np.zeros((len(vocab),), dtype=np.float32)
    if material in vocab:
        arr[vocab.index(material)] = 1.0
    return arr


def _parse_groups_from_schedule_json(p: Path) -> Tuple[Optional[float], Optional[List[List[str]]], Dict]:
    """
    Supports:
    1) normal schedule: obj["groups"] -> each group has "laterals"
    2) fault schedule: obj["postfault_reopt"]["groups"] -> each has "laterals"
       plus fault_effect.* lists for masks.
    Returns:
      H0, groups, raw_obj
    """
    obj = json.loads(p.read_text(encoding="utf-8"))
    H0 = obj.get("H0", None)

    # normal
    if isinstance(obj.get("groups", None), list):
        groups = []
        for g in obj["groups"]:
            if isinstance(g, dict) and isinstance(g.get("laterals", None), list):
                groups.append([str(x).strip() for x in g["laterals"]])
        if groups:
            return (float(H0) if H0 is not None else None), groups, obj

    # fault-post
    post = obj.get("postfault_reopt", None)
    if isinstance(post, dict) and isinstance(post.get("groups", None), list):
        groups = []
        for g in post["groups"]:
            if isinstance(g, dict) and isinstance(g.get("laterals", None), list):
                groups.append([str(x).strip() for x in g["laterals"]])
        if groups:
            return (float(H0) if H0 is not None else None), groups, obj

    return None, None, obj


def _collect_schedule_jsons(opt_root: Path, fault_root: Optional[Path]) -> List[Path]:
    paths: List[Path] = []
    if opt_root and opt_root.exists():
        paths += list(opt_root.rglob("schedule.json"))
    if fault_root and fault_root.exists():
        paths += list(fault_root.rglob("schedule.json"))
    # stable ordering
    paths = sorted(paths, key=lambda x: str(x).lower())
    return paths


def build_static_graph(
    nodes_path: str,
    pipes_path: str,
    root: str,
    Hmin: float,
) -> Tuple[Dict, Dict[str, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Returns:
      meta_dict, node_id_to_idx, node_x, edge_index, edge_attr, lateral_to_node_idx, side_idx, node_ids, lateral_ids
    """
    nodes = te.load_nodes_xlsx(nodes_path)
    edges = te.load_pipes_xlsx(pipes_path)

    # Build evaluator once (topology only)
    evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=root, H0=25.0, Hmin=Hmin)

    node_ids = sorted(nodes.keys(), key=lambda x: (_node_num(x) if _node_num(x) is not None else 10**9, x))
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # node features: [z, depth, subtree_size, is_field, is_mainline]
    z = np.array([nodes[nid].z for nid in node_ids], dtype=np.float32)
    z_mean, z_std = float(z.mean()), float(z.std() + 1e-6)
    z_norm = (z - z_mean) / z_std

    # compute depth and subtree sizes from evaluator children
    depth = np.zeros((len(node_ids),), dtype=np.float32)
    subtree = np.ones((len(node_ids),), dtype=np.float32)

    # BFS depth
    stack = [(root, 0)]
    visited = set()
    while stack:
        u, d = stack.pop()
        visited.add(u)
        depth[node_id_to_idx[u]] = float(d)
        for c in evaluator.children[u]:
            stack.append((c, d + 1))

    # subtree size (postorder)
    def dfs_size(u: str) -> int:
        s = 1
        for c in evaluator.children[u]:
            s += dfs_size(c)
        subtree[node_id_to_idx[u]] = float(s)
        return s

    dfs_size(root)

    depth_norm = (depth - depth.mean()) / (depth.std() + 1e-6)
    subtree_norm = (subtree - subtree.mean()) / (subtree.std() + 1e-6)

    is_field = np.array([1.0 if te.is_field_node_id(nid) else 0.0 for nid in node_ids], dtype=np.float32)
    is_mainline = np.array([0.0 if te.is_field_node_id(nid) else 1.0 for nid in node_ids], dtype=np.float32)

    node_x = np.stack([z_norm, depth_norm, subtree_norm, is_field, is_mainline], axis=1)

    # laterals
    field_nodes = [nid for nid in node_ids if te.is_field_node_id(nid)]
    lateral_ids, lateral_to_node = te.build_lateral_ids_for_field_nodes(field_nodes)
    lateral_ids = sorted(lateral_ids)
    lateral_to_node_idx = np.array([node_id_to_idx[lateral_to_node[lid]] for lid in lateral_ids], dtype=np.int64)
    side_idx = np.array([0 if lid.endswith("_L") else 1 for lid in lateral_ids], dtype=np.int64)

    # edge features with direction flags (rooted tree)
    materials = sorted({e.material.strip().upper() for e in edges})
    # undirected edges in xlsx are stored once. we create BOTH directions.
    edge_src = []
    edge_dst = []
    edge_attr = []

    for e in edges:
        u = e.u
        v = e.v

        # Determine parent->child direction using evaluator.parent
        # exactly one of these should be true for a tree:
        u_is_parent = (evaluator.parent.get(v) == u)
        v_is_parent = (evaluator.parent.get(u) == v)

        # base attrs
        base = np.array([float(e.L), float(e.D)], dtype=np.float32)
        mat_oh = _material_onehot(e.material, materials)

        def add_dir(a: str, b: str, dir_flag: float):
            edge_src.append(node_id_to_idx[a])
            edge_dst.append(node_id_to_idx[b])
            edge_attr.append(np.concatenate([base, mat_oh, np.array([dir_flag], dtype=np.float32)], axis=0))

        if u_is_parent:
            add_dir(u, v, +1.0)  # downstream
            add_dir(v, u, -1.0)  # upstream
        elif v_is_parent:
            add_dir(v, u, +1.0)
            add_dir(u, v, -1.0)
        else:
            # fallback (shouldn't happen if inputs are consistent tree)
            add_dir(u, v, 0.0)
            add_dir(v, u, 0.0)

    edge_index = np.stack([np.array(edge_src, dtype=np.int64), np.array(edge_dst, dtype=np.int64)], axis=0)
    edge_attr = np.stack(edge_attr, axis=0).astype(np.float32)

    meta = {
        "root": root,
        "node_features": ["z_norm", "depth_norm", "subtree_norm", "is_field", "is_mainline"],
        "edge_features": ["L_m", "D_m"] + [f"mat_{m}" for m in materials] + ["dir_flag(+1 down, -1 up)"],
        "materials_vocab": materials,
        "counts": {"N": len(node_ids), "E_undirected": len(edges), "E_directed": edge_index.shape[1], "L": len(lateral_ids)},
        "notes": {
            "target_is_multihot": "target is multi-hot vector over laterals (exactly 2-4 ones).",
            "fault_samples_are_post_only": "fault scenarios contribute only post-fault rescheduling steps.",
        },
    }

    return (
        meta,
        node_id_to_idx,
        torch.from_numpy(node_x),
        torch.from_numpy(edge_index),
        torch.from_numpy(edge_attr),
        torch.from_numpy(lateral_to_node_idx),
        torch.from_numpy(side_idx),
        node_ids,
        lateral_ids,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--pipes", required=True)
    ap.add_argument("--root", default="J0")
    ap.add_argument("--Hmin", type=float, default=11.59)

    ap.add_argument("--opt_root", required=True, help="Root folder of offline optimal schedules (runs_opt_groupstats).")
    ap.add_argument("--fault_root", default="", help="Optional root folder of fault scenarios (each contains schedule.json).")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_json", type=int, default=0, help="Cap # schedule.json processed (0 = all).")
    args = ap.parse_args()

    opt_root = Path(args.opt_root)
    fault_root = Path(args.fault_root) if args.fault_root.strip() else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, node_id_to_idx, node_x, edge_index, edge_attr, lateral_to_node_idx, side_idx, node_ids, lateral_ids = build_static_graph(
        nodes_path=args.nodes, pipes_path=args.pipes, root=args.root, Hmin=args.Hmin
    )

    lateral_id_to_idx = {lid: i for i, lid in enumerate(lateral_ids)}

    schedule_paths = _collect_schedule_jsons(opt_root, fault_root)
    if args.max_json and args.max_json > 0:
        schedule_paths = schedule_paths[: args.max_json]

    H0_list: List[float] = []
    irrigated_list: List[np.ndarray] = []
    reachable_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    target_size_list: List[int] = []
    scenario_type_list: List[int] = []
    fault_node_idx_list: List[int] = []
    step_idx_list: List[int] = []
    source_list: List[str] = []

    skipped = 0
    kept = 0

    for p in schedule_paths:
        H0, groups, obj = _parse_groups_from_schedule_json(p)
        if H0 is None or groups is None:
            skipped += 1
            continue

        # Determine whether this is a fault schedule (postfault groups)
        is_fault = ("postfault_reopt" in obj) and isinstance(obj.get("postfault_reopt"), dict)

        # Build initial masks
        L = len(lateral_ids)
        irrigated = np.zeros((L,), dtype=np.uint8)
        reachable = np.ones((L,), dtype=np.uint8)

        fault_node_idx = -1
        step0 = 0

        if is_fault:
            fe = obj.get("fault_effect", {}) if isinstance(obj.get("fault_effect"), dict) else {}
            irrig_before = fe.get("irrigated_before_fault", []) if isinstance(fe.get("irrigated_before_fault", []), list) else []
            blocked = fe.get("blocked_laterals", []) if isinstance(fe.get("blocked_laterals", []), list) else []

            # masks from JSON lists
            for lid in irrig_before:
                lid = str(lid).strip()
                if lid in lateral_id_to_idx:
                    irrigated[lateral_id_to_idx[lid]] = 1
            for lid in blocked:
                lid = str(lid).strip()
                if lid in lateral_id_to_idx:
                    reachable[lateral_id_to_idx[lid]] = 0

            m = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
            fn = m.get("fault_node", None)
            if fn is not None and str(fn).strip() in node_id_to_idx:
                fault_node_idx = node_id_to_idx[str(fn).strip()]

            # step0 = 0 for postfault sequence
            step0 = 0

        # Build per-step samples
        for t, g in enumerate(groups):
            # target multi-hot
            tgt = np.zeros((L,), dtype=np.uint8)
            valid = True
            for lid in g:
                lid = str(lid).strip()
                if lid not in lateral_id_to_idx:
                    valid = False
                    break
                li = lateral_id_to_idx[lid]
                # if already irrigated or unreachable, this group is inconsistent with masks; skip whole schedule
                if irrigated[li] == 1 or reachable[li] == 0:
                    valid = False
                    break
                tgt[li] = 1

            if not valid:
                valid = False
                break

            size = int(tgt.sum())
            if size < 2 or size > 4:
                valid = False
                break

            # append sample
            H0_list.append(float(H0))
            irrigated_list.append(irrigated.copy())
            reachable_list.append(reachable.copy())
            target_list.append(tgt)
            target_size_list.append(size)
            scenario_type_list.append(1 if is_fault else 0)
            fault_node_idx_list.append(fault_node_idx)
            step_idx_list.append(step0 + t)
            source_list.append(str(p))

            # update irrigated with teacher action
            irrigated = np.maximum(irrigated, tgt).astype(np.uint8)

        if not valid:
            skipped += 1
            continue

        kept += 1

    # Save
    meta_out = {
        **meta,
        "node_ids": node_ids,
        "lateral_ids": lateral_ids,
        "id_notes": {
            "lateral_ids": "Two laterals per field node: <node>_L, <node>_R (from tree_evaluator.build_lateral_ids_for_field_nodes).",
            "fault_node_idx": "index into node_ids; -1 means normal sample.",
        },
        "build_stats": {"schedule_json_found": len(schedule_paths), "kept": kept, "skipped": skipped, "samples": len(H0_list)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save(
        {
            "node_x": node_x.contiguous(),
            "edge_index": edge_index.contiguous(),
            "edge_attr": edge_attr.contiguous(),
            "lateral_to_node_idx": lateral_to_node_idx.contiguous(),
            "side_idx": side_idx.contiguous(),
        },
        out_dir / "graph_tensors.pt",
    )

    # stack samples (may be large)
    H0_t = torch.tensor(np.array(H0_list, dtype=np.float32))
    irrig_t = torch.from_numpy(np.stack(irrigated_list, axis=0)).to(torch.uint8)
    reach_t = torch.from_numpy(np.stack(reachable_list, axis=0)).to(torch.uint8)
    tgt_t = torch.from_numpy(np.stack(target_list, axis=0)).to(torch.uint8)
    size_t = torch.tensor(np.array(target_size_list, dtype=np.int64))
    scen_t = torch.tensor(np.array(scenario_type_list, dtype=np.uint8))
    fn_t = torch.tensor(np.array(fault_node_idx_list, dtype=np.int64))
    step_t = torch.tensor(np.array(step_idx_list, dtype=np.int64))

    # `source_list` is a python list (kept outside tensor for readability)
    torch.save(
        {
            "H0": H0_t,
            "irrigated": irrig_t,
            "reachable": reach_t,
            "target": tgt_t,
            "target_size": size_t,
            "scenario_type": scen_t,
            "fault_node_idx": fn_t,
            "step_idx": step_t,
            "source": source_list,
        },
        out_dir / "samples.pt",
    )

    print(f"[DONE] schedules found={len(schedule_paths)}, kept={kept}, skipped={skipped}, samples={len(H0_list)}")
    print(f"       wrote: {out_dir/'meta.json'}")
    print(f"              {out_dir/'graph_tensors.pt'}")
    print(f"              {out_dir/'samples.pt'}")


if __name__ == "__main__":
    main()
