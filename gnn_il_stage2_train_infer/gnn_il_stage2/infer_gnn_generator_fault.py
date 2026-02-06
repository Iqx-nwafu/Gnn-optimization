
# infer_gnn_generator_fault.py
# Extend Stage2 inference to support: H0 + fault_node + fault_time (after N groups), auto reachable_mask, continue scheduling.
from __future__ import annotations

import argparse, csv, json, itertools, random
from pathlib import Path
from typing import List, Optional, Dict, Set

import numpy as np
import torch

from model_gnn import GroupGenerator, FeasibilityClassifier
from utils_data import load_graph

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError("tree_evaluator.py must be importable for evaluator hard-check during inference.") from e


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--generator_ckpt", required=True)
    ap.add_argument("--feas_ckpt", default="")

    ap.add_argument("--nodes", required=True)
    ap.add_argument("--pipes", required=True)
    ap.add_argument("--root", default="J0")
    ap.add_argument("--H0", type=float, required=True)
    ap.add_argument("--Hmin", type=float, default=11.59)
    ap.add_argument("--q_lateral", type=float, default=0.012)

    ap.add_argument("--out", required=True)

    # decoding
    ap.add_argument("--top_k", type=int, default=14)
    ap.add_argument("--max_expand", type=int, default=24)
    ap.add_argument("--feas_threshold", type=float, default=0.35)
    ap.add_argument("--max_random_tries", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)

    # fault controls
    ap.add_argument("--fault_node", default="", help="e.g., J12. empty means no fault.")
    ap.add_argument("--fault_after_groups", type=int, default=-1, help="trigger fault AFTER executing this many groups. -1 means no fault.")
    ap.add_argument("--prefault_force_k", type=int, default=0, help="0=use predicted size; 4/3/2=force group size BEFORE fault (engineering scenario).")

    return ap.parse_args()


def write_csv(path: Path, header: List[str], rows: List[List[object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def descendants(evaluator: "te.TreeHydraulicEvaluator", fault_node: str) -> Set[str]:
    """Return set of nodes in subtree rooted at fault_node (including fault_node)."""
    fault_node = str(fault_node).strip()
    out: Set[str] = set()
    stack = [fault_node]
    while stack:
        u = stack.pop()
        if u in out:
            continue
        out.add(u)
        for c in evaluator.children.get(u, []):
            stack.append(c)
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.dataset_dir) / "meta.json").read_text(encoding="utf-8"))
    lateral_ids: List[str] = meta["lateral_ids"]

    # Build lateral_to_node mapping (from Excel, ground truth)
    nodes = te.load_nodes_xlsx(args.nodes)
    edges = te.load_pipes_xlsx(args.pipes)
    field_nodes = [nid for nid in nodes.keys() if te.is_field_node_id(nid)]
    _, lat_to_node_true = te.build_lateral_ids_for_field_nodes(field_nodes)
    lateral_to_node = {lid: lat_to_node_true[lid] for lid in lateral_ids}

    evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    device = torch.device(args.device)
    graph = load_graph(args.dataset_dir, device=str(device))

    # generator
    gckpt = torch.load(args.generator_ckpt, map_location=device)
    gcfg = gckpt["config"]
    generator = GroupGenerator(
        node_in_dim=gcfg["node_in_dim"],
        edge_attr_dim=gcfg["edge_attr_dim"],
        hidden_dim=gcfg["hidden"],
        num_layers=gcfg["layers"],
        dropout=gcfg["dropout"]
    ).to(device)
    generator.load_state_dict(gckpt["state_dict"])
    generator.eval()

    # feasibility gating (optional)
    feas = None
    if args.feas_ckpt.strip():
        fckpt = torch.load(args.feas_ckpt, map_location=device)
        fcfg = fckpt["config"]
        feas = FeasibilityClassifier(
            node_in_dim=fcfg["node_in_dim"],
            edge_attr_dim=fcfg["edge_attr_dim"],
            hidden_dim=fcfg["hidden"],
            num_layers=fcfg["layers"],
            dropout=fcfg["dropout"]
        ).to(device)
        feas.load_state_dict(fckpt["state_dict"])
        feas.eval()

    L = len(lateral_ids)
    irrigated = torch.zeros((1, L), dtype=torch.uint8, device=device)
    reachable = torch.ones((1, L), dtype=torch.uint8, device=device)
    H0 = torch.tensor([args.H0], dtype=torch.float32, device=device)

    groups: List[Dict] = []
    group_rows: List[List[object]] = []
    lat_rows: List[List[object]] = []

    # fault bookkeeping
    fault_triggered = False
    fault_effect = {
        "fault_node": None,
        "fault_after_groups": None,
        "blocked_nodes": [],
        "blocked_laterals": [],
        "lost_laterals": [],
        "irrigated_before_fault": [],
    }

    def eval_group(lids: List[str]):
        res = evaluator.evaluate_group(lids, lateral_to_node=lateral_to_node, q_lateral=args.q_lateral)
        pressures = {lid: float(res.pressures[lateral_to_node[lid]]) for lid in lids}
        return bool(res.ok), float(res.min_margin), float(res.min_pressure_head), pressures

    def trigger_fault():
        nonlocal fault_triggered, reachable
        if fault_triggered:
            return
        fnode = str(args.fault_node).strip()
        if not fnode:
            return
        blocked_nodes = descendants(evaluator, fnode)
        blocked_laterals = [lid for lid in lateral_ids if lateral_to_node[lid] in blocked_nodes]

        irr_idx = torch.nonzero(irrigated.squeeze(0) == 1).view(-1).detach().cpu().numpy().tolist()
        irr_lids = [lateral_ids[i] for i in irr_idx]

        # set reachable=0 for blocked laterals
        for lid in blocked_laterals:
            i = lateral_ids.index(lid)
            reachable[0, i] = 0

        lost = []
        for lid in blocked_laterals:
            i = lateral_ids.index(lid)
            if irrigated[0, i].item() == 0:
                lost.append(lid)

        fault_effect.update({
            "fault_node": fnode,
            "fault_after_groups": int(args.fault_after_groups),
            "blocked_nodes": sorted(list(blocked_nodes)),
            "blocked_laterals": blocked_laterals,
            "lost_laterals": lost,
            "irrigated_before_fault": irr_lids,
        })
        fault_triggered = True

    while True:
        avail = (reachable.bool() & (~irrigated.bool())).squeeze(0)
        remain = int(avail.sum().item())
        if remain == 0 or remain < 2:
            break

        # trigger fault after executing N groups
        if (not fault_triggered) and args.fault_node.strip() and args.fault_after_groups >= 0:
            if len(groups) == int(args.fault_after_groups):
                trigger_fault()

        with torch.no_grad():
            tok_logits, size_logits = generator(
                graph.node_x, graph.edge_index, graph.edge_attr,
                graph.lateral_to_node_idx, graph.side_idx,
                H0, irrigated, reachable
            )
            tok_logits = tok_logits.squeeze(0)
            size_logits = size_logits.squeeze(0)

        # group size decision
        if (not fault_triggered) and args.prefault_force_k in (2, 3, 4):
            k = int(args.prefault_force_k)
        else:
            k = int(size_logits.argmax().item()) + 2

        if remain < k:
            k = min(4, remain)
            if k < 2:
                k = 2

        masked = tok_logits.clone()
        masked[~avail] = -1e9

        selected: Optional[List[int]] = None
        selected_feas_prob: Optional[float] = None

        for K in [min(args.top_k, remain), min(args.max_expand, remain)]:
            top_idx = torch.topk(masked, k=K).indices.detach().cpu().numpy().tolist()
            combos = list(itertools.combinations(top_idx, k))
            scored = [(float(masked[list(c)].sum().item()), c) for c in combos]
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:min(800, len(scored))]

            trial = [(c, None) for _, c in scored]

            if feas is not None:
                cand_pad = []
                cand_size = []
                for _, c in scored:
                    pad = [-1, -1, -1, -1]
                    for ii, v in enumerate(c):
                        pad[ii] = int(v)
                    cand_pad.append(pad)
                    cand_size.append(k)
                cand_pad_t = torch.tensor(cand_pad, dtype=torch.int64, device=device)
                cand_size_t = torch.tensor(cand_size, dtype=torch.int64, device=device)
                with torch.no_grad():
                    flog = feas(
                        graph.node_x, graph.edge_index, graph.edge_attr,
                        graph.lateral_to_node_idx, graph.side_idx,
                        H0.expand(cand_pad_t.size(0)),
                        irrigated.expand(cand_pad_t.size(0), -1),
                        reachable.expand(cand_pad_t.size(0), -1),
                        cand_pad_t, cand_size_t
                    )
                    fprob = torch.sigmoid(flog).detach().cpu().numpy().tolist()

                trial2 = []
                for (_, c), fp in zip(scored, fprob):
                    if fp >= args.feas_threshold:
                        trial2.append((c, float(fp)))
                if trial2:
                    trial = trial2

            for c, fp in trial:
                lids = [lateral_ids[i] for i in c]
                ok, mm, mp, pressures = eval_group(lids)
                if ok:
                    selected = list(c)
                    selected_feas_prob = (float(fp) if fp is not None else None)
                    break

            if selected is not None:
                break

        if selected is None:
            avail_idx = np.flatnonzero(avail.detach().cpu().numpy()).tolist()
            for _ in range(args.max_random_tries):
                c = rng.sample(avail_idx, k)
                lids = [lateral_ids[i] for i in c]
                ok, mm, mp, pressures = eval_group(lids)
                if ok:
                    selected = c
                    selected_feas_prob = None
                    break

        if selected is None:
            break

        lids = [lateral_ids[i] for i in selected]
        ok, mm, mp, pressures = eval_group(lids)
        assert ok

        stage = "postfault" if fault_triggered else "prefault"
        gid = len(groups) + 1
        surplus = [pressures[lid] - args.Hmin for lid in lids]
        gmean = float(np.mean(surplus))
        gstd = float(np.std(surplus))

        groups.append({
            "group_id": gid,
            "stage": stage,
            "laterals": lids,
            "size": len(lids),
            "min_pressure_head": mp,
            "min_margin": mm,
            "group_mean_surplus": gmean,
            "group_std_surplus": gstd,
            "feas_prob": selected_feas_prob,
        })

        group_rows.append([args.H0, gid, stage, len(lids), mp, mm, gmean, gstd, selected_feas_prob if selected_feas_prob is not None else "", " ".join(lids)])

        for lid in lids:
            lat_rows.append([args.H0, gid, stage, lid, lateral_to_node[lid], pressures[lid], pressures[lid] - args.Hmin])

        for i in selected:
            irrigated[0, i] = 1

    # objective over group means
    group_means_all = [g["group_mean_surplus"] for g in groups]
    mu_all = float(np.mean(group_means_all)) if group_means_all else float("nan")
    std_all = float(np.std(group_means_all)) if group_means_all else float("nan")
    J_all = 0.5 * mu_all + 0.5 * std_all if group_means_all else float("nan")

    post_means = [g["group_mean_surplus"] for g in groups if g["stage"] == "postfault"]
    mu_post = float(np.mean(post_means)) if post_means else float("nan")
    std_post = float(np.std(post_means)) if post_means else float("nan")
    J_post = 0.5 * mu_post + 0.5 * std_post if post_means else float("nan")

    write_csv(out_dir / "group_metrics.csv",
              ["H0", "group_id", "stage", "group_size", "min_pressure_head", "min_margin", "group_mean_surplus", "group_std_surplus", "feas_prob", "laterals"],
              group_rows)

    write_csv(out_dir / "lateral_metrics.csv",
              ["H0", "group_id", "stage", "lateral_id", "node_id", "pressure_head", "surplus_head"],
              lat_rows)

    payload = {
        "H0": args.H0,
        "Hmin": args.Hmin,
        "global": {
            "n_groups": len(groups),
            "mu_group_mean": mu_all,
            "std_group_mean": std_all,
            "J": J_all,
            "postfault": {"mu_group_mean": mu_post, "std_group_mean": std_post, "J": J_post, "n_groups": int(len(post_means))},
        },
        "fault": (fault_effect if fault_triggered else None),
        "groups": groups,
        "decode": {
            "top_k": args.top_k,
            "max_expand": args.max_expand,
            "feas_used": bool(args.feas_ckpt.strip()),
            "feas_threshold": args.feas_threshold,
            "max_random_tries": args.max_random_tries,
            "prefault_force_k": args.prefault_force_k,
        }
    }

    (out_dir / "schedule.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload["global"], ensure_ascii=False, indent=2), encoding="utf-8")

    if fault_triggered:
        print(f"[FAULT] node={fault_effect['fault_node']} after_groups={fault_effect['fault_after_groups']} "
              f"blocked_laterals={len(fault_effect['blocked_laterals'])} lost={len(fault_effect['lost_laterals'])}")

    print(f"[DONE] groups={len(groups)}  J_all={J_all:.4f}  J_post={J_post:.4f}")
    print(f"       wrote: {out_dir/'schedule.json'}")


if __name__ == "__main__":
    main()
