
# infer_gnn_generator.py
from __future__ import annotations

import argparse, csv, json, itertools, random
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
    ap.add_argument("--top_k", type=int, default=14)
    ap.add_argument("--max_expand", type=int, default=24)
    ap.add_argument("--feas_threshold", type=float, default=0.35)
    ap.add_argument("--max_random_tries", type=int, default=800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def write_csv(path: Path, header: List[str], rows: List[List[object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(args.dataset_dir)/"meta.json").read_text(encoding="utf-8"))
    lateral_ids: List[str] = meta["lateral_ids"]

    nodes = te.load_nodes_xlsx(args.nodes)
    edges = te.load_pipes_xlsx(args.pipes)
    field_nodes = [nid for nid in nodes.keys() if te.is_field_node_id(nid)]
    _, lat_to_node_true = te.build_lateral_ids_for_field_nodes(field_nodes)
    lateral_to_node = {lid: lat_to_node_true[lid] for lid in lateral_ids}

    evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)

    device = torch.device(args.device)
    graph = load_graph(args.dataset_dir, device=str(device))

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

    groups: List[List[str]] = []
    group_rows: List[List[object]] = []
    lat_rows: List[List[object]] = []

    def eval_group(lids: List[str]):
        res = evaluator.evaluate_group(lids, lateral_to_node=lateral_to_node, q_lateral=args.q_lateral)
        pressures = {lid: float(res.pressures[lateral_to_node[lid]]) for lid in lids}
        return bool(res.ok), float(res.min_margin), float(res.min_pressure_head), pressures

    while True:
        avail = (reachable.bool() & (~irrigated.bool())).squeeze(0)
        remain = int(avail.sum().item())
        if remain == 0:
            break
        if remain < 2:
            break

        with torch.no_grad():
            tok_logits, size_logits = generator(
                graph.node_x, graph.edge_index, graph.edge_attr,
                graph.lateral_to_node_idx, graph.side_idx,
                H0, irrigated, reachable
            )
            tok_logits = tok_logits.squeeze(0)
            size_logits = size_logits.squeeze(0)

        k = int(size_logits.argmax().item()) + 2
        if remain < k:
            k = min(4, remain)
            if k < 2:
                k = 2

        masked = tok_logits.clone()
        masked[~avail] = -1e9

        selected: Optional[List[int]] = None

        for K in [min(args.top_k, remain), min(args.max_expand, remain)]:
            top_idx = torch.topk(masked, k=K).indices.detach().cpu().numpy().tolist()
            combos = list(itertools.combinations(top_idx, k))
            scored = [(float(masked[list(c)].sum().item()), c) for c in combos]
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:min(600, len(scored))]

            trial = [(c, None) for _, c in scored]
            if feas is not None:
                cand_pad = []
                cand_size = []
                for _, c in scored:
                    pad = [-1,-1,-1,-1]
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
                        trial2.append((c, fp))
                if trial2:
                    trial = trial2

            for c, _fp in trial:
                lids = [lateral_ids[i] for i in c]
                ok, mm, mp, pressures = eval_group(lids)
                if ok:
                    selected = list(c)
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
                    break

        if selected is None:
            break

        lids = [lateral_ids[i] for i in selected]
        ok, mm, mp, pressures = eval_group(lids)
        assert ok

        groups.append(lids)
        gid = len(groups)
        surplus = [pressures[lid]-args.Hmin for lid in lids]
        gmean = float(np.mean(surplus))
        gstd = float(np.std(surplus))
        group_rows.append([args.H0, gid, len(lids), mp, mm, gmean, gstd, " ".join(lids)])

        for lid in lids:
            lat_rows.append([args.H0, gid, lid, lateral_to_node[lid], pressures[lid], pressures[lid]-args.Hmin])

        for i in selected:
            irrigated[0, i] = 1

    group_means = [r[5] for r in group_rows]
    mu = float(np.mean(group_means)) if group_means else float("nan")
    std = float(np.std(group_means)) if group_means else float("nan")
    J = 0.5*mu + 0.5*std if group_means else float("nan")

    write_csv(out_dir/"group_metrics.csv",
              ["H0","group_id","group_size","min_pressure_head","min_margin","group_mean_surplus","group_std_surplus","laterals"],
              group_rows)
    write_csv(out_dir/"lateral_metrics.csv",
              ["H0","group_id","lateral_id","node_id","pressure_head","surplus_head"],
              lat_rows)

    payload = {
        "H0": args.H0,
        "Hmin": args.Hmin,
        "global": {"n_groups": len(groups), "mu_group_mean": mu, "std_group_mean": std, "J": J},
        "groups": [{"group_id": i+1, "laterals": g} for i, g in enumerate(groups)],
        "decode": {"top_k": args.top_k, "max_expand": args.max_expand, "feas_used": bool(args.feas_ckpt.strip())}
    }
    (out_dir/"schedule.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir/"summary.json").write_text(json.dumps(payload["global"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] groups={len(groups)} mu={mu:.4f} std={std:.4f} J={J:.4f}")


if __name__ == "__main__":
    main()
