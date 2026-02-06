
# verify_dataset.py
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    meta = json.loads((ddir / "meta.json").read_text(encoding="utf-8"))
    graph = torch.load(ddir / "graph_tensors.pt", map_location="cpu")
    samples = torch.load(ddir / "samples.pt", map_location="cpu")

    H0 = samples["H0"].numpy()
    scen = samples["scenario_type"].numpy()
    size = samples["target_size"].numpy()

    print("=== META ===")
    print("root:", meta["root"])
    print("counts:", meta["counts"])
    print("build_stats:", meta["build_stats"])

    print("\n=== GRAPH TENSORS ===")
    print("node_x:", tuple(graph["node_x"].shape))
    print("edge_index:", tuple(graph["edge_index"].shape))
    print("edge_attr:", tuple(graph["edge_attr"].shape))
    print("lateral_to_node_idx:", tuple(graph["lateral_to_node_idx"].shape))
    print("side_idx:", tuple(graph["side_idx"].shape))

    print("\n=== SAMPLES ===")
    print("S:", H0.shape[0])
    print("H0 range:", float(H0.min()), "to", float(H0.max()))
    print("scenario_type counts:", {int(k): int(v) for k, v in zip(*np.unique(scen, return_counts=True))})
    print("target_size counts:", {int(k): int(v) for k, v in zip(*np.unique(size, return_counts=True))})

    # sanity checks
    tgt = samples["target"].numpy().astype(np.uint8)
    tcnt = tgt.sum(axis=1)
    bad = np.where((tcnt < 2) | (tcnt > 4))[0]
    print("bad target sizes:", int(bad.size))

    irrig = samples["irrigated"].numpy().astype(np.uint8)
    reach = samples["reachable"].numpy().astype(np.uint8)
    # target should not intersect irrigated, and must be reachable
    inter1 = np.where((tgt & irrig).sum(axis=1) > 0)[0]
    inter2 = np.where((tgt & (1 - reach)).sum(axis=1) > 0)[0]
    print("target∩irrigated >0:", int(inter1.size))
    print("target∩unreachable >0:", int(inter2.size))


if __name__ == "__main__":
    main()
