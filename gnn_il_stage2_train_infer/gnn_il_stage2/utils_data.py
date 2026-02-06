
# utils_data.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import torch


@dataclass
class GraphData:
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    lateral_to_node_idx: torch.Tensor
    side_idx: torch.Tensor


def load_graph(dataset_dir: str, device: str = "cpu") -> GraphData:
    ddir = Path(dataset_dir)
    g = torch.load(ddir / "graph_tensors.pt", map_location=device)
    return GraphData(
        node_x=g["node_x"].to(device),
        edge_index=g["edge_index"].to(device),
        edge_attr=g["edge_attr"].to(device),
        lateral_to_node_idx=g["lateral_to_node_idx"].to(device),
        side_idx=g["side_idx"].to(device),
    )


def load_samples(dataset_dir: str, device: str = "cpu") -> Dict:
    ddir = Path(dataset_dir)
    return torch.load(ddir / "samples.pt", map_location=device)


def train_val_split(N: int, val_ratio: float, seed: int = 123) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(N, generator=g)
    n_val = int(round(N * val_ratio))
    return idx[n_val:], idx[:n_val]
