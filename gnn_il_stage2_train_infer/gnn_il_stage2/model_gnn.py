
# model_gnn.py
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeEdgeFusionLayer(nn.Module):
    def __init__(self, h: int, edge_attr_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = MLP(in_dim=2 * h + edge_attr_dim, hidden=2 * h, out_dim=h, dropout=dropout)
        self.node_mlp = MLP(in_dim=h + h, hidden=2 * h, out_dim=h, dropout=dropout)
        self.ln_node = nn.LayerNorm(h)
        self.ln_edge = nn.LayerNorm(h)

    def forward(
        self,
        node_h: torch.Tensor,      # [N,H]
        edge_h: torch.Tensor,      # [E,H]
        edge_index: torch.Tensor,  # [2,E]
        edge_attr: torch.Tensor,   # [E,Fe]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src = edge_index[0]
        dst = edge_index[1]
        hs = node_h[src]
        hd = node_h[dst]
        edge_in = torch.cat([hs, hd, edge_attr], dim=-1)
        msg = self.edge_mlp(edge_in)
        edge_h = self.ln_edge(edge_h + msg)

        agg = scatter_add(edge_h, dst, dim_size=node_h.size(0))
        node_in = torch.cat([node_h, agg], dim=-1)
        upd = self.node_mlp(node_in)
        node_h = self.ln_node(node_h + upd)
        return node_h, edge_h


class NodeEdgeFusionEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_attr_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_attr_dim, hidden_dim)
        self.layers = nn.ModuleList([NodeEdgeFusionLayer(hidden_dim, edge_attr_dim, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        node_h = self.dropout(F.relu(self.node_proj(node_x)))
        edge_h = self.dropout(F.relu(self.edge_proj(edge_attr)))
        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h, edge_index, edge_attr)
        return node_h


class GroupGenerator(nn.Module):
    def __init__(self, node_in_dim: int, edge_attr_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = NodeEdgeFusionEncoder(node_in_dim, edge_attr_dim, hidden_dim, num_layers, dropout)
        self.side_emb = nn.Embedding(2, hidden_dim)
        self.h0_mlp = MLP(1, hidden_dim, hidden_dim, dropout=dropout)

        self.token_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lateral_to_node_idx: torch.Tensor,
        side_idx: torch.Tensor,
        H0: torch.Tensor,          # [B]
        irrigated: torch.Tensor,   # [B,L]
        reachable: torch.Tensor,   # [B,L]
    ):
        B = H0.size(0)
        node_h = self.encoder(node_x, edge_index, edge_attr)  # [N,H]
        lat_h = node_h[lateral_to_node_idx] + self.side_emb(side_idx)  # [L,H]
        lat_h = lat_h.unsqueeze(0).expand(B, -1, -1)  # [B,L,H]

        h0_emb = self.h0_mlp(H0.view(B, 1))  # [B,H]
        h0_expand = h0_emb.unsqueeze(1).expand(B, lat_h.size(1), -1)  # [B,L,H]

        irr = irrigated.float().unsqueeze(-1)
        rea = reachable.float().unsqueeze(-1)
        mask_feats = torch.cat([irr, rea], dim=-1)  # [B,L,2]

        token_in = torch.cat([lat_h, h0_expand, mask_feats], dim=-1)
        token_logits = self.token_head(token_in).squeeze(-1)  # [B,L]

        avail = (reachable.bool() & (~irrigated.bool())).float()  # [B,L]
        denom = avail.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (lat_h * avail.unsqueeze(-1)).sum(dim=1) / denom  # [B,H]

        size_in = torch.cat([pooled, h0_emb], dim=-1)
        size_logits = self.size_head(size_in)  # [B,3]
        return token_logits, size_logits


class FeasibilityClassifier(nn.Module):
    def __init__(self, node_in_dim: int, edge_attr_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = NodeEdgeFusionEncoder(node_in_dim, edge_attr_dim, hidden_dim, num_layers, dropout)
        self.side_emb = nn.Embedding(2, hidden_dim)
        self.h0_mlp = MLP(1, hidden_dim, hidden_dim, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        lateral_to_node_idx: torch.Tensor,
        side_idx: torch.Tensor,
        H0: torch.Tensor,          # [B]
        irrigated: torch.Tensor,   # [B,L]
        reachable: torch.Tensor,   # [B,L]
        cand_pad: torch.Tensor,    # [B,4] idx or -1
        cand_size: torch.Tensor,   # [B] 2/3/4
    ) -> torch.Tensor:
        B = H0.size(0)
        node_h = self.encoder(node_x, edge_index, edge_attr)
        lat_h = node_h[lateral_to_node_idx] + self.side_emb(side_idx)  # [L,H]
        lat_h = lat_h.unsqueeze(0).expand(B, -1, -1)  # [B,L,H]

        h0_emb = self.h0_mlp(H0.view(B, 1))  # [B,H]

        avail = (reachable.bool() & (~irrigated.bool())).float()
        denom = avail.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_avail = (lat_h * avail.unsqueeze(-1)).sum(dim=1) / denom

        cand_mask = (cand_pad >= 0)
        cand_idx = cand_pad.clamp_min(0)
        cand_vec = torch.gather(lat_h, 1, cand_idx.unsqueeze(-1).expand(-1, -1, lat_h.size(-1)))  # [B,4,H]
        cand_vec = cand_vec * cand_mask.unsqueeze(-1).float()
        denom2 = cand_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        cand_mean = cand_vec.sum(dim=1) / denom2

        s = (cand_size.long() - 2).clamp(0, 2)
        s_oh = torch.zeros(B, 3, device=H0.device)
        s_oh.scatter_(1, s.view(B, 1), 1.0)

        x = torch.cat([cand_mean, pooled_avail, h0_emb, s_oh], dim=-1)
        logit = self.mlp(x).squeeze(-1)
        return logit
