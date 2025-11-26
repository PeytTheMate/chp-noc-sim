from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


@dataclass
class RiskAwareGATConfig:
    """
    Lightweight GAT policy for risk-aware routing.

    Node features: [queue_length, temperature]
    Edge features: [distance, estimated_loss_variance]
    """
    in_node_dim: int = 2
    in_edge_dim: int = 2
    hidden_dim: int = 32
    num_gat_heads: int = 2
    num_gat_layers: int = 2
    dropout: float = 0.1


class RiskAwareGATPolicy(nn.Module):
    """
    GNN policy π(a | G, src, dst) implemented with GAT.

    The model produces a logit per directed edge. At decision time, we restrict
    to edges whose source == current node and normalize over those neighbors.
    """

    def __init__(self, config: RiskAwareGATConfig) -> None:
        super().__init__()
        self.config: Final = config

        heads = config.num_gat_heads
        self.gat_layers = nn.ModuleList()
        in_dim = config.in_node_dim

        # Stacked GATv2Conv layers with edge features.
        for _ in range(config.num_gat_layers):
            layer = GATv2Conv(
                in_channels=in_dim,
                out_channels=config.hidden_dim,
                heads=heads,
                edge_dim=config.in_edge_dim,
                dropout=config.dropout,
                add_self_loops=True,
                share_weights=True,
            )
            self.gat_layers.append(layer)
            in_dim = config.hidden_dim * heads

        # Edge scorer: f(h_src, h_dst, h_goal, edge_attr) -> scalar logit
        edge_input_dim = in_dim * 3 + config.in_edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, data: Data, dest_index: int) -> Tensor:
        """
        Args:
            data: PyG Data with x, edge_index, edge_attr.
            dest_index: index of destination node in data.x

        Returns:
            edge_logits: Tensor[num_edges] — unnormalized scores per directed edge.
        """
        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        edge_attr: Tensor = data.edge_attr

        h: Tensor = x
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr)
            h = F.elu(h)

        # Destination embedding; broadcast to all edges.
        goal_emb: Tensor = h[dest_index]  # [H]
        num_edges = edge_index.size(1)
        goal_rep: Tensor = goal_emb.unsqueeze(0).expand(num_edges, -1)

        src_emb: Tensor = h[edge_index[0]]  # [E, H]
        dst_emb: Tensor = h[edge_index[1]]  # [E, H]

        edge_input: Tensor = torch.cat(
            [src_emb, dst_emb, goal_rep, edge_attr],
            dim=-1,
        )
        logits: Tensor = self.edge_mlp(edge_input).squeeze(-1)
        return logits  # [num_edges]
