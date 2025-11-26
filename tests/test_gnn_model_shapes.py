# tests/test_gnn_model_shapes.py
from __future__ import annotations

import pathlib
import sys

# Add project root to sys.path when tests are run directly.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.data import Data

from gnn_agent.model import RiskAwareGATConfig, RiskAwareGATPolicy


def test_gnn_forward_edge_logits_shape() -> None:
    cfg = RiskAwareGATConfig()
    model = RiskAwareGATPolicy(cfg)

    # Tiny toy graph: 3 nodes, 2 directed edges
    x = torch.tensor(
        [
            [0.0, 300.0],
            [1.0, 305.0],
            [2.0, 295.0],
        ],
        dtype=torch.float32,
    )  # [3, in_node_dim]

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=torch.long,
    )  # [2, num_edges]

    edge_attr = torch.tensor(
        [
            [100.0, 0.01],
            [120.0, 0.02],
        ],
        dtype=torch.float32,
    )  # [num_edges, in_edge_dim]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    logits = model(data, dest_index=2)
    assert logits.shape == (edge_index.size(1),)
    # logits must be finite
    assert torch.isfinite(logits).all()
