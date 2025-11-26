from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import random

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.data import Data

from chp_noc_sim.config import PhysicsConfig
from .model import RiskAwareGATConfig, RiskAwareGATPolicy


@dataclass
class SyntheticGridConfig:
    rows: int = 6
    cols: int = 6
    hotspot_prob: float = 0.15  # fraction of edges with high variance
    base_distance_microns: float = 200.0
    queue_max: float = 8.0


def generate_synthetic_grid(
    grid_cfg: SyntheticGridConfig,
    physics_cfg: PhysicsConfig,
    rng: random.Random,
) -> Data:
    """
    Create a random grid graph with:
      - Node features: [queue_length, temperature]
      - Edge features: [distance_microns, estimated_loss_variance]
    Some edges are "hotspots" with elevated variance.
    """
    rows, cols = grid_cfg.rows, grid_cfg.cols
    num_nodes = rows * cols

    # Node features
    x = torch.zeros((num_nodes, 2), dtype=torch.float32)
    node_temps = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            queue_len = rng.uniform(0.0, grid_cfg.queue_max)
            temp_k = physics_cfg.BASE_TEMPERATURE_K + rng.uniform(-15.0, 15.0)
            x[idx, 0] = queue_len
            x[idx, 1] = temp_k
            node_temps.append(temp_k)

    # Edges (4-neighbor grid, bidirectional)
    src_list = []
    dst_list = []
    edge_feat_list = []

    base_dist = grid_cfg.base_distance_microns
    base_var_per_micron = physics_cfg.PROPAGATION_LOSS_STD_DB_PER_MICRON ** 2

    def add_edge(u: int, v: int) -> None:
        distance = base_dist
        # baseline variance
        var = base_var_per_micron * distance
        # hotspot edges get large variance
        if rng.random() < grid_cfg.hotspot_prob:
            var *= 10.0

        src_list.append(u)
        dst_list.append(v)
        edge_feat_list.append([distance, var])

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                v = r * cols + (c + 1)
                add_edge(u, v)
                add_edge(v, u)
            # Down neighbor
            if r + 1 < rows:
                v = (r + 1) * cols + c
                add_edge(u, v)
                add_edge(v, u)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = num_nodes
    return data


def run_episode(
    model: RiskAwareGATPolicy,
    data: Data,
    src_index: int,
    dest_index: int,
    physics_cfg: PhysicsConfig,
    rng: random.Random,
    max_steps: int | None = None,
) -> Tuple[float, float, list[Tensor]]:
    """
    Roll out one trajectory using the current policy.

    Returns:
        path_effective_loss: risk-adjusted path metric (μ + λ * σ).
        total_loss_scalar: path_effective_loss + spike if > LINK_BUDGET.
        log_probs: list of log π(a_t | s_t) tensors for REINFORCE.
    """
    if max_steps is None:
        max_steps = data.num_nodes * 4

    current = src_index
    link_budget = physics_cfg.LINK_BUDGET_DB

    # For risk-aware expected loss, track mean and variance separately.
    path_mu = 0.0
    path_var = 0.0

    log_probs: list[Tensor] = []

    # Fixed optical mean loss per micron for synthetic env
    alpha_mean = physics_cfg.PROPAGATION_LOSS_MEAN_DB_PER_MICRON

    for _ in range(max_steps):
        if current == dest_index:
            break

        edge_logits = model(data, dest_index=dest_index)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Candidate outgoing edges from 'current'
        mask = edge_index[0] == current
        candidate_idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if candidate_idxs.numel() == 0:
            # Dead end
            break

        candidate_logits = edge_logits[candidate_idxs]
        probs = F.softmax(candidate_logits, dim=0)
        dist = torch.distributions.Categorical(probs)

        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        log_probs.append(log_prob)

        chosen_edge_idx = candidate_idxs[action_idx].item()
        chosen_edge_attr = edge_attr[chosen_edge_idx]

        distance = float(chosen_edge_attr[0].item())
        var_edge = float(chosen_edge_attr[1].item())

        # Expected loss contribution and variance accumulation
        mu_edge = alpha_mean * distance
        path_mu += mu_edge
        path_var += var_edge

        next_node = int(edge_index[1, chosen_edge_idx].item())
        current = next_node

    # Risk-aware effective loss: μ + λ * σ
    risk_lambda = 1.0
    sigma = math.sqrt(max(path_var, 0.0))
    path_effective_loss = path_mu + risk_lambda * sigma

    # Spike penalty if effective loss breaches budget
    penalty_scale = 5.0
    penalty = 0.0
    if path_effective_loss > link_budget:
        over = path_effective_loss - link_budget
        penalty = penalty_scale * (over ** 2)

    total_loss_scalar = path_effective_loss + penalty
    return path_effective_loss, total_loss_scalar, log_probs


def train_risk_aware_policy(
    num_epochs: int = 50,
    episodes_per_epoch: int = 64,
    seed: int = 1234,
) -> RiskAwareGATPolicy:
    """
    Train the GAT policy on synthetic grid graphs to minimize expected
    accumulated loss and strongly penalize paths that exceed the link budget.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    physics_cfg = PhysicsConfig()
    grid_cfg = SyntheticGridConfig()

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model_cfg = RiskAwareGATConfig()
    model = RiskAwareGATPolicy(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Moving baseline for REINFORCE
    baseline = 0.0
    baseline_momentum = 0.9

    for epoch in range(num_epochs):
        epoch_losses: list[float] = []
        epoch_effective: list[float] = []

        for _ in range(episodes_per_epoch):
            data = generate_synthetic_grid(grid_cfg, physics_cfg, rng)
            data = data.to(device)

            num_nodes = data.num_nodes
            # Sample distinct src/dest
            src_index = rng.randrange(num_nodes)
            dest_index = rng.randrange(num_nodes)
            while dest_index == src_index:
                dest_index = rng.randrange(num_nodes)

            path_effective, total_loss_scalar, log_probs = run_episode(
                model=model,
                data=data,
                src_index=src_index,
                dest_index=dest_index,
                physics_cfg=physics_cfg,
                rng=rng,
            )

            if not log_probs:
                # If the agent couldn't move, skip gradient update.
                continue

            epoch_losses.append(total_loss_scalar)
            epoch_effective.append(path_effective)

            # REINFORCE-style policy gradient:
            # minimize E[total_loss_scalar] via log-prob gradients.
            advantage = total_loss_scalar - baseline
            log_prob_sum = torch.stack(log_probs).sum()
            policy_loss = advantage * log_prob_sum

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # Update baseline (moving average of total loss)
            baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * total_loss_scalar

        if epoch_losses:
            avg_tot_loss = sum(epoch_losses) / len(epoch_losses)
            avg_eff = sum(epoch_effective) / len(epoch_effective)
            print(
                f"[Epoch {epoch + 1:03d}] "
                f"avg_effective_loss={avg_eff:.3f} dB, "
                f"avg_total_loss={avg_tot_loss:.3f}, baseline={baseline:.3f}"
            )

    return model

def risk_aware_loss(
    path_mu: float,
    path_var: float,
    link_budget: float,
    risk_lambda: float = 1.0,
    penalty_scale: float = 5.0,
) -> float:
    sigma = math.sqrt(max(path_var, 0.0))
    path_effective_loss = path_mu + risk_lambda * sigma

    penalty = 0.0
    if path_effective_loss > link_budget:
        over = path_effective_loss - link_budget
        penalty = penalty_scale * (over ** 2)

    return path_effective_loss + penalty


if __name__ == "__main__":
    # Simple training entrypoint:
    trained_model = train_risk_aware_policy(num_epochs=10, episodes_per_epoch=32)
    torch.save(trained_model.state_dict(), "risk_aware_gnn.pt")
