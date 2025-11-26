from __future__ import annotations

import pathlib
import sys

# Add project root to sys.path so `gnn_agent` can be imported
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnn_agent.train import risk_aware_loss


def test_risk_penalty_no_spike_below_budget() -> None:
    link_budget = 20.0
    value = risk_aware_loss(
        path_mu=10.0,
        path_var=1.0,
        link_budget=link_budget,
        risk_lambda=1.0,
        penalty_scale=5.0,
    )
    # If below budget, total loss == effective loss (no penalty)
    # effective_loss = 10 + sqrt(1) = 11
    assert abs(value - 11.0) < 1e-6


def test_risk_penalty_spikes_above_budget() -> None:
    link_budget = 20.0
    value = risk_aware_loss(
        path_mu=25.0,
        path_var=4.0,  # sigma = 2
        link_budget=link_budget,
        risk_lambda=1.0,
        penalty_scale=5.0,
    )
    # effective_loss = 25 + 2 = 27 > 20
    effective_loss = 27.0
    over = effective_loss - link_budget  # 7
    expected = effective_loss + 5.0 * (over ** 2)
    assert abs(value - expected) < 1e-6
    # And the penalty is huge relative to effective_loss
    assert value > effective_loss * 2
