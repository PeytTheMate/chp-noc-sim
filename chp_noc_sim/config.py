from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    """
    Configurable physics constants for the CHP-NoC Monte Carlo simulator.

    All values are expressed in convenient units for chip-scale optics:
    - Lengths: microns
    - Loss: dB
    - Time: nanoseconds
    """

    # Propagation loss (mean and manufacturing variation).
    PROPAGATION_LOSS_MEAN_DB_PER_MICRON: float = 0.03
    PROPAGATION_LOSS_STD_DB_PER_MICRON: float = 0.005

    # Discrete bending loss per 90-degree turn.
    BENDING_LOSS_DB_PER_TURN: float = 1.2

    # Switching latency (dominated by microring thermal tuning) in nanoseconds.
    SWITCHING_LATENCY_NS: float = 1000.0  # 1 microsecond

    # Maximum allowable end-to-end loss before signal is considered "dead".
    LINK_BUDGET_DB: float = 20.0

    # Additional knobs for Monte Carlo noise modeling.
    BASE_TEMPERATURE_K: float = 300.0
    TEMPERATURE_NOISE_COEFF_DB_PER_K: float = 0.001
    LINK_NOISE_STD_DB: float = 0.05
