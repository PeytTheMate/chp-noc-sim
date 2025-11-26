from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final
import math
import random

from .config import PhysicsConfig


@dataclass
class PhysicsEngine:
    """
    Encapsulates all loss and latency calculations for the CHP-NoC.

    This object is deliberately decoupled from routing decisions: it exposes
    pure functions for deterministic and stochastic physics, while the
    RoutingAgent only uses the deterministic components for path planning.
    """
    config: PhysicsConfig
    rng: random.Random = field(default_factory=random.Random)

    # Physical constant: approximate speed-of-light delay in silicon (~2e8 m/s).
    # Used for a tiny propagation delay contribution (dominated by switching).
    _C_EFFECTIVE_MM_PER_NS: Final[float] = 0.2  # speed in mm/ns

    def compute_link_loss(self, length_microns: float, num_bends: int = 0) -> float:
        """
        Deterministic loss component (no randomness).

        Loss = alpha_prop * L + alpha_bend * N_bends.
        """
        prop = (
            length_microns
            * self.config.PROPAGATION_LOSS_MEAN_DB_PER_MICRON
        )
        bend = num_bends * self.config.BENDING_LOSS_DB_PER_TURN
        return prop + bend

    def sample_link_noise(
        self,
        length_microns: float,
        temp_src_k: float,
        temp_dst_k: float,
    ) -> float:
        """
        Sample a Gaussian noise term for manufacturing and thermal variation.

        We scale the standard deviation with sqrt(length) to reflect that
        longer paths accumulate more stochastic variation.
        """
        std_per_micron = self.config.PROPAGATION_LOSS_STD_DB_PER_MICRON
        sigma_prop = std_per_micron * math.sqrt(max(length_microns, 0.0))
        avg_temp_k = 0.5 * (temp_src_k + temp_dst_k)
        temp_delta_k = avg_temp_k - self.config.BASE_TEMPERATURE_K
        temp_mean_shift_db = (
            temp_delta_k * self.config.TEMPERATURE_NOISE_COEFF_DB_PER_K
        )
        sigma_total = sigma_prop + self.config.LINK_NOISE_STD_DB
        return self.rng.gauss(temp_mean_shift_db, sigma_total)

    def compute_link_latency(self, length_microns: float) -> float:
        """
        Compute link latency in nanoseconds.

        Propagation delay is tiny at chip scale but we include a small term
        for completeness; switching latency dominates.
        """
        length_mm = length_microns * 1e-3
        # t = d / v
        prop_delay_ns = length_mm / self._C_EFFECTIVE_MM_PER_NS
        return self.config.SWITCHING_LATENCY_NS + prop_delay_ns
