from __future__ import annotations

import math
import random
import os
import sys

# Ensure project root (containing chp_noc_sim/) is on sys.path for direct runs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from chp_noc_sim.config import PhysicsConfig
from chp_noc_sim.entities import Link, Node, Packet
from chp_noc_sim.physics import PhysicsEngine


def test_compute_link_loss_linear() -> None:
    config = PhysicsConfig()
    physics = PhysicsEngine(config=config)
    length = 100.0
    loss = physics.compute_link_loss(length_microns=length, num_bends=0)
    expected = length * config.PROPAGATION_LOSS_MEAN_DB_PER_MICRON
    assert math.isclose(loss, expected, rel_tol=1e-6)


def test_link_traverse_increases_loss() -> None:
    config = PhysicsConfig()
    rng = random.Random(0)
    physics = PhysicsEngine(config=config, rng=rng)
    src = Node(id="A", temperature_k=config.BASE_TEMPERATURE_K)
    dst = Node(id="B", temperature_k=config.BASE_TEMPERATURE_K)
    link = Link(id="L1", src=src, dst=dst, length_microns=50.0, is_bent=False)
    packet = Packet(id=1, source="A", dest="B", creation_time_ns=0.0)
    initial_loss = packet.current_loss_db
    _ = link.traverse(
        packet=packet,
        physics=physics,
        current_time_ns=0.0,
        is_turn=False,
    )
    # After traversal, loss should have changed (increased on average).
    assert packet.current_loss_db != initial_loss
    assert packet.current_loss_db > initial_loss - 1.0  # very loose lower bound
