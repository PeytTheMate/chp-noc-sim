from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class SwitchState(Enum):
    """Simple model of a router's switching state."""
    IDLE = auto()
    ACTIVE = auto()
    BLOCKED = auto()


@dataclass
class Packet:
    """
    Represents a single optical packet / burst in the CHP-NoC.
    """
    id: int
    source: str
    dest: str
    creation_time_ns: float
    current_loss_db: float = 0.0
    path: List[str] = field(default_factory=list)
    is_dead: bool = False
    arrival_time_ns: Optional[float] = None


@dataclass
class Node:
    """
    A routing node / router in the CHP-NoC.
    """
    id: str
    temperature_k: float = 300.0
    switch_state: SwitchState = SwitchState.IDLE


@dataclass
class Link:
    """
    Directed optical link between two nodes.
    """
    id: str
    src: Node
    dst: Node
    length_microns: float
    is_bent: bool = True  # Whether this waveguide segment includes a 90° bend.

    def traverse(
        self,
        packet: Packet,
        physics: "PhysicsEngine",
        current_time_ns: float,
        is_turn: bool = False,
    ) -> float:
        """
        Apply propagation + bending + stochastic noise loss to the packet and
        return the updated simulation time (ns) after the traversal.
        """
        deterministic_loss_db = physics.compute_link_loss(
            length_microns=self.length_microns,
            num_bends=1 if self.is_bent or is_turn else 0,
        )
        noise_db = physics.sample_link_noise(
            length_microns=self.length_microns,
            temp_src_k=self.src.temperature_k,
            temp_dst_k=self.dst.temperature_k,
        )
        packet.current_loss_db += deterministic_loss_db + noise_db
        latency_ns = physics.compute_link_latency(self.length_microns)
        return current_time_ns + latency_ns
