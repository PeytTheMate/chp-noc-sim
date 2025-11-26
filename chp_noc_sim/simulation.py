from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import PhysicsConfig
from .entities import Link, Node, Packet
from .physics import PhysicsEngine
from .routing import RoutingAgent


@dataclass
class NetworkTopology:
    """
    Convenience wrapper to hold nodes, links, and adjacency information.
    """
    nodes: Dict[str, Node]
    links: List[Link]
    adjacency: Dict[str, List[Link]]


@dataclass
class NetworkSimulation:
    """
    Discrete-time Monte Carlo simulation for the CHP-NoC.
    """
    physics: PhysicsEngine
    routing_agent: RoutingAgent[str]
    topology: NetworkTopology
    current_time_ns: float = 0.0
    packets: List[Packet] = field(default_factory=list)

    def inject_and_route_packet(self, packet_id: int) -> Packet:
        """
        Inject a single packet with a random source/destination and let it
        traverse the planned path.
        """
        # Select distinct source and destination.
        node_ids = list(self.topology.nodes.keys())
        src_id = self.physics.rng.choice(node_ids)
        dst_id = self.physics.rng.choice([nid for nid in node_ids if nid != src_id])

        packet = Packet(
            id=packet_id,
            source=src_id,
            dest=dst_id,
            creation_time_ns=self.current_time_ns,
        )

        path_links = self.routing_agent.plan_path(src_id, dst_id)
        if not path_links:
            # No feasible path under mean physics; packet "dies" immediately.
            packet.is_dead = True
            self.packets.append(packet)
            return packet

        # Store path as node-id sequence for debugging / inspection.
        packet.path = [src_id] + [link.dst.id for link in path_links]

        time_ns = self.current_time_ns
        prev_link: Link | None = None

        for link in path_links:
            is_turn = prev_link is not None
            time_ns = link.traverse(
                packet=packet,
                physics=self.physics,
                current_time_ns=time_ns,
                is_turn=is_turn,
            )
            if packet.current_loss_db > self.physics.config.LINK_BUDGET_DB:
                packet.is_dead = True
                break
            prev_link = link

        if not packet.is_dead:
            packet.arrival_time_ns = time_ns

        self.current_time_ns = time_ns
        self.packets.append(packet)
        return packet

    def run(self, num_packets: int) -> float:
        """
        Run the Monte Carlo simulation for a fixed number of injected packets.

        Returns:
            Survival rate as a fraction in [0, 1].
        """
        for packet_id in range(num_packets):
            self.inject_and_route_packet(packet_id)

        if not self.packets:
            return 0.0

        survivors = sum(1 for p in self.packets if not p.is_dead)
        return survivors / len(self.packets)


def build_ring_topology(
    config: PhysicsConfig,
    num_nodes: int = 8,
    link_length_microns: float = 250.0,
) -> NetworkTopology:
    """
    Build a simple unidirectional ring topology for demonstration.

    Each link is modeled as containing a 90-degree bend so that
    BENDING_LOSS_DB_PER_TURN is exercised in the physics model.
    """
    nodes: Dict[str, Node] = {}
    links: List[Link] = []
    adjacency: Dict[str, List[Link]] = {}

    # Initialize nodes with slight temperature variation.
    import random

    rng = random.Random(42)
    for i in range(num_nodes):
        node_id = f"N{i}"
        temp_k = config.BASE_TEMPERATURE_K + rng.uniform(-10.0, 10.0)
        node = Node(id=node_id, temperature_k=temp_k)
        nodes[node_id] = node
        adjacency[node_id] = []

    # Create directed ring links.
    for i in range(num_nodes):
        src = nodes[f"N{i}"]
        dst = nodes[f"N{(i + 1) % num_nodes}"]
        link_id = f"L{i}"
        link = Link(
            id=link_id,
            src=src,
            dst=dst,
            length_microns=link_length_microns,
            is_bent=True,
        )
        links.append(link)
        adjacency[src.id].append(link)

    return NetworkTopology(nodes=nodes, links=links, adjacency=adjacency)
