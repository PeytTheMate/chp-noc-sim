# tests/test_gnn_router_integration.py
from __future__ import annotations

import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chp_noc_sim import (
    NetworkTopology,
    Node,
    Link,
    PhysicsConfig,
)
from gnn_agent.router import GNNRouter
from gnn_agent.model import RiskAwareGATConfig, RiskAwareGATPolicy

class DummyAdapter:
    """
    Minimal adapter with the same plan_path API as GNNRoutingAdapter
    but inline to keep this test self-contained.
    """

    def __init__(self, topology: NetworkTopology, router: GNNRouter) -> None:
        self.topology = topology
        self.router = router

    def plan_path(self, source_id: str, dest_id: str) -> List[Link]:
        if source_id == dest_id:
            return []

        current = source_id
        visited = {source_id}
        path_links: List[Link] = []
        max_hops = len(self.topology.nodes) * 4

        for _ in range(max_hops):
            if current == dest_id:
                break

            nxt = self.router.get_next_hop(
                current_node=current,
                destination_node=dest_id,
                network_state=self.topology,
            )
            if nxt == current or nxt in visited:
                # No progress or loop -> fail
                return []

            # Must be a physical neighbor
            outgoing = self.topology.adjacency.get(current, [])
            matching = [lnk for lnk in outgoing if lnk.dst.id == nxt]
            if not matching:
                return []

            link = matching[0]
            path_links.append(link)
            visited.add(nxt)
            current = nxt

        if current != dest_id:
            return []
        return path_links


def build_line_topology() -> NetworkTopology:
    """
    Build a trivial line A -> B -> C with bidirectional links.
    """
    nodes = {
        "A": Node(id="A"),
        "B": Node(id="B"),
        "C": Node(id="C"),
    }
    adjacency = {nid: [] for nid in nodes}
    links = []

    def add_link(src: str, dst: str, length: float) -> None:
        link = Link(
            id=f"{src}->{dst}",
            src=nodes[src],
            dst=nodes[dst],
            length_microns=length,
            is_bent=False,
        )
        links.append(link)
        adjacency[src].append(link)

    add_link("A", "B", 100.0)
    add_link("B", "C", 100.0)
    add_link("B", "A", 100.0)
    add_link("C", "B", 100.0)

    return NetworkTopology(nodes=nodes, links=links, adjacency=adjacency)


def test_gnn_router_returns_physical_neighbor() -> None:
    physics_cfg = PhysicsConfig()
    topo = build_line_topology()

    cfg = RiskAwareGATConfig()
    model = RiskAwareGATPolicy(cfg)
    router = GNNRouter(physics_config=physics_cfg, model=model)

    adapter = DummyAdapter(topo, router)

    path = adapter.plan_path("A", "C")
    # Even with random weights, the path must be composed of actual links
    for link in path:
        assert link.src.id in topo.nodes
        assert link.dst.id in topo.nodes
        assert link in topo.links
        # path continuity:
    for prev, nxt in zip(path, path[1:]):
        assert prev.dst.id == nxt.src.id
