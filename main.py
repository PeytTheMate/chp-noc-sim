from __future__ import annotations

import os
from typing import Dict, List

import torch

from chp_noc_sim import (
    NetworkSimulation,
    NetworkTopology,
    PhysicsConfig,
    PhysicsEngine,
    RoutingAgent,
    build_ring_topology,
)
from gnn_agent import build_default_gnn_router, GNNRouter


class GNNRoutingAdapter:
    """
    Adapter that wraps a GNNRouter and exposes a plan_path(...) method so it
    can be passed into NetworkSimulation in place of the naive RoutingAgent.

    The adapter builds a path by repeatedly calling GNNRouter.get_next_hop(...)
    until we hit the destination or fail to make progress.
    """

    def __init__(self, topology: NetworkTopology, gnn_router: GNNRouter) -> None:
        self.topology: NetworkTopology = topology
        self.gnn_router: GNNRouter = gnn_router

    def plan_path(self, source_id: str, dest_id: str) -> List:
        """
        Construct a list[Link] representing the path from source_id to dest_id,
        by repeatedly asking the GNN for the next hop.

        If we get stuck (no neighbor, loop, or missing link), we return [].
        """
        if source_id == dest_id:
            return []

        current_id = source_id
        visited: set[str] = {source_id}
        max_hops = len(self.topology.nodes) * 4

        path_links: List = []

        for _ in range(max_hops):
            if current_id == dest_id:
                break

            next_id = self.gnn_router.get_next_hop(
                current_node=current_id,
                destination_node=dest_id,
                network_state=self.topology,
            )

            # If GNN says "stay" or loops, give up.
            if next_id == current_id or next_id in visited:
                return []

            # Find a link from current -> next_id.
            outgoing = self.topology.adjacency.get(current_id, [])
            candidate_links = [lnk for lnk in outgoing if lnk.dst.id == next_id]
            if not candidate_links:
                # Topology says there's no physical link for this hop.
                return []

            link = candidate_links[0]
            path_links.append(link)
            visited.add(next_id)
            current_id = next_id

        # Did we actually reach the destination?
        if current_id != dest_id:
            return []

        return path_links


def maybe_load_trained_weights(gnn_router: GNNRouter, path: str = "risk_aware_gnn.pt") -> None:
    """
    Optional helper: if a trained checkpoint exists, load it into the router.
    Otherwise we just run with random (untrained) weights.
    """
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=gnn_router.device)
        gnn_router.model.load_state_dict(state_dict)
        print(f"[GNN] Loaded trained weights from {path}")
    else:
        print(f"[GNN] No trained weights found at {path}; using random init.")


def main() -> None:
    # Shared physics + topology
    config = PhysicsConfig()
    topology = build_ring_topology(
        config=config,
        num_nodes=8,
        link_length_microns=250.0,
    )
    physics = PhysicsEngine(config=config)

    # -------------------------------
    # 1) Baseline deterministic router
    # -------------------------------
    baseline_routing: RoutingAgent[str] = RoutingAgent(
        nodes=topology.nodes,
        adjacency=topology.adjacency,
        physics_config=config,
    )
    baseline_sim = NetworkSimulation(
        physics=physics,
        routing_agent=baseline_routing,
        topology=topology,
    )
    baseline_survival_rate = baseline_sim.run(num_packets=1000)
    print(f"[Baseline] Survival Rate: {baseline_survival_rate * 100:.2f}%.")

    # -------------------------------
    # 2) GNN-driven router (Phase 2)
    # -------------------------------
    gnn_router = build_default_gnn_router(config)
    maybe_load_trained_weights(gnn_router, path="risk_aware_gnn.pt")

    gnn_adapter = GNNRoutingAdapter(
        topology=topology,
        gnn_router=gnn_router,
    )

    gnn_sim = NetworkSimulation(
        physics=physics,
        routing_agent=gnn_adapter,  # duck-typed: only needs plan_path(...)
        topology=topology,
    )
    gnn_survival_rate = gnn_sim.run(num_packets=1000)
    print(f"[GNN] Survival Rate: {gnn_survival_rate * 100:.2f}%.")


if __name__ == "__main__":
    main()
