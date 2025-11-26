from __future__ import annotations

from chp_noc_sim import (
    NetworkSimulation,
    PhysicsConfig,
    PhysicsEngine,
    RoutingAgent,
    build_ring_topology,
)


def main() -> None:
    config = PhysicsConfig()
    topology = build_ring_topology(config=config, num_nodes=8, link_length_microns=250.0)
    physics = PhysicsEngine(config=config)
    routing_agent: RoutingAgent[str] = RoutingAgent(
        nodes=topology.nodes,
        adjacency=topology.adjacency,
        physics_config=config,
    )
    sim = NetworkSimulation(
        physics=physics,
        routing_agent=routing_agent,
        topology=topology,
    )
    survival_rate = sim.run(num_packets=1000)
    print(f"Survival Rate: {survival_rate * 100:.2f}%.")


if __name__ == "__main__":
    main()
