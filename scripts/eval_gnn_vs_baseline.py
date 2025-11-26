# scripts/eval_gnn_vs_baseline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict

import argparse
import csv
import os
import random
import statistics

from chp_noc_sim import (
    NetworkSimulation,
    PhysicsConfig,
    PhysicsEngine,
    RoutingAgent,
    build_ring_topology,
)
from gnn_agent import build_default_gnn_router, GNNRoutingAdapter


@dataclass
class SimulationMetrics:
    survival_rate: float
    avg_latency_ns: float
    avg_loss_db: float
    fraction_dead: float
    fraction_over_budget: float


def compute_metrics(sim: NetworkSimulation) -> SimulationMetrics:
    packets = sim.packets
    if not packets:
        return SimulationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    cfg = sim.physics.config
    link_budget = cfg.LINK_BUDGET_DB

    survivors = [p for p in packets if not p.is_dead and p.arrival_time_ns is not None]
    dead = [p for p in packets if p.is_dead]

    if survivors:
        avg_latency = sum(
            (p.arrival_time_ns - p.creation_time_ns)  # type: ignore[operator]
            for p in survivors
        ) / len(survivors)
        avg_loss = sum(p.current_loss_db for p in survivors) / len(survivors)
    else:
        avg_latency = 0.0
        avg_loss = 0.0

    fraction_dead = len(dead) / len(packets)
    fraction_over_budget = sum(
        1 for p in packets if p.current_loss_db > link_budget
    ) / len(packets)

    survival_rate = 1.0 - fraction_dead
    return SimulationMetrics(
        survival_rate=survival_rate,
        avg_latency_ns=avg_latency,
        avg_loss_db=avg_loss,
        fraction_dead=fraction_dead,
        fraction_over_budget=fraction_over_budget,
    )


def run_single_simulation(
    make_router: Callable[[PhysicsConfig, Dict[str, object], Dict[str, object]], object],
    seed: int,
    num_packets: int,
    num_nodes: int,
    link_length_microns: float,
) -> SimulationMetrics:
    random.seed(seed)

    config = PhysicsConfig()
    topology = build_ring_topology(
        config=config,
        num_nodes=num_nodes,
        link_length_microns=link_length_microns,
    )
    physics_rng = random.Random(seed)
    physics = PhysicsEngine(config=config, rng=physics_rng)

    router = make_router(config, topology.nodes, topology.adjacency)

    sim = NetworkSimulation(
        physics=physics,
        routing_agent=router,  # duck-typed: must expose plan_path(...)
        topology=topology,
    )
    sim.run(num_packets=num_packets)
    return compute_metrics(sim)


def make_baseline_router(
    config: PhysicsConfig, nodes: Dict[str, object], adjacency: Dict[str, object]
) -> RoutingAgent[str]:
    return RoutingAgent(
        nodes=nodes,
        adjacency=adjacency,
        physics_config=config,
    )


def make_gnn_router(
    config: PhysicsConfig, nodes: Dict[str, object], adjacency: Dict[str, object]
) -> GNNRoutingAdapter:
    # NOTE: we build a topology again purely for the adapter; you can
    # refactor to share if desired.
    from chp_noc_sim import build_ring_topology

    topo = build_ring_topology(config=config, num_nodes=len(nodes), link_length_microns=250.0)
    gnn_router = build_default_gnn_router(config)
    adapter = GNNRoutingAdapter(
        topology=topo,
        gnn_router=gnn_router,
    )
    return adapter


def summarize(label: str, metrics_list: List[SimulationMetrics]) -> None:
    def mean_std(xs: List[float]) -> tuple[float, float]:
        if not xs:
            return 0.0, 0.0
        if len(xs) == 1:
            return xs[0], 0.0
        return statistics.mean(xs), statistics.pstdev(xs)

    sr = [m.survival_rate for m in metrics_list]
    lat = [m.avg_latency_ns for m in metrics_list]
    loss = [m.avg_loss_db for m in metrics_list]
    over = [m.fraction_over_budget for m in metrics_list]

    sr_m, sr_s = mean_std(sr)
    lat_m, lat_s = mean_std(lat)
    loss_m, loss_s = mean_std(loss)
    over_m, over_s = mean_std(over)

    print(f"\n[{label}]")
    print(f"  Survival rate:      {sr_m * 100:.2f}% ± {sr_s * 100:.2f}%")
    print(f"  Avg latency:        {lat_m:.2f} ns ± {lat_s:.2f} ns")
    print(f"  Avg loss (survivor):{loss_m:.2f} dB ± {loss_s:.2f} dB")
    print(f"  Over-budget frac:   {over_m * 100:.2f}% ± {over_s * 100:.2f}%")


def write_csv(
    csv_path: str,
    num_nodes: int,
    link_length_microns: float,
    num_packets: int,
    baseline_metrics: List[SimulationMetrics],
    gnn_metrics: List[SimulationMetrics],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "router_type",
                "seed",
                "num_nodes",
                "link_length_microns",
                "num_packets",
                "survival_rate",
                "avg_latency_ns",
                "avg_loss_db",
                "fraction_dead",
                "fraction_over_budget",
            ],
        )
        writer.writeheader()
        for seed, m in enumerate(baseline_metrics):
            writer.writerow(
                {
                    "router_type": "baseline",
                    "seed": seed,
                    "num_nodes": num_nodes,
                    "link_length_microns": link_length_microns,
                    "num_packets": num_packets,
                    "survival_rate": m.survival_rate,
                    "avg_latency_ns": m.avg_latency_ns,
                    "avg_loss_db": m.avg_loss_db,
                    "fraction_dead": m.fraction_dead,
                    "fraction_over_budget": m.fraction_over_budget,
                }
            )
        for seed, m in enumerate(gnn_metrics):
            writer.writerow(
                {
                    "router_type": "gnn",
                    "seed": seed,
                    "num_nodes": num_nodes,
                    "link_length_microns": link_length_microns,
                    "num_packets": num_packets,
                    "survival_rate": m.survival_rate,
                    "avg_latency_ns": m.avg_latency_ns,
                    "avg_loss_db": m.avg_loss_db,
                    "fraction_dead": m.fraction_dead,
                    "fraction_over_budget": m.fraction_over_budget,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs GNN routing on the CHP-NoC simulator."
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-packets", type=int, default=2000)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--link-length-microns", type=float, default=250.0)
    parser.add_argument("--csv-path", type=str, default="results/gnn_eval.csv")
    args = parser.parse_args()

    baseline_metrics: List[SimulationMetrics] = []
    gnn_metrics: List[SimulationMetrics] = []

    for seed in range(args.num_seeds):
        print(f"Running seed {seed}...")

        base = run_single_simulation(
            make_baseline_router,
            seed,
            args.num_packets,
            args.num_nodes,
            args.link_length_microns,
        )
        baseline_metrics.append(base)

        gnn = run_single_simulation(
            make_gnn_router,
            seed,
            args.num_packets,
            args.num_nodes,
            args.link_length_microns,
        )
        gnn_metrics.append(gnn)

    summarize("Baseline (Dijkstra, mean-loss)", baseline_metrics)
    summarize("GNN (risk-aware)", gnn_metrics)

    write_csv(
        csv_path=args.csv_path,
        num_nodes=args.num_nodes,
        link_length_microns=args.link_length_microns,
        num_packets=args.num_packets,
        baseline_metrics=baseline_metrics,
        gnn_metrics=gnn_metrics,
    )
    print(f"\n[INFO] Wrote CSV to {args.csv_path}")


if __name__ == "__main__":
    main()
