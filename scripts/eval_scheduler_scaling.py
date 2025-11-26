# scripts/eval_scheduler_scaling.py
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

from scheduler import (
    generate_synthetic_workload,
    simulate_fifo_baseline,
    simulate_epoch_based,
    REQUEST_PAYLOAD_NS,
)


def compute_raw_link_rate_bytes_per_ns() -> float:
    return 8.0 / REQUEST_PAYLOAD_NS  # 8 bytes per payload


def run_scaling(
    num_requests: int,
    num_regions: int,
    mean_interarrival_ns: float,
    epoch_ns: float,
    capacities: List[int],
) -> Tuple[float, float, float, List[Tuple[int, float, float, float]]]:
    """
    Returns:
        fifo_bw, fifo_lat, R, results
    where results is a list of (train_capacity, train_bw, utilization, avg_latency).
    """
    R = compute_raw_link_rate_bytes_per_ns()

    workload = generate_synthetic_workload(
        num_requests=num_requests,
        num_regions=num_regions,
        mean_interarrival_ns=mean_interarrival_ns,
        raw_seed=123,
    )

    fifo_bw, fifo_lat = simulate_fifo_baseline(workload)
    print(f"FIFO baseline: BW={fifo_bw:.6f} bytes/ns, avg latency={fifo_lat:.2f} ns")
    print(f"Raw link rate R ≈ {R:.6f} bytes/ns\n")

    results: List[Tuple[int, float, float, float]] = []

    for cap in capacities:
        train_bw, train_lat = simulate_epoch_based(
            workload,
            epoch_ns=epoch_ns,
            train_capacity=cap,
        )
        utilization = train_bw / R if R > 0 else 0.0
        results.append((cap, train_bw, utilization, train_lat))
        print(
            f"N={cap:4d} | BW={train_bw:.6f} bytes/ns | Util={utilization*100:6.2f}% | "
            f"AvgLat={train_lat:.2f} ns"
        )

    return fifo_bw, fifo_lat, R, results


def write_csv(
    csv_path: str,
    num_requests: int,
    num_regions: int,
    mean_interarrival_ns: float,
    epoch_ns: float,
    fifo_bw: float,
    fifo_lat: float,
    raw_link_rate: float,
    results: List[Tuple[int, float, float, float]],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_capacity",
                "train_bw_bytes_per_ns",
                "utilization",
                "avg_latency_ns",
                "num_requests",
                "num_regions",
                "mean_interarrival_ns",
                "epoch_ns",
                "fifo_bw_bytes_per_ns",
                "fifo_avg_latency_ns",
                "raw_link_rate_bytes_per_ns",
            ],
        )
        writer.writeheader()
        for cap, train_bw, util, lat in results:
            writer.writerow(
                {
                    "train_capacity": cap,
                    "train_bw_bytes_per_ns": train_bw,
                    "utilization": util,
                    "avg_latency_ns": lat,
                    "num_requests": num_requests,
                    "num_regions": num_regions,
                    "mean_interarrival_ns": mean_interarrival_ns,
                    "epoch_ns": epoch_ns,
                    "fifo_bw_bytes_per_ns": fifo_bw,
                    "fifo_avg_latency_ns": fifo_lat,
                    "raw_link_rate_bytes_per_ns": raw_link_rate,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bandwidth scaling of epoch-based Train scheduler."
    )
    parser.add_argument("--num-requests", type=int, default=50000)
    parser.add_argument("--num-regions", type=int, default=4)
    parser.add_argument("--mean-interarrival-ns", type=float, default=5.0)
    parser.add_argument("--epoch-ns", type=float, default=500.0)
    parser.add_argument(
        "--csv-path", type=str, default="results/scheduler_scaling.csv"
    )
    args = parser.parse_args()

    capacities = [1, 2, 4, 8, 16, 32, 64, 128]

    fifo_bw, fifo_lat, R, results = run_scaling(
        num_requests=args.num_requests,
        num_regions=args.num_regions,
        mean_interarrival_ns=args.mean_interarrival_ns,
        epoch_ns=args.epoch_ns,
        capacities=capacities,
    )

    write_csv(
        csv_path=args.csv_path,
        num_requests=args.num_requests,
        num_regions=args.num_regions,
        mean_interarrival_ns=args.mean_interarrival_ns,
        epoch_ns=args.epoch_ns,
        fifo_bw=fifo_bw,
        fifo_lat=fifo_lat,
        raw_link_rate=R,
        results=results,
    )
    print(f"\n[INFO] Wrote CSV to {args.csv_path}")


if __name__ == "__main__":
    main()
