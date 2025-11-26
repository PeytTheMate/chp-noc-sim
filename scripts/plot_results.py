# scripts/plot_results.py
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_gnn_csv(csv_path: str) -> Dict[str, List[Dict[str, float]]]:
    """
    Load gnn_eval.csv and group rows by router_type ('baseline', 'gnn').
    """
    groups: Dict[str, List[Dict[str, float]]] = {"baseline": [], "gnn": []}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rtype = row["router_type"]
            if rtype not in groups:
                groups[rtype] = []
            groups[rtype].append(
                {
                    "seed": float(row["seed"]),
                    "survival_rate": float(row["survival_rate"]),
                    "avg_latency_ns": float(row["avg_latency_ns"]),
                    "avg_loss_db": float(row["avg_loss_db"]),
                    "fraction_over_budget": float(row["fraction_over_budget"]),
                }
            )
    return groups


def load_scheduler_csv(csv_path: str) -> Tuple[List[int], List[float], List[float], List[float], float, float]:
    """
    Returns:
        capacities, train_bw, utilization, avg_lat, raw_link_rate, fifo_bw
    """
    capacities: List[int] = []
    train_bw: List[float] = []
    utilization: List[float] = []
    avg_lat: List[float] = []
    raw_link_rate = 0.0
    fifo_bw = 0.0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            capacities.append(int(row["train_capacity"]))
            bw = float(row["train_bw_bytes_per_ns"])
            train_bw.append(bw)
            util = float(row["utilization"])
            utilization.append(util)
            avg_lat.append(float(row["avg_latency_ns"]))
            raw_link_rate = float(row["raw_link_rate_bytes_per_ns"])
            fifo_bw = float(row["fifo_bw_bytes_per_ns"])

    return capacities, train_bw, utilization, avg_lat, raw_link_rate, fifo_bw


def plot_gnn_results(groups: Dict[str, List[Dict[str, float]]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    baseline = groups.get("baseline", [])
    gnn = groups.get("gnn", [])

    if not baseline or not gnn:
        print("[WARN] Missing baseline or gnn rows in gnn_eval.csv; skipping GNN plots.")
        return

    baseline_sr = [g["survival_rate"] for g in baseline]
    gnn_sr = [g["survival_rate"] for g in gnn]
    baseline_over = [g["fraction_over_budget"] for g in baseline]
    gnn_over = [g["fraction_over_budget"] for g in gnn]

    seeds = [g["seed"] for g in baseline]

    # Survival rate per seed
    plt.figure()
    plt.plot(seeds, [s * 100 for s in baseline_sr], marker="o", label="Baseline")
    plt.plot(seeds, [s * 100 for s in gnn_sr], marker="x", label="GNN")
    plt.xlabel("Seed")
    plt.ylabel("Survival rate (%)")
    plt.title("Survival rate per seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gnn_survival_per_seed.png"))

    # Over-budget fraction per seed
    plt.figure()
    plt.plot(seeds, [s * 100 for s in baseline_over], marker="o", label="Baseline")
    plt.plot(seeds, [s * 100 for s in gnn_over], marker="x", label="GNN")
    plt.xlabel("Seed")
    plt.ylabel("Over-budget fraction (%)")
    plt.title("Over-budget fraction per seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gnn_over_budget_per_seed.png"))

    # Quick health check: mean improvements
    import statistics

    base_sr_mean = statistics.mean(baseline_sr)
    gnn_sr_mean = statistics.mean(gnn_sr)
    base_over_mean = statistics.mean(baseline_over)
    gnn_over_mean = statistics.mean(gnn_over)

    print("\n[GNN diagnostic]")
    print(f"  Mean survival (baseline) = {base_sr_mean*100:.2f}%")
    print(f"  Mean survival (gnn)      = {gnn_sr_mean*100:.2f}%")
    print(f"  Mean over-budget (base)  = {base_over_mean*100:.2f}%")
    print(f"  Mean over-budget (gnn)   = {gnn_over_mean*100:.2f}%")

    sr_delta = gnn_sr_mean - base_sr_mean
    over_delta = gnn_over_mean - base_over_mean

    # Heuristic thresholds; tweak them as you experiment
    if sr_delta < 0.005 and over_delta > -0.005:
        print(
            "  [WARN] GNN is not clearly outperforming baseline "
            "(<0.5 percentage point improvement in survival / over-budget). "
            "Consider revisiting training, features, or reward shaping."
        )
    else:
        print("  [OK] GNN shows a meaningful difference vs baseline.")


def plot_scheduler_results(
    capacities: List[int],
    train_bw: List[float],
    utilization: List[float],
    avg_lat: List[float],
    raw_link_rate: float,
    fifo_bw: float,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Bandwidth vs train capacity
    plt.figure()
    plt.plot(capacities, train_bw, marker="o")
    plt.axhline(raw_link_rate, linestyle="--")
    plt.axhline(fifo_bw, linestyle=":")
    plt.xlabel("Train capacity N")
    plt.ylabel("Train BW (bytes/ns)")
    plt.title("Train bandwidth vs capacity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scheduler_bandwidth_vs_capacity.png"))

    # Utilization vs train capacity
    plt.figure()
    plt.plot(capacities, [u * 100 for u in utilization], marker="o")
    plt.xlabel("Train capacity N")
    plt.ylabel("Utilization (%)")
    plt.title("Scheduler utilization vs capacity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scheduler_utilization_vs_capacity.png"))

    # Health check: monotonic-ish utilization
    print("\n[Scheduler diagnostic]")
    print(f"  Raw link rate R = {raw_link_rate:.6f} bytes/ns")
    print(f"  FIFO BW         = {fifo_bw:.6f} bytes/ns")

    if len(utilization) >= 2:
        deltas = [utilization[i + 1] - utilization[i] for i in range(len(utilization) - 1)]
        non_increasing_steps = sum(1 for d in deltas if d < -1e-3)
        if non_increasing_steps > 0:
            print(
                "  [WARN] Utilization is not monotonically increasing with capacity "
                f"({non_increasing_steps} decreases observed). Check for bugs or noisy workloads."
            )
        else:
            print("  [OK] Utilization increases with capacity as expected.")

        if utilization[-1] < 0.7:
            print(
                "  [WARN] Even at highest train capacity, utilization < 70% of raw link rate. "
                "Consider adjusting epoch_ns, workload intensity, or modeling assumptions."
            )
        else:
            print("  [OK] High-capacity trains approach the theoretical link limit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and diagnose GNN and scheduler results from CSVs."
    )
    parser.add_argument(
        "--gnn-csv",
        type=str,
        default="results/gnn_eval.csv",
        help="Path to gnn_eval.csv produced by eval_gnn_vs_baseline.py",
    )
    parser.add_argument(
        "--scheduler-csv",
        type=str,
        default="results/scheduler_scaling.csv",
        help="Path to scheduler_scaling.csv produced by eval_scheduler_scaling.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    if os.path.exists(args.gnn_csv):
        groups = load_gnn_csv(args.gnn_csv)
        plot_gnn_results(groups, args.out_dir)
    else:
        print(f"[INFO] GNN CSV not found at {args.gnn_csv}; skipping GNN plots.")

    if os.path.exists(args.scheduler_csv):
        capacities, bw, util, lat, R, fifo_bw = load_scheduler_csv(args.scheduler_csv)
        plot_scheduler_results(capacities, bw, util, lat, R, fifo_bw, args.out_dir)
    else:
        print(f"[INFO] Scheduler CSV not found at {args.scheduler_csv}; skipping scheduler plots.")


if __name__ == "__main__":
    main()
