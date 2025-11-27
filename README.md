# CHP-NoC Monte Carlo Simulator  
**Coupled Hybrid Plasmonic Network-on-Chip with GNN Routing & Train Scheduler**

This repository implements a high-fidelity **Monte Carlo simulator** for a  
**Coupled Hybrid Plasmonic Network-on-Chip (CHP-NoC)**, with three tightly integrated phases:

- **Phase 1 – Physics & Discrete-Event Simulation**  
  Optical loss + latency under manufacturing variance, routing, and packet-level simulation.

- **Phase 2 – GNN Routing Agent (“The Brain”)**  
  A risk-aware Graph Neural Network (GAT-based) that learns to route packets to minimize signal death probability under stochastic loss.

- **Phase 3 – Train Scheduler (“Time-Train” Compiler Pass)**  
  An epoch-based “traffic coalescing” scheduler that batches memory requests into trains to amortize a 1 µs switching penalty and approach the raw link bandwidth.

This is a **v1 research prototype**: the focus is on a clean, testable, reproducible framework that can be iterated on — not on claiming final “best possible” metrics.

---

## 1. Problem Setting & Goals

Modern CHP-NoCs face two major challenges:

1. **Random signal loss** from process variation, temperature, and optical path differences.
2. **Massive switching overhead** (~1 µs) compared to nanosecond-scale packet times.

This repo explores two orthogonal optimizations:

- **Routing-side**: Use a risk-aware GNN router that “sees” loss variance in the topology and prefers safer paths, improving survival under a fixed link budget (e.g., 20 dB).
- **Scheduling-side**: Use an epoch-based Train scheduler that groups many small memory requests into larger “time-trains”, so the 1 µs switching penalty is paid per train instead of per request, asymptotically approaching the physical link capacity.

Everything is wired to be:

- **Strictly typed** (Python 3.10+ type hints, dataclasses).
- **Tested** (pytest).
- **Reproducible** (scripted experiments → CSVs → figures).

---

## 2. Repository Structure

```text
chp_noc_sim_project/
├── chp_noc_sim/
│   ├── __init__.py
│   ├── config.py          # Physics constants: loss, switching latency, link budget, etc.
│   ├── entities.py        # Packet, Node, Link dataclasses
│   ├── physics.py         # PhysicsEngine: Monte Carlo loss & latency
│   ├── routing.py         # Baseline RoutingAgent (mean-loss Dijkstra / deterministic)
│   └── simulation.py      # NetworkSimulation: time-stepped sim, traffic injection
│
├── gnn_agent/
│   ├── __init__.py
│   ├── model.py           # RiskAwareGATPolicy: GAT-based policy over nodes/edges
│   ├── train.py           # RL-style training loop on synthetic grid graphs
│   └── router.py          # GNNRouter + GNNRoutingAdapter to plug into NetworkSimulation
│
├── scheduler.py           # Phase 3: Train scheduler, FIFO vs epoch-based simulators
│
├── scripts/
│   ├── eval_gnn_vs_baseline.py      # Evaluate baseline routing vs GNN routing, write CSV
│   ├── eval_scheduler_scaling.py    # Evaluate train capacity scaling, write CSV
│   └── plot_results.py              # Load CSVs, generate figures + diagnostics
│
├── tests/
│   ├── test_physics.py              # Physics sanity checks
│   ├── test_gnn_model_shapes.py     # GNN forward shapes + finite outputs
│   ├── test_gnn_router_integration.py  # GNN router produces valid paths
│   ├── test_risk_penalty_behavior.py   # Risk-aware penalty math (over-budget spike)
│   └── test_gnn_training_smoke.py      # Small training run sanity check
│
├── results/                          # (Generated) CSV logs from experiments
├── figures/                          # (Generated) PNG plots
├── main.py                           # Example main entrypoint / demo
└── Makefile                          # Convenience targets (test, benchmarks)
```

## 3. Core Components

### 3.1 Phase 1 – Physics & Simulation

This is the “world” the packets live in.

- **`PhysicsConfig` (`chp_noc_sim/config.py`)**  
  Holds all the tunable physical constants:
  - Propagation loss per micron (mean + stddev).
  - Bending loss per 90° turn.
  - Switching latency (e.g., 1 µs = 1000 ns).
  - Link budget (max allowed loss before signal death).

- **`PhysicsEngine` (`chp_noc_sim/physics.py`)**  
  Given a `Link` and a `Packet`, it:
  - Computes deterministic loss (length, bends).
  - Adds Gaussian noise for Monte Carlo variation.
  - Updates `packet.current_loss_db` and latency.

- **`NetworkSimulation` (`chp_noc_sim/simulation.py`)**  
  Runs a discrete-event style simulation:
  - Injects packets into the topology over time.
  - Asks a routing agent for a path (`plan_path(source, dest)`).
  - Applies physics along each hop.
  - Marks packets as **DEAD** if loss > `LINK_BUDGET_DB`.
  - Tracks arrival times, survival rate, etc.

Phase 1 knows nothing about GNNs or train scheduling — it’s pure physics + routing.

---

### 3.2 Phase 2 – GNN Routing Agent (“The Brain”)

This replaces the naive deterministic router with a GNN that “sees” risk.

- **Graph representation**
  - Nodes: routers, with features like `[queue_length, temperature]`.
  - Edges: links, with features `[distance_microns, estimated_loss_variance]`.

- **Model (`gnn_agent/model.py`)**
  - `RiskAwareGATPolicy`: a small Graph Attention Network (GAT) that:
    - Takes the network graph as input.
    - Outputs a **logit per directed edge** (how good that edge is).
    - Uses attention to weight safer neighbors higher than lossy ones.

- **Router integration (`gnn_agent/router.py`)**
  - `GNNRouter`: builds a PyTorch Geometric `Data` object from the current `NetworkTopology` and applies the GNN.
  - `GNNRoutingAdapter`: exposes:

    ```python
    def plan_path(self, source_id: str, dest_id: str) -> List[Link]:
        ...
    ```

    so it can be dropped into `NetworkSimulation` just like the baseline `RoutingAgent`.

- **Training (`gnn_agent/train.py`)**
  - Generates synthetic grid graphs with “hotspot” edges (high variance).
  - Runs an RL-style loop where each episode is a routing path.
  - Loss:
    - Minimize **expected accumulated loss**.
    - Add a big **risk penalty** if the path exceeds the 20 dB budget.
  - The goal is to push the policy toward routes that:
    - Have low mean loss, and
    - Low variance / low chance of blowing the budget.

---

### 3.3 Phase 3 – Train Scheduler (“Time-Train” Compiler Pass)

This is the “compiler” that batches memory traffic to beat the 1 µs switching overhead.

- **Data structures (`scheduler.py`)**
  - `MemRequest`: a 64-bit read/write to a given address/region with an arrival time.
  - `Train`: a container of up to `N` `MemRequest`s going to the **same dest_region**.
    - All requests inside a `Train` **pay one** 1 µs switching penalty.
    - Requests are kept in **arrival-time order** to avoid Read-After-Write hazards.

- **Coalescing (`CoalescingQueue`)**
  - Maintains a holding buffer of incoming `MemRequest`s.
  - Every `T_epoch` (e.g., 500 ns):
    - Looks at all requests that arrived before that epoch’s end.
    - Groups them by `dest_region`.
    - Sorts each group by `(arrival_time, request_id)` to enforce ordering.
    - Packs them into `Train`s of size at most `train_capacity`.
  - Uses `heapq` to keep trains sorted by `release_time_ns` (epoch boundary).

- **Two schedulers for comparison**
  - **Baseline FIFO**: send each request immediately.
    - Every request pays `1 µs + payload_time`.
  - **Epoch-based Train scheduler**:
    - Every train pays `1 µs`, then each request in the train just pays its payload time.

Mathematically (and verified in code), as `Train_Size → ∞`, the train scheduler’s throughput approaches the raw link rate, while FIFO is stuck below that due to always paying 1 µs per request.

---

## 4. Setup & Dependencies

- **Python**: 3.10+
- Recommended: virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install pytest matplotlib torch
# plus torch-geometric; see their install docs for the right wheel:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

Optional (but nice):

```bash
mypy chp_noc_sim gnn_agent scheduler.py

```

## 5. Running Tests

Run the full suite (from the root) by:

```bash
pytest -q
```

This checks:

* Physics calculations behave as expected.
* GNN model forward passes have correct shapes and finite values.
* GNNRouter+adapter only produce valid paths over real links.
* Risk-aware loss spikes when paths exceed the link budget.
* A tiny training run executes without NaNs or shape errors.

## 6. Reproducible Experiments & Figures

All experiments follow the pattern:

1. Run a script in scripts/ to generate a CSV under results/.
2. Run plot_results.py to turn CSVs into PNGs under figures/.

### 6.1 GNN vs Baseline Routing
```bash
python -m scripts.eval_gnn_vs_baseline \
  --num-seeds 10 \
  --num-packets 5000 \
  --num-nodes 8 \
  --link-length-microns 250 \
  --csv-path results/gnn_eval.csv
  ```

This evaluates:

* Baseline deterministic routing.
* GNN-based routing.

And logs per-seed metrics:
* survival_rate
* avg_latency_ns
* avg_loss_db
* fraction_over_budget

### 6.2 Train Scheduler Scaling
```bash
python -m scripts.eval_scheduler_scaling \
  --num-requests 50000 \
  --num-regions 4 \
  --mean-interarrival-ns 5 \
  --epoch-ns 500 \
  --csv-path results/scheduler_scaling.csv
```

This evaluates:

* FIFO scheduler (1 µs per request).
* Epoch-based trains (1 µs per train) for different train_capacity values.

### 6.3 Plots & Diagnostics

```bash
python -m scripts.plot_results \
  --gnn-csv results/gnn_eval.csv \
  --scheduler-csv results/scheduler_scaling.csv \
  --out-dir figures
```

This generates:

* figures/gnn_survival_per_seed.png
* figures/gnn_over_budget_per_seed.png
* figures/scheduler_bandwidth_vs_capacity.png
* figures/scheduler_utilization_vs_capacity.png

and prints quick “health checks” so you can see whether:

* The GNN is actually better than the baseline.
* The Train scheduler’s utilization increases with capacity and approaches the theoretical link limit.

