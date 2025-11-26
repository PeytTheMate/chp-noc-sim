from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import heapq
import math
import random

### Data Structures ###


@dataclass(frozen=True)
class MemRequest:
    """
    A small 64-bit memory request (read or write).

    In a real system, this would correspond to a cache-line or word-size
    access, but here we fix the payload to 64 bits (8 bytes) for simplicity.
    """
    request_id: int
    address: int          # byte-addressed location
    size_bytes: int       # should be <= 8 for "64-bit" semantics
    is_write: bool
    dest_region: int      # logical memory controller / region ID
    arrival_time_ns: float


@dataclass
class Train:
    """
    A "Time-Train" that batches up to N MemRequests aimed at the same
    destination region. All requests in a train incur a single 1µs
    switching penalty at the head of the train.

    RAW hazards: we preserve the original arrival-time order of requests
    within the same Train, so if a write arrives before a read to the same
    address, it will also be ordered earlier in the Train. That ensures
    the read sees the updated value once the train executes.
    """
    train_id: int
    dest_region: int
    requests: List[MemRequest] = field(default_factory=list)
    epoch_start_time_ns: float = 0.0
    release_time_ns: float = 0.0  # when the train becomes eligible to enter the network

    @property
    def total_bytes(self) -> int:
        return sum(req.size_bytes for req in self.requests)



### Coalescing Queue ###


@dataclass
class CoalescingQueue:
    """
    Epoch-based traffic coalescer.

    - Incoming requests are placed in a holding buffer.
    - Every T_epoch, we scan the buffer for all requests whose arrival_time
      is <= current epoch boundary.
    - We group those requests by dest_region and pack them into Train
      objects of at most train_capacity each.
    - Trains are placed on a min-heap keyed by release_time_ns.

    RAW hazard handling:
    - Within each dest_region batch, requests are sorted by arrival_time_ns
      and *kept in that order* inside the Train.
    - This preserves program order for all accesses to the same address
      and thus avoids Read-After-Write (RAW) hazards without needing to
      split trains further.
    """

    epoch_ns: float
    train_capacity: int

    holding_buffer: List[MemRequest] = field(default_factory=list)
    next_epoch_boundary_ns: float = 0.0
    _train_sequence: int = 0
    _ready_trains_heap: List[Tuple[float, int, Train]] = field(default_factory=list)

    def push_request(self, request: MemRequest) -> None:
        """
        Add a new MemRequest into the holding buffer.

        The caller is responsible for calling advance_time(...) with a
        monotonically increasing current_time to trigger epoch formation.
        """
        self.holding_buffer.append(request)

    def advance_time(self, current_time_ns: float) -> None:
        """
        Advance simulated time for the coalescer. For each epoch boundary
        that passes, we:
            - Take all requests with arrival_time <= epoch_boundary.
            - Group by dest_region.
            - Pack into Train objects (capacity-limited).
            - Push each Train onto the ready-trains heap.
        """
        # Process all epochs up to current_time_ns
        while self.next_epoch_boundary_ns <= current_time_ns:
            epoch_start = self.next_epoch_boundary_ns
            epoch_end = epoch_start + self.epoch_ns

            # Collect all requests that arrived in [epoch_start, epoch_end]
            ready_reqs: List[MemRequest] = []
            remaining_reqs: List[MemRequest] = []
            for req in self.holding_buffer:
                if req.arrival_time_ns <= epoch_end:
                    ready_reqs.append(req)
                else:
                    remaining_reqs.append(req)
            self.holding_buffer = remaining_reqs

            # Group by destination region
            by_dest: Dict[int, List[MemRequest]] = {}
            for req in ready_reqs:
                by_dest.setdefault(req.dest_region, []).append(req)

            # Build trains for each destination
            for dest, reqs in by_dest.items():
                # Preserve RAW order: sort by arrival time, stable by request_id
                reqs.sort(key=lambda r: (r.arrival_time_ns, r.request_id))

                # Chunk into trains of capacity -  train_capacity
                for i in range(0, len(reqs), self.train_capacity):
                    chunk = reqs[i : i + self.train_capacity]
                    self._train_sequence += 1
                    train = Train(
                        train_id=self._train_sequence,
                        dest_region=dest,
                        requests=chunk,
                        epoch_start_time_ns=epoch_start,
                        release_time_ns=epoch_end,
                    )
                    # Use a heap for efficient "next train to release" queries
                    heapq.heappush(
                        self._ready_trains_heap,
                        (train.release_time_ns, self._train_sequence, train),
                    )

            # Move to next epoch boundary
            self.next_epoch_boundary_ns += self.epoch_ns

            # If we've passed the current_time_ns then stop
            if self.next_epoch_boundary_ns > current_time_ns:
                break

    def drain_ready_trains(self) -> List[Train]:
        """
        Pop all trains from the heap in release-time order.
        Intended to be called after advance_time(...) with a large enough
        current_time so all epochs of interest have been processed.
        """
        trains: List[Train] = []
        while self._ready_trains_heap:
            _, _, train = heapq.heappop(self._ready_trains_heap)
            trains.append(train)
        return trains


### Simulation / Benchmark Logic ###


# Constants for the link model
SWITCHING_DELAY_NS: float = 1000.0   # 1 µs switching penalty
REQUEST_PAYLOAD_NS: float = 10.0     # time to transmit one 64-bit payload


def simulate_fifo_baseline(
    requests: Iterable[MemRequest],
) -> Tuple[float, float]:
    """
    FIFO baseline:
        - Each request is sent immediately when the link is free.
        - Every request pays the full 1µs switching penalty.

    Returns:
        (effective_bandwidth_bytes_per_ns, average_latency_ns)

    Effective bandwidth is:
        total_bytes_transferred / total_elapsed_ns
    where:
        total_elapsed_ns = last_completion_time - first_arrival_time

    Mathematical throughput model (comment proof):

    Let:
        B = payload bits per request
        R = raw link rate (bits/ns)    -> payload time per request τ = B / R
        T_s = SWITCHING_DELAY_NS

    Baseline per-request service time:
        T_baseline = T_s + τ

    So steady-state throughput in bits/ns is:

        Throughput_baseline = B / (T_s + τ)

    This is strictly less than the theoretical link limit R because T_s > 0.
    """

    reqs = sorted(requests, key=lambda r: r.arrival_time_ns)
    if not reqs:
        return 0.0, 0.0

    total_bytes = 0
    total_latency = 0.0

    link_time_ns = 0.0
    first_arrival_ns = reqs[0].arrival_time_ns
    last_completion_ns = 0.0

    for req in reqs:
        start_ns = max(req.arrival_time_ns, link_time_ns)
        completion_ns = start_ns + SWITCHING_DELAY_NS + REQUEST_PAYLOAD_NS
        latency_ns = completion_ns - req.arrival_time_ns

        total_bytes += req.size_bytes
        total_latency += latency_ns
        link_time_ns = completion_ns
        last_completion_ns = completion_ns

    total_time_ns = max(last_completion_ns - first_arrival_ns, 1e-9)
    effective_bandwidth = total_bytes / total_time_ns
    average_latency = total_latency / len(reqs)
    return effective_bandwidth, average_latency


def generate_trains_from_requests(
    requests: Iterable[MemRequest],
    epoch_ns: float,
    train_capacity: int,
) -> List[Train]:
    """
    Run the epoch-based coalescing scheduler over the entire request trace,
    and return all trains ordered by their release time.
    """
    reqs = sorted(requests, key=lambda r: r.arrival_time_ns)
    cq = CoalescingQueue(epoch_ns=epoch_ns, train_capacity=train_capacity)

    for req in reqs:
        cq.push_request(req)
        # Advance time up to this request's arrival to potentially form
        # epochs that end before/at this arrival
        cq.advance_time(req.arrival_time_ns)

    if reqs:
        last_arrival = reqs[-1].arrival_time_ns
    else:
        last_arrival = 0.0

    # Flush any remaining epochs after the last arrival
    cq.advance_time(last_arrival + 10 * epoch_ns)
    trains = cq.drain_ready_trains()
    # Trains are already in release-time order due to heap
    return trains


def simulate_epoch_based(
    requests: Iterable[MemRequest],
    epoch_ns: float,
    train_capacity: int,
) -> Tuple[float, float]:
    """
    Epoch-based Train scheduler:
        - Requests are grouped every T_epoch into trains of capacity N.
        - Each train pays the switching penalty once (1µs per train).
        - Each request inside the train only pays its payload time.

    Returns:
        (effective_bandwidth_bytes_per_ns, average_latency_ns)

    Mathematical throughput model (comment proof):

    Let:
        N = train_capacity (number of requests in a train)
        B = payload bits per request
        R = raw link rate (bits/ns)        -> τ = B / R
        T_s = SWITCHING_DELAY_NS

    For a full train of N requests, total data bits = N * B.
    Time to transmit that train:

        T_train = T_s + N * τ

    So steady-state throughput in bits/ns is:

        Throughput_train(N) = (N * B) / (T_s + N * τ)

    Divide numerator and denominator by N:

        Throughput_train(N) = B / (T_s / N + τ)

    As N → inf, T_s / N → 0 and thus:

        lim_{N->inf} Throughput_train(N)
            = B / (0 + τ)
            = B / (B / R)
            = R

    which equals the theoretical raw link rate. Therefore, the Train
    approach asymptotically achieves the link's physical capacity as the
    train size grows, unlike the FIFO baseline which is bounded away from R.
    """

    reqs = sorted(requests, key=lambda r: r.arrival_time_ns)
    if not reqs:
        return 0.0, 0.0

    trains = generate_trains_from_requests(
        requests=reqs,
        epoch_ns=epoch_ns,
        train_capacity=train_capacity,
    )

    # Process trains on a single link.
    total_bytes = 0
    total_latency = 0.0
    num_requests = 0

    if not trains:
        return 0.0, 0.0

    first_arrival_ns = reqs[0].arrival_time_ns
    link_time_ns = 0.0
    last_completion_ns = 0.0

    for train in trains:
        # Train can't start before its release (epoch end) or the link being free.
        start_ns = max(train.release_time_ns, link_time_ns)

        # First pay the switching delay at the head of the train.
        train_head_ns = start_ns + SWITCHING_DELAY_NS

        # Each request then pays a payload time in sequence.
        for idx, req in enumerate(train.requests):
            completion_ns = train_head_ns + (idx + 1) * REQUEST_PAYLOAD_NS
            latency_ns = completion_ns - req.arrival_time_ns

            total_bytes += req.size_bytes
            total_latency += latency_ns
            num_requests += 1
            last_completion_ns = completion_ns

        # Link is occupied until the last request in the train finishes.
        link_time_ns = last_completion_ns

    total_time_ns = max(last_completion_ns - first_arrival_ns, 1e-9)
    effective_bandwidth = total_bytes / total_time_ns
    average_latency = total_latency / num_requests
    return effective_bandwidth, average_latency



### Simple Workload Generator ###


def generate_synthetic_workload(
    num_requests: int,
    num_regions: int,
    mean_interarrival_ns: float,
    raw_seed: int = 42,
) -> List[MemRequest]:
    """
    Generate a synthetic stream of memory requests with:
        - Exponential inter-arrival times (Poisson process).
        - Uniformly random destination regions.
        - Some repeated addresses to create potential RAW hazards.
    """
    rng = random.Random(raw_seed)
    requests: List[MemRequest] = []

    time_ns = 0.0
    addr_space = 1_000_000

    for rid in range(num_requests):
        # Exponential inter-arrival with mean 'mean_interarrival_ns'.
        gap = rng.expovariate(1.0 / mean_interarrival_ns)
        time_ns += gap

        dest_region = rng.randrange(num_regions)
        # Occasionally reuse addresses to create RAW hazards.
        if rng.random() < 0.3:
            address = rng.randrange(0, addr_space // 1000) * 8
        else:
            address = rng.randrange(0, addr_space) * 8

        is_write = rng.random() < 0.5
        req = MemRequest(
            request_id=rid,
            address=address,
            size_bytes=8,
            is_write=is_write,
            dest_region=dest_region,
            arrival_time_ns=time_ns,
        )
        requests.append(req)

    return requests



### Benchmark Script Entry Point ###


def run_benchmark() -> None:
    """
    Compare:
        - Baseline FIFO (1µs per request).
        - Epoch-based Train scheduler (1µs per train).

    Prints effective bandwidth and average latency for both.
    """
    num_requests = 10_000
    num_regions = 8
    mean_interarrival_ns = 50.0
    epoch_ns = 500.0
    train_capacity = 32

    print(
        f"Benchmarking {num_requests} requests, "
        f"{num_regions} regions, mean inter-arrival {mean_interarrival_ns} ns, "
        f"epoch = {epoch_ns} ns, train_capacity = {train_capacity}"
    )

    workload = generate_synthetic_workload(
        num_requests=num_requests,
        num_regions=num_regions,
        mean_interarrival_ns=mean_interarrival_ns,
        raw_seed=123,
    )

    fifo_bw, fifo_lat = simulate_fifo_baseline(workload)
    train_bw, train_lat = simulate_epoch_based(
        workload,
        epoch_ns=epoch_ns,
        train_capacity=train_capacity,
    )

    print("\n--- Results ---")
    print(f"Baseline FIFO:")
    print(f"  Effective Bandwidth: {fifo_bw:.6f} bytes/ns")
    print(f"  Average Latency:     {fifo_lat:.3f} ns")

    print(f"\nEpoch-based Train Scheduler:")
    print(f"  Effective Bandwidth: {train_bw:.6f} bytes/ns")
    print(f"  Average Latency:     {train_lat:.3f} ns")

    # Optional but I'm doing it anyway - prints relative improvement
    if fifo_bw > 0:
        bw_gain = (train_bw / fifo_bw - 1.0) * 100.0
        lat_change = (train_lat / fifo_lat - 1.0) * 100.0
        print(f"\nBandwidth improvement: {bw_gain:+.2f}%")
        print(f"Latency change:        {lat_change:+.2f}%")


if __name__ == "__main__":
    run_benchmark()
