"""Benchmark: ConnectionManager throughput with and without backpressure.

Measures how long it takes to broadcast a single message to N simultaneous
clients and compares two modes:

    with_backpressure   — BackpressureController (DROP_OLDEST, queue=64)
    without_backpressure — direct asyncio.gather over raw send coroutines

Client counts tested: 10, 50, 100, 500, 1 000.

Metrics reported per run:
    mean_ms   — arithmetic mean broadcast latency in milliseconds
    p95_ms    — 95th-percentile latency
    p99_ms    — 99th-percentile latency
    throughput_msg_s — messages delivered per second

Run::

    python -m benchmarks.connection_throughput
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from patterns.backpressure import BackpressureConfig, BackpressureController, BackpressureStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLIENT_COUNTS = [10, 50, 100, 500, 1000]
BROADCAST_ROUNDS = 20  # number of broadcast rounds per (N, mode) measurement
MESSAGE_PAYLOAD: dict[str, Any] = {
    "type": "metrics_update",
    "cpu_pct": 42.5,
    "mem_mb": 1024,
    "request_rate": 350,
}


# ---------------------------------------------------------------------------
# Fake send callable — simulates I/O without real network overhead
# ---------------------------------------------------------------------------


@dataclass
class ClientBuffer:
    """Accumulates messages sent to a single simulated client.

    Attributes:
        client_id: Unique identifier for this buffer.
        received: All messages delivered to this client.
        latencies_ms: Per-message send latency in milliseconds.
        _delay_s: Artificial delay injected per send (simulates slow clients).
    """

    client_id: str
    received: list[dict[str, Any]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    _delay_s: float = 0.0  # set > 0 to simulate a slow client

    async def send(self, data: dict[str, Any]) -> None:
        """Simulate delivering a message, optionally with artificial delay."""
        t0 = time.monotonic()
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        self.received.append(data)
        self.latencies_ms.append((time.monotonic() - t0) * 1_000)


# ---------------------------------------------------------------------------
# Without backpressure: raw gather broadcast
# ---------------------------------------------------------------------------


async def broadcast_raw(clients: list[ClientBuffer], payload: dict[str, Any]) -> float:
    """Broadcast payload to all clients with a bare asyncio.gather.

    No queue, no flow control. If clients are slow, the caller blocks until
    every send completes (or raises).

    Args:
        clients: Target client buffers.
        payload: Message to broadcast.

    Returns:
        Wall-clock broadcast duration in milliseconds.
    """
    t0 = time.monotonic()
    await asyncio.gather(*(c.send(payload) for c in clients))
    return (time.monotonic() - t0) * 1_000


# ---------------------------------------------------------------------------
# With backpressure: enqueue into BackpressureController then drain
# ---------------------------------------------------------------------------


async def broadcast_with_bp(
    controller: BackpressureController,
    clients: list[ClientBuffer],
    payload: dict[str, Any],
) -> float:
    """Broadcast payload via BackpressureController queues.

    Each client has a bounded send queue. Messages are enqueued, then all
    queues are drained concurrently — simulating the production pattern where
    a drain task runs on a configurable interval.

    Args:
        controller: Shared backpressure controller.
        clients: Target client buffers.
        payload: Message to broadcast.

    Returns:
        Wall-clock broadcast duration in milliseconds.
    """
    t0 = time.monotonic()

    # Enqueue for all clients (non-blocking for DROP_OLDEST)
    enqueue_tasks = []
    for c in clients:
        q = controller.get_queue(c.client_id)
        if q is not None:
            enqueue_tasks.append(q.enqueue(payload))
    await asyncio.gather(*enqueue_tasks)

    # Drain all queues — each client's send fn is called here
    drain_tasks = []
    for c in clients:
        q = controller.get_queue(c.client_id)
        if q is not None:
            drain_tasks.append(q.drain(c.send, batch_size=10))
    await asyncio.gather(*drain_tasks)

    return (time.monotonic() - t0) * 1_000


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Statistics for one (N, mode) benchmark run.

    Attributes:
        n_clients: Number of simultaneous clients.
        mode: "with_backpressure" or "without_backpressure".
        mean_ms: Mean broadcast latency in ms.
        p95_ms: 95th-percentile latency in ms.
        p99_ms: 99th-percentile latency in ms.
        throughput_msg_s: Messages delivered per second.
        rounds: Number of broadcast rounds measured.
    """

    n_clients: int
    mode: str
    mean_ms: float
    p95_ms: float
    p99_ms: float
    throughput_msg_s: float
    rounds: int


def _percentile(data: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile of ``data`` (nearest-rank method).

    Args:
        data: Sorted or unsorted list of floats.
        pct: Percentile in [0, 100].

    Returns:
        Percentile value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = max(0, int(len(sorted_data) * pct / 100) - 1)
    return sorted_data[k]


async def run_benchmark(n_clients: int, mode: str) -> BenchmarkResult:
    """Measure broadcast latency for ``n_clients`` over ``BROADCAST_ROUNDS`` rounds.

    Args:
        n_clients: Number of concurrent client buffers to create.
        mode: Either "with_backpressure" or "without_backpressure".

    Returns:
        Populated :class:`BenchmarkResult`.
    """
    clients = [ClientBuffer(client_id=f"client-{i}") for i in range(n_clients)]

    controller: BackpressureController | None = None
    if mode == "with_backpressure":
        bp_config = BackpressureConfig(
            max_queue_size=64,
            strategy=BackpressureStrategy.DROP_OLDEST,
        )
        controller = BackpressureController(bp_config)
        for c in clients:
            controller.add_connection(c.client_id)

    latencies: list[float] = []

    for _ in range(BROADCAST_ROUNDS):
        if controller is not None:
            lat = await broadcast_with_bp(controller, clients, MESSAGE_PAYLOAD)
        else:
            lat = await broadcast_raw(clients, MESSAGE_PAYLOAD)
        latencies.append(lat)

    mean = statistics.mean(latencies)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    # throughput: n_clients messages per broadcast, BROADCAST_ROUNDS broadcasts
    total_messages = n_clients * BROADCAST_ROUNDS
    total_time_s = sum(latencies) / 1_000  # ms → s
    throughput = total_messages / max(total_time_s, 1e-9)

    return BenchmarkResult(
        n_clients=n_clients,
        mode=mode,
        mean_ms=round(mean, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        throughput_msg_s=round(throughput, 1),
        rounds=BROADCAST_ROUNDS,
    )


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def print_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted ASCII table.

    Args:
        results: List of benchmark results to display.
    """
    header = f"{'N':>6}  {'Mode':<24}  {'mean_ms':>9}  {'p95_ms':>8}  {'p99_ms':>8}  {'msg/s':>10}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for r in results:
        print(
            f"{r.n_clients:>6}  {r.mode:<24}  {r.mean_ms:>9.3f}  "
            f"{r.p95_ms:>8.3f}  {r.p99_ms:>8.3f}  {r.throughput_msg_s:>10.1f}"
        )
    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all benchmark combinations and print the results table."""
    print(f"\nWebSocket ConnectionManager Throughput Benchmark")
    print(f"  Broadcast rounds per run : {BROADCAST_ROUNDS}")
    print(f"  Client counts            : {CLIENT_COUNTS}")
    print(f"  Modes                    : with_backpressure, without_backpressure")

    results: list[BenchmarkResult] = []

    for n in CLIENT_COUNTS:
        for mode in ("without_backpressure", "with_backpressure"):
            print(f"  Running n={n:>5}, mode={mode}…", end=" ", flush=True)
            result = await run_benchmark(n, mode)
            results.append(result)
            print(f"mean={result.mean_ms:.3f}ms  p95={result.p95_ms:.3f}ms")

    print_table(results)

    # Quick analysis
    print("Analysis:")
    for n in CLIENT_COUNTS:
        without = next(r for r in results if r.n_clients == n and r.mode == "without_backpressure")
        with_bp = next(r for r in results if r.n_clients == n and r.mode == "with_backpressure")
        overhead = with_bp.mean_ms - without.mean_ms
        print(f"  N={n:>5}: backpressure overhead = {overhead:+.3f}ms (mean)")


if __name__ == "__main__":
    asyncio.run(main())
