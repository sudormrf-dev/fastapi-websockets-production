"""Realtime dashboard example using WebSocket production patterns.

Simulates a live metrics dashboard where multiple clients connect, subscribe
to a data feed, receive streaming updates, then disconnect gracefully.

Demonstrates:
    - ConnectionManager built from ConnectionRegistry + BackpressureController
    - BackpressureController with DROP_OLDEST for live telemetry
    - InMemoryMessageBus pub/sub for broadcasting updates to all subscribers
    - Graceful client disconnect sequence (unsubscribe → deregister)
    - 5 simulated clients joining and leaving over ~3 seconds

Run::

    python -m examples.realtime_dashboard
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from patterns.backpressure import BackpressureConfig, BackpressureController, BackpressureStrategy
from patterns.graceful_shutdown import ConnectionInfo, ConnectionRegistry, make_connection_id
from patterns.pubsub_distributed import InMemoryMessageBus, Subscriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FEED_CHANNEL = "metrics:dashboard"
NUM_CLIENTS = 5
SIMULATION_DURATION_S = 3.0
TICK_INTERVAL_S = 0.2  # emit a metrics update every 200 ms


# ---------------------------------------------------------------------------
# Lightweight stand-in for a real WebSocket send callable
# ---------------------------------------------------------------------------


@dataclass
class FakeWebSocket:
    """Simulates a WebSocket connection without a real network layer.

    Attributes:
        client_id: Human-readable label for this fake client.
        received: List of all messages delivered to this socket.
        closed: True once disconnect() has been called.
    """

    client_id: str
    received: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def send_json(self, data: dict[str, Any]) -> None:
        """Record an outbound JSON message (simulates ws.send_json)."""
        if self.closed:
            msg = f"Socket {self.client_id} already closed"
            raise RuntimeError(msg)
        self.received.append(data)

    async def disconnect(self) -> None:
        """Mark the socket as closed (simulates client-initiated close)."""
        self.closed = True
        logger.info("Client %s disconnected (received %d messages)", self.client_id, len(self.received))


# ---------------------------------------------------------------------------
# ConnectionManager: ties Registry + BackpressureController together
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages active dashboard WebSocket connections.

    Combines :class:`ConnectionRegistry` (lifecycle) with
    :class:`BackpressureController` (flow control) so each client gets
    a bounded send queue with DROP_OLDEST semantics — correct for live
    telemetry where stale data is worthless.

    Args:
        registry: Shared connection registry (from lifespan).
        bp_config: Backpressure configuration for send queues.
    """

    def __init__(self, registry: ConnectionRegistry, bp_config: BackpressureConfig | None = None) -> None:
        self._registry = registry
        self._bp = BackpressureController(bp_config)

    def connect(self, ws: FakeWebSocket, task: asyncio.Task[None]) -> str:
        """Register a new connection and create its send queue.

        Args:
            ws: The fake WebSocket for this client.
            task: The asyncio task running the client handler.

        Returns:
            Assigned connection ID.
        """
        conn_id = make_connection_id()
        info = ConnectionInfo(
            connection_id=conn_id,
            client_id=ws.client_id,
            connected_at=time.time(),
            task=task,
        )
        self._registry.register(info)
        self._bp.add_connection(conn_id)
        logger.info("Client %s connected → conn_id=%s", ws.client_id, conn_id)
        return conn_id

    def disconnect(self, conn_id: str, ws: FakeWebSocket) -> None:
        """Deregister the connection and remove its send queue.

        Args:
            conn_id: Connection identifier to remove.
            ws: Associated WebSocket (for logging).
        """
        self._registry.deregister(conn_id)
        self._bp.remove_connection(conn_id)
        logger.info("Connection %s (%s) cleaned up", conn_id, ws.client_id)

    @property
    def active_count(self) -> int:
        """Number of currently active connections."""
        return self._registry.active_count

    def global_bp_stats(self) -> dict[str, Any]:
        """Return aggregated backpressure statistics."""
        return self._bp.global_stats()


# ---------------------------------------------------------------------------
# Metrics producer: simulates time-series data
# ---------------------------------------------------------------------------


async def metrics_producer(bus: InMemoryMessageBus, stop_event: asyncio.Event) -> None:
    """Publish synthetic metrics to FEED_CHANNEL until stop_event is set.

    Emits a JSON payload with cpu_pct, mem_mb, and request_rate every
    TICK_INTERVAL_S seconds.  In a real dashboard this would be replaced by
    Prometheus scrapes, sensor reads, or database queries.

    Args:
        bus: Message bus to publish metrics onto.
        stop_event: Set this event to halt the producer.
    """
    tick = 0
    while not stop_event.is_set():
        payload: dict[str, Any] = {
            "type": "metrics_update",
            "tick": tick,
            "timestamp": time.time(),
            "cpu_pct": 20.0 + (tick % 40),
            "mem_mb": 512 + tick * 3,
            "request_rate": 100 + (tick % 50) * 2,
        }
        delivered = await bus.publish(FEED_CHANNEL, payload, publisher_id="producer")
        logger.debug("Tick %d published to %d subscribers", tick, delivered)
        tick += 1
        await asyncio.sleep(TICK_INTERVAL_S)


# ---------------------------------------------------------------------------
# Client handler: subscribe → receive updates → disconnect
# ---------------------------------------------------------------------------


async def client_handler(
    ws: FakeWebSocket,
    manager: ConnectionManager,
    bus: InMemoryMessageBus,
    lifetime_s: float,
) -> None:
    """Simulate a dashboard client's full lifecycle.

    Connects, subscribes to the metrics feed, consumes updates for
    ``lifetime_s`` seconds, then disconnects gracefully.

    Args:
        ws: Fake WebSocket for this client.
        manager: Shared connection manager.
        bus: Message bus to subscribe on.
        lifetime_s: How long this client stays connected.
    """
    task: asyncio.Task[None] = asyncio.current_task()  # type: ignore[assignment]
    conn_id = manager.connect(ws, task)
    subscriber: Subscriber = bus.subscribe(FEED_CHANNEL, subscriber_id=conn_id)

    try:
        deadline = asyncio.get_event_loop().time() + lifetime_s
        async for msg in bus.messages(subscriber):
            await ws.send_json(msg.payload)
            if asyncio.get_event_loop().time() >= deadline:
                break
    except asyncio.CancelledError:
        logger.info("Client %s handler cancelled (graceful shutdown path)", ws.client_id)
    finally:
        bus.unsubscribe(subscriber)
        await ws.disconnect()
        manager.disconnect(conn_id, ws)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


async def run_simulation() -> None:
    """Orchestrate the full dashboard simulation.

    1. Initialise registry, manager, and in-memory bus.
    2. Start the metrics producer.
    3. Spawn 5 clients with staggered connect times and varying lifetimes.
    4. Wait for all clients to finish, then stop the producer.
    5. Print a summary of received messages and backpressure stats.
    """
    registry = ConnectionRegistry(shutdown_timeout_s=2.0)
    bp_config = BackpressureConfig(
        max_queue_size=64,
        strategy=BackpressureStrategy.DROP_OLDEST,
        warn_threshold_pct=0.75,
    )
    manager = ConnectionManager(registry, bp_config)
    bus = InMemoryMessageBus()
    stop_event = asyncio.Event()

    producer_task = asyncio.create_task(metrics_producer(bus, stop_event))

    # Stagger client connections: each joins 0.2 s after the previous one
    client_tasks: list[asyncio.Task[None]] = []
    sockets: list[FakeWebSocket] = []

    for i in range(NUM_CLIENTS):
        await asyncio.sleep(0.2)
        ws = FakeWebSocket(client_id=f"client-{i + 1}")
        sockets.append(ws)
        lifetime = SIMULATION_DURATION_S - i * 0.4  # later clients stay shorter
        task: asyncio.Task[None] = asyncio.create_task(
            client_handler(ws, manager, bus, lifetime_s=max(0.4, lifetime))
        )
        client_tasks.append(task)
        logger.info(
            "Spawned %s (lifetime=%.1fs, active_connections=%d)",
            ws.client_id,
            max(0.4, lifetime),
            manager.active_count,
        )

    await asyncio.gather(*client_tasks, return_exceptions=True)
    stop_event.set()
    await producer_task

    # Summary
    print("\n--- Simulation Summary ---")
    print(f"Clients simulated : {NUM_CLIENTS}")
    for ws in sockets:
        print(f"  {ws.client_id:12s} → {len(ws.received):3d} messages received")
    stats = manager.global_bp_stats()
    print(f"\nBackpressure stats: {stats}")
    print(f"Registry active   : {registry.active_count} (should be 0)")
    print("--------------------------\n")


if __name__ == "__main__":
    asyncio.run(run_simulation())
