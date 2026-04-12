"""WebSocket backpressure: flow control to prevent buffer overflow.

Without backpressure, a slow client causes the server to buffer unlimited
messages, eventually consuming all memory. This module provides strategies
for detecting and handling slow consumers.

Strategies:
    DROP_OLDEST: When queue full, drop oldest message (live data streams)
    DROP_NEWEST: When queue full, drop incoming message (critical data)
    BLOCK_SENDER: Apply backpressure to message producers
    DISCONNECT: Close the connection if client is too slow

Usage::

    controller = BackpressureController(
        strategy=BackpressureStrategy.DROP_OLDEST,
        max_queue_size=100,
    )
    await controller.send(connection_id="abc", message={"data": "..."}, send_fn=ws.send_json)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BackpressureStrategy(str, Enum):
    """How to handle a full send queue."""

    DROP_OLDEST = "drop_oldest"  # Discard head of queue, add new message
    DROP_NEWEST = "drop_newest"  # Discard incoming message when full
    BLOCK_SENDER = "block_sender"  # Await until space available (slow path)
    DISCONNECT = "disconnect"  # Close the slow connection


@dataclass
class SendStats:
    """Per-connection send statistics.

    Attributes:
        connection_id: Connection this stats object belongs to.
        messages_queued: Total messages successfully queued.
        messages_dropped: Messages dropped due to backpressure.
        messages_sent: Messages successfully delivered.
        send_errors: Failed send operations.
        high_water_mark: Peak queue depth observed.
    """

    connection_id: str
    messages_queued: int = 0
    messages_dropped: int = 0
    messages_sent: int = 0
    send_errors: int = 0
    high_water_mark: int = 0

    @property
    def drop_rate(self) -> float:
        """Fraction of messages dropped (0.0-1.0)."""
        total = self.messages_queued + self.messages_dropped
        if total == 0:
            return 0.0
        return self.messages_dropped / total


class SendQueue:
    """Bounded FIFO queue for outbound WebSocket messages.

    Provides backpressure via the configured strategy when full.

    Args:
        connection_id: Owner connection identifier.
        max_size: Maximum queued messages.
        strategy: What to do when the queue is full.
    """

    def __init__(
        self,
        connection_id: str,
        max_size: int = 256,
        strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
    ) -> None:
        self._id = connection_id
        self._max = max_size
        self._strategy = strategy
        self._queue: deque[dict[str, Any]] = deque(
            maxlen=max_size if strategy == BackpressureStrategy.DROP_OLDEST else None
        )
        self._async_queue: asyncio.Queue[dict[str, Any]] | None = (
            asyncio.Queue(maxsize=max_size)
            if strategy in {BackpressureStrategy.BLOCK_SENDER, BackpressureStrategy.DROP_NEWEST}
            else None
        )
        self.stats = SendStats(connection_id=connection_id)

    @property
    def depth(self) -> int:
        """Current queue depth."""
        if self._async_queue is not None:
            return self._async_queue.qsize()
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """True if the queue has reached its maximum capacity."""
        return self.depth >= self._max

    async def enqueue(self, message: dict[str, Any]) -> bool:
        """Add a message to the send queue.

        Applies backpressure strategy if full.

        Args:
            message: JSON-serializable message dict.

        Returns:
            True if enqueued, False if dropped or connection should close.
        """
        self.stats.high_water_mark = max(self.stats.high_water_mark, self.depth)

        if self._strategy == BackpressureStrategy.DROP_OLDEST:
            # deque(maxlen=N) auto-drops oldest on append
            self._queue.append(message)
            if len(self._queue) == self._max:
                self.stats.messages_dropped += 1
            else:
                self.stats.messages_queued += 1
            return True

        if self._strategy == BackpressureStrategy.DROP_NEWEST:
            if self._async_queue is not None and self._async_queue.full():
                self.stats.messages_dropped += 1
                logger.debug("DROP_NEWEST: dropped message for %s", self._id)
                return True
            if self._async_queue is not None:
                await self._async_queue.put(message)
            self.stats.messages_queued += 1
            return True

        if self._strategy == BackpressureStrategy.BLOCK_SENDER:
            if self._async_queue is not None:
                await self._async_queue.put(message)
            self.stats.messages_queued += 1
            return True

        # DISCONNECT
        if self.is_full:
            logger.warning("DISCONNECT: %s queue full, closing", self._id)
            return False
        self._queue.append(message)
        self.stats.messages_queued += 1
        return True

    async def drain(
        self,
        send_fn: Any,
        batch_size: int = 10,
    ) -> int:
        """Drain queued messages by sending them.

        Args:
            send_fn: Async callable that accepts a dict and sends it.
            batch_size: Max messages to send per drain call.

        Returns:
            Number of messages sent.
        """
        sent = 0
        for _ in range(batch_size):
            msg: dict[str, Any] | None = None
            if self._async_queue is not None:
                try:
                    msg = self._async_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            elif self._queue:
                msg = self._queue.popleft()
            else:
                break

            if msg is None:
                break
            try:
                await send_fn(msg)
                self.stats.messages_sent += 1
                sent += 1
            except Exception:
                self.stats.send_errors += 1
                break
        return sent


@dataclass
class BackpressureConfig:
    """Configuration for the backpressure controller.

    Attributes:
        max_queue_size: Maximum buffered messages per connection.
        strategy: Backpressure strategy.
        warn_threshold_pct: Log warning when queue exceeds this fraction.
        drain_interval_ms: Milliseconds between drain cycles.
    """

    max_queue_size: int = 256
    strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST
    warn_threshold_pct: float = 0.8
    drain_interval_ms: float = 10.0


class BackpressureController:
    """Manages send queues with backpressure for multiple connections.

    Args:
        config: Backpressure configuration.
    """

    def __init__(self, config: BackpressureConfig | None = None) -> None:
        self._cfg = config or BackpressureConfig()
        self._queues: dict[str, SendQueue] = {}

    def add_connection(self, connection_id: str) -> SendQueue:
        """Create a send queue for a new connection.

        Args:
            connection_id: Connection identifier.

        Returns:
            The new :class:`SendQueue`.
        """
        queue = SendQueue(
            connection_id=connection_id,
            max_size=self._cfg.max_queue_size,
            strategy=self._cfg.strategy,
        )
        self._queues[connection_id] = queue
        return queue

    def remove_connection(self, connection_id: str) -> None:
        """Remove and discard a connection's send queue.

        Args:
            connection_id: Connection to remove.
        """
        self._queues.pop(connection_id, None)

    def get_queue(self, connection_id: str) -> SendQueue | None:
        """Return the send queue for a connection, or None.

        Args:
            connection_id: Connection identifier.
        """
        return self._queues.get(connection_id)

    def lagging_connections(self) -> list[str]:
        """Return IDs of connections whose queues exceed the warn threshold.

        Returns:
            List of connection IDs with high queue depth.
        """
        threshold = int(self._cfg.max_queue_size * self._cfg.warn_threshold_pct)
        return [cid for cid, q in self._queues.items() if q.depth >= threshold]

    def global_stats(self) -> dict[str, Any]:
        """Aggregate stats across all connections.

        Returns:
            Dict with total queued, dropped, sent, and per-connection counts.
        """
        total_queued = sum(q.stats.messages_queued for q in self._queues.values())
        total_dropped = sum(q.stats.messages_dropped for q in self._queues.values())
        total_sent = sum(q.stats.messages_sent for q in self._queues.values())
        return {
            "connections": len(self._queues),
            "total_queued": total_queued,
            "total_dropped": total_dropped,
            "total_sent": total_sent,
            "overall_drop_rate": (total_dropped / max(1, total_queued + total_dropped)),
        }
