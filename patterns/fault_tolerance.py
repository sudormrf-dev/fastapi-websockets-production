"""WebSocket fault tolerance: heartbeats, reconnection, dead client detection.

Production WebSocket connections fail silently. TCP keepalive is too slow
(default: 2h idle → probe). Application-level heartbeats detect dead clients
in seconds, enabling fast cleanup and preventing resource leaks.

Pattern:
    Server sends ping every N seconds
    Client must pong within timeout
    If no pong: close connection, deregister, log

This module provides:
    - HeartbeatManager: periodic ping with pong tracking
    - ReconnectPolicy: client-side exponential backoff config
    - DeadClientDetector: identifies connections with no recent activity

Usage::

    hb = HeartbeatManager(interval_s=30, timeout_s=10)
    async with hb.run(connection_id="abc123", send_fn=ws.send_text) as tracker:
        # tracker.is_alive is True while heartbeats are acknowledged
        async for msg in ws.iter_text():
            if msg == "__pong__":
                tracker.record_pong()
            else:
                handle(msg)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

type SendFn = Callable[[str], Coroutine[Any, Any, None]]


@dataclass
class HeartbeatTracker:
    """Tracks heartbeat state for a single connection.

    Attributes:
        connection_id: Identifier for this connection.
        last_pong_at: Timestamp of the most recent pong received.
        ping_count: Total pings sent.
        pong_count: Total pongs received.
        missed_pongs: Consecutive pongs not received.
        alive: Whether the connection is considered alive.
    """

    connection_id: str
    last_pong_at: float = field(default_factory=time.monotonic)
    ping_count: int = 0
    pong_count: int = 0
    missed_pongs: int = 0
    alive: bool = True

    def record_pong(self) -> None:
        """Record receipt of a pong message."""
        self.last_pong_at = time.monotonic()
        self.pong_count += 1
        self.missed_pongs = 0

    def seconds_since_pong(self) -> float:
        """Elapsed seconds since the last pong."""
        return time.monotonic() - self.last_pong_at


class HeartbeatManager:
    """Manages application-level ping/pong heartbeats for WebSocket connections.

    Sends a configurable ping string at regular intervals and tracks
    whether clients respond within the timeout window.

    Args:
        interval_s: Seconds between pings.
        timeout_s: Seconds to wait for a pong before considering client dead.
        ping_message: Message to send as a ping.
        max_missed: Consecutive missed pongs before marking connection dead.
    """

    def __init__(
        self,
        interval_s: float = 30.0,
        timeout_s: float = 10.0,
        ping_message: str = "__ping__",
        max_missed: int = 2,
    ) -> None:
        self._interval = interval_s
        self._timeout = timeout_s
        self._ping_msg = ping_message
        self._max_missed = max_missed

    @asynccontextmanager
    async def run(
        self,
        connection_id: str,
        send_fn: SendFn,
    ) -> AsyncGenerator[HeartbeatTracker, None]:
        """Context manager that runs heartbeats for a connection.

        Yields a :class:`HeartbeatTracker` that handlers use to record pongs.
        Cancels the heartbeat loop when the context exits.

        Args:
            connection_id: Identifier for logging/tracking.
            send_fn: Async callable that sends a string to the WebSocket.

        Yields:
            :class:`HeartbeatTracker` for this connection.
        """
        tracker = HeartbeatTracker(connection_id=connection_id)
        task = asyncio.create_task(self._heartbeat_loop(tracker, send_fn))
        try:
            yield tracker
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _heartbeat_loop(self, tracker: HeartbeatTracker, send_fn: SendFn) -> None:
        """Internal ping loop."""
        while tracker.alive:
            await asyncio.sleep(self._interval)
            tracker.ping_count += 1
            try:
                await send_fn(self._ping_msg)
            except Exception:
                tracker.alive = False
                break

            # Wait for pong
            await asyncio.sleep(self._timeout)
            if tracker.seconds_since_pong() > self._interval + self._timeout:
                tracker.missed_pongs += 1
                logger.warning(
                    "Connection %s missed pong %d/%d",
                    tracker.connection_id,
                    tracker.missed_pongs,
                    self._max_missed,
                )
                if tracker.missed_pongs >= self._max_missed:
                    tracker.alive = False
                    logger.warning("Connection %s marked dead — closing", tracker.connection_id)
                    break


@dataclass
class ReconnectPolicy:
    """Client-side reconnection policy with exponential backoff.

    Provides the delay schedule for client reconnection attempts.
    Documented here for reference implementation in frontend clients
    and for server-side reconnection simulation in tests.

    Attributes:
        initial_delay_s: First reconnect delay in seconds.
        max_delay_s: Maximum delay cap.
        multiplier: Exponential backoff factor.
        jitter_fraction: Random jitter as fraction of delay (0.0-1.0).
        max_attempts: Maximum reconnection attempts (0 = infinite).
    """

    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    multiplier: float = 2.0
    jitter_fraction: float = 0.2
    max_attempts: int = 0  # 0 = infinite

    def delay_for(self, attempt: int) -> float:
        """Compute reconnection delay for the given attempt number.

        Args:
            attempt: Zero-indexed attempt number.

        Returns:
            Delay in seconds before attempting reconnection.
        """
        import random

        raw = min(self.initial_delay_s * (self.multiplier**attempt), self.max_delay_s)
        jitter = raw * self.jitter_fraction * (random.random() * 2 - 1)  # noqa: S311
        return max(0.0, raw + jitter)

    def should_retry(self, attempt: int) -> bool:
        """Return True if another reconnection attempt should be made.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            True if max_attempts not reached (or unlimited).
        """
        if self.max_attempts == 0:
            return True
        return attempt < self.max_attempts


@dataclass
class ConnectionHealth:
    """Aggregated health metrics for a WebSocket connection.

    Attributes:
        connection_id: Connection identifier.
        connected_at: Unix timestamp of connection establishment.
        messages_sent: Total messages sent to this client.
        messages_received: Total messages received from this client.
        last_activity_at: Timestamp of most recent message activity.
        errors: Count of send/receive errors encountered.
    """

    connection_id: str
    connected_at: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0
    last_activity_at: float = field(default_factory=time.time)
    errors: int = 0

    def record_send(self) -> None:
        """Record a message sent to the client."""
        self.messages_sent += 1
        self.last_activity_at = time.time()

    def record_receive(self) -> None:
        """Record a message received from the client."""
        self.messages_received += 1
        self.last_activity_at = time.time()

    def record_error(self) -> None:
        """Record a communication error."""
        self.errors += 1

    @property
    def idle_seconds(self) -> float:
        """Seconds since last message activity."""
        return time.time() - self.last_activity_at

    @property
    def uptime_seconds(self) -> float:
        """Seconds since connection was established."""
        return time.time() - self.connected_at


class DeadClientDetector:
    """Identifies and removes idle or dead WebSocket connections.

    Runs a periodic sweep over tracked connections and marks those
    that haven't had activity within the idle threshold.

    Args:
        idle_threshold_s: Seconds of inactivity before marking a connection dead.
        sweep_interval_s: Seconds between sweeps.
    """

    def __init__(
        self,
        idle_threshold_s: float = 60.0,
        sweep_interval_s: float = 30.0,
    ) -> None:
        self._idle_threshold = idle_threshold_s
        self._sweep_interval = sweep_interval_s
        self._connections: dict[str, ConnectionHealth] = {}

    def track(self, health: ConnectionHealth) -> None:
        """Register a connection for dead-client detection.

        Args:
            health: :class:`ConnectionHealth` object for the connection.
        """
        self._connections[health.connection_id] = health

    def untrack(self, connection_id: str) -> None:
        """Remove a connection from tracking.

        Args:
            connection_id: ID to remove.
        """
        self._connections.pop(connection_id, None)

    def find_dead(self) -> list[ConnectionHealth]:
        """Return connections that exceed the idle threshold.

        Returns:
            List of :class:`ConnectionHealth` objects for dead connections.
        """
        return [h for h in self._connections.values() if h.idle_seconds > self._idle_threshold]

    async def run_sweep_loop(
        self,
        on_dead: Callable[[ConnectionHealth], Coroutine[Any, Any, None]],
    ) -> None:
        """Run periodic sweeps, calling on_dead for each dead connection.

        Args:
            on_dead: Async callback invoked for each dead connection.
        """
        while True:
            await asyncio.sleep(self._sweep_interval)
            dead = self.find_dead()
            for health in dead:
                with suppress(Exception):
                    await on_dead(health)
                self.untrack(health.connection_id)
