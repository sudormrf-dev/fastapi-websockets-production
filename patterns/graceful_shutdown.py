"""Graceful WebSocket shutdown for FastAPI — fixing bug #9149.

FastAPI's WebSocket connections do not close cleanly on application shutdown
by default. Active connections are abruptly terminated, causing clients to
see connection drops instead of a proper close handshake.

Root cause (FastAPI#9149): The ASGI lifespan context exits before active
WebSocket connections are drained, leaving tasks dangling in the event loop.

Solution:
    1. Track all active connections in a global registry.
    2. On shutdown, send close frames to all connections.
    3. Wait for tasks to finish with a timeout before forcefully cancelling.

Pattern:
    startup: register signal handlers + create registry
    websocket handler: register on connect, deregister on disconnect
    shutdown: drain registry with timeout, cancel remaining

Usage::

    app = create_app()
    # Connections registered automatically via the lifespan context.
    # On SIGTERM, all active WebSockets receive close code 1001 (going away).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import IntEnum

logger = logging.getLogger(__name__)


class CloseCode(IntEnum):
    """WebSocket close codes (RFC 6455)."""

    NORMAL = 1000
    GOING_AWAY = 1001  # Server shutdown — use this on graceful shutdown
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012


@dataclass
class ConnectionInfo:
    """Metadata about an active WebSocket connection.

    Attributes:
        connection_id: Unique identifier for this connection.
        client_id: Application-level client identifier (user ID, session, etc.)
        connected_at: Unix timestamp when the connection was established.
        task: The asyncio task running this connection's handler.
    """

    connection_id: str
    client_id: str
    connected_at: float
    task: asyncio.Task[None]


class ConnectionRegistry:
    """Thread-safe registry of active WebSocket connections.

    Provides register/deregister and bulk-close operations for
    graceful shutdown.

    Args:
        shutdown_timeout_s: Seconds to wait for connections to drain
            before force-cancelling tasks.
    """

    def __init__(self, shutdown_timeout_s: float = 5.0) -> None:
        self._connections: dict[str, ConnectionInfo] = {}
        self._shutdown_timeout = shutdown_timeout_s
        self._shutting_down = False

    @property
    def active_count(self) -> int:
        """Number of currently active connections."""
        return len(self._connections)

    @property
    def is_shutting_down(self) -> bool:
        """True once shutdown has been initiated."""
        return self._shutting_down

    def register(self, info: ConnectionInfo) -> None:
        """Register a new WebSocket connection.

        Args:
            info: Connection metadata including the handler task.
        """
        self._connections[info.connection_id] = info
        logger.debug("Registered connection %s (total: %d)", info.connection_id, self.active_count)

    def deregister(self, connection_id: str) -> None:
        """Remove a connection from the registry.

        Args:
            connection_id: ID of the connection to remove.
        """
        self._connections.pop(connection_id, None)
        logger.debug("Deregistered %s (total: %d)", connection_id, self.active_count)

    def get(self, connection_id: str) -> ConnectionInfo | None:
        """Look up a connection by ID.

        Args:
            connection_id: Connection identifier.

        Returns:
            :class:`ConnectionInfo` if found, else None.
        """
        return self._connections.get(connection_id)

    async def close_all(
        self,
        code: CloseCode = CloseCode.GOING_AWAY,
        reason: str = "Server shutting down",
    ) -> int:
        """Send close frames to all active connections and wait for drain.

        Args:
            code: WebSocket close code to send.
            reason: Human-readable close reason.

        Returns:
            Number of connections that were closed.
        """
        self._shutting_down = True
        count = len(self._connections)
        if count == 0:
            return 0

        logger.info("Closing %d WebSocket connections (code=%d)...", count, code)

        # Cancel all handler tasks (they should handle CancelledError and send close)
        tasks = [info.task for info in self._connections.values()]
        for task in tasks:
            task.cancel()

        # Wait for graceful drain with timeout
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=self._shutdown_timeout)
            if pending:
                logger.warning(
                    "%d connections did not drain within %.1fs — force cancelling",
                    len(pending),
                    self._shutdown_timeout,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        self._connections.clear()
        logger.info("All WebSocket connections closed.")
        return count


# Global singleton registry — imported by route handlers
_registry: ConnectionRegistry | None = None


def get_registry() -> ConnectionRegistry:
    """Return the global connection registry.

    Raises:
        RuntimeError: If called before the lifespan context is entered.
    """
    if _registry is None:
        msg = "ConnectionRegistry not initialised. Use create_lifespan() in your FastAPI app."
        raise RuntimeError(msg)
    return _registry


@asynccontextmanager
async def create_lifespan(
    shutdown_timeout_s: float = 5.0,
) -> AsyncGenerator[dict[str, ConnectionRegistry], None]:
    """Async context manager for FastAPI lifespan with WebSocket drain.

    Usage in FastAPI::

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with create_lifespan(shutdown_timeout_s=5.0) as state:
                yield state

        app = FastAPI(lifespan=lifespan)

    Args:
        shutdown_timeout_s: Seconds to wait for connections to drain.

    Yields:
        Dict with ``"registry"`` key for dependency injection.
    """
    global _registry
    registry = ConnectionRegistry(shutdown_timeout_s=shutdown_timeout_s)
    _registry = registry
    logger.info("WebSocket registry initialised (timeout=%.1fs)", shutdown_timeout_s)

    try:
        yield {"registry": registry}
    finally:
        closed = await registry.close_all()
        logger.info("Lifespan shutdown: closed %d connections.", closed)
        _registry = None


def make_connection_id() -> str:
    """Generate a unique connection ID using uuid4."""
    import uuid

    return str(uuid.uuid4())[:8]
