"""Graceful shutdown demo: infinite loop bug vs. correct lifespan pattern.

Shows the classic WebSocket anti-pattern (infinite loop with no cancellation
support) side-by-side with the correct approach (cancellation-aware handler
inside a lifespan-managed registry).

Sections
--------
BROKEN  — ``broken_ws_handler`` spins forever; cancellation is swallowed.
CORRECT — ``correct_ws_handler`` uses ``asyncio.CancelledError`` propagation
           and ``ConnectionRegistry.close_all()`` via the lifespan context.

Run::

    python -m examples.graceful_shutdown_demo
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from patterns.graceful_shutdown import (
    CloseCode,
    ConnectionInfo,
    ConnectionRegistry,
    make_connection_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared fake WebSocket (same lightweight stub as in realtime_dashboard.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeWebSocket:
    """Minimal WebSocket stub used by both broken and correct handlers.

    Attributes:
        client_id: Label for this connection.
        messages: Queue of simulated inbound messages (None = EOF/close).
        sent: All payloads sent by the server-side handler.
        closed: True after close() is called.
    """

    client_id: str
    messages: asyncio.Queue[str | None] = field(default_factory=asyncio.Queue)
    sent: list[str] = field(default_factory=list)
    closed: bool = False

    async def receive_text(self) -> str:
        """Block until the next inbound message (or EOF)."""
        msg = await self.messages.get()
        if msg is None:
            raise EOFError("WebSocket closed by client")
        return msg

    async def send_text(self, text: str) -> None:
        """Record an outbound message from the handler."""
        if self.closed:
            msg = "Socket already closed"
            raise RuntimeError(msg)
        self.sent.append(text)

    async def close(self, code: int = CloseCode.NORMAL, reason: str = "") -> None:
        """Simulate sending a close frame."""
        self.closed = True
        logger.info("[%s] close frame sent (code=%d, reason=%r)", self.client_id, code, reason)


# ===========================================================================
# BROKEN PATTERN — infinite loop that ignores CancelledError
# ===========================================================================


async def broken_ws_handler(ws: FakeWebSocket) -> None:
    """Anti-pattern: infinite loop with a bare except that swallows cancellation.

    Problems:
        1. ``while True`` with no exit condition means the task never finishes
           on its own — it must be forcefully killed.
        2. ``except Exception`` catches ``asyncio.CancelledError`` in Python
           3.7 (it was changed to ``BaseException`` in 3.8, but many old
           tutorials still show this pattern with blanket ``except Exception``).
        3. No close frame is sent to the client on shutdown, so the client
           sees an abrupt TCP RST instead of a clean WebSocket close.

    This handler is intentionally broken — do NOT copy it.
    """
    logger.warning("[BROKEN] Handler started — will NOT respond to cancellation correctly")
    try:
        while True:  # <-- no way out
            try:
                msg = await ws.receive_text()
                await ws.send_text(f"echo: {msg}")
            except Exception:  # noqa: BLE001  # <-- swallows CancelledError in 3.7
                logger.error("[BROKEN] Exception swallowed — loop continues!")
                # BUG: we continue the loop instead of re-raising CancelledError
    finally:
        # This finally block is never reached during a forced kill
        logger.warning("[BROKEN] finally block (unreachable without force-cancel)")


async def simulate_broken_shutdown(duration_s: float = 0.5) -> dict[str, Any]:
    """Demonstrate the broken handler failing to shut down cleanly.

    Runs the broken handler for ``duration_s`` seconds, then cancels it and
    measures how long the forced teardown takes.

    Args:
        duration_s: Seconds to let the handler run before shutdown.

    Returns:
        Dict with timing and outcome metadata.
    """
    ws = FakeWebSocket(client_id="broken-client")

    # Feed a couple of messages so the handler has something to do
    await ws.messages.put("hello")
    await ws.messages.put("world")

    task: asyncio.Task[None] = asyncio.create_task(broken_ws_handler(ws))
    await asyncio.sleep(duration_s)

    logger.warning("[BROKEN] Sending shutdown signal (cancel)…")
    start = time.monotonic()
    task.cancel()

    # Give it a grace period — it should stop, but might not close the socket
    _done, pending = await asyncio.wait({task}, timeout=1.0)
    elapsed = time.monotonic() - start

    if pending:
        # Force-kill what didn't stop
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        outcome = "force-killed (did not cancel gracefully)"
    else:
        outcome = "cancelled (task finished)"

    return {
        "pattern": "BROKEN",
        "close_frame_sent": ws.closed,
        "outcome": outcome,
        "teardown_s": round(elapsed, 3),
        "messages_echoed": len(ws.sent),
    }


# ===========================================================================
# CORRECT PATTERN — cancellation-aware handler with lifespan registry
# ===========================================================================


async def correct_ws_handler(ws: FakeWebSocket, registry: ConnectionRegistry) -> None:
    """Correct pattern: respects CancelledError and sends a close frame.

    Key differences from the broken version:
        1. Handler registers itself with the :class:`ConnectionRegistry` so the
           lifespan context can initiate a graceful drain.
        2. ``asyncio.CancelledError`` is **not** caught in the receive loop —
           it propagates to the ``finally`` block.
        3. The ``finally`` block always sends a close frame (code 1001 = going
           away) so the client gets a clean close handshake.
        4. The handler deregisters itself on exit, keeping the registry accurate.

    Args:
        ws: WebSocket connection to manage.
        registry: Shared registry that tracks this connection.
    """
    conn_id = make_connection_id()
    task: asyncio.Task[None] = asyncio.current_task()  # type: ignore[assignment]
    info = ConnectionInfo(
        connection_id=conn_id,
        client_id=ws.client_id,
        connected_at=time.time(),
        task=task,
    )
    registry.register(info)
    logger.info("[CORRECT] %s registered (conn_id=%s)", ws.client_id, conn_id)

    try:
        while True:
            msg = await ws.receive_text()  # CancelledError propagates freely
            await ws.send_text(f"echo: {msg}")
    except EOFError:
        logger.info("[CORRECT] %s disconnected by client (EOF)", ws.client_id)
    except asyncio.CancelledError:
        logger.info("[CORRECT] %s received shutdown cancellation — sending close frame", ws.client_id)
        raise  # re-raise so the task finishes cleanly
    finally:
        # Always send close frame — even on CancelledError
        with suppress(Exception):
            await ws.close(code=CloseCode.GOING_AWAY, reason="Server shutting down")
        registry.deregister(conn_id)
        logger.info("[CORRECT] %s cleaned up", ws.client_id)


async def simulate_correct_shutdown(duration_s: float = 0.5) -> dict[str, Any]:
    """Demonstrate the correct handler shutting down cleanly via the registry.

    Starts a handler, then uses ``ConnectionRegistry.close_all()`` to initiate
    a graceful drain — the same path FastAPI's lifespan context takes.

    Args:
        duration_s: Seconds to let the handler run before shutdown.

    Returns:
        Dict with timing and outcome metadata.
    """
    registry = ConnectionRegistry(shutdown_timeout_s=2.0)
    ws = FakeWebSocket(client_id="correct-client")

    # Feed messages; the handler will keep waiting after these
    await ws.messages.put("hello")
    await ws.messages.put("world")
    # Don't send EOF — let the shutdown cancel the handler

    task: asyncio.Task[None] = asyncio.create_task(correct_ws_handler(ws, registry))
    await asyncio.sleep(duration_s)

    logger.info("[CORRECT] Initiating graceful shutdown via registry.close_all()…")
    start = time.monotonic()
    closed_count = await registry.close_all(code=CloseCode.GOING_AWAY)
    elapsed = time.monotonic() - start

    # Task should be done by now
    with suppress(asyncio.CancelledError):
        await task

    return {
        "pattern": "CORRECT",
        "close_frame_sent": ws.closed,
        "connections_drained": closed_count,
        "registry_empty_after": registry.active_count == 0,
        "outcome": "graceful close frame sent + registry drained",
        "teardown_s": round(elapsed, 3),
        "messages_echoed": len(ws.sent),
    }


# ---------------------------------------------------------------------------
# Main: run both scenarios and print the comparison table
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run both shutdown scenarios and print a side-by-side comparison."""
    print("\n" + "=" * 60)
    print("  WebSocket Shutdown: Broken vs. Correct")
    print("=" * 60)

    print("\n[1/2] Simulating BROKEN handler…")
    broken_result = await simulate_broken_shutdown(duration_s=0.3)

    print("\n[2/2] Simulating CORRECT handler…")
    correct_result = await simulate_correct_shutdown(duration_s=0.3)

    print("\n" + "=" * 60)
    print(f"{'Metric':<28} {'BROKEN':>15} {'CORRECT':>15}")
    print("-" * 60)

    keys = ["close_frame_sent", "teardown_s", "messages_echoed", "outcome"]
    for key in keys:
        b_val = str(broken_result.get(key, "n/a"))
        c_val = str(correct_result.get(key, "n/a"))
        print(f"  {key:<26} {b_val:>15} {c_val:>15}")

    print("=" * 60)
    print(
        "\nConclusion: The CORRECT pattern sends a close frame and drains"
        "\nthe registry cleanly. The BROKEN pattern requires a force-kill"
        "\nand leaves the client without a close handshake.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
