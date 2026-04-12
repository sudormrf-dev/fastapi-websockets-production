# fastapi-websockets-production

Production-grade WebSocket patterns for FastAPI: graceful shutdown, distributed pub/sub, fault tolerance, and backpressure.

## Patterns

### Graceful Shutdown (`patterns/graceful_shutdown.py`)
- `ConnectionRegistry` — tracks active WebSocket tasks; `close_all()` cancels them with configurable timeout
- `create_lifespan()` — FastAPI lifespan context manager that shuts down cleanly on SIGTERM
- `make_connection_id()` — 8-char hex connection IDs
- `CloseCode` — WebSocket close codes 1000 (normal) and 1001 (going away)

### Distributed Pub/Sub (`patterns/pubsub_distributed.py`)
- `InMemoryMessageBus` — subscribe/publish/unsubscribe with per-subscriber async queues
- Lag detection: `publish(skip_lagging=True)` skips subscribers whose queue is > 80% full
- `messages()` — async generator that yields until subscriber is closed
- `RedisPubSubConfig` — configuration dataclass for Redis-backed bus (drop-in upgrade path)

### Fault Tolerance (`patterns/fault_tolerance.py`)
- `HeartbeatManager` — async context manager that sends periodic pings; cancels background task on exit
- `HeartbeatTracker` — tracks missed pongs, last pong time, alive status
- `ReconnectPolicy` — exponential backoff with jitter and max-attempts guard
- `ConnectionHealth` — per-connection counters (sent/received/errors) with idle and uptime timers
- `DeadClientDetector` — finds connections idle beyond a threshold

### Backpressure (`patterns/backpressure.py`)
- `BackpressureStrategy` — DROP_OLDEST / DROP_NEWEST / BLOCK_SENDER / DISCONNECT
- `SendQueue` — bounded queue; DROP_OLDEST uses `deque(maxlen=N)` for O(1) eviction
- `BackpressureController` — manages per-connection queues; `lagging_connections()` reports slow clients
- `SendStats` — per-connection counters with `drop_rate` property

## Quick Start

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from patterns import ConnectionRegistry, BackpressureController, BackpressureStrategy, BackpressureConfig

registry = ConnectionRegistry(shutdown_timeout_s=5.0)
bp = BackpressureController(BackpressureConfig(strategy=BackpressureStrategy.DROP_OLDEST))

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await registry.close_all()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(ws: WebSocket, client_id: str):
    await ws.accept()
    queue = bp.add_connection(client_id)
    try:
        while True:
            data = await ws.receive_json()
            await queue.enqueue(data)
            await queue.drain(ws.send_json)
    finally:
        bp.remove_connection(client_id)
```

## Installation

```bash
pip install -e ".[dev]"
pytest -q
```

## Requirements

- Python 3.12+
- FastAPI, Starlette (WebSocket support)
- pytest, pytest-asyncio (tests)
