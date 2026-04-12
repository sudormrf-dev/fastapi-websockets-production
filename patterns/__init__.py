"""FastAPI WebSocket production patterns.

Graceful shutdown, distributed pub/sub, fault tolerance, and backpressure.
"""

from .backpressure import (
    BackpressureConfig,
    BackpressureController,
    BackpressureStrategy,
    SendQueue,
    SendStats,
)
from .fault_tolerance import (
    ConnectionHealth,
    DeadClientDetector,
    HeartbeatManager,
    HeartbeatTracker,
    ReconnectPolicy,
)
from .graceful_shutdown import (
    CloseCode,
    ConnectionInfo,
    ConnectionRegistry,
    create_lifespan,
    make_connection_id,
)
from .pubsub_distributed import (
    InMemoryMessageBus,
    Message,
    RedisPubSubConfig,
    Subscriber,
)

__all__ = [
    "BackpressureConfig",
    "BackpressureController",
    "BackpressureStrategy",
    "CloseCode",
    "ConnectionHealth",
    "ConnectionInfo",
    "ConnectionRegistry",
    "DeadClientDetector",
    "HeartbeatManager",
    "HeartbeatTracker",
    "InMemoryMessageBus",
    "Message",
    "ReconnectPolicy",
    "RedisPubSubConfig",
    "SendQueue",
    "SendStats",
    "Subscriber",
    "create_lifespan",
    "make_connection_id",
]
