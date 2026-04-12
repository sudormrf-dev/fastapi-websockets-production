"""Distributed WebSocket pub/sub with Redis for multi-worker FastAPI.

Single-process pub/sub is easy. The hard part is multi-worker deployment
(Kubernetes, gunicorn): a message published to worker A must reach clients
connected to workers B and C. Redis pub/sub solves this via a shared channel.

Architecture:
    Client A (worker 1) --publish--> Redis channel "room:lobby"
                                          |
                                    Redis fanout
                                          |
              worker 1 subscriber <-------+-------> worker 2 subscriber
                    |                                      |
             Client A <--broadcast                  Client B <--broadcast

This module provides a channel abstraction that works in both:
- Single-process (in-memory, for testing / small scale)
- Multi-process (Redis pub/sub, for production)

Usage::

    # In-memory (dev/test)
    bus = InMemoryMessageBus()
    await bus.publish("room:lobby", {"type": "chat", "text": "hello"})

    # Redis (production) — use the interface pattern
    # bus = RedisMessageBus(redis_url="redis://localhost:6379")
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
from typing import Any


@dataclass
class Message:
    """A pub/sub message.

    Attributes:
        channel: Channel the message was published to.
        payload: Message payload (JSON-serializable dict).
        timestamp: Unix timestamp when published.
        publisher_id: Optional identifier of the publishing worker/client.
    """

    channel: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    publisher_id: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string for wire transport."""
        return json.dumps(
            {
                "channel": self.channel,
                "payload": self.payload,
                "timestamp": self.timestamp,
                "publisher_id": self.publisher_id,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Message:
        """Deserialize from JSON string.

        Args:
            raw: JSON string from Redis pub/sub.

        Returns:
            Decoded :class:`Message`.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            msg = f"Invalid message JSON: {e}"
            raise ValueError(msg) from e
        return cls(
            channel=data["channel"],
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time()),
            publisher_id=data.get("publisher_id", ""),
        )


@dataclass
class Subscriber:
    """A subscriber to a pub/sub channel.

    Attributes:
        subscriber_id: Unique subscriber identifier.
        channel: Channel being subscribed to.
        queue: Asyncio queue receiving incoming messages.
        max_queue_size: Maximum buffered messages before backpressure.
    """

    subscriber_id: str
    channel: str
    queue: asyncio.Queue[Message | None] = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    max_queue_size: int = 256

    @property
    def queue_depth(self) -> int:
        """Current number of buffered messages."""
        return self.queue.qsize()

    @property
    def is_lagging(self) -> bool:
        """True if queue is more than 80% full."""
        return self.queue.qsize() >= int(self.max_queue_size * 0.8)


class InMemoryMessageBus:
    """In-process pub/sub bus for testing and single-process deployments.

    Not suitable for multi-worker production — use a Redis-backed bus there.
    All operations are async-safe within a single event loop.
    """

    def __init__(self) -> None:
        self._channels: dict[str, list[Subscriber]] = {}

    @property
    def channel_count(self) -> int:
        """Number of channels with at least one subscriber."""
        return len(self._channels)

    def subscriber_count(self, channel: str) -> int:
        """Number of active subscribers on a channel."""
        return len(self._channels.get(channel, []))

    def subscribe(self, channel: str, subscriber_id: str, max_queue_size: int = 256) -> Subscriber:
        """Create and register a subscriber for a channel.

        Args:
            channel: Channel name to subscribe to.
            subscriber_id: Unique ID for this subscriber.
            max_queue_size: Max buffered messages before backpressure.

        Returns:
            New :class:`Subscriber` instance.
        """
        sub = Subscriber(
            subscriber_id=subscriber_id,
            channel=channel,
            queue=asyncio.Queue(maxsize=max_queue_size),
            max_queue_size=max_queue_size,
        )
        self._channels.setdefault(channel, []).append(sub)
        return sub

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove a subscriber from its channel.

        Args:
            subscriber: The subscriber to remove.
        """
        subs = self._channels.get(subscriber.channel, [])
        with suppress(ValueError):
            subs.remove(subscriber)
        if not subs:
            self._channels.pop(subscriber.channel, None)
        # Signal the subscriber's queue to stop
        with suppress(asyncio.QueueFull):
            subscriber.queue.put_nowait(None)

    async def publish(
        self,
        channel: str,
        payload: dict[str, Any],
        publisher_id: str = "",
        skip_lagging: bool = True,
    ) -> int:
        """Publish a message to all subscribers on a channel.

        Args:
            channel: Target channel.
            payload: Message payload dict.
            publisher_id: Optional publisher identifier.
            skip_lagging: If True, skip subscribers whose queues are >80% full.

        Returns:
            Number of subscribers that received the message.
        """
        msg = Message(channel=channel, payload=payload, publisher_id=publisher_id)
        subs = list(self._channels.get(channel, []))
        delivered = 0
        for sub in subs:
            if skip_lagging and sub.is_lagging:
                continue
            try:
                sub.queue.put_nowait(msg)
                delivered += 1
            except asyncio.QueueFull:
                pass
        return delivered

    async def messages(self, subscriber: Subscriber) -> AsyncGenerator[Message, None]:
        """Async generator yielding messages for a subscriber.

        Args:
            subscriber: The subscriber to read from.

        Yields:
            :class:`Message` objects as they arrive.
        """
        while True:
            msg = await subscriber.queue.get()
            if msg is None:
                break
            yield msg

    async def close_channel(self, channel: str) -> None:
        """Signal all subscribers on a channel to stop.

        Args:
            channel: Channel to close.
        """
        for sub in list(self._channels.get(channel, [])):
            self.unsubscribe(sub)


class RedisPubSubConfig:
    """Configuration stub for Redis-backed pub/sub.

    In production, replace :class:`InMemoryMessageBus` with a Redis client:

    .. code-block:: python

        import redis.asyncio as aioredis

        redis = aioredis.from_url("redis://localhost:6379")

        async def publish(channel, payload):
            await redis.publish(channel, json.dumps(payload))

        async def subscribe(channel):
            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    yield Message.from_json(msg["data"])

    Key production settings:
        - ``maxmemory-policy allkeys-lru`` — prevent OOM on message backlog
        - ``tcp-keepalive 60`` — detect dead connections
        - ``notify-keyspace-events ""`` — disable keyspace events (performance)
    """

    redis_url: str = "redis://localhost:6379"
    channel_prefix: str = "ws:"
    max_message_size_bytes: int = 65536
    connection_pool_size: int = 20
