"""Tests for pubsub_distributed.py."""

from __future__ import annotations

import asyncio
import json

import pytest

from patterns.pubsub_distributed import InMemoryMessageBus, Message, Subscriber


class TestMessage:
    def test_to_json_round_trip(self):
        msg = Message(channel="test", payload={"key": "val"}, publisher_id="p1")
        raw = msg.to_json()
        restored = Message.from_json(raw)
        assert restored.channel == "test"
        assert restored.payload == {"key": "val"}
        assert restored.publisher_id == "p1"

    def test_from_json_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid message JSON"):
            Message.from_json("not json")

    def test_from_json_sets_defaults(self):
        data = json.dumps({"channel": "c", "payload": {}})
        msg = Message.from_json(data)
        assert msg.publisher_id == ""

    def test_timestamp_set(self):
        msg = Message(channel="c", payload={})
        assert msg.timestamp > 0


class TestInMemoryMessageBus:
    def test_subscribe_creates_subscriber(self):
        bus = InMemoryMessageBus()
        sub = bus.subscribe("room:test", "sub1")
        assert isinstance(sub, Subscriber)
        assert sub.channel == "room:test"
        assert sub.subscriber_id == "sub1"

    def test_channel_count_increments(self):
        bus = InMemoryMessageBus()
        bus.subscribe("ch1", "s1")
        bus.subscribe("ch2", "s2")
        assert bus.channel_count == 2

    def test_subscriber_count(self):
        bus = InMemoryMessageBus()
        bus.subscribe("ch1", "s1")
        bus.subscribe("ch1", "s2")
        assert bus.subscriber_count("ch1") == 2

    async def test_publish_delivers_message(self):
        bus = InMemoryMessageBus()
        sub = bus.subscribe("news", "s1")
        count = await bus.publish("news", {"text": "hello"})
        assert count == 1
        msg = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
        assert msg is not None
        assert msg.payload["text"] == "hello"

    async def test_publish_to_empty_channel(self):
        bus = InMemoryMessageBus()
        count = await bus.publish("no-subs", {"x": 1})
        assert count == 0

    async def test_messages_generator(self):
        bus = InMemoryMessageBus()
        sub = bus.subscribe("feed", "s1")
        await bus.publish("feed", {"n": 1})
        await bus.publish("feed", {"n": 2})
        bus.unsubscribe(sub)

        received = [msg.payload["n"] async for msg in bus.messages(sub)]
        assert received == [1, 2]

    def test_unsubscribe_removes_subscriber(self):
        bus = InMemoryMessageBus()
        sub = bus.subscribe("ch", "s1")
        bus.unsubscribe(sub)
        assert bus.subscriber_count("ch") == 0

    async def test_publish_skips_lagging_subscriber(self):
        bus = InMemoryMessageBus()
        _sub = bus.subscribe("ch", "s1", max_queue_size=4)
        # Fill queue to >80% (4 * 0.8 = 3.2 -> 4 messages needed)
        for i in range(4):
            await bus.publish("ch", {"i": i}, skip_lagging=False)
        # Now queue is full, skip_lagging=True should skip
        count = await bus.publish("ch", {"i": 99}, skip_lagging=True)
        assert count == 0

    async def test_close_channel(self):
        bus = InMemoryMessageBus()
        bus.subscribe("ch", "s1")
        await bus.close_channel("ch")
        assert bus.subscriber_count("ch") == 0
