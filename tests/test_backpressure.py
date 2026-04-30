"""Tests for backpressure.py."""

from __future__ import annotations

from patterns.backpressure import (
    BackpressureConfig,
    BackpressureController,
    BackpressureStrategy,
    SendQueue,
    SendStats,
)


class TestSendStats:
    def test_drop_rate_zero_when_empty(self):
        s = SendStats(connection_id="c1")
        assert s.drop_rate == 0.0

    def test_drop_rate_calculation(self):
        s = SendStats(connection_id="c1", messages_queued=8, messages_dropped=2)
        assert abs(s.drop_rate - 0.2) < 1e-9


class TestSendQueueDropOldest:
    async def test_enqueue_returns_true(self):
        q = SendQueue("c1", max_size=4, strategy=BackpressureStrategy.DROP_OLDEST)
        result = await q.enqueue({"n": 1})
        assert result is True

    async def test_depth_increases(self):
        q = SendQueue("c1", max_size=4, strategy=BackpressureStrategy.DROP_OLDEST)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        assert q.depth == 2

    async def test_full_drops_oldest(self):
        q = SendQueue("c1", max_size=2, strategy=BackpressureStrategy.DROP_OLDEST)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        # Adding a third: deque(maxlen=2) auto-drops oldest
        await q.enqueue({"n": 3})
        assert q.depth == 2

    async def test_stats_high_water_mark(self):
        q = SendQueue("c1", max_size=10, strategy=BackpressureStrategy.DROP_OLDEST)
        for i in range(5):
            await q.enqueue({"n": i})
        assert q.stats.high_water_mark >= 4

    async def test_drain_sends_messages(self):
        q = SendQueue("c1", max_size=10, strategy=BackpressureStrategy.DROP_OLDEST)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        received = []

        async def send_fn(msg: dict[str, object]) -> None:
            received.append(msg)

        sent = await q.drain(send_fn, batch_size=10)
        assert sent == 2
        assert len(received) == 2

    async def test_drain_respects_batch_size(self):
        q = SendQueue("c1", max_size=10, strategy=BackpressureStrategy.DROP_OLDEST)
        for i in range(5):
            await q.enqueue({"n": i})
        received = []

        async def send_fn(msg: dict[str, object]) -> None:
            received.append(msg)

        sent = await q.drain(send_fn, batch_size=2)
        assert sent == 2

    async def test_drain_increments_messages_sent(self):
        q = SendQueue("c1", max_size=10, strategy=BackpressureStrategy.DROP_OLDEST)
        await q.enqueue({"n": 1})

        async def send_fn(msg: dict[str, object]) -> None:
            pass

        await q.drain(send_fn)
        assert q.stats.messages_sent == 1

    async def test_drain_handles_send_error(self):
        q = SendQueue("c1", max_size=10, strategy=BackpressureStrategy.DROP_OLDEST)
        await q.enqueue({"n": 1})

        async def failing_send(msg: dict[str, object]) -> None:
            _ = msg
            err = "send failed"
            raise RuntimeError(err)

        sent = await q.drain(failing_send)
        assert sent == 0
        assert q.stats.send_errors == 1


class TestSendQueueDropNewest:
    async def test_full_drops_incoming(self):
        q = SendQueue("c1", max_size=2, strategy=BackpressureStrategy.DROP_NEWEST)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        result = await q.enqueue({"n": 3})
        # Returns True (not disconnect) but drops the message
        assert result is True
        assert q.stats.messages_dropped == 1

    async def test_not_full_enqueues(self):
        q = SendQueue("c1", max_size=5, strategy=BackpressureStrategy.DROP_NEWEST)
        result = await q.enqueue({"n": 1})
        assert result is True
        assert q.stats.messages_queued == 1


class TestSendQueueBlockSender:
    async def test_enqueue_succeeds(self):
        q = SendQueue("c1", max_size=4, strategy=BackpressureStrategy.BLOCK_SENDER)
        result = await q.enqueue({"n": 1})
        assert result is True
        assert q.stats.messages_queued == 1


class TestSendQueueDisconnect:
    async def test_not_full_returns_true(self):
        q = SendQueue("c1", max_size=4, strategy=BackpressureStrategy.DISCONNECT)
        result = await q.enqueue({"n": 1})
        assert result is True

    async def test_full_returns_false(self):
        q = SendQueue("c1", max_size=2, strategy=BackpressureStrategy.DISCONNECT)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        result = await q.enqueue({"n": 3})
        assert result is False


class TestBackpressureController:
    def test_add_connection_returns_queue(self):
        ctrl = BackpressureController()
        q = ctrl.add_connection("c1")
        assert isinstance(q, SendQueue)

    def test_get_queue_returns_queue(self):
        ctrl = BackpressureController()
        ctrl.add_connection("c1")
        assert ctrl.get_queue("c1") is not None

    def test_get_queue_missing_returns_none(self):
        ctrl = BackpressureController()
        assert ctrl.get_queue("nope") is None

    def test_remove_connection(self):
        ctrl = BackpressureController()
        ctrl.add_connection("c1")
        ctrl.remove_connection("c1")
        assert ctrl.get_queue("c1") is None

    def test_remove_missing_no_error(self):
        ctrl = BackpressureController()
        ctrl.remove_connection("nonexistent")  # Should not raise

    async def test_lagging_connections(self):
        cfg = BackpressureConfig(max_queue_size=4, warn_threshold_pct=0.5)
        ctrl = BackpressureController(cfg)
        q = ctrl.add_connection("c1")
        # Fill past 50% threshold (2 messages)
        await q.enqueue({"n": 1})
        await q.enqueue({"n": 2})
        lagging = ctrl.lagging_connections()
        assert "c1" in lagging

    def test_lagging_empty_when_all_ok(self):
        ctrl = BackpressureController()
        ctrl.add_connection("c1")
        assert ctrl.lagging_connections() == []

    def test_global_stats_empty(self):
        ctrl = BackpressureController()
        stats = ctrl.global_stats()
        assert stats["connections"] == 0
        assert stats["total_queued"] == 0

    async def test_global_stats_aggregates(self):
        ctrl = BackpressureController()
        q1 = ctrl.add_connection("c1")
        q2 = ctrl.add_connection("c2")
        await q1.enqueue({"n": 1})
        await q2.enqueue({"n": 2})
        stats = ctrl.global_stats()
        assert stats["connections"] == 2
        assert stats["total_queued"] == 2


class TestBackpressureConfig:
    def test_defaults(self):
        cfg = BackpressureConfig()
        assert cfg.max_queue_size == 256
        assert cfg.strategy == BackpressureStrategy.DROP_OLDEST
        assert cfg.warn_threshold_pct == 0.8

    def test_custom_values(self):
        cfg = BackpressureConfig(
            max_queue_size=64,
            strategy=BackpressureStrategy.DISCONNECT,
            warn_threshold_pct=0.5,
        )
        assert cfg.max_queue_size == 64
        assert cfg.strategy == BackpressureStrategy.DISCONNECT
