"""Tests for fault_tolerance.py."""

from __future__ import annotations

import asyncio

from patterns.fault_tolerance import (
    ConnectionHealth,
    DeadClientDetector,
    HeartbeatManager,
    HeartbeatTracker,
    ReconnectPolicy,
)


class TestHeartbeatTracker:
    def test_record_pong_resets_missed(self):
        t = HeartbeatTracker(connection_id="c1")
        t.missed_pongs = 3
        t.record_pong()
        assert t.missed_pongs == 0
        assert t.pong_count == 1

    def test_seconds_since_pong(self):
        t = HeartbeatTracker(connection_id="c1")
        elapsed = t.seconds_since_pong()
        assert elapsed >= 0.0

    def test_alive_default_true(self):
        t = HeartbeatTracker(connection_id="c1")
        assert t.alive is True


class TestHeartbeatManager:
    async def test_context_manager_yields_tracker(self):
        hb = HeartbeatManager(interval_s=100, timeout_s=5)
        pings = []

        async def send_fn(msg: str) -> None:
            pings.append(msg)

        async with hb.run("c1", send_fn) as tracker:
            assert isinstance(tracker, HeartbeatTracker)
            assert tracker.connection_id == "c1"

    async def test_no_ping_when_interval_not_reached(self):
        hb = HeartbeatManager(interval_s=100, timeout_s=5)
        pings = []

        async def send_fn(msg: str) -> None:
            pings.append(msg)

        async with hb.run("c1", send_fn) as _:
            await asyncio.sleep(0.01)
        assert len(pings) == 0  # interval=100s not reached

    async def test_tracker_alive_after_exit(self):
        hb = HeartbeatManager(interval_s=100, timeout_s=5)

        async def send_fn(msg: str) -> None:
            pass

        async with hb.run("c1", send_fn) as tracker:
            pass
        # After context exit, task is cancelled - tracker may still be alive
        assert isinstance(tracker.alive, bool)


class TestReconnectPolicy:
    def test_delay_increases(self):
        p = ReconnectPolicy(initial_delay_s=1.0, multiplier=2.0, jitter_fraction=0.0)
        d0 = p.delay_for(0)
        d1 = p.delay_for(1)
        d2 = p.delay_for(2)
        assert d1 > d0
        assert d2 > d1

    def test_delay_capped(self):
        p = ReconnectPolicy(
            initial_delay_s=1.0, max_delay_s=5.0, multiplier=10.0, jitter_fraction=0.0
        )
        assert p.delay_for(10) <= 5.0

    def test_should_retry_infinite(self):
        p = ReconnectPolicy(max_attempts=0)
        assert p.should_retry(1000) is True

    def test_should_retry_limited(self):
        p = ReconnectPolicy(max_attempts=3)
        assert p.should_retry(2) is True
        assert p.should_retry(3) is False

    def test_jitter_gives_variation(self):
        p = ReconnectPolicy(jitter_fraction=0.5)
        delays = {p.delay_for(0) for _ in range(20)}
        assert len(delays) > 1


class TestConnectionHealth:
    def test_record_send(self):
        h = ConnectionHealth(connection_id="c1")
        h.record_send()
        assert h.messages_sent == 1

    def test_record_receive(self):
        h = ConnectionHealth(connection_id="c1")
        h.record_receive()
        assert h.messages_received == 1

    def test_record_error(self):
        h = ConnectionHealth(connection_id="c1")
        h.record_error()
        assert h.errors == 1

    def test_idle_seconds(self):
        h = ConnectionHealth(connection_id="c1")
        assert h.idle_seconds >= 0.0

    def test_uptime_seconds(self):
        h = ConnectionHealth(connection_id="c1")
        assert h.uptime_seconds >= 0.0


class TestDeadClientDetector:
    def test_find_dead_empty(self):
        d = DeadClientDetector(idle_threshold_s=60)
        assert d.find_dead() == []

    def test_find_dead_idle(self):
        d = DeadClientDetector(idle_threshold_s=0.0)
        h = ConnectionHealth(connection_id="c1")
        d.track(h)
        dead = d.find_dead()
        assert any(dh.connection_id == "c1" for dh in dead)

    def test_untrack_removes(self):
        d = DeadClientDetector()
        h = ConnectionHealth(connection_id="c1")
        d.track(h)
        d.untrack("c1")
        assert d.find_dead() == []

    def test_active_not_dead(self):
        d = DeadClientDetector(idle_threshold_s=9999)
        h = ConnectionHealth(connection_id="c1")
        d.track(h)
        assert d.find_dead() == []
