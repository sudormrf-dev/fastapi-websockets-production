"""Tests for graceful_shutdown.py."""

from __future__ import annotations

import asyncio

from patterns.graceful_shutdown import (
    CloseCode,
    ConnectionInfo,
    ConnectionRegistry,
    create_lifespan,
    make_connection_id,
)


class TestConnectionRegistry:
    def _make_registry(self) -> ConnectionRegistry:
        return ConnectionRegistry(shutdown_timeout_s=0.1)

    async def _make_info(self, cid: str) -> ConnectionInfo:
        task = asyncio.create_task(asyncio.sleep(100))
        return ConnectionInfo(
            connection_id=cid,
            client_id="user1",
            connected_at=0.0,
            task=task,
        )

    async def test_register_increases_count(self):
        reg = self._make_registry()
        info = await self._make_info("c1")
        reg.register(info)
        assert reg.active_count == 1
        info.task.cancel()

    async def test_deregister_decreases_count(self):
        reg = self._make_registry()
        info = await self._make_info("c1")
        reg.register(info)
        reg.deregister("c1")
        assert reg.active_count == 0
        info.task.cancel()

    async def test_get_returns_info(self):
        reg = self._make_registry()
        info = await self._make_info("c1")
        reg.register(info)
        assert reg.get("c1") is info
        info.task.cancel()

    def test_get_missing_returns_none(self):
        reg = self._make_registry()
        assert reg.get("nonexistent") is None

    def test_deregister_missing_no_error(self):
        reg = self._make_registry()
        reg.deregister("nonexistent")  # Should not raise

    async def test_close_all_empty(self):
        reg = self._make_registry()
        count = await reg.close_all()
        assert count == 0

    async def test_close_all_cancels_tasks(self):
        reg = self._make_registry()
        info = await self._make_info("c1")
        reg.register(info)
        count = await reg.close_all()
        assert count == 1
        assert info.task.cancelled() or info.task.done()

    async def test_close_all_sets_shutting_down(self):
        reg = self._make_registry()
        await reg.close_all()
        assert reg.is_shutting_down is True

    async def test_close_all_clears_registry(self):
        reg = self._make_registry()
        info = await self._make_info("c1")
        reg.register(info)
        await reg.close_all()
        assert reg.active_count == 0


class TestCreateLifespan:
    async def test_lifespan_yields_registry(self):
        async with create_lifespan(shutdown_timeout_s=0.1) as state:
            assert "registry" in state
            assert isinstance(state["registry"], ConnectionRegistry)

    async def test_lifespan_closes_on_exit(self):
        async with create_lifespan(shutdown_timeout_s=0.1) as state:
            reg = state["registry"]
        # After exit, registry should be cleaned up
        assert reg.active_count == 0


class TestMakeConnectionId:
    def test_returns_string(self):
        assert isinstance(make_connection_id(), str)

    def test_unique_ids(self):
        ids = {make_connection_id() for _ in range(20)}
        assert len(ids) == 20

    def test_length(self):
        cid = make_connection_id()
        assert len(cid) == 8


class TestCloseCode:
    def test_going_away_value(self) -> None:
        assert CloseCode.GOING_AWAY.value == 1001

    def test_normal_value(self) -> None:
        assert CloseCode.NORMAL.value == 1000
