"""The SIGTERM drain must be BOUNDED by its own grace budget.

``drain_background_tasks`` cancels every in-flight render then awaits them. Each
task's ``CancelledError`` handler awaits a **shielded** ``_stamp_interrupted``
(``compile.py``), and a shield cannot be cancelled — so one stalled pgbouncer
write blocked the whole ``gather`` for as long as that write took.

Consequence, and it costs money: when the drain overruns Kubernetes'
``terminationGracePeriodSeconds`` the pod is SIGKILLed, no
``interrupt_reason='drain'`` is ever written, and the sibling's reclaim falls to
the SQL's ``ELSE 1`` — charging a poison attempt against a render that was only
rolled out. Migration 008's exemption protects the graceful path and the
ungraceful path is exactly the one it cannot cover, so the drain has to STAY
graceful: a single stalled write must never consume the grace period.
"""

from __future__ import annotations

import asyncio

import pytest

from src.api.routes import compile as compile_routes

pytestmark = pytest.mark.asyncio


async def _stalled_task() -> asyncio.Task[None]:
    """A render task whose cancellation handler blocks on a shielded write, exactly
    like ``_spawn._run``'s ``CancelledError`` branch does."""
    running = asyncio.Event()

    async def _body() -> None:
        try:
            running.set()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # The shield is the point: this await CANNOT be cancelled.
            await asyncio.shield(asyncio.sleep(3600))
            raise

    task = asyncio.create_task(_body())
    await running.wait()
    return task


async def test_drain_returns_within_its_grace_budget_despite_a_stalled_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before the fix: the unbounded ``gather`` waits on the shielded write for
    the full hour, so the drain never returns inside the grace period."""
    monkeypatch.setattr(compile_routes, "_DRAIN_GRACE_SECONDS", 0.2)
    task = await _stalled_task()
    compile_routes._background_tasks.add(task)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await asyncio.wait_for(compile_routes.drain_background_tasks(), timeout=5.0)
        assert loop.time() - started < 2.0, "drain overran its 0.2s grace budget"
    finally:
        task.cancel()
        compile_routes._background_tasks.discard(task)


async def test_drain_grace_is_below_the_pods_termination_grace_period() -> None:
    """The budget only helps if it expires BEFORE the kubelet's SIGKILL. 30s is the
    k8s default ``terminationGracePeriodSeconds``; the drain must leave headroom for
    the rest of the shutdown (reaper stop, metering flush, engine dispose)."""
    assert 0 < compile_routes._DRAIN_GRACE_SECONDS < 30.0


async def test_a_healthy_drain_still_awaits_every_task_to_completion() -> None:
    """GUARD: the bound must not turn the drain into fire-and-forget. A render that
    cancels promptly is still fully awaited — that is what lets its handler stamp
    ``interrupted`` and keeps the sibling's reclaim attempt-exempt."""
    stamped = asyncio.Event()

    async def _body() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stamped.set()
            raise

    task = asyncio.create_task(_body())
    await asyncio.sleep(0)
    compile_routes._background_tasks.add(task)
    try:
        await compile_routes.drain_background_tasks()
        assert stamped.is_set()
        assert task.done()
    finally:
        compile_routes._background_tasks.discard(task)


async def test_drain_never_propagates_a_render_failure() -> None:
    """GUARD: a task that dies with a real exception must not abort the shutdown
    path (the old ``gather(return_exceptions=True)`` contract)."""

    async def _boom() -> None:
        raise RuntimeError("render blew up")

    task = asyncio.create_task(_boom())
    compile_routes._background_tasks.add(task)
    try:
        await compile_routes.drain_background_tasks()
        assert task.done(), "drain returned without reaping the failed task"
    finally:
        compile_routes._background_tasks.discard(task)
