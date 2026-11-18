"""The CLIENT-RESEND rerun path must carry a REAL ceiling.

``compile_job_store.accept`` re-runs an ``interrupted`` job on a client resend and
guards that rerun with the reaper's two caps (attempts / wall-clock lifetime). But
BOTH inputs to that guard — ``attempts`` and ``started_at`` — used to be written
ONLY by ``reclaim_next_compile_job``, and the insert explicitly stamps
``started_at=None``. So for a job the reaper never claims (the normal case: the
orchestrator's ``BaseServiceClient`` retries the POST immediately, long before the
~30s reaper sweep; also ``reaper_enabled=false``, also a pod already at
``max_concurrent_compile_jobs`` where the sweep breaks):

    attempts >= max_attempts  →  0 >= 3        — False forever
    over_lifetime             →  started_at is None — False forever

…so the gate could not fire on the very path it guards, and drain → resend →
drain → resend re-ran a PAID render without bound.

The fix mirrors ``reclaim_next_compile_job``'s claim EXACTLY on this path:
``started_at`` is anchored unconditionally (COALESCE, first-start wins) and the
attempt is charged for every rerun EXCEPT one whose row was flipped by a graceful
drain. That preserves the deliberate drain exemption (a rollout must not spend a
poison life) while giving the drain path the wall-clock backstop the exemption's
own rationale names.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import CompileJob
from src.services import compile_job_store as job_store
from tests.conftest import TENANT_A

pytestmark = pytest.mark.asyncio

_MAX_CONCURRENT = 8
_OPTIONS: dict[str, object] = {"shots": []}


async def _submit(
    factory: async_sessionmaker[AsyncSession],
    *,
    key: str,
    max_attempts: int = 3,
    max_lifetime_seconds: int = 7200,
    instance_id: str = "pod-A",
) -> job_store.AcceptResult:
    """One submit/resend in its own committed txn (mirrors a route request)."""
    async with factory() as session:
        result = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key=key,
            kind="compile",
            instance_id=instance_id,
            active_count=0,
            max_concurrent=_MAX_CONCURRENT,
            job_options=_OPTIONS,
            lease_seconds=60,
            max_attempts=max_attempts,
            max_lifetime_seconds=max_lifetime_seconds,
        )
        await session.commit()
        return result


async def _interrupt(
    factory: async_sessionmaker[AsyncSession],
    job_id: str,
    *,
    reason: str,
    instance_id: str = "pod-A",
) -> None:
    async with factory() as session:
        await job_store.mark_interrupted(session, job_id, "boom", instance_id, reason=reason)
        await session.commit()


async def _row(factory: async_sessionmaker[AsyncSession], job_id: str) -> CompileJob:
    async with factory() as session:
        return (
            await session.execute(select(CompileJob).where(CompileJob.job_id == job_id))
        ).scalar_one()


async def _backdate_start(
    factory: async_sessionmaker[AsyncSession], job_id: str, seconds: int
) -> None:
    """Age the job's RUN anchor without touching anything else."""
    row = await _row(factory, job_id)
    assert row.started_at is not None, "nothing to backdate — started_at was never stamped"
    async with factory() as session:
        await session.execute(
            update(CompileJob)
            .where(CompileJob.job_id == job_id)
            .values(started_at=row.started_at - timedelta(seconds=seconds))
        )
        await session.commit()


async def test_resend_rerun_anchors_started_at_so_the_lifetime_cap_can_fire(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A rerun IS a start of real render work, so it must anchor the lifetime cap.

    RED before the fix: ``started_at`` stayed NULL through any number of reruns, so
    ``over_lifetime`` short-circuited on ``started_at is not None`` forever.
    """
    first = await _submit(db_session_factory, key="anchor")
    assert first.action == "spawn"
    assert (await _row(db_session_factory, first.job_id)).started_at is None

    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
    again = await _submit(db_session_factory, key="anchor")
    assert again.action == "rerun"

    row = await _row(db_session_factory, first.job_id)
    assert row.started_at is not None, "the rerun left the lifetime cap unanchored"


async def test_first_anchor_wins_a_later_rerun_does_not_reset_the_lifetime_clock(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """COALESCE, not overwrite: otherwise every resend refreshes the clock and the
    wall-clock cap is defeated by resending often enough."""
    first = await _submit(db_session_factory, key="coalesce")
    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
    await _submit(db_session_factory, key="coalesce")
    anchored = (await _row(db_session_factory, first.job_id)).started_at
    assert anchored is not None

    await _backdate_start(db_session_factory, first.job_id, 3600)
    aged = (await _row(db_session_factory, first.job_id)).started_at

    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
    assert (await _submit(db_session_factory, key="coalesce")).action == "rerun"
    assert (await _row(db_session_factory, first.job_id)).started_at == aged


async def test_drain_resend_loop_is_bounded_by_the_wall_clock_lifetime_cap(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """drain → resend → drain → resend is the NORMAL rollout case and is attempt-EXEMPT
    by design; the lifetime cap must therefore terminate it.

    RED before the fix: the resend never anchored ``started_at``, so a drained job
    could be re-run (and re-billed) forever.
    """
    first = await _submit(db_session_factory, key="drain-loop")
    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
    assert (await _submit(db_session_factory, key="drain-loop")).action == "rerun"

    # The render has now been alive far longer than the configured ceiling.
    await _backdate_start(db_session_factory, first.job_id, 10_000)
    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)

    poisoned = await _submit(db_session_factory, key="drain-loop", max_lifetime_seconds=7200)
    assert poisoned.action == "dedup", "an over-lifetime drain resend was re-run again"
    row = await _row(db_session_factory, first.job_id)
    assert row.status == "failed"
    assert "poison" in (row.error or "")
    assert row.lease_expires_at is None


async def test_non_drain_resend_loop_is_bounded_by_the_attempt_cap(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A crash-looping (non-drain) render resent by the client must burn poison lives
    exactly as a reaper reclaim does, and hard-fail at the cap.

    RED before the fix: ``attempts`` stayed 0 forever on this path, so the rerun
    UPDATE fired on every one of an unbounded number of resends.
    """
    first = await _submit(db_session_factory, key="crash-loop", max_attempts=3)
    assert first.action == "spawn"

    for expected in (1, 2, 3):
        await _interrupt(
            db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_RENDER_FAILURE
        )
        assert (await _submit(db_session_factory, key="crash-loop", max_attempts=3)).action == (
            "rerun"
        )
        assert (await _row(db_session_factory, first.job_id)).attempts == expected

    await _interrupt(
        db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_RENDER_FAILURE
    )
    poisoned = await _submit(db_session_factory, key="crash-loop", max_attempts=3)
    assert poisoned.action == "dedup", "the client-resend path re-ran a job at the poison cap"
    row = await _row(db_session_factory, first.job_id)
    assert row.status == "failed"
    assert "poison" in (row.error or "")


async def test_drain_resend_charges_zero_attempts_exemption_preserved(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """GUARD (green before AND after): the deliberate drain exemption must survive.

    ``reclaim_max_attempts`` is 3, so with an unconditional increment on the resend
    path four rollouts would poison-fail a perfectly healthy PAID render — the exact
    regression migration 008's exemption exists to prevent. Charging the attempt
    unconditionally here fails this test.
    """
    first = await _submit(db_session_factory, key="rollouts", max_attempts=3)
    for _ in range(6):
        await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
        assert (await _submit(db_session_factory, key="rollouts", max_attempts=3)).action == (
            "rerun"
        ), "a rollout spent a poison life"
    row = await _row(db_session_factory, first.job_id)
    assert row.attempts == 0
    assert row.status == "processing"


async def test_a_real_fault_after_drains_still_burns_a_life(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The exemption is an ALLOWLIST OF ONE: only ``drain`` is free. A NULL/unknown
    reason (legacy row, older pod) still costs an attempt."""
    first = await _submit(db_session_factory, key="mixed", max_attempts=3)
    await _interrupt(db_session_factory, first.job_id, reason=job_store.INTERRUPT_REASON_DRAIN)
    await _submit(db_session_factory, key="mixed", max_attempts=3)
    assert (await _row(db_session_factory, first.job_id)).attempts == 0

    async with db_session_factory() as session:
        await session.execute(
            update(CompileJob)
            .where(CompileJob.job_id == first.job_id)
            .values(status="interrupted", interrupt_reason=None)
        )
        await session.commit()
    await _submit(db_session_factory, key="mixed", max_attempts=3)
    assert (await _row(db_session_factory, first.job_id)).attempts == 1


async def test_route_passes_both_ceilings_into_accept() -> None:
    """Pin the PRODUCTION call site: the store-side ceiling is inert unless the route
    forwards both settings on every submit."""
    import inspect

    from src.api.routes import compile as compile_route

    src = inspect.getsource(compile_route._accept)
    assert "max_attempts=settings.reclaim_max_attempts" in src
    assert "max_lifetime_seconds=settings.reclaim_max_lifetime_seconds" in src
    assert "lease_seconds=settings.job_lease_seconds" in src
