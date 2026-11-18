"""Crash-resumption reaper: the proof that a killed render resumes exactly once.

The durable store (residual #2) stops a *resend* from double-rendering, but a render
whose pod dies mid-flight used to just stall as a stuck ``processing`` row. These tests
prove the reaper layer:

- ``reclaim_next`` claims a crash-orphan (interrupted, or processing past its lapsed
  lease) and returns its reconstruction bundle — while NEVER claiming a render whose
  lease is still live (a healthy pod is heartbeating), an unmanaged row (no job_options),
  or a poison/over-lifetime job (which it hard-fails instead);
- the pre-existing MONEY BUG is fixed: a superseded terminal write loses the
  ``WHERE status='processing'`` race and ``mark_completed`` returns False, so its cost
  event is suppressed — with an executable REVERT-PROOF that ungating it double-emits;
- a re-dispatch of a ``file://`` (pod-affine) input fail-fasts to ``failed`` instead of
  looping or crashing the reaper.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.database.models import CompileJob
from src.services import compile_job_store as job_store
from tests.conftest import TENANT_A

pytestmark = pytest.mark.asyncio

_HTTP_REQUEST: dict[str, object] = {
    "shots": [{"line_id": "l1", "seconds": 2.0, "image_url": "https://minio/x.png"}],
}
_FILE_REQUEST: dict[str, object] = {
    "shots": [{"line_id": "l1", "seconds": 2.0, "image_url": "file:///tmp/pod-local.png"}],
}


def _options(request: dict[str, object], *, kind: str = "compile") -> dict[str, object]:
    return {
        "kind": kind,
        "request": request,
        "tenant_id": str(TENANT_A),
        "license_id": "",
        "subject_id": "",
        "work_timeout": 60,
    }


async def _seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    status: str,
    lease_delta_s: float | None,
    job_options: dict[str, object] | None,
    instance_id: str = "dead-pod",
    attempts: int = 0,
    started_delta_s: float | None = None,
    idempotency_key: str | None = None,
    interrupt_reason: str | None = None,
) -> str:
    """Insert one compile_jobs row; return its job_id."""
    now = datetime.now(UTC)
    async with factory() as session:
        job = CompileJob(
            tenant_id=TENANT_A,
            idempotency_key=idempotency_key,
            kind="compile",
            status=status,
            instance_id=instance_id,
            attempts=attempts,
            interrupt_reason=interrupt_reason,
            job_options=job_options,
            lease_expires_at=(now + timedelta(seconds=lease_delta_s))
            if lease_delta_s is not None
            else None,
            started_at=(now + timedelta(seconds=started_delta_s))
            if started_delta_s is not None
            else None,
        )
        session.add(job)
        await session.commit()
        return job.job_id


async def _reclaim(
    factory: async_sessionmaker[AsyncSession],
    *,
    worker: str = "live-pod",
    max_attempts: int = 3,
    max_lifetime: int = 0,
) -> job_store.ClaimedJob | job_store._PoisonSkipped | None:
    async with factory() as session:
        claimed = await job_store.reclaim_next(
            session,
            worker_id=worker,
            lease_seconds=120,
            max_attempts=max_attempts,
            max_lifetime_seconds=max_lifetime,
        )
        await session.commit()
        return claimed


async def _get(factory: async_sessionmaker[AsyncSession], job_id: str) -> CompileJob:
    async with factory() as session:
        return (
            await session.execute(select(CompileJob).where(CompileJob.job_id == job_id))
        ).scalar_one()


# ---------------------------------------------------------------------------
# reclaim_next — the SQL claim core
# ---------------------------------------------------------------------------


async def test_reclaims_crash_orphan_interrupted_job_with_options(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A CRASH-orphan (interrupted, reason crash_orphan) with job_options is claimed + charged.

    The reason matters: only ``drain`` is exempt from the attempt increment, so a row the
    startup reclaim declared orphaned still costs one of the poison budget's lives.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        interrupt_reason=job_store.INTERRUPT_REASON_CRASH_ORPHAN,
    )
    claimed = await _reclaim(db_session_factory, worker="live-pod")
    assert isinstance(claimed, job_store.ClaimedJob)
    assert claimed.job_id == job_id
    assert claimed.options["request"] == _HTTP_REQUEST
    row = await _get(db_session_factory, job_id)
    assert row.status == "processing"
    assert row.instance_id == "live-pod"  # ownership transferred
    assert row.attempts == 1  # a reclaim consumed one attempt
    assert row.lease_expires_at is not None and row.lease_expires_at > datetime.now(UTC)


async def test_reclaims_processing_with_lapsed_lease(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A 'processing' row whose lease has LAPSED (owning pod died) is reclaimable."""
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=-5,
        job_options=_options(_HTTP_REQUEST),
    )
    claimed = await _reclaim(db_session_factory)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id


async def test_never_reclaims_live_lease(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A 'processing' row with a LIVE lease (a healthy pod is heartbeating) is untouched.

    This is the guard against a spurious double-render: the reaper must never grab a job
    still running on a live pod.
    """
    await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=+300,
        job_options=_options(_HTTP_REQUEST),
    )
    assert await _reclaim(db_session_factory) is None


async def test_never_reclaims_unmanaged_row(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A row with NO job_options (unmanaged / pre-reaper) is never reclaimed."""
    await _seed(db_session_factory, status="interrupted", lease_delta_s=-30, job_options=None)
    assert await _reclaim(db_session_factory) is None


async def test_poison_job_is_hard_failed_not_reclaimed(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A job already at the attempt cap is failed as poison instead of re-run."""
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=3,
        interrupt_reason=job_store.INTERRUPT_REASON_CRASH_ORPHAN,
    )
    claimed = await _reclaim(db_session_factory, max_attempts=3)
    # POISON_SKIPPED (not None): the poison job was CONSUMED here, so the reaper's sweep
    # must CONTINUE past it, not break and stall the healthy orphans queued behind it.
    assert claimed is job_store.POISON_SKIPPED
    row = await _get(db_session_factory, job_id)
    assert row.status == "failed"
    assert row.error is not None and "poison" in row.error


async def test_max_attempts_zero_disables_poison_gate_not_fails_first_reclaim(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`p_max_attempts=0` means 'attempts cap disabled', NOT 'poison on first reclaim'.

    Without the ``p_max_attempts > 0`` guard, ``COALESCE(0,0) >= 0`` is true and a
    freshly-interrupted job (attempts=0) would be poison-failed before it ever ran
    again. Unreachable via config (bounded ge=1) but a latent inconsistency; the
    lifetime clause is already guarded the same way.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        interrupt_reason=job_store.INTERRUPT_REASON_CRASH_ORPHAN,
    )
    claimed = await _reclaim(db_session_factory, max_attempts=0)  # cap disabled
    assert isinstance(claimed, job_store.ClaimedJob)  # reclaimed for re-dispatch, NOT poison-failed
    assert claimed.job_id == job_id
    row = await _get(db_session_factory, job_id)
    assert row.status == "processing"
    assert "poison" not in (row.error or "")


async def test_over_lifetime_job_is_hard_failed(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A job that has RUN longer than the lifetime bound is failed, not re-run."""
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        started_delta_s=-7200,  # started 2h ago
    )
    claimed = await _reclaim(db_session_factory, max_lifetime=3600)  # 1h ceiling
    assert claimed is job_store.POISON_SKIPPED  # over-lifetime consumed → sweep continues
    row = await _get(db_session_factory, job_id)
    assert row.status == "failed"


async def test_concurrent_reapers_claim_at_most_once(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """FOR UPDATE SKIP LOCKED: two reapers racing the SAME orphan → exactly one claim."""
    import asyncio

    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
    )
    a, b = await asyncio.gather(
        _reclaim(db_session_factory, worker="pod-a"),
        _reclaim(db_session_factory, worker="pod-b"),
    )
    winners = [c for c in (a, b) if isinstance(c, job_store.ClaimedJob) and c.job_id == job_id]
    assert len(winners) == 1  # never both


# ---------------------------------------------------------------------------
# interrupt_reason: a ROLLOUT must not spend a poison life (2026-08-06)
#
# A graceful shutdown drain writes status='interrupted'; a sibling replica reclaims it
# within ~30s and the claim used to charge attempts+1 unconditionally. With
# reclaim_max_attempts=3 that meant FOUR rollouts poison-failed a healthy PAID render.
# The claim now exempts the increment for interrupted+'drain' ONLY — an allowlist of one.
# ---------------------------------------------------------------------------


async def test_drain_interrupt_does_not_consume_an_attempt(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A reclaim of a DRAIN-interrupted row is free: attempts stays put, the row re-runs.

    MUTATION-PROOF: change ``THEN 0`` to ``THEN 1`` in reclaim_next_compile_job's CASE and
    attempts becomes 1 — the pre-fix behaviour this test exists to forbid.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        interrupt_reason=job_store.INTERRUPT_REASON_DRAIN,
    )
    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    row = await _get(db_session_factory, job_id)
    assert row.status == "processing"  # re-dispatched
    assert row.attempts == 0  # the rollout cost NOTHING
    # Cleared on the claim: a stale 'drain' would exempt every LATER crash of this row.
    assert row.interrupt_reason is None


async def test_four_rollout_drains_do_not_poison_a_healthy_render(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """THE REPORTED BUG: four rollouts must not hard-fail a healthy paid render.

    reclaim_max_attempts is 3, so with an unconditional attempts+1 the fourth drain→reclaim
    cycle poison-failed a render that had never once faulted. Each iteration re-stamps the
    drain exactly as ``_stamp_interrupted(..., reason=drain)`` does on SIGTERM.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        interrupt_reason=job_store.INTERRUPT_REASON_DRAIN,
    )
    for cycle in range(4):
        claimed = await _reclaim(db_session_factory, worker="pod-live", max_attempts=3)
        assert isinstance(claimed, job_store.ClaimedJob), f"poisoned on rollout {cycle + 1}"
        row = await _get(db_session_factory, job_id)
        assert row.status == "processing"
        assert row.attempts == 0
        # The next rollout drains it again (what mark_interrupted writes on SIGTERM).
        async with db_session_factory() as session:
            await job_store.mark_interrupted(
                session,
                job_id,
                "cancelled — service shutting down",
                "pod-live",
                reason=job_store.INTERRUPT_REASON_DRAIN,
            )
            await session.commit()

    row = await _get(db_session_factory, job_id)
    assert row.status == "interrupted"  # never hard-failed
    assert "poison" not in (row.error or "")
    assert row.attempts == 0


async def test_lifetime_cap_still_poisons_an_endlessly_drained_render(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """THE BACKSTOP: the drain exemption is bounded by wall clock, not by attempts.

    ``test_four_rollout_drains_do_not_poison_a_healthy_render`` proves a drain costs no
    attempt — which alone would be an UNBOUNDED re-render of a paid job. The claim anchors
    ``started_at`` unconditionally, so ``reclaim_max_lifetime_seconds`` still terminates
    the drain->rerun loop. This is the invariant the PR's headline rests on and nothing
    else exercises it: every other drain test uses ``_reclaim``'s ``max_lifetime=0``
    default (cap disabled), which is exactly why ``Settings.reclaim_max_lifetime_seconds``
    is now ``ge=1`` — the backstop must not be switchable off.

    REVERT-PROOF: delete the ``p_max_lifetime_seconds > 0 AND v_started < ...`` disjunct
    from the poison gate in ``reclaim_next_compile_job`` and this fails — the drained row
    is claimed forever (attempts stays 0, so the attempt disjunct never fires).
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,  # a drain never charged one — the attempt cap can NEVER fire here
        started_delta_s=-7200,  # ran 2h, past the 3600s ceiling
        interrupt_reason=job_store.INTERRUPT_REASON_DRAIN,
    )

    claimed = await _reclaim(db_session_factory, max_attempts=3, max_lifetime=3600)

    # Poison sentinel, not a claim: the reaper CONTINUES its sweep past it.
    assert isinstance(claimed, job_store._PoisonSkipped), claimed
    row = await _get(db_session_factory, job_id)
    assert row.status == "failed"
    assert row.attempts == 0  # proves the ATTEMPT cap did not do this — the clock did
    assert "poison" in (row.error or "")


async def test_resend_of_a_drained_job_clears_the_stale_interrupt_reason(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A client resend of a drained row must CLEAR ``interrupt_reason`` on its claim.

    ``accept()``'s rerun claim is the second way a row goes interrupted -> processing (the
    reaper's ``reclaim_next_compile_job`` is the first, and it clears the column too). A
    stale ``'drain'`` surviving here would exempt the NEXT genuine crash of this same row
    from the poison-attempt increment, making a crash-looping paid render unbounded.

    REVERT-PROOF: delete ``interrupt_reason=None`` from the claim ``.values()`` in
    ``compile_job_store.accept`` and this fails with a leftover ``'drain'``.
    """
    key = "drained-resend-key"
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        idempotency_key=key,
        interrupt_reason=job_store.INTERRUPT_REASON_DRAIN,
    )
    async with db_session_factory() as session:
        res = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key=key,
            kind="compile",
            instance_id="live-pod",
            active_count=0,
            max_concurrent=8,
            job_options=_options(_HTTP_REQUEST),
            lease_seconds=120,
            max_attempts=3,
            max_lifetime_seconds=3600,
        )
        await session.commit()
    assert res.action == "rerun"
    assert res.job_id == job_id

    row = await _get(db_session_factory, job_id)
    assert row.status == "processing"
    assert row.interrupt_reason is None  # the stale 'drain' is gone


async def test_zombie_prune_interrupt_still_consumes_an_attempt(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A zombie-pruned row (its render task died) is a FAULT — the reclaim charges it.

    MUTATION-PROOF against the forbidden shortcut "never increment for interrupted": the
    zombie prune flips genuine crash-loops to ``interrupted`` too, so blanket-exempting the
    status would make a crash-looping render unbounded.
    """
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
    )
    # Age the heartbeat past the prune's work-timeout window so the zombie flip fires.
    async with db_session_factory() as session:
        await session.execute(
            text("UPDATE compile_jobs SET heartbeat_at = now() - interval '1 hour'")
        )
        await session.commit()
    async with db_session_factory() as session:
        await job_store.prune(session, ttl_seconds=86400, work_timeout_seconds=60)
        await session.commit()

    row = await _get(db_session_factory, job_id)
    assert row.status == "interrupted"
    assert row.interrupt_reason == job_store.INTERRUPT_REASON_CRASH_ZOMBIE

    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    assert (await _get(db_session_factory, job_id)).attempts == 1  # the fault was charged


async def test_startup_reclaim_interrupt_still_consumes_an_attempt(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A startup-reclaimed orphan (its pod died without draining) is charged an attempt."""
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        instance_id="dead-pod",
        attempts=0,
    )
    async with db_session_factory() as session:
        assert await job_store.reclaim_orphans(session, "booting-pod") == 1
        await session.commit()

    row = await _get(db_session_factory, job_id)
    assert row.status == "interrupted"
    assert row.interrupt_reason == job_store.INTERRUPT_REASON_CRASH_ORPHAN

    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    assert (await _get(db_session_factory, job_id)).attempts == 1


async def test_legacy_null_interrupt_reason_consumes_an_attempt(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A NULL reason (a pre-migration row, or one written by an older pod mid-rollout)
    fails SAFE and costs an attempt.

    MUTATION-PROOF against rewriting the CASE as a DENYLIST (e.g.
    ``WHEN v_reason IS DISTINCT FROM 'render_failure' THEN 0``): NULL compares UNKNOWN, so
    only an explicit allowlist of one keeps legacy rows on the charging path.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        interrupt_reason=None,  # legacy
    )
    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    assert (await _get(db_session_factory, job_id)).attempts == 1


async def test_render_failure_reason_consumes_an_attempt(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A transient ffmpeg crash (RetryableRenderError → 'render_failure') IS charged.

    Pairs with ``test_transient_render_error_is_interrupted_not_failed`` in
    test_render_lifecycle.py, which proves that path passes ``reason='render_failure'``.
    Mislabelling it 'drain' would make a crash-looping render reclaimable forever — this
    is the DB half of that guarantee.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        interrupt_reason=job_store.INTERRUPT_REASON_RENDER_FAILURE,
    )
    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    assert (await _get(db_session_factory, job_id)).attempts == 1


async def test_drain_row_already_at_the_cap_is_still_poison_failed(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The drain exemption spares the INCREMENT, never the CAP.

    A row that already burned ``max_attempts`` on REAL faults and is then drained must
    still be hard-failed — otherwise a single rollout would resurrect every poison job.
    MUTATION-PROOF: move the reason test into the poison gate (skip the gate when
    reason='drain') and this row is re-dispatched instead of failed.
    """
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=3,
        interrupt_reason=job_store.INTERRUPT_REASON_DRAIN,
    )
    claimed = await _reclaim(db_session_factory, max_attempts=3)
    assert claimed is job_store.POISON_SKIPPED  # the sentinel, not a re-dispatch
    row = await _get(db_session_factory, job_id)
    assert row.status == "failed"
    assert row.error is not None and "poison" in row.error


async def test_sql_case_pins_the_python_drain_literal() -> None:
    """The exempting SQL literal and the Python constant must be the SAME string.

    They live in different files with no import between them, so a rename on either side
    would silently stop exempting drains (every value would fall to ELSE 1) with no test
    failure anywhere else. MUTATION-PROOF: change either spelling and this fails.
    """
    from src.database.rls_resolvers import RECLAIM_NEXT_FN_SQL

    assert f"'{job_store.INTERRUPT_REASON_DRAIN}'" in RECLAIM_NEXT_FN_SQL


# ---------------------------------------------------------------------------
# The money bug — mark_completed win-bool gates cost metering
# ---------------------------------------------------------------------------


async def test_superseded_terminal_write_loses_and_reports_false(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Only the FIRST terminal write wins the WHERE status='processing' race.

    The loser (a superseded original racing a reclaimed re-run of the same job_id) gets
    False from mark_completed — which is exactly what the route gates the cost event on,
    so the loser never emits a second paid metering event.
    """
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=+120,
        job_options=_options(_HTTP_REQUEST),
    )
    async with db_session_factory() as s1:
        won1 = await job_store.mark_completed(s1, job_id, {"artifact_url": "a"})
        await s1.commit()
    async with db_session_factory() as s2:
        won2 = await job_store.mark_completed(s2, job_id, {"artifact_url": "b"})
        await s2.commit()
    assert won1 is True
    assert won2 is False  # the money guard: the loser is silenced


async def test_emit_cost_is_gated_on_the_terminal_write_winner(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """REVERT-PROOF for the money bug: cost is emitted iff THIS attempt won the row.

    Drives the real ``_spawn`` completion path with a stubbed render + a counting metering
    emit. Two runs of the SAME job_id (a reclaim overlapping the original) must emit
    EXACTLY ONE cost event. Ungating (emitting regardless of the win-bool) would emit two —
    the pre-fix double-bill.
    """
    import asyncio

    from src.api.routes import compile as compile_routes

    emits: list[str] = []
    monkeypatch.setattr(
        compile_routes, "_emit_cost", lambda **kw: emits.append(str(kw.get("job_type")))
    )

    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=+120,
        job_options=_options(_HTTP_REQUEST),
    )

    async def _work() -> dict[str, object]:
        return {"artifact_url": "x", "duration_seconds": 3.0}

    # Two concurrent completions of the SAME job_id — only the first flips the row.
    compile_routes._spawn(
        job_id, _work, tenant_id=TENANT_A, work_timeout=60.0, job_type="compile", lease_seconds=0
    )
    compile_routes._spawn(
        job_id, _work, tenant_id=TENANT_A, work_timeout=60.0, job_type="compile", lease_seconds=0
    )
    await asyncio.gather(*list(compile_routes._background_tasks), return_exceptions=True)

    assert emits == ["compile"]  # exactly one paid cost event, never two


# ---------------------------------------------------------------------------
# Re-dispatch reconstructability — fail-fast, never loop
# ---------------------------------------------------------------------------


async def test_redispatch_file_input_fails_fast(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A reclaimed job whose input is a pod-affine file:// URL is failed, not re-run.

    A file:// input can't be resolved on another pod, so re-running would render a broken
    compile (or crash). The reaper must stamp it 'failed' (terminal, not reclaimable) so it
    dies after ONE cycle instead of looping.
    """
    from src.api.routes.compile import redispatch_reclaimed

    # reclaim_next stamps the row's instance_id to the reaper's before redispatch, so the
    # row is owned by "live-pod" when the fenced terminal write lands.
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=+120,
        job_options=_options(_FILE_REQUEST),
        instance_id="live-pod",
    )
    claimed = job_store.ClaimedJob(
        job_id=job_id, tenant_id=TENANT_A, options=_options(_FILE_REQUEST)
    )
    await redispatch_reclaimed(claimed, lease_seconds=0, instance_id="live-pod")

    row = await _get(db_session_factory, job_id)
    assert row.status == "failed"
    assert row.error is not None and "reconstructable" in row.error


# ---------------------------------------------------------------------------
# Adversarial hardening (2026-07-23): the reaper must not over-admit past the
# per-pod render cap, and must never double-render a job THIS pod is still running.
# ---------------------------------------------------------------------------


async def test_reaper_respects_concurrency_cap(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """One sweep never dispatches past ``max_concurrent_compile_jobs``.

    A Recreate can leave a large backlog of reclaimable orphans; without the capacity
    gate the reaper would fire ALL of them (up to _MAX_RECLAIMS_PER_SWEEP) as detached
    renders in one sweep, blowing past the cap and piling that many scratch dirs into
    /tmp. REVERT-PROOF: drop the gate in reap_once and this dispatches 5, not 2.
    """
    from fastapi import FastAPI

    from src.api.routes import compile as compile_routes
    from src.services.reaper_service import ReaperService

    for _ in range(5):
        await _seed(
            db_session_factory,
            status="interrupted",
            lease_delta_s=-30,
            job_options=_options(_HTTP_REQUEST),
        )

    dispatched: list[str] = []
    # The in-flight gauge climbs as each reclaimed render is dispatched (a real _spawn
    # adds to _background_tasks synchronously), so the reaper sees the count rise.
    monkeypatch.setattr(compile_routes, "_active_render_count", lambda: len(dispatched))

    async def _fake_redispatch(
        claimed: job_store.ClaimedJob, *, lease_seconds: int, instance_id: str
    ) -> None:
        dispatched.append(claimed.job_id)

    monkeypatch.setattr(compile_routes, "redispatch_reclaimed", _fake_redispatch)

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._max_concurrent = 2
    n = await reaper.reap_once()

    assert n == 2  # capped, not 5
    assert len(dispatched) == 2
    # The remaining 3 orphans are untouched and still reclaimable next sweep.
    assert await _reclaim(db_session_factory) is not None


async def test_reaper_continues_past_head_of_queue_poison_job(
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A poison job at HEAD-of-queue must not stall a healthy orphan queued behind it.

    reclaim_next consumes (hard-fails) the poison job and returns ``POISON_SKIPPED`` — a
    sentinel DISTINCT from ``None`` — so the reaper CONTINUES its sweep and re-dispatches
    the healthy orphan in the SAME sweep. REVERT-PROOF: change the reaper's
    ``if claimed is POISON_SKIPPED: continue`` back to a ``break`` (the pre-fix behaviour
    where poison and empty-queue were indistinguishable) and the healthy orphan is NEVER
    dispatched — this asserts it IS.
    """
    from fastapi import FastAPI

    from src.api.routes import compile as compile_routes
    from src.services.reaper_service import ReaperService

    # Seeded FIRST → strictly-older created_at → head of the ORDER BY created_at queue.
    await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=5,  # already past the cap → poison, consumed on the first claim
    )
    healthy_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
    )

    dispatched: list[str] = []
    monkeypatch.setattr(compile_routes, "_active_render_count", lambda: 0)  # under cap

    async def _fake_redispatch(
        claimed: job_store.ClaimedJob, *, lease_seconds: int, instance_id: str
    ) -> None:
        dispatched.append(claimed.job_id)

    monkeypatch.setattr(compile_routes, "redispatch_reclaimed", _fake_redispatch)

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._max_attempts = 3  # the poison job (attempts=5) exceeds this
    n = await reaper.reap_once()

    assert dispatched == [healthy_id]  # the healthy orphan ran despite the poison at head
    assert n == 1


async def test_self_reclaim_of_live_render_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """redispatch_reclaimed refuses to re-run a job THIS pod is already rendering.

    If a live render's lease lapses (e.g. a hung heartbeat renew under DB saturation) and
    the same-pod reaper reclaims it, a second dispatch would double-render paid compute
    (the money guard dedups the BILL, not the work). The in-process _active_job_ids guard
    prevents the second spawn. REVERT-PROOF: remove the guard and the second call spawns.
    """
    from src.api.routes import compile as compile_routes

    valid_req: dict[str, object] = {
        "shots": [{"id": "s1", "seconds": 2.0, "image_url": "https://minio/x.png"}],
    }
    spawned: list[str] = []
    monkeypatch.setattr(compile_routes, "_spawn", lambda job_id, *a, **k: spawned.append(job_id))

    job_id = "job-live-123"
    claimed = job_store.ClaimedJob(job_id=job_id, tenant_id=TENANT_A, options=_options(valid_req))

    # Not currently rendering here → the reaper re-dispatches (spawns) it.
    await compile_routes.redispatch_reclaimed(claimed, lease_seconds=0, instance_id="live-pod")
    assert spawned == [job_id]

    # THIS pod is already rendering it → the reaper must NOT spawn a second render.
    spawned.clear()
    compile_routes._active_job_ids.add(job_id)
    try:
        await compile_routes.redispatch_reclaimed(claimed, lease_seconds=0, instance_id="live-pod")
    finally:
        compile_routes._active_job_ids.discard(job_id)
    assert spawned == []  # veto held — no double render


async def test_heartbeat_bounds_a_hung_renew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung lease renew is abandoned (asyncio.wait_for) and retried, never wedging.

    Without the wait_for bound a single stuck renew (a connection wedged on a saturated
    pgbouncer) blocks the heartbeat loop forever, the lease lapses under a healthy render,
    and the same-pod reaper reclaims it. REVERT-PROOF: unwrap the renew and the loop stalls
    on the first hung call — stop is never observed, so `await wait_for(hb)` times out and
    this test fails.
    """
    import asyncio

    from src.services import durable_dispatch

    attempts = 0

    async def _hang(*_a: object, **_k: object) -> None:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(3600)  # never returns on its own

    monkeypatch.setattr(durable_dispatch, "_renew_once", _hang)

    stop = asyncio.Event()
    hb = asyncio.create_task(durable_dispatch._heartbeat("j", "pod", 120, stop, interval=0.05))
    await asyncio.sleep(0.35)  # several beats: each hung renew abandoned after ~0.025s
    stop.set()
    await asyncio.wait_for(hb, timeout=2.0)  # completes promptly iff the loop wasn't wedged

    assert attempts >= 2  # renews kept being retried, never blocked on the first


# ---------------------------------------------------------------------------
# Adversarial round 2 (2026-07-24): a same-pod reclaim of a lease-lapsed-but-LIVE
# render must not touch it at the CLAIM (bump attempts / overwrite instance_id /
# poison-fail a completing render), a client resend must honour the poison/lifetime
# caps, the started_at lifetime anchor must survive a reclaim, and the none_as_null
# invariant is enforced at the storage layer.
# ---------------------------------------------------------------------------


async def test_reclaim_skips_a_job_this_pod_is_actively_rendering(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The SQL claim excludes ``active_job_ids`` — never claims a live self-render.

    If a live render's lease lapses (a multi-beat DB stall) the same-pod reaper used to
    claim its OWN running job: reclaim_next bumps ``attempts``, re-stamps ``instance_id``
    + lease, and at the cap POISON-FAILS a render about to complete. The post-claim veto
    could not undo those side-effects — they already happened inside the function. Passing
    the in-process live set to the claim skips the row BEFORE any mutation.

    REVERT-PROOF inline: with the same row and an EMPTY active set it IS claimed
    (attempts→1), proving the exclusion is the load-bearing guard.
    """
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=-5,  # lapsed
        job_options=_options(_HTTP_REQUEST),
        instance_id="live-pod",
        attempts=0,
    )
    # This pod is actively rendering it → excluded → not claimed, row untouched.
    async with db_session_factory() as session:
        claimed = await job_store.reclaim_next(
            session,
            worker_id="live-pod",
            lease_seconds=120,
            max_attempts=3,
            max_lifetime_seconds=0,
            active_job_ids=[job_id],
        )
        await session.commit()
    assert claimed is None
    row = await _get(db_session_factory, job_id)
    assert row.status == "processing"  # not flipped
    assert row.attempts == 0  # NOT bumped — the poison spiral can never start
    assert row.instance_id == "live-pod"  # NOT overwritten — heartbeat stays matched

    # Revert-proof: drop the exclusion (empty set) and the very same row is claimed.
    async with db_session_factory() as session:
        claimed2 = await job_store.reclaim_next(
            session,
            worker_id="live-pod",
            lease_seconds=120,
            max_attempts=3,
            max_lifetime_seconds=0,
            active_job_ids=[],
        )
        await session.commit()
    assert isinstance(claimed2, job_store.ClaimedJob)
    assert claimed2.job_id == job_id
    assert (await _get(db_session_factory, job_id)).attempts == 1


async def test_reaper_worker_id_is_the_app_instance_id_not_hostname() -> None:
    """The reaper claims under ``app.state.instance_id`` — the SAME id the render path
    stamps and its heartbeat renews with — so a self-reclaim can't orphan the heartbeat.

    REVERT-PROOF: bind ``self._pod_id`` to a hostname again and a self-reclaim would
    overwrite ``instance_id`` so ``renew_lease WHERE instance_id=<uuid7>`` matches 0 rows.
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService

    app = FastAPI()
    app.state.instance_id = "instance-uuid7-abc"
    assert ReaperService(app)._pod_id == "instance-uuid7-abc"


async def test_resend_of_a_poison_job_is_refused_not_rerun(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A client resend of a job already at the reaper attempt cap is hard-failed, not run.

    The poison/lifetime caps live in reclaim_next; accept()'s interrupted→processing
    rerun bypassed them, so a resend (BaseServiceClient retries POSTs) re-rendered an
    already-poison job unbounded. REVERT-PROOF: with ``max_attempts=0`` (cap disabled)
    the same seed reruns.
    """
    key = "poison-key"
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=3,
        idempotency_key=key,
    )

    async with db_session_factory() as session:
        res = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key=key,
            kind="compile",
            instance_id="live-pod",
            active_count=0,
            max_concurrent=8,
            job_options=_options(_HTTP_REQUEST),
            lease_seconds=120,
            max_attempts=3,
            max_lifetime_seconds=0,
        )
        await session.commit()
    assert res.action == "dedup"  # refused, NOT "rerun"
    assert (await _get(db_session_factory, job_id)).status == "failed"

    # Revert-proof: disable the cap and the same interrupted row reruns.
    await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=3,
        idempotency_key="ok-key",
    )
    async with db_session_factory() as session:
        res2 = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key="ok-key",
            kind="compile",
            instance_id="live-pod",
            active_count=0,
            max_concurrent=8,
            job_options=_options(_HTTP_REQUEST),
            lease_seconds=120,
            max_attempts=0,  # cap disabled
            max_lifetime_seconds=0,
        )
        await session.commit()
    assert res2.action == "rerun"


async def test_resend_of_an_over_lifetime_job_is_refused(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A resend of a job past ``max_lifetime`` is hard-failed, not re-rendered."""
    key = "aged-key"
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=0,
        started_delta_s=-36000,  # 10h old
        idempotency_key=key,
    )
    async with db_session_factory() as session:
        res = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key=key,
            kind="compile",
            instance_id="live-pod",
            active_count=0,
            max_concurrent=8,
            job_options=_options(_HTTP_REQUEST),
            lease_seconds=120,
            max_attempts=0,
            max_lifetime_seconds=3600,
        )
        await session.commit()
    assert res.action == "dedup"
    assert (await _get(db_session_factory, job_id)).status == "failed"


async def test_reclaim_preserves_the_started_at_lifetime_anchor(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``started_at`` is the first-claim anchor for the lifetime cap; a reclaim must NEVER
    reset it, or a flapping job would never age out.

    REVERT-PROOF: regress ``started_at = COALESCE(started_at, now())`` to ``= now()`` in
    reclaim_next_compile_job and this fails (the anchor jumps to now).
    """
    now = datetime.now(UTC)
    job_id = await _seed(
        db_session_factory,
        status="interrupted",
        lease_delta_s=-30,
        job_options=_options(_HTTP_REQUEST),
        attempts=1,
        started_delta_s=-3500,  # under a 3600s lifetime → still reclaimable
    )
    claimed = await _reclaim(db_session_factory, max_lifetime=3600)
    assert isinstance(claimed, job_store.ClaimedJob) and claimed.job_id == job_id
    row = await _get(db_session_factory, job_id)
    assert row.started_at is not None
    assert row.started_at < now - timedelta(seconds=3000)  # preserved, not reset to now()


async def test_job_options_jsonb_null_is_rejected_by_the_check(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The storage CHECK forbids the JSONB scalar ``'null'`` for job_options.

    A ``'null'`` bundle reads as IS NOT NULL and would mis-mark a row reaper-managed with
    no reconstruction request. ``none_as_null`` prevents it ORM-side; this proves the DB
    refuses it even via raw SQL.
    """
    from sqlalchemy import text as _text
    from sqlalchemy.exc import IntegrityError

    # Provide created_at/updated_at (NOT NULL, Python-side default only) so the ONLY
    # possible violation is the job_options CHECK — otherwise the IntegrityError could
    # come from a missing timestamp and the test would pass for the wrong reason.
    with pytest.raises(IntegrityError, match="ck_compile_jobs_job_options_not_json_null"):
        async with db_session_factory() as session:
            await session.execute(
                _text(
                    "INSERT INTO compile_jobs "
                    "(job_id, tenant_id, kind, status, job_options, created_at, updated_at) "
                    "VALUES (:jid, :tid, 'compile', 'processing', 'null'::jsonb, now(), now())"
                ),
                {"jid": "json-null-row", "tid": str(TENANT_A)},
            )
            await session.commit()


# ---------------------------------------------------------------------------
# Multi-replica cross-pod safety (2026-07-24): a terminal write must be fenced to
# the CURRENT owner, and the startup reclaim must not interrupt a LIVE sibling.
# ---------------------------------------------------------------------------


async def test_terminal_write_is_fenced_to_the_current_owner(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """mark_completed/mark_failed only win for the pod that currently owns the row.

    At replicas>1 a reclaiming pod flips the row back to ``processing`` under ITS
    instance_id; the displaced ORIGINAL render (lease lapsed under a DB stall) would
    otherwise still see ``processing`` and clobber the reclaimer's in-flight re-run. The
    instance_id fence makes only the current owner win. REVERT-PROOF inline: with an empty
    instance_id (fence disabled) the same non-owner write WINS via status-only.
    """
    job_id = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=60,
        job_options=_options(_HTTP_REQUEST),
        instance_id="pod-A",
    )
    # A displaced non-owner (pod-B) must NOT win.
    async with db_session_factory() as s:
        won_b = await job_store.mark_completed(
            s, job_id, {"artifact_url": "b"}, instance_id="pod-B"
        )
        await s.commit()
    assert won_b is False
    assert (await _get(db_session_factory, job_id)).status == "processing"  # not clobbered

    # Revert-proof: with the fence disabled (instance_id="") the SAME non-owner write wins.
    async with db_session_factory() as s:
        won_unfenced = await job_store.mark_failed(s, job_id, "unfenced", instance_id="")
        await s.commit()
    assert won_unfenced is True  # status-only guard alone cannot tell pods apart

    # And the true owner wins on a fresh row.
    job2 = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=60,
        job_options=_options(_HTTP_REQUEST),
        instance_id="pod-A",
    )
    async with db_session_factory() as s:
        won_a = await job_store.mark_completed(s, job2, {"artifact_url": "a"}, instance_id="pod-A")
        await s.commit()
    assert won_a is True
    assert (await _get(db_session_factory, job2)).status == "completed"


async def test_reaper_sweep_failure_escalates_once_on_crossing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A sweep failing EVERY time escalates ONCE on crossing the threshold, not per sweep.

    The pre-fix loop logged an identical traceback every interval forever with no other
    signal, so a permanently-broken reclaim (e.g. a partially-applied migration leaving the
    SQL function referencing a column that doesn't exist yet) reclaimed nothing, alerted
    nobody, and never crashed. REVERT-PROOF: drop the modulo gate and 10 sweeps emit 10
    criticals (noise at a louder level); drop the escalation entirely and it emits 0.
    """
    import logging

    from fastapi import FastAPI

    from src.services.reaper_service import _ESCALATE_EVERY_N_FAILURES, ReaperService

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _boom() -> int:
        raise RuntimeError("column compile_jobs.interrupt_reason does not exist")

    monkeypatch.setattr(reaper, "reap_once", _boom)

    caplog.set_level(logging.DEBUG, logger="src.services.reaper_service")
    for _ in range(_ESCALATE_EVERY_N_FAILURES):
        await reaper._sweep()  # never raises: a broken sweep must not kill the pod

    criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(criticals) == 1, [r.getMessage() for r in criticals]
    assert "REAPER STUCK" in criticals[0].getMessage()
    assert "NOT being reclaimed" in criticals[0].getMessage()
    assert reaper.consecutive_failures == _ESCALATE_EVERY_N_FAILURES
    assert reaper.last_failure is not None and "does not exist" in reaper.last_failure


async def test_single_sweep_failure_then_success_resets_and_does_not_escalate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A transient blip is NOT an incident: the counter resets and nothing escalates.

    The counter must be CONSECUTIVE — a running total would page on a pgbouncer hiccup
    accumulated over a long-lived pod's lifetime.
    """
    import logging

    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _boom() -> int:
        raise RuntimeError("transient blip")

    async def _ok() -> int:
        return 0

    caplog.set_level(logging.DEBUG, logger="src.services.reaper_service")
    monkeypatch.setattr(reaper, "reap_once", _boom)
    await reaper._sweep()
    assert reaper.consecutive_failures == 1

    monkeypatch.setattr(reaper, "reap_once", _ok)
    await reaper._sweep()

    assert reaper.consecutive_failures == 0  # the run was broken → not an incident
    assert reaper.last_failure is not None  # but the last error stays as a diagnostic
    assert [r for r in caplog.records if r.levelno == logging.CRITICAL] == []


async def test_reaper_prune_still_fires_when_reap_once_fails_every_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reap_once that raises on EVERY sweep must not starve ``_prune``.

    Pre-fix, ``self._elapsed_since_prune += self._interval`` sat AFTER the
    ``await self.reap_once()`` call inside the same ``try`` — a permanently-failing
    reap_once (the concrete hazard: a partially-applied migration leaves
    ``reclaim_next``'s SQL body referencing a column a later migration adds, so it
    raises forever) jumped straight to ``except Exception`` and the counter never
    advanced, so ``_prune`` was never reached no matter how many sweeps ran.

    interval=10 / prune_interval=25 is chosen so prune must NOT fire on sweep 1 or 2
    (elapsed 10, then 20 — both under 25) and MUST fire on sweep 3 (elapsed 30 >= 25):
    a test where prune fires on sweep 1 regardless of the interval would pass for the
    wrong reason.
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._interval = 10.0
    reaper._prune_interval = 25.0

    async def _boom() -> int:
        raise RuntimeError("column compile_jobs.interrupt_reason does not exist")

    prune_calls = 0

    async def _count_prune() -> None:
        nonlocal prune_calls
        prune_calls += 1

    monkeypatch.setattr(reaper, "reap_once", _boom)
    monkeypatch.setattr(reaper, "_prune", _count_prune)

    await reaper._sweep()
    await reaper._sweep()
    assert prune_calls == 0  # not yet due (elapsed=20 < prune_interval=25)

    await reaper._sweep()
    assert prune_calls == 1  # due (elapsed crossed 25) despite reap_once failing every time


async def test_reaper_prune_cadence_unchanged_on_healthy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: with reap_once succeeding, prune fires at the same cadence as
    before the fix — the fix must not change healthy-path timing.

    Same interval=10 / prune_interval=25 as the failure-path test: sweeps 3 and 6 cross
    the threshold, so 6 sweeps must produce EXACTLY 2 prune calls (not merely ">0").
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._interval = 10.0
    reaper._prune_interval = 25.0

    async def _ok() -> int:
        return 0

    prune_calls = 0

    async def _count_prune() -> None:
        nonlocal prune_calls
        prune_calls += 1

    monkeypatch.setattr(reaper, "reap_once", _ok)
    monkeypatch.setattr(reaper, "_prune", _count_prune)

    for _ in range(6):
        await reaper._sweep()

    assert prune_calls == 2


async def test_reaper_failure_counter_and_metrics_unaffected_by_unconditional_prune_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prune-starvation fix must not disturb the failure-escalation bookkeeping:
    consecutive failures still accumulate across a failing streak (even while a prune
    fires mid-streak) and a subsequent success still resets the counter/gauge to 0.
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService, reaper_consecutive_sweep_failures

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._interval = 10.0
    reaper._prune_interval = 25.0

    async def _boom() -> int:
        raise RuntimeError("transient-looking but permanent failure")

    async def _noop_prune() -> None:
        return None

    monkeypatch.setattr(reaper, "reap_once", _boom)
    monkeypatch.setattr(reaper, "_prune", _noop_prune)

    await reaper._sweep()
    await reaper._sweep()
    await reaper._sweep()  # crosses prune_interval while reap_once is still failing

    assert reaper.consecutive_failures == 3
    assert reaper_consecutive_sweep_failures._value.get() == 3

    async def _ok() -> int:
        return 0

    monkeypatch.setattr(reaper, "reap_once", _ok)
    await reaper._sweep()

    assert reaper.consecutive_failures == 0
    assert reaper_consecutive_sweep_failures._value.get() == 0


async def test_reaper_loop_survives_prune_raising_every_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``_prune()`` that raises on EVERY due sweep must NOT kill the reaper task.

    Drives the REAL ``_loop()`` (not ``_sweep()`` directly) as an ``asyncio.Task``,
    because the bug this guards is specifically that an unguarded exception from
    ``_prune()`` escapes ``_sweep()`` into ``_loop()``, which only catches
    ``asyncio.CancelledError`` — so the task dies silently and permanently, with
    ``consecutive_failures`` stuck at 0 and ``last_failure`` stuck at None (reap_once
    is healthy throughout — only prune is broken). That is measurably worse than the
    starvation bug this whole change started from: a starved-but-alive reaper still
    increments a counter an operator can alert on; a dead task increments nothing.

    interval == prune_interval == a few ms so prune is due on every sweep, driving
    many raises quickly without a long-running test.
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService, reaper_prune_failures_total

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)
    reaper._interval = 0.01
    reaper._prune_interval = 0.01

    async def _ok() -> int:
        return 0

    async def _prune_boom() -> None:
        raise RuntimeError("prunefail")

    monkeypatch.setattr(reaper, "reap_once", _ok)
    monkeypatch.setattr(reaper, "_prune", _prune_boom)

    before = reaper_prune_failures_total._value.get()
    task = asyncio.create_task(reaper._loop())
    try:
        await asyncio.sleep(0.15)  # several sweeps' worth at interval=0.01

        assert task.done() is False, "reaper loop task died — a prune fault must never kill it"
        # The reap side is untouched (reap_once never failed) — proves this is
        # specifically the prune path, not a repeat of the reap-starvation bug.
        assert reaper.consecutive_failures == 0
        assert reaper.last_failure is None
        # ...but the prune fault is OBSERVABLE, not silently dropped.
        assert reaper.last_prune_failure is not None and "prunefail" in reaper.last_prune_failure
        assert reaper_prune_failures_total._value.get() > before
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_the_real_prune_reports_its_failure_instead_of_swallowing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the real ``_prune``, not a stand-in that raises.

    The test above monkeypatches ``_prune`` to raise, so it proves the CALLER nets
    and counts a raising prune — and it passed for as long as the real ``_prune``
    wrapped its own body in ``except Exception: logger.warning(...)``. With that
    swallow in place the caller's net was dead code:
    ``reaper_prune_failures_total`` sat at 0 and ``last_prune_failure`` at ``None``
    however broken prune was, which is the invisible-failure class this module
    exists to close. A prune wedged forever means terminal rows are never evicted
    and the idempotency dedup index grows unbounded, with nothing on /metrics
    saying so.

    So this one breaks the DB call underneath and asserts the fault surfaces.
    """
    from fastapi import FastAPI

    from src.services import reaper_service as mod
    from src.services.reaper_service import ReaperService, reaper_prune_failures_total

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _prune_boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("pgbouncer refused the prune")

    monkeypatch.setattr(mod.job_store, "prune", _prune_boom)

    async def _ok() -> int:
        return 0

    monkeypatch.setattr(reaper, "reap_once", _ok)
    reaper._elapsed_since_prune = reaper._prune_interval  # prune is due on this sweep

    before = reaper_prune_failures_total._value.get()
    await reaper._sweep()

    assert reaper_prune_failures_total._value.get() > before, (
        "the real _prune swallowed its own failure, so the caller's net never "
        "fired and the counter can never move off zero"
    )
    assert reaper.last_prune_failure is not None
    assert "pgbouncer refused the prune" in reaper.last_prune_failure
    # The reap side must be untouched — the two counters are deliberately separate.
    assert reaper.consecutive_failures == 0


async def test_health_reports_reaper_failures_and_stays_200(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Liveness SURFACES the failing reaper and keeps returning 200.

    The count must be queryable without log-diving, and the status code must NOT change:
    failing liveness on a broken reaper would restart-loop a pod that is otherwise serving
    submits/polls perfectly — turning invisible housekeeping breakage into a real outage.
    """
    from src.services.reaper_service import ReaperService

    reaper = ReaperService(app)
    app.state.reaper = reaper
    # Started (not merely constructed) so the probe reports `running` and therefore
    # publishes the count — the sweeps below are still driven by hand; the loop's own
    # first tick is a full interval away.
    await reaper.start()

    async def _boom() -> int:
        raise RuntimeError("reclaim broken")

    async def _ok() -> int:
        return 0

    try:
        monkeypatch.setattr(reaper, "reap_once", _boom)
        await reaper._sweep()
        await reaper._sweep()

        r = await client.get("/health")
        assert r.status_code == 200  # NEGATIVE CONTROL: unchanged while the reaper is broken
        body = r.json()
        assert body["status"] == "ok"
        assert body["reaper_state"] == "running"
        assert body["reaper_consecutive_failures"] == 2
        assert "reclaim broken" in body["reaper_last_failure"]
        assert body["reaper_last_failure_at"] is not None

        # After a successful sweep the count is back to 0 on the live endpoint too.
        monkeypatch.setattr(reaper, "reap_once", _ok)
        await reaper._sweep()
        r2 = await client.get("/healthz")
        assert r2.status_code == 200
        assert r2.json()["reaper_consecutive_failures"] == 0
    finally:
        await reaper.stop()


async def test_escalation_fires_on_exactly_the_nth_failure_not_the_one_before(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The threshold is at EXACTLY N — silent at N-1, critical on N.

    MUTANT-PROOF. The sibling test pins "exactly one critical per 10 sweeps", which an
    off-by-one gate (``(self._consecutive_failures + 1) % N``, escalating at 9/19/29)
    satisfies perfectly — it still emits exactly one critical per ten sweeps, just at the
    wrong sweep. That matters because the threshold is the whole argument for the feature:
    it is chosen as "long enough that a transient DB/pgbouncer blip is excluded", so an
    escalation that fires early pages on the blip the design explicitly rules out. This
    test brackets the boundary from BOTH sides, which is what makes it kill the mutant.
    """
    import logging

    from fastapi import FastAPI

    from src.services.reaper_service import _ESCALATE_EVERY_N_FAILURES, ReaperService

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _boom() -> int:
        raise RuntimeError("standing fault")

    monkeypatch.setattr(reaper, "reap_once", _boom)
    caplog.set_level(logging.DEBUG, logger="src.services.reaper_service")

    # N-1 failures: the streak is real but not yet long enough to be a standing fault.
    for _ in range(_ESCALATE_EVERY_N_FAILURES - 1):
        await reaper._sweep()
    assert reaper.consecutive_failures == _ESCALATE_EVERY_N_FAILURES - 1
    assert [r for r in caplog.records if r.levelno == logging.CRITICAL] == [], (
        f"escalated early, at failure {_ESCALATE_EVERY_N_FAILURES - 1} of "
        f"{_ESCALATE_EVERY_N_FAILURES}"
    )

    # The Nth failure — and only now — crosses the threshold.
    await reaper._sweep()
    criticals = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(criticals) == 1, f"expected exactly one critical ON failure N, got {len(criticals)}"
    assert f"{_ESCALATE_EVERY_N_FAILURES} consecutive sweep failures" in criticals[0].getMessage()


async def test_streak_logs_one_traceback_not_one_per_sweep(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure streak emits ONE traceback, then single-line warnings.

    The stated problem this change exists to fix is "a repeating traceback nobody reads".
    Logging a traceback per sweep would leave that problem exactly as it was and merely add
    a counter next to it: at a 30s interval a permanently-broken reaper emits 2880
    identical stack dumps a day. Below the escalation threshold, so the CRITICAL path
    (which carries its own traceback deliberately) is not what is being counted here.
    """
    import logging

    from fastapi import FastAPI

    from src.services.reaper_service import _ESCALATE_EVERY_N_FAILURES, ReaperService

    streak = 5
    assert streak < _ESCALATE_EVERY_N_FAILURES  # keeps this test off the escalation path

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _boom() -> int:
        raise RuntimeError("standing fault")

    async def _ok() -> int:
        return 0

    caplog.set_level(logging.DEBUG, logger="src.services.reaper_service")
    monkeypatch.setattr(reaper, "reap_once", _boom)
    for _ in range(streak):
        await reaper._sweep()

    with_traceback = [r for r in caplog.records if r.exc_info is not None]
    assert len(with_traceback) == 1, (
        f"{streak} failures emitted {len(with_traceback)} tracebacks; the whole point is 1"
    )
    # The streak is still visible in a log tail — quiet, not silent.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == streak - 1
    assert "consecutive=5" in warnings[-1].getMessage()

    # A NEW streak after a recovery is a new incident: it gets its own traceback, because
    # the second failure's cause may be nothing like the first's.
    caplog.clear()
    monkeypatch.setattr(reaper, "reap_once", _ok)
    await reaper._sweep()
    monkeypatch.setattr(reaper, "reap_once", _boom)
    await reaper._sweep()
    assert len([r for r in caplog.records if r.exc_info is not None]) == 1


async def test_sweep_failures_are_exported_as_prometheus_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure state reaches ``/metrics``, the only surface that can be collected.

    The liveness fields are `kubectl exec`-only: the kubelet discards the probe body, so
    nothing can alert on them. ``/metrics`` is already mounted over the default REGISTRY in
    ``src/asgi.py``, so these series are one gitops scrape-annotation away from alerting.
    Asserted through ``generate_latest()`` — the exact bytes the mounted app serves —
    rather than the metric objects' internals, so a metric that exists but is not exported
    (e.g. registered to a private registry) fails here.
    """
    from fastapi import FastAPI
    from prometheus_client import REGISTRY, generate_latest

    from src.services.reaper_service import ReaperService

    def _gauge() -> float | None:
        return REGISTRY.get_sample_value("w2v_reaper_consecutive_sweep_failures")

    def _total() -> float:
        return REGISTRY.get_sample_value("w2v_reaper_sweep_failures_total") or 0.0

    app = FastAPI()
    app.state.instance_id = "live-pod"
    reaper = ReaperService(app)

    async def _boom() -> int:
        raise RuntimeError("reclaim broken")

    async def _ok() -> int:
        return 0

    total_before = _total()  # cumulative across the process — compare a DELTA, not a value

    monkeypatch.setattr(reaper, "reap_once", _boom)
    await reaper._sweep()
    await reaper._sweep()
    await reaper._sweep()

    assert _gauge() == 3.0  # gauge tracks the CONSECUTIVE count, matching the health body
    assert _total() - total_before == 3.0
    assert reaper.consecutive_failures == 3  # the two surfaces cannot drift

    # Recovery drops the gauge to 0 (so an alert on it clears) while the counter keeps the
    # history (so a flapping reaper that never reaches the threshold is still visible).
    monkeypatch.setattr(reaper, "reap_once", _ok)
    await reaper._sweep()

    assert _gauge() == 0.0
    assert _total() - total_before == 3.0

    exported = generate_latest().decode()
    assert "w2v_reaper_consecutive_sweep_failures 0.0" in exported
    assert "w2v_reaper_sweep_failures_total" in exported


async def test_health_reports_absent_when_no_reaper_is_installed(client: AsyncClient) -> None:
    """No ``app.state.reaper`` must not 500 the probe — and must NOT read as healthy.

    With ``reaper_enabled`` on (the default), nothing installed is a real fault: it used to
    report ``reaper_consecutive_failures == 0``, indistinguishable from a reaper that swept
    cleanly. REVERT-PROOF: restore the ``or 0`` mask and the count is 0 here again.
    """
    r = await client.get("/health/live")
    assert r.status_code == 200
    body = r.json()
    assert body["reaper_state"] == "absent"
    assert body["reaper_consecutive_failures"] is None


async def test_startup_reclaim_spares_a_live_sibling_render(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """reclaim_orphaned_compile_jobs only flips rows whose lease has ALSO lapsed.

    At replicas>1 a booting pod's startup reclaim must NOT interrupt a LIVE sibling's
    in-flight render (its lease is in the future because the sibling is heartbeating). Only
    a genuinely-dead pod's rows (lapsed lease) are reclaimed. REVERT-PROOF: drop the
    lease clause and the live sibling's row is ALSO flipped (n==2).
    """
    live = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=300,  # live sibling, heartbeating
        job_options=_options(_HTTP_REQUEST),
        instance_id="sibling-B",
    )
    dead = await _seed(
        db_session_factory,
        status="processing",
        lease_delta_s=-30,  # dead pod, lapsed lease
        job_options=_options(_HTTP_REQUEST),
        instance_id="dead-C",
    )
    async with db_session_factory() as s:
        n = await job_store.reclaim_orphans(s, "booting-pod")
        await s.commit()
    assert n == 1  # only the dead pod's row
    assert (await _get(db_session_factory, live)).status == "processing"  # live sibling SPARED
    assert (await _get(db_session_factory, dead)).status == "interrupted"  # dead pod reclaimed
