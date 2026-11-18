"""Cancelling a run must stop the render it is paying for.

Before ``POST /api/jobs/cancel`` there was NO channel from the orchestrator's cancel into
this service: a user pressing Cancel stopped the UI, the compile ran to completion, and
the render — the most expensive stage in the product — was billed in full. These tests
pin the three things that make a cancel real:

1. the durable flip is tenant-scoped, prefix-literal, and terminal (a resend of the same
   Idempotency-Key DEDUPS instead of resurrecting the render);
2. the in-flight ffmpeg on THIS replica is actually killed, and the row ends ``cancelled``
   — not ``interrupted``, which would leave it re-runnable and undo the cancel;
3. a render on a SIBLING replica aborts itself when its lease renew comes back empty.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.routes import compile as compile_routes
from src.services import compile_job_store as job_store
from src.services import durable_dispatch
from tests.conftest import TENANT_A, TENANT_B

pytestmark = pytest.mark.asyncio

_MAX = 8


async def _accept(
    factory: async_sessionmaker[AsyncSession], *, tenant_id: uuid.UUID | None, key: str
) -> str:
    async with factory() as session:
        result = await job_store.accept(
            session,
            tenant_id=tenant_id,
            idempotency_key=key,
            kind="compile",
            instance_id="pod-A",
            active_count=0,
            max_concurrent=_MAX,
        )
        await session.commit()
        return result.job_id


async def _status(factory: async_sessionmaker[AsyncSession], job_id: str, tenant: uuid.UUID) -> str:
    async with factory() as session:
        row = await job_store.get(session, job_id, tenant)
    assert row is not None
    return row.status


async def _cancel(
    factory: async_sessionmaker[AsyncSession], *, tenant_id: uuid.UUID | None, prefix: str
) -> list[str]:
    async with factory() as session:
        ids = await job_store.cancel_by_key_prefix(session, tenant_id=tenant_id, key_prefix=prefix)
        await session.commit()
        return ids


# ---------------------------------------------------------------------------
# 1. The durable flip
# ---------------------------------------------------------------------------


async def test_cancel_reaps_every_stage_of_one_run(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """One run owns several compiles (``{run_id}:{stage}``) — the prefix reaps them ALL,
    and leaves another run's render alone."""
    a = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:video_render")
    b = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:promo_render:redo")
    other = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-2:video_render")

    cancelled = await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="run-1:")

    assert sorted(cancelled) == sorted([a, b])
    assert await _status(db_session_factory, a, TENANT_A) == "cancelled"
    assert await _status(db_session_factory, b, TENANT_A) == "cancelled"
    assert await _status(db_session_factory, other, TENANT_A) == "processing"


async def test_cancelled_job_is_not_rerunnable_by_a_resend(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The orchestrator replays its Idempotency-Key (poll-loss resubmit, resume-after-
    timeout re-dispatch). A cancelled job must DEDUP, never re-run — otherwise the very
    next retry undoes the cancel and the money is spent anyway."""
    job_id = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:video_render")
    await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="run-1:")

    async with db_session_factory() as session:
        again = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key="run-1:video_render",
            kind="compile",
            instance_id="pod-A",
            active_count=0,
            max_concurrent=_MAX,
        )
        await session.commit()

    assert again.action == "dedup"
    assert again.job_id == job_id
    assert await _status(db_session_factory, job_id, TENANT_A) == "cancelled"


async def test_cancel_is_tenant_scoped(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Tenant B's identical key is untouched — a cancel can never reach across tenants."""
    mine = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:video_render")
    theirs = await _accept(db_session_factory, tenant_id=TENANT_B, key="run-1:video_render")

    cancelled = await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="run-1:")

    assert cancelled == [mine]
    assert await _status(db_session_factory, theirs, TENANT_B) == "processing"


async def test_prefix_wildcards_are_literal_not_like_patterns(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``%`` / ``_`` in the prefix must match themselves. Unescaped they are LIKE
    wildcards, and a cancel would silently reap a WIDER set than the caller named — every
    other run of the tenant, in the ``%`` case."""
    literal = await _accept(db_session_factory, tenant_id=TENANT_A, key="a%b:video_render")
    wildcard_victim = await _accept(db_session_factory, tenant_id=TENANT_A, key="axb:video_render")
    underscore_victim = await _accept(
        db_session_factory, tenant_id=TENANT_A, key="aXb:video_render"
    )

    cancelled = await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="a%b:")
    assert cancelled == [literal]
    assert await _status(db_session_factory, wildcard_victim, TENANT_A) == "processing"

    cancelled = await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="a_b:")
    assert cancelled == []
    assert await _status(db_session_factory, underscore_victim, TENANT_A) == "processing"


async def test_cancel_claims_an_interrupted_job_but_not_a_terminal_one(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An ``interrupted`` (restart-orphan) row is still QUEUED work that would be re-run
    and billed — it must be cancellable. A ``completed`` one is history: leave it."""
    orphan = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:a")
    done = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:b")
    async with db_session_factory() as session:
        # Complete one FIRST, then let a restart orphan the other: reclaim_orphans is a
        # bulk flip of every processing row, so the order is what makes the two rows
        # land in different states.
        await job_store.mark_completed(session, done, {"artifact_url": "x"})
        await job_store.reclaim_orphans(session, "pod-B")
        await session.commit()
    assert await _status(db_session_factory, orphan, TENANT_A) == "interrupted"

    cancelled = await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="run-1:")

    assert cancelled == [orphan]
    assert await _status(db_session_factory, done, TENANT_A) == "completed"


# ---------------------------------------------------------------------------
# 2. The local kill (same replica)
# ---------------------------------------------------------------------------


def _blocking_compile(started: asyncio.Event, cancelled: asyncio.Event):  # type: ignore[no-untyped-def]
    """A stand-in for ``compile_shots`` that behaves like a live ffmpeg render: it blocks,
    and it observes its own cancellation (which is what ``renderer._run`` turns into
    ``proc.kill()`` in production)."""

    async def _fn(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return Path("/tmp/never-cancel.mp4"), 1.0, None, None, 30

    return _fn


async def test_cancel_kills_the_inflight_render_and_lands_cancelled_not_interrupted(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The end-to-end money proof: a live render dies and the row is terminal.

    The ``!= "interrupted"`` assertion is the one that matters: the render's own
    ``CancelledError`` handler stamps ``interrupted``, which is RE-RUNNABLE, so a cancel
    that ends there is undone by the next resend and bought nothing. (Measured: the flip
    also converges from kill-first order, because ``interrupted`` is itself cancellable —
    so this assertion pins the OUTCOME, not the statement order.)
    """
    started = asyncio.Event()
    render_cancelled = asyncio.Event()
    monkeypatch.setattr(
        "src.api.routes.compile.compile_shots", _blocking_compile(started, render_cancelled)
    )

    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    submit = await client.post(
        "/api/compile", json=body, headers={"Idempotency-Key": "run-9:video_render"}
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    await asyncio.wait_for(started.wait(), timeout=5)

    resp = await client.post("/api/jobs/cancel", json={"idempotency_key_prefix": "run-9:"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"cancelled": [job_id], "killed": 1}
    # The render itself observed the cancellation — in production this is the
    # proc.kill() in renderer._run, i.e. the ffmpeg child actually dying.
    await asyncio.wait_for(render_cancelled.wait(), timeout=5)

    from src.database.session import get_db_session

    async with get_db_session() as session:
        row = await job_store.get(session, job_id, TENANT_A)
    assert row is not None
    assert row.status == "cancelled", row.status
    assert row.status != "interrupted", "a cancel that leaves the job re-runnable is not a cancel"
    assert row.lease_expires_at is None  # no reaper can reclaim it

    poll = await client.get(f"/api/jobs/{job_id}")
    assert poll.json()["status"] == "cancelled"
    assert poll.json()["error"] == job_store.CANCELLED_ERROR


async def test_cancel_of_an_unknown_prefix_is_a_clean_noop(client: AsyncClient) -> None:
    """A cancel for a run with no live compile must not 4xx/5xx — the orchestrator calls
    this best-effort on EVERY cancel, including runs that never reached a render stage."""
    resp = await client.post("/api/jobs/cancel", json={"idempotency_key_prefix": "run-none:"})
    assert resp.status_code == 200
    assert resp.json() == {"cancelled": [], "killed": 0}


async def test_cancel_registry_is_emptied_when_a_render_finishes(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """``_job_tasks`` must not leak: it is a module-global that would otherwise pin every
    finished render's Task for the pod's whole life."""
    started = asyncio.Event()
    render_cancelled = asyncio.Event()
    monkeypatch.setattr(
        "src.api.routes.compile.compile_shots", _blocking_compile(started, render_cancelled)
    )
    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]},
        headers={"Idempotency-Key": "run-10:video_render"},
    )
    job_id = submit.json()["job_id"]
    await asyncio.wait_for(started.wait(), timeout=5)
    assert job_id in compile_routes._job_tasks

    await client.post("/api/jobs/cancel", json={"idempotency_key_prefix": "run-10:"})
    await asyncio.wait_for(render_cancelled.wait(), timeout=5)
    for _ in range(50):
        if job_id not in compile_routes._job_tasks:
            break
        await asyncio.sleep(0.02)
    assert job_id not in compile_routes._job_tasks


# ---------------------------------------------------------------------------
# 3. The cross-replica kill (lease loss)
# ---------------------------------------------------------------------------


async def test_lease_renew_reports_loss_when_the_job_was_cancelled(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``renew_lease`` must return False on a cancelled row. That boolean IS the
    cross-replica cancel signal; if it stays True the sibling pod renders on obliviously
    until its work timeout, spending the money the cancel was supposed to save."""
    job_id = await _accept(db_session_factory, tenant_id=TENANT_A, key="run-1:video_render")
    async with db_session_factory() as session:
        assert await job_store.renew_lease(session, job_id, "pod-A", 120) is True
        await session.commit()

    await _cancel(db_session_factory, tenant_id=TENANT_A, prefix="run-1:")

    async with db_session_factory() as session:
        assert await job_store.renew_lease(session, job_id, "pod-A", 120) is False


async def test_heartbeat_aborts_the_render_on_lease_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost lease cancels the render task; a HELD lease never does. The negative half
    matters most: an over-eager abort would kill healthy renders."""
    monkeypatch.setattr(durable_dispatch, "_MIN_BEAT_SECONDS", 0.01)

    async def _lost(job_id: str, instance_id: str, lease_seconds: int) -> bool:
        return False

    async def _held(job_id: str, instance_id: str, lease_seconds: int) -> bool:
        return True

    async def _errors(job_id: str, instance_id: str, lease_seconds: int) -> bool:
        raise RuntimeError("pgbouncer blip")

    async def _work() -> str:
        await asyncio.sleep(3600)
        return "done"

    monkeypatch.setattr(durable_dispatch, "_renew_once", _lost)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            durable_dispatch.run_with_lease(
                job_id="j", instance_id="pod-A", lease_seconds=1, work=_work()
            ),
            timeout=5,
        )

    # A held lease — and, critically, a lease renew that ERRORS (unknown ownership, not a
    # cancel) — must leave the render running.
    for renew in (_held, _errors):
        monkeypatch.setattr(durable_dispatch, "_renew_once", renew)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                durable_dispatch.run_with_lease(
                    job_id="j", instance_id="pod-A", lease_seconds=1, work=_work()
                ),
                timeout=0.3,
            )


async def test_outer_cancellation_still_reaches_the_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the SIGTERM drain: ``run_with_lease`` now awaits a Task rather
    than the coroutine directly, and a cancel of the OUTER task must still propagate into
    the work (otherwise a rolling deploy abandons a live ffmpeg)."""
    monkeypatch.setattr(durable_dispatch, "_MIN_BEAT_SECONDS", 0.01)

    async def _held(job_id: str, instance_id: str, lease_seconds: int) -> bool:
        return True

    monkeypatch.setattr(durable_dispatch, "_renew_once", _held)
    observed = asyncio.Event()

    async def _work() -> str:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            observed.set()
            raise
        return "done"

    task = asyncio.create_task(
        durable_dispatch.run_with_lease(
            job_id="j", instance_id="pod-A", lease_seconds=1, work=_work()
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed.is_set()
