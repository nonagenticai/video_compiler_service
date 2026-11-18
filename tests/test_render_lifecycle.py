"""Render-lifecycle terminal-status classification + billing correctness.

- a transient DB blip on the COMPLETION write must NOT flip a fully-rendered job to
  ``failed`` (which dedups terminal + orphans the uploaded artifact) — it leaves the row
  ``processing`` for the reaper to resume;
- billing meters the PROBED output duration, not the nominal per-shot sum (which
  over-reports a seeked no-narration clip);
- ``begin_drain`` marks the pod draining before renders are cancelled;
- a submit does NOT stamp ``started_at`` (the lifetime cap anchors on first reclaim).
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes import compile as compile_routes
from src.services import compile_job_store as job_store
from tests.conftest import TENANT_A


async def _ok_completed(*_a: object, **_k: object) -> bool:
    """A completion write that succeeds without a DB (``lease_seconds=0`` spawns)."""
    return True


def test_billed_duration_prefers_probed_actual() -> None:
    # nominal (per-shot sum) over-reports a seeked clip; the probed final-mux wins.
    assert compile_routes._billed_duration(
        {"duration_seconds": 20.0, "actual_duration_seconds": 5.0}
    ) == (5.0, "probed")
    # no probe → fall back to nominal, and SAY so.
    assert compile_routes._billed_duration({"duration_seconds": 20.0}) == (20.0, "nominal")
    # a 0.0 probe means ffprobe failed → fall back to nominal, not bill 0. The
    # fallback is an over-report of unmeasured work, so it is labelled, not silent.
    assert compile_routes._billed_duration(
        {"duration_seconds": 20.0, "actual_duration_seconds": 0.0}
    ) == (20.0, "nominal")


def test_begin_drain_sets_draining() -> None:
    assert compile_routes._draining is False
    try:
        compile_routes.begin_drain()
        assert compile_routes._draining is True
    finally:
        compile_routes._draining = False


async def test_completion_write_failure_leaves_processing_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DB blip on mark_completed must not stamp the (successful) render ``failed``.

    REVERT-PROOF: remove the inner try/except around ``_stamp_completed`` and the raise
    propagates to ``except Exception`` → ``_stamp_failed`` is called (this asserts it is NOT).
    """
    stamped_failed: list[str] = []
    emitted: list[object] = []

    async def _boom(*_a: object, **_k: object) -> bool:
        raise RuntimeError("pgbouncer blip on completion write")

    async def _record_failed(job_id: str, *_a: object, **_k: object) -> None:
        stamped_failed.append(job_id)

    monkeypatch.setattr(compile_routes, "_stamp_completed", _boom)
    monkeypatch.setattr(compile_routes, "_stamp_failed", _record_failed)
    monkeypatch.setattr(compile_routes, "_emit_cost", lambda **kw: emitted.append(kw))
    monkeypatch.setattr(
        compile_routes, "_COMPLETION_WRITE_BACKOFF", 0.0, raising=False
    )  # exhaust retries fast

    async def _work() -> dict[str, object]:
        return {"artifact_url": "u", "duration_seconds": 5.0, "actual_duration_seconds": 5.0}

    before = set(compile_routes._background_tasks)
    compile_routes._spawn(
        "job-h1",
        _work,
        tenant_id=TENANT_A,
        work_timeout=30.0,
        job_type="compile",
        lease_seconds=0,  # no heartbeat → no DB needed
        instance_id="pod-A",
    )
    task = next(iter(compile_routes._background_tasks - before))
    await task  # the completion-write failure is swallowed (logged), task ends cleanly

    assert stamped_failed == []  # a successful render was NOT flipped to failed
    assert emitted == []  # nor billed (won never returned)


async def test_completion_write_transient_blip_retries_then_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TRANSIENT completion-write blip is absorbed by the bounded retry: the render
    completes and bills exactly once, without leaving the row ``processing`` for the
    reaper to do a full paid re-render of an already-successful render.

    REVERT-PROOF: collapse the retry loop back to a single ``_stamp_completed`` and the
    first blip returns early (row left processing, nothing billed) → ``emitted`` stays
    empty; this asserts exactly ONE bill after the retry wins.
    """
    calls = {"n": 0}
    stamped_failed: list[str] = []
    emitted: list[object] = []

    async def _flaky(*_a: object, **_k: object) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pgbouncer blip on first completion write")
        return True  # the retry wins the terminal write

    async def _record_failed(job_id: str, *_a: object, **_k: object) -> None:
        stamped_failed.append(job_id)

    monkeypatch.setattr(compile_routes, "_stamp_completed", _flaky)
    monkeypatch.setattr(compile_routes, "_stamp_failed", _record_failed)
    monkeypatch.setattr(compile_routes, "_emit_cost", lambda **kw: emitted.append(kw))
    monkeypatch.setattr(
        compile_routes, "_COMPLETION_WRITE_BACKOFF", 0.0, raising=False
    )  # no real sleep

    async def _work() -> dict[str, object]:
        return {"artifact_url": "u", "duration_seconds": 5.0, "actual_duration_seconds": 5.0}

    before = set(compile_routes._background_tasks)
    compile_routes._spawn(
        "job-blip1",
        _work,
        tenant_id=TENANT_A,
        work_timeout=30.0,
        job_type="compile",
        lease_seconds=0,  # no heartbeat → no DB needed
        instance_id="pod-A",
    )
    task = next(iter(compile_routes._background_tasks - before))
    await task

    assert calls["n"] == 2  # first blip retried, second attempt won
    assert stamped_failed == []  # a successful render was NOT mis-stamped failed
    assert len(emitted) == 1  # billed exactly once (won on the retry)


async def test_submit_does_not_stamp_started_at(
    db_session: AsyncSession,
) -> None:
    """A managed submit leaves started_at NULL; the reaper's first claim anchors it."""
    res = await job_store.accept(
        db_session,
        tenant_id=TENANT_A,
        idempotency_key=None,
        kind="compile",
        instance_id="pod-A",
        active_count=0,
        max_concurrent=8,
        job_options={"kind": "compile", "request": {"shots": [{"id": "s1", "seconds": 2.0}]}},
        lease_seconds=120,
    )
    await db_session.commit()
    row = await job_store.get(db_session, res.job_id, TENANT_A)
    assert row is not None
    assert row.started_at is None  # NOT anchored at submit
    assert row.lease_expires_at is not None  # but it IS reaper-managed (has a lease)


async def test_transient_render_error_is_interrupted_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RetryableRenderError (ffmpeg subprocess timeout / OOM signal-kill) stamps
    'interrupted' (re-runnable), while a genuine FfmpegError stays terminal 'failed'.

    REVERT-PROOF: remove the `except RetryableRenderError` handler and the transient error
    falls to `except Exception` → _stamp_failed (this asserts _stamp_interrupted instead).

    Also pins the interrupt REASON to 'render_failure', never 'drain': a drain-labelled
    interrupt is exempt from the reclaim attempt increment, so mislabelling this genuine
    ffmpeg crash would let a crash-looping render be reclaimed and re-billed forever.
    """
    from src.services import compile_job_store as job_store
    from src.services.renderer import FfmpegError, RetryableRenderError

    calls: dict[str, list[str]] = {"failed": [], "interrupted": []}
    reasons: list[object] = []

    async def _rec_failed(job_id: str, *_a: object, **_k: object) -> None:
        calls["failed"].append(job_id)

    async def _rec_interrupted(job_id: str, *_a: object, **_k: object) -> None:
        calls["interrupted"].append(job_id)
        reasons.append(_k.get("reason"))

    monkeypatch.setattr(compile_routes, "_stamp_failed", _rec_failed)
    monkeypatch.setattr(compile_routes, "_stamp_interrupted", _rec_interrupted)

    async def _spawn_and_wait(job_id: str, exc: Exception) -> None:
        async def _work() -> dict[str, object]:
            raise exc

        before = set(compile_routes._background_tasks)
        compile_routes._spawn(
            job_id,
            _work,
            tenant_id=TENANT_A,
            work_timeout=30.0,
            job_type="compile",
            lease_seconds=0,
            instance_id="pod-A",
        )
        await next(iter(compile_routes._background_tasks - before))

    await _spawn_and_wait("job-transient", RetryableRenderError("ffmpeg killed by signal 9"))
    assert calls["interrupted"] == ["job-transient"]  # retryable → re-runnable
    assert calls["failed"] == []
    assert reasons == [job_store.INTERRUPT_REASON_RENDER_FAILURE]  # a FAULT, not a rollout

    await _spawn_and_wait("job-permanent", FfmpegError("ffmpeg exit=234: bad filter"))
    assert calls["failed"] == ["job-permanent"]  # deterministic encode error → terminal


async def test_a_superseded_tasks_callback_does_not_unshield_the_live_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_done`` must drop ``_active_job_ids`` only for ITS OWN task.

    ``_active_job_ids`` is the SOLE input to both same-pod double-render guards —
    the reaper's reclaim exclusion (``reaper_service``: ``active_job_ids=``) and the
    ``/internal/claim`` check. A rerun of the same ``job_id`` (an interrupted job
    re-accepted under the same key) installs a new task; when the SUPERSEDED task's
    completion callback then discards the id unconditionally, the LIVE render is
    un-shielded and the very same paid ffmpeg pass can be claimed and run twice.
    The sibling ``_job_tasks`` pop three lines below already carries this guard.

    REVERT-PROOF: move ``_active_job_ids.discard(job_id)`` back above the
    ``if _job_tasks.get(job_id) is t`` guard in ``_done`` and the first assertion
    below fails — the old task evicts the live entry.
    """
    monkeypatch.setattr(compile_routes, "_stamp_completed", _ok_completed)
    monkeypatch.setattr(compile_routes, "_emit_cost", lambda **kw: None)

    job_id = "job-rerun-supersede"
    first_may_finish = asyncio.Event()
    second_may_finish = asyncio.Event()

    def _gated(gate: asyncio.Event):  # type: ignore[no-untyped-def]
        async def _work() -> dict[str, object]:
            await gate.wait()
            return {"artifact_url": "u", "duration_seconds": 1.0, "actual_duration_seconds": 1.0}

        return _work

    before = set(compile_routes._background_tasks)
    try:
        compile_routes._spawn(
            job_id,
            _gated(first_may_finish),
            tenant_id=TENANT_A,
            work_timeout=30.0,
            job_type="compile",
            lease_seconds=0,
            instance_id="pod-A",
        )
        first = next(iter(compile_routes._background_tasks - before))
        # The RERUN: same job_id, a second task installed while the first still runs.
        compile_routes._spawn(
            job_id,
            _gated(second_may_finish),
            tenant_id=TENANT_A,
            work_timeout=30.0,
            job_type="compile",
            lease_seconds=0,
            instance_id="pod-A",
        )
        second = next(iter(compile_routes._background_tasks - before - {first}))
        assert compile_routes._job_tasks[job_id] is second
        assert job_id in compile_routes._active_job_ids

        # The SUPERSEDED task finishes; its done-callback must not touch either entry.
        first_may_finish.set()
        await first
        await asyncio.sleep(0)  # let the done-callback run

        assert job_id in compile_routes._active_job_ids, (
            "the superseded task's callback evicted the LIVE rerun from "
            "_active_job_ids — both same-pod double-render guards are now blind to it"
        )
        assert compile_routes._job_tasks.get(job_id) is second

        # The OWNING task's callback still clears both — the guard adds no leak.
        second_may_finish.set()
        await second
        await asyncio.sleep(0)
        assert job_id not in compile_routes._active_job_ids
        assert job_id not in compile_routes._job_tasks
    finally:
        first_may_finish.set()
        second_may_finish.set()
        compile_routes._active_job_ids.discard(job_id)
        compile_routes._job_tasks.pop(job_id, None)
