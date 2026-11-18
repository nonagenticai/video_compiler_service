"""Row-retention TTL must NOT be the presigned-URL expiry.

``compile_jobs`` rows ARE the Idempotency-Key dedup index — a deleted row means the
next submit carrying the SAME stable key inserts a brand-new job and fires a SECOND
paid render (``_emit_cost`` runs once per winning terminal write). So the delete TTL
in ``prune_compile_jobs`` is the dedup HORIZON, i.e. how long a resend is safe.

Both prune call sites passed ``settings.minio_url_expiry_seconds`` — the presigned
artifact-URL expiry, deployed at 3600 — so the dedup horizon was one hour, while the
orchestrator resends the same ``f"{run_id}:{stage}"`` key on a resumable stage timeout
hours later. Retention now has its OWN setting so tuning URL expiry can never move it.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.services import compile_job_store as job_store
from src.services.reaper_service import ReaperService
from tests.conftest import TENANT_A

pytestmark = pytest.mark.asyncio

# The stable key shape the orchestrator sends (``f"{run_id}:{stage}"``) — the same
# key comes back on a resumable-timeout re-dispatch, not a fresh per-submit uuid.
_STABLE_KEY = "run-abc:video_compile"


async def _accept(factory: async_sessionmaker[AsyncSession], *, key: str) -> job_store.AcceptResult:
    async with factory() as session:
        result = await job_store.accept(
            session,
            tenant_id=TENANT_A,
            idempotency_key=key,
            kind="compile",
            instance_id="pod-A",
            active_count=0,
            max_concurrent=8,
        )
        await session.commit()
        return result


def _deployed_retention_seconds(app: FastAPI) -> int:
    """The TTL the reaper ACTUALLY passes to ``prune`` in production wiring.

    Read off ``ReaperService`` rather than hardcoded so this test tracks the real
    call site instead of asserting a constant back to itself.
    """
    return int(ReaperService(app)._ttl_seconds)


async def test_dedup_survives_a_two_hour_resume_gap(
    app: FastAPI, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A completed job resent with the same stable key 2h later must still dedup.

    Under the old wiring the row was evicted after ``minio_url_expiry_seconds`` (1h),
    so the resend spawned a second full paid render.
    """
    settings = get_settings()
    retention = _deployed_retention_seconds(app)
    # Preconditions — none of these may be vacuous.
    assert settings.minio_url_expiry_seconds == 3600, (
        "fixture drift: the presign expiry this test contrasts against is not the "
        f"deployed 3600 ({settings.minio_url_expiry_seconds})"
    )
    first = await _accept(db_session_factory, key=_STABLE_KEY)
    assert first.action == "spawn"
    async with db_session_factory() as session:
        assert await job_store.mark_completed(session, first.job_id, {"artifact_url": "u"}, "")
        await session.commit()
    immediate = await _accept(db_session_factory, key=_STABLE_KEY)
    assert immediate.action == "dedup", "precondition: same-key dedup must work at t+0"
    assert immediate.job_id == first.job_id

    # The stage times out resumably and the orchestrator re-dispatches after a gap.
    # Derive the gap from the DEPLOYED retention rather than hard-coding one: #153
    # chose 7200s, so a literal "2 hours" would sit exactly ON the eviction boundary
    # and this test would assert the boundary instead of the behaviour. Sit clearly
    # inside the window; the boundary itself is covered by the bound assertions in
    # test_retention_is_not_bound_to_presigned_url_expiry.
    gap_seconds = retention - 60
    assert gap_seconds > settings.minio_url_expiry_seconds, (
        f"the gap under test ({gap_seconds}s) must exceed the presign expiry "
        f"({settings.minio_url_expiry_seconds}s), or this proves nothing about the old bug"
    )
    async with db_session_factory() as session:
        await session.execute(
            text(
                "UPDATE compile_jobs SET updated_at = now() - make_interval(secs => :g) "
                "WHERE job_id=:j"
            ),
            {"g": gap_seconds, "j": first.job_id},
        )
        await session.commit()
    async with db_session_factory() as session:
        await job_store.prune(
            session, ttl_seconds=retention, work_timeout_seconds=settings.job_work_timeout_seconds
        )
        await session.commit()

    resend = await _accept(db_session_factory, key=_STABLE_KEY)
    assert resend.action == "dedup", (
        "the completed job's dedup row was evicted inside the resume window — "
        f"a resend re-renders and RE-BILLS (retention ttl={retention}s)"
    )
    assert resend.job_id == first.job_id


async def test_retention_is_not_bound_to_presigned_url_expiry(app: FastAPI) -> None:
    """The dedup horizon must have its own knob, not ride on the artifact-URL expiry.

    An operator shortening ``MINIO_URL_EXPIRY_SECONDS`` (a CDN/security decision) must
    not silently shorten how long a resend is protected from double-billing.
    """
    settings = get_settings()
    assert settings.minio_url_expiry_seconds == 3600  # precondition: the deployed value
    retention = _deployed_retention_seconds(app)
    assert retention != settings.minio_url_expiry_seconds, (
        "row retention is wired to the presigned-URL expiry; tuning one moves the other"
    )
    # The floor is the ORCHESTRATOR'S RESUME ENVELOPE, not an arbitrary round number.
    # #153 sized this deliberately: normal re-dispatch is ~90s (reconciler_interval);
    # the real exposure was a 3600-7200s dispatch gap (a wedged rollout, a lost node,
    # drained replicas), so the horizon must clear the top of that range.
    # NB an earlier draft of this test asserted >= 86400 — that encoded a competing
    # 7-day proposal, not the semantics that shipped, and would have failed main.
    resume_envelope_seconds = 7200
    assert retention >= resume_envelope_seconds, (
        f"dedup horizon is {retention}s, below the {resume_envelope_seconds}s "
        "orchestrator resume envelope — a re-dispatch after a wedged rollout would "
        "miss the dedup row and start a second full PAID render"
    )
