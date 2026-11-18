"""Durable compile-job store: the residual-#2 proof (dedup survives a pod restart).

The former in-memory job store + Idempotency-Key index lost all state on a pod
restart, so a resume-after-timeout re-submit (same stable key) started a SECOND
paid render. These tests prove the Postgres-backed store:

- dedups a resend to a COMPLETED job across a simulated restart (no 2nd render),
- with an executable REVERT-PROOF: wiping the row (the old in-memory-loss on
  restart) makes the resend fire a second render — so the dedup test is not a
  tautology,
- re-runs an INTERRUPTED (restart-orphaned) job under the SAME job_id,
- never re-runs a genuinely FAILED job,
- FAILS LOUD on a tenant-less submit/poll (no '' sentinel bucket to share),
- never dedups keyless submits,
- rolls back an at-capacity new insert (no phantom row),
- reclaims only orphans, never the live pod's own in-flight rows.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import get_settings
from src.database.models import CompileJob
from src.services import compile_job_store as job_store
from src.services.compile_job_store import CapacityError, MissingTenantError
from tests.conftest import TENANT_A, TENANT_B

pytestmark = pytest.mark.asyncio

_MAX = 8


async def _accept(
    factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: uuid.UUID | str | None,
    key: str | None,
    instance_id: str = "pod-A",
    active_count: int = 0,
) -> job_store.AcceptResult:
    """One submit in its own committed txn (mirrors a route request)."""
    async with factory() as session:
        result = await job_store.accept(
            session,
            tenant_id=tenant_id,
            idempotency_key=key,
            kind="compile",
            instance_id=instance_id,
            active_count=active_count,
            max_concurrent=_MAX,
        )
        await session.commit()
        return result


async def _mark(factory: async_sessionmaker[AsyncSession], job_id: str, status: str) -> None:
    async with factory() as session:
        if status == "completed":
            await job_store.mark_completed(session, job_id, {"artifact_url": "x"})
        elif status == "failed":
            await job_store.mark_failed(session, job_id, "boom")
        await session.commit()


async def _count(factory: async_sessionmaker[AsyncSession]) -> int:
    async with factory() as session:
        return (await session.execute(select(func.count()).select_from(CompileJob))).scalar_one()


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


async def test_completed_job_dedups_never_reruns(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A resend of a COMPLETED job's key returns the SAME job — never a 2nd render."""
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    assert first.action == "spawn"
    await _mark(db_session_factory, first.job_id, "completed")

    again = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    assert again.action == "dedup"
    assert again.job_id == first.job_id
    assert await _count(db_session_factory) == 1  # no second row


async def test_processing_job_dedups(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    again = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    assert again.action == "dedup"
    assert again.job_id == first.job_id


async def test_interrupted_job_reruns_under_same_job_id(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A restart orphan (interrupted) is re-run under the SAME job_id, not duplicated."""
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k", instance_id="pod-A")
    # Simulate a pod restart: a NEW pod reclaims orphaned processing rows.
    async with db_session_factory() as session:
        flipped = await job_store.reclaim_orphans(session, "pod-B")
        await session.commit()
    assert flipped == 1

    again = await _accept(db_session_factory, tenant_id=TENANT_A, key="k", instance_id="pod-B")
    assert again.action == "rerun"
    assert again.job_id == first.job_id  # SAME job, re-run — not a duplicate
    async with db_session_factory() as session:
        row = await job_store.get(session, first.job_id, TENANT_A)
    assert row is not None
    assert row.status == "processing"  # re-run transitioned it back
    assert row.instance_id == "pod-B"  # now owned by the new pod
    assert await _count(db_session_factory) == 1


async def test_failed_job_stays_terminal_no_rerun(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A genuinely FAILED render is returned as-is (dedup), never re-run in a loop."""
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    await _mark(db_session_factory, first.job_id, "failed")

    again = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    assert again.action == "dedup"
    assert again.job_id == first.job_id


@pytest.mark.parametrize("absent", [None, "", "   ", "t1", "not-a-uuid"])
async def test_tenantless_submit_raises_and_writes_nothing(
    db_session_factory: async_sessionmaker[AsyncSession], absent: uuid.UUID | str | None
) -> None:
    """A submit that forwards no REAL tenant uuid raises and persists NOTHING.

    Replaces ``test_null_tenant_dedups_nulls_not_distinct``, which proved that two
    tenant-less submits shared one dedup bucket — the defect. ``tenant_id`` was
    ``varchar NOT NULL DEFAULT ''``, so every unattributed caller collapsed onto the
    same empty-string tenant: one shared row, readable by all of them under the strict
    per-tenant RLS policy, and a second caller's resend deduped onto a FIRST caller's
    render. The column is now ``uuid NOT NULL`` with no server_default and ``accept()``
    coerces through ``require_tenant``, so there is no bucket to share — the submit
    fails loud instead.

    The blank/garbage cases matter as much as ``None``: an empty string used to BE the
    sentinel, and ``"t1"`` is the shape every legacy test passed.
    """
    with pytest.raises(MissingTenantError):
        await _accept(db_session_factory, tenant_id=absent, key="k")
    assert await _count(db_session_factory) == 0  # nothing was written


async def test_tenantless_read_raises_rather_than_scanning_a_sentinel_bucket(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A tenant-less poll cannot read ANY row — it raises instead of matching ``''``.

    ``get()`` used to filter ``tenant_id == (tenant_id or '')``, so an unattributed
    caller read every unattributed row. Now the scope is un-defaultable.
    """
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    async with db_session_factory() as session:
        with pytest.raises(MissingTenantError):
            await job_store.get(session, first.job_id, None)
        with pytest.raises(MissingTenantError):
            await job_store.get(session, first.job_id, "")
        # the real tenant still reads its own row — the guard is not a blanket denial.
        assert await job_store.get(session, first.job_id, TENANT_A) is not None


async def test_keyless_submits_never_dedup(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key=None)
    second = await _accept(db_session_factory, tenant_id=TENANT_A, key=None)
    assert first.action == second.action == "spawn"
    assert first.job_id != second.job_id


async def test_cross_tenant_get_returns_none(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k")
    async with db_session_factory() as session:
        assert await job_store.get(session, first.job_id, TENANT_B) is None
        assert await job_store.get(session, first.job_id, TENANT_A) is not None


async def test_at_capacity_new_insert_rolls_back_no_phantom_row(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A spawn refused at capacity leaves NO row — the uncommitted insert rolls back."""
    async with db_session_factory() as session:
        with pytest.raises(CapacityError):
            await job_store.accept(
                session,
                tenant_id=TENANT_A,
                idempotency_key="k",
                kind="compile",
                instance_id="pod-A",
                active_count=_MAX,  # already at the cap
                max_concurrent=_MAX,
            )
        await session.rollback()
    assert await _count(db_session_factory) == 0


async def test_reclaim_excludes_the_live_pods_own_rows(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Reclaim flips orphans (other instances) but NOT the calling pod's own in-flight rows."""
    await _accept(db_session_factory, tenant_id=TENANT_A, key="k", instance_id="pod-A")
    # Same pod calling reclaim must not flip its own live job.
    async with db_session_factory() as session:
        assert await job_store.reclaim_orphans(session, "pod-A") == 0
        await session.commit()
    async with db_session_factory() as session:
        row = await job_store.get(
            session, (await session.execute(select(CompileJob.job_id))).scalar_one(), TENANT_A
        )
    assert row is not None
    assert row.status == "processing"
    # A different pod (a real restart) flips it.
    async with db_session_factory() as session:
        assert await job_store.reclaim_orphans(session, "pod-B") == 1
        await session.commit()


async def test_concurrent_resend_of_interrupted_reruns_exactly_once(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """TWO concurrent same-key resends of an INTERRUPTED job must yield exactly one
    ``rerun`` (one render) + one ``dedup`` — never two renders. Guards the atomic
    ``UPDATE … WHERE status='interrupted' RETURNING`` claim: reverting it to an
    unconditional UPDATE makes BOTH return ``rerun`` and this assertion fails."""
    first = await _accept(db_session_factory, tenant_id=TENANT_A, key="k", instance_id="pod-A")
    async with db_session_factory() as session:
        await job_store.reclaim_orphans(session, "pod-B")
        await session.commit()

    async def _resend() -> job_store.AcceptResult:
        async with db_session_factory() as session:
            result = await job_store.accept(
                session,
                tenant_id=TENANT_A,
                idempotency_key="k",
                kind="compile",
                instance_id="pod-B",
                active_count=0,
                max_concurrent=_MAX,
            )
            await session.commit()
            return result

    r1, r2 = await asyncio.gather(_resend(), _resend())
    assert sorted([r1.action, r2.action]) == ["dedup", "rerun"]  # exactly ONE reruns
    assert r1.job_id == r2.job_id == first.job_id
    assert await _count(db_session_factory) == 1


async def test_prune_flips_zombie_and_evicts_aged_terminal(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The SECURITY DEFINER prune flips a zombie ``processing`` (dead task past the
    work timeout) → ``interrupted``, and deletes terminal rows past the TTL."""
    zombie = await _accept(db_session_factory, tenant_id=TENANT_A, key="z")
    old = await _accept(db_session_factory, tenant_id=TENANT_A, key="o")
    await _mark(db_session_factory, old.job_id, "completed")
    # Backdate: the zombie's attempt started 2h ago; the terminal row aged 2h ago.
    async with db_session_factory() as session:
        await session.execute(
            text(
                "UPDATE compile_jobs SET heartbeat_at = now() - interval '2 hours' WHERE job_id=:j"
            ),
            {"j": zombie.job_id},
        )
        await session.execute(
            text("UPDATE compile_jobs SET updated_at = now() - interval '2 hours' WHERE job_id=:j"),
            {"j": old.job_id},
        )
        await session.commit()

    async with db_session_factory() as session:
        n = await job_store.prune(session, ttl_seconds=3600, work_timeout_seconds=3600)
        await session.commit()
    assert n == 2  # 1 zombie flipped + 1 terminal evicted

    async with db_session_factory() as session:
        z = await job_store.get(session, zombie.job_id, TENANT_A)
        o = await job_store.get(session, old.job_id, TENANT_A)
    assert z is not None
    assert z.status == "interrupted"
    assert o is None  # evicted past TTL


@pytest.mark.asyncio
async def test_retention_outlives_the_presign_ttl_so_a_late_resend_still_dedups(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A terminal row must survive the orchestrator's whole resume window.

    The compile_jobs row IS the ``(tenant_id, idempotency_key)`` dedup index
    (uq_compile_jobs_tenant_idempotency_key). ``prune()`` used to be driven by
    ``minio_url_expiry_seconds`` (3600) — the PRESIGN expiry — so a resend of the same
    key 70 minutes later found no row, hit no conflict, inserted fresh and returned
    ``action="spawn"``: a SECOND full paid render plus a second ``_emit_cost``.

    The orchestrator's compile idempotency key is resume-stable (``{run_id}:{stage}``)
    and it resumes runs whose ``updated_at`` is up to ``reconciler_max_age_seconds``
    (7200) old, so 2h is the real outer bound on a same-key resend. Retention is now a
    separate knob defaulting to 7200.

    This asserts the BOUNDARY: a 90-minute-old terminal row (past the old 3600s presign
    TTL, inside the 7200s retention) must still be present and must still dedup.
    """
    settings = get_settings()
    assert settings.compile_job_retention_seconds >= 7200, (
        "retention must cover the orchestrator's reconciler_max_age_seconds (7200)"
    )
    assert settings.compile_job_retention_seconds > settings.minio_url_expiry_seconds, (
        "dedup retention must not be coupled to the presigned-URL lifetime"
    )

    accepted = await _accept(db_session_factory, tenant_id=TENANT_A, key="late-resend")
    await _mark(db_session_factory, accepted.job_id, "completed")
    # 90 minutes old: past the old 3600s presign TTL, inside the 7200s retention.
    async with db_session_factory() as session:
        await session.execute(
            text(
                "UPDATE compile_jobs SET updated_at = now() - interval '90 minutes' WHERE job_id=:j"
            ),
            {"j": accepted.job_id},
        )
        await session.commit()

    async with db_session_factory() as session:
        await job_store.prune(
            session,
            ttl_seconds=settings.compile_job_retention_seconds,
            work_timeout_seconds=settings.job_work_timeout_seconds,
        )
        await session.commit()

    # The row survived, so the key still dedups instead of spawning a paid re-render.
    async with db_session_factory() as session:
        survivor = await job_store.get(session, accepted.job_id, TENANT_A)
    assert survivor is not None, (
        "terminal row was pruned inside the orchestrator's resume window — a resend "
        "of this key would spawn a second PAID render"
    )

    resend = await _accept(db_session_factory, tenant_id=TENANT_A, key="late-resend")
    assert resend.action == "dedup", (
        f"late resend must dedup, got action={resend.action!r} (a paid re-render)"
    )
    assert resend.job_id == accepted.job_id


async def test_prune_call_sites_use_retention_not_the_presign_ttl() -> None:
    """The WIRING, not just the config value.

    The bug was never the default — it was that both prune() call sites read
    ``minio_url_expiry_seconds``. A test that calls prune() with the right argument
    itself cannot catch a regression at the CALL SITE, so assert the call sites
    directly: the reaper's cached TTL, and the source of the startup prune.
    """
    from fastapi import FastAPI

    from src.services.reaper_service import ReaperService

    settings = get_settings()
    app = FastAPI()
    app.state.instance_id = "pod-test"
    reaper = ReaperService(app)
    assert reaper._ttl_seconds == settings.compile_job_retention_seconds, (
        "reaper prune must use the dedup retention window, not the presigned-URL expiry"
    )
    assert reaper._ttl_seconds != settings.minio_url_expiry_seconds, (
        "retention and presign expiry must not be the same knob — that coupling is the bug"
    )

    startup = Path("src/asgi.py").read_text()
    assert "ttl_seconds=settings.compile_job_retention_seconds" in startup, (
        "the startup prune in asgi.py must also use retention, not presign expiry"
    )
    assert "ttl_seconds=settings.minio_url_expiry_seconds" not in startup


# ---------------------------------------------------------------------------
# End-to-end restart dedup + the executable revert-proof (residual-#2 headline)
# ---------------------------------------------------------------------------


def _counting_compile_and_upload(monkeypatch: pytest.MonkeyPatch, renders: dict[str, int]) -> None:
    fake_mp4 = Path("/tmp/fake-durable.mp4")
    fake_mp4.write_bytes(b"\x00")

    async def counting_compile(shots, settings, **kwargs):  # type: ignore[no-untyped-def]
        renders["n"] += 1
        return fake_mp4, 1.0, None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", counting_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)


async def _poll_done(client: AsyncClient, job_id: str) -> dict[str, object]:
    import asyncio

    # Generous budget: the default audio-master runs ffmpeg several times on the
    # (fake) output, so a stubbed render still takes a few seconds to finish.
    for _ in range(1500):
        r = await client.get(f"/api/jobs/{job_id}")
        if r.status_code == 200 and r.json()["status"] != "processing":
            return r.json()
        await asyncio.sleep(0.02)
    raise AssertionError("job never left processing")


async def test_resend_after_restart_dedups_no_second_render(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The residual-#2 fix: a resend with the same key AFTER the first render
    completed dedupes to the finished job — zero second render — because the dedup
    index is durable in Postgres (it would have been lost with the old in-memory
    index on a pod restart)."""
    renders = {"n": 0}
    _counting_compile_and_upload(monkeypatch, renders)
    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    headers = {"Idempotency-Key": "durable-k"}

    first = await client.post("/api/compile", json=body, headers=headers)
    assert first.status_code == 202
    job_id = first.json()["job_id"]
    assert (await _poll_done(client, job_id))["status"] == "completed"
    assert renders["n"] == 1

    # A resume-after-timeout re-dispatch resends the SAME stable key. The durable
    # row survives (a pod restart would have wiped the old in-memory index).
    resend = await client.post("/api/compile", json=body, headers=headers)
    assert resend.status_code == 202
    assert resend.json()["job_id"] == job_id  # deduped to the finished job
    assert renders["n"] == 1  # NO second paid render


async def test_revert_proof_wiping_the_row_fires_a_second_render(
    monkeypatch: pytest.MonkeyPatch,
    client: AsyncClient,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """REVERT-PROOF: delete the persisted row (what the OLD in-memory store
    effectively did on a pod restart) → the resend NO LONGER dedupes and fires a
    SECOND render with a NEW job_id. Proves the dedup test above pins durability,
    not a tautology."""
    from sqlalchemy import delete

    renders = {"n": 0}
    _counting_compile_and_upload(monkeypatch, renders)
    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    headers = {"Idempotency-Key": "revert-k"}

    first = await client.post("/api/compile", json=body, headers=headers)
    first_job = first.json()["job_id"]
    assert (await _poll_done(client, first_job))["status"] == "completed"
    assert renders["n"] == 1

    # Simulate the pre-fix in-memory loss: the store is gone after a restart.
    async with db_session_factory() as session:
        await session.execute(delete(CompileJob))
        await session.commit()

    resend = await client.post("/api/compile", json=body, headers=headers)
    second_job = resend.json()["job_id"]
    assert (await _poll_done(client, second_job))["status"] == "completed"
    assert second_job != first_job  # a NEW job — no dedup without the durable row
    assert renders["n"] == 2  # the second (duplicate) PAID render the fix prevents
