"""Durable compile-job store + Idempotency-Key dedup (replaces the in-memory dicts).

The former per-pod ``app.state.compile_jobs`` / ``compile_idem`` dicts lost all
state on a pod restart, so a resume-after-timeout re-submit (same stable
``Idempotency-Key``) started a SECOND paid render. This module persists the job
store + dedup index in Postgres so the dedup survives a restart.

## Accept decision table (on an existing row for the same tenant+key)
- ``completed``   → dedup: return the finished job. **Never re-render a success.**
- ``processing``  → dedup: an in-flight render on this pod owns it.
- ``failed``      → dedup: terminal; return it (the orchestrator's own attempt
                    cap governs retry — we don't loop a doomed render).
- ``interrupted`` → **rerun**: a restart orphaned the render (its in-process task
                    died). Transition back to ``processing`` under the SAME
                    ``job_id`` and re-run with the resend's fresh inputs. This is
                    the completion of interrupted work, not a duplicate.

A brand-new key inserts a fresh ``processing`` row (action ``spawn``). Both
``spawn`` and ``rerun`` respect the per-pod in-flight render cap: at capacity the
uncommitted row is rolled back and the caller returns 429 — a dedup/terminal
return is never rate-limited.

Backpressure counts LIVE in-process render tasks (passed in as ``active_count``),
not DB rows: under ``TENANT_RLS_MODE=enforce`` the runtime role only sees its own
tenant's rows, so a DB count would undercount globally — and the task set is the
accurate live-render gauge anyway.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

from sqlalchemy import ColumnElement, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CompileJob

AcceptAction = Literal["spawn", "dedup", "rerun"]
# ``cancelled`` is terminal and belongs here: a caller-cancelled render must NOT be
# resurrected by a resend of the same Idempotency-Key (the orchestrator's poll loop and
# its loss-resubmit both replay the key), or a cancel would be undone by the very next
# retry and the money would be spent anyway. Listing it is EXPLICITNESS, not the
# guarantee — measured: with it omitted, a cancelled row falls into the interrupted
# branch below whose claim is ``WHERE status='interrupted'`` and so still dedups. Naming
# it here is what stops that from being an accident one refactor away from breaking.
_NON_RERUNNABLE = ("completed", "processing", "failed", "cancelled")

# Terminal note stamped on a caller-cancelled job. Surfaced verbatim on the poll.
CANCELLED_ERROR: Final = "cancelled by caller"
# Statuses a cancel may claim: an in-flight render and a restart-orphan awaiting a
# re-run. ``completed``/``failed`` are already terminal (the money is spent or lost);
# re-stamping them would rewrite history for no benefit.
_CANCELLABLE = ("processing", "interrupted")

# --- interrupt_reason vocabulary -------------------------------------------------------
# WHY a row was flipped to ``interrupted``. The reclaim claim in
# ``reclaim_next_compile_job`` reads it to decide whether that reclaim costs a poison
# attempt: ONLY ``drain`` is exempt (a rollout is not a fault), everything else — and a
# NULL written by a legacy row or an older pod — charges one. Deliberately NOT a DB CHECK
# constraint: unknown values already fail safe, and a CHECK widened out of step with this
# literal is a known fleet trap.
INTERRUPT_REASON_DRAIN: Final = "drain"
INTERRUPT_REASON_RENDER_FAILURE: Final = "render_failure"
INTERRUPT_REASON_CRASH_ZOMBIE: Final = "crash_zombie"
INTERRUPT_REASON_CRASH_ORPHAN: Final = "crash_orphan"
InterruptReason = Literal["drain", "render_failure", "crash_zombie", "crash_orphan"]


class MissingTenantError(ValueError):
    """No real tenant was forwarded for a write/read that MUST be tenant-scoped.

    FAIL LOUD, never substitute. ``compile_jobs.tenant_id`` used to carry an
    empty-string sentinel (schema ``server_default``) and the metering path a nil-uuid
    one (``DEFAULT_TENANT_ID``) — two spellings of the same lie. Every row must carry a
    REAL registered tenant forwarded from the requestor, so an absent or unparseable one
    raises this instead of inventing an identity that RLS would then isolate into a
    shared bucket every unattributed caller can read.
    """

    def __init__(self, field: str = "tenant_id") -> None:
        super().__init__(
            f"{field} is required and must be a registered tenant UUID forwarded by the "
            "caller — no sentinel, no default, no substitution. Nothing was written."
        )


class NilTenantIdError(MissingTenantError):
    """The all-zeros UUID was forwarded as a tenant.

    ``uuid.UUID("00000000-0000-0000-0000-000000000000")`` parses cleanly, so the
    ``ValueError`` arm below never sees it: without an explicit BY-VALUE check the
    nil uuid walked straight into ``compile_jobs.tenant_id`` and into every metering
    event. It is not a customer — it is the removed pre-0.67 sentinel, and it is
    *registered* in ``licenses.tenants`` as ``default-anon``, so it also survives a
    dirty-row audit while collapsing every unattributed caller into one RLS bucket
    they can all read. Rejecting the four spellings ``uuid.UUID`` normalises
    (dashed, undashed, braced, ``urn:uuid:``) is free once ``.int`` is read.

    Subclasses :class:`MissingTenantError` on purpose: ``_caller_tenant`` already
    maps that to **400**, which is the right shape at THIS seam — the tenant arrives
    in a forwarded header/job payload, so a bad one really is a bad request. (The
    sibling seam in ``video_production_service`` answers 401 because there the value
    can only be absent, never caller-chosen. See its ``src/api/tenant.py``.)
    """

    def __init__(self, field: str = "tenant_id") -> None:
        ValueError.__init__(
            self,
            f"{field} is the nil uuid, which is a removed sentinel and not a registered "
            "tenant — no substitution, no default. Nothing was written.",
        )


def require_tenant(tenant_id: object, field: str = "tenant_id") -> uuid.UUID:
    """Coerce a forwarded tenant to ``uuid.UUID`` or raise :class:`MissingTenantError`.

    The nil uuid is refused BY VALUE (:class:`NilTenantIdError`) on every arm,
    including the already-``uuid.UUID`` fast path.
    """
    if isinstance(tenant_id, uuid.UUID):
        return _reject_nil(tenant_id, field)
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise MissingTenantError(field)
    try:
        parsed = uuid.UUID(tenant_id.strip())
    except ValueError as exc:
        raise MissingTenantError(field) from exc
    return _reject_nil(parsed, field)


def _reject_nil(tenant: uuid.UUID, field: str) -> uuid.UUID:
    """Return ``tenant`` unless it is the nil uuid, which raises."""
    if tenant.int == 0:
        raise NilTenantIdError(field)
    return tenant


class CapacityError(Exception):
    """The per-pod in-flight render cap is reached for a would-be spawn/rerun (→ 429)."""


@dataclass(frozen=True)
class AcceptResult:
    """Outcome of an accept: the job to return + whether the caller must spawn work."""

    job_id: str
    is_new: bool
    action: AcceptAction


@dataclass(frozen=True)
class ClaimedJob:
    """A crash-orphaned job the reaper claimed for re-dispatch on this pod."""

    job_id: str
    tenant_id: uuid.UUID
    options: dict[str, object]


class _PoisonSkipped:
    """Distinct singleton type for the ``POISON_SKIPPED`` reclaim sentinel.

    A dedicated class (not a bare ``object()``) so ``reclaim_next``'s return annotation
    can name it precisely and the caller's ``is POISON_SKIPPED`` identity check is
    ty-checkable rather than widening to ``object``."""

    __slots__ = ()


# Returned by ``reclaim_next`` when the claim consumed a poison / over-lifetime job
# (hard-failed inside the SQL function) rather than handing out a re-dispatchable job.
# Distinct from ``None`` (queue genuinely empty) so the reaper CONTINUES its sweep past a
# head-of-queue poison job instead of breaking and stalling the healthy orphans behind it.
POISON_SKIPPED: Final = _PoisonSkipped()


def _now() -> datetime:
    return datetime.now(UTC)


def _rerun_accounting(
    existing: CompileJob, now: datetime
) -> dict[str, ColumnElement[int] | ColumnElement[datetime | None]]:
    """The poison-accounting half of a CLIENT-RESEND rerun claim.

    Mirrors ``reclaim_next_compile_job``'s claim EXACTLY, because both are a claim of
    the same interrupted row and the caps in ``accept`` read what this writes:

    - ``started_at`` is anchored UNCONDITIONALLY with a COALESCE (first start wins). A
      rerun IS the start of real render work, so it is the same anchor a reaper claim
      stamps; without it the wall-clock lifetime cap short-circuits on
      ``started_at is None`` forever. COALESCE, not overwrite — an overwrite would let
      a client defeat the wall clock simply by resending often enough.
    - ``attempts`` is charged +1 for every rerun EXCEPT one whose row was flipped by a
      graceful shutdown drain. That is the SAME allowlist-of-one as the SQL: a rollout
      is not a fault, and charging it would poison-fail a healthy PAID render after
      ``reclaim_max_attempts`` deploys (migration 008's whole point). A NULL/unknown
      reason falls to +1, so a crash-loop resent by the client is bounded.

    The drain exemption therefore stays intact on BOTH paths, and the wall-clock cap —
    which the exemption's own rationale names as its backstop — finally has an anchor
    on this one.
    """
    charge = 0 if existing.interrupt_reason == INTERRUPT_REASON_DRAIN else 1
    return {
        "attempts": func.coalesce(CompileJob.attempts, 0) + charge,
        "started_at": func.coalesce(CompileJob.started_at, now),
    }


async def accept(
    session: AsyncSession,
    *,
    tenant_id: str | uuid.UUID | None,
    idempotency_key: str | None,
    kind: str,
    instance_id: str,
    active_count: int,
    max_concurrent: int,
    job_options: dict[str, object] | None = None,
    lease_seconds: int = 0,
    max_attempts: int = 0,
    max_lifetime_seconds: int = 0,
) -> AcceptResult:
    """Insert-or-dedup a submit; return the job + the action the caller must take.

    Race-free against TWO concurrent same-key resends on every path:
    - a brand-new key: ``INSERT … ON CONFLICT DO NOTHING`` — only one insert wins;
    - an ``interrupted`` (restart-orphan) key: the re-run is claimed by a
      **conditional** ``UPDATE … WHERE status='interrupted' RETURNING`` — only the
      request that actually flips interrupted→processing gets ``action="rerun"``;
      a loser sees 0 rows and dedups. Without this guard two resends could both
      SELECT ``interrupted`` and both spawn — a duplicate PAID render, the exact
      bug this store prevents.
    Raises :class:`CapacityError` (before the txn commits, so nothing is
    persisted) when a spawn/rerun would exceed the in-flight cap. The 2-iteration
    loop retries only the astronomically-rare case where the conflicting row was
    concurrently pruned between the INSERT and the follow-up SELECT.
    """
    now = _now()
    # FAIL LOUD on an absent tenant. This used to coalesce None into the empty
    # string for the NOT NULL column, minting a sentinel every unattributed submit
    # shared — one bucket, cross-tenant-readable under the strict RLS policy.
    tenant = require_tenant(tenant_id)
    # A fresh lease + the reconstruction bundle make this row reaper-managed: if this
    # pod dies mid-render, another pod's reaper reclaims the row (once the lease lapses)
    # and re-runs it from job_options. A submit that passes no job_options (lease_seconds
    # left 0) stays unmanaged (job_options NULL ⇒ the reaper ignores it) — the pre-reaper
    # behaviour, so tests and any non-render caller are unaffected.
    lease_until = now + timedelta(seconds=lease_seconds) if job_options is not None else None
    for _attempt in range(2):
        insert_stmt = (
            pg_insert(CompileJob)
            .values(
                job_id=uuid.uuid7().hex,
                tenant_id=tenant,
                idempotency_key=idempotency_key,
                kind=kind,
                status="processing",
                instance_id=instance_id,
                job_options=job_options,
                lease_expires_at=lease_until,
                # started_at is the first-REAPER-CLAIM anchor for the lifetime cap (set by
                # reclaim_next_compile_job's COALESCE), NULL until then. NOT stamped at
                # submit: the cap must measure accumulated run/reclaim time, not wall-clock
                # since submit, or a job legitimately queued/back-pressured for hours would
                # be false-poisoned on its first reclaim despite ~0s of real render.
                started_at=None,
                created_at=now,
                updated_at=now,
                heartbeat_at=now,
            )
            .on_conflict_do_nothing(
                index_elements=["tenant_id", "idempotency_key"],
                index_where=text("idempotency_key IS NOT NULL"),
            )
            .returning(CompileJob.job_id)
        )
        inserted = (await session.execute(insert_stmt)).scalar_one_or_none()
        if inserted is not None:
            # Brand-new job. Enforce the in-flight cap; raising rolls the insert back.
            if active_count >= max_concurrent:
                raise CapacityError
            return AcceptResult(job_id=inserted, is_new=True, action="spawn")

        # Conflict → a row already exists for this (tenant, key). Load + decide.
        existing = (
            await session.execute(
                select(CompileJob).where(
                    CompileJob.tenant_id == tenant,
                    CompileJob.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            # The conflicting row was concurrently pruned between the INSERT and
            # this SELECT. Retry: the next INSERT has no conflict → a clean spawn.
            continue

        if existing.status in _NON_RERUNNABLE:
            return AcceptResult(job_id=existing.job_id, is_new=False, action="dedup")

        # status == "interrupted". The reaper's poison/lifetime caps live in
        # reclaim_next_compile_job; a CLIENT resend must honour the same ceiling or it is
        # an unbounded re-render bypass (BaseServiceClient retries POSTs). A job already
        # at the reaper's attempt cap, or older than the lifetime bound, is hard-failed
        # here and dedup'd — NOT re-run. Note the ``drain`` exemption in
        # reclaim_next_compile_job spares the INCREMENT, never this CAP: a drained row
        # that has already accumulated ``max_attempts`` from real faults is still
        # poison-failed right here.
        #
        # ⚠️ Both INPUTS to this gate are written by the rerun claim below (see
        # ``_rerun_accounting``). They used to be written ONLY by
        # reclaim_next_compile_job, and the insert stamps ``started_at=None`` — so for a
        # job the reaper never claims (the NORMAL case: BaseServiceClient retries the
        # POST immediately, long before the ~30s sweep; also ``reaper_enabled=false``;
        # also a pod at ``max_concurrent_compile_jobs``, where the sweep breaks) this
        # read ``0 >= max_attempts`` and ``started_at is None`` on every resend. Both
        # halves were False forever and the ceiling this docstring promises could not
        # fire on the very path it guards — an unbounded PAID re-render.
        over_lifetime = (
            max_lifetime_seconds > 0
            and existing.started_at is not None
            and existing.started_at < now - timedelta(seconds=max_lifetime_seconds)
        )
        if (max_attempts > 0 and (existing.attempts or 0) >= max_attempts) or over_lifetime:
            await session.execute(
                update(CompileJob)
                .where(CompileJob.job_id == existing.job_id, CompileJob.status == "interrupted")
                .values(
                    status="failed",
                    error="exceeded max reclaim attempts / lifetime — failed as poison job",
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            return AcceptResult(job_id=existing.job_id, is_new=False, action="dedup")

        # atomically claim the re-run (see docstring).
        if active_count >= max_concurrent:
            raise CapacityError
        claimed = (
            await session.execute(
                update(CompileJob)
                .where(CompileJob.job_id == existing.job_id, CompileJob.status == "interrupted")
                .values(
                    status="processing",
                    instance_id=instance_id,
                    # Refresh the reconstruction bundle + lease to the RESEND's payload
                    # (its inputs are freshly presigned): a later reaper reclaim must
                    # re-run the current inputs, not the stale first-submit ones. The
                    # fresh lease also stops the reaper racing this just-claimed rerun.
                    job_options=job_options,
                    # Clear the stale interrupt reason for the same reason the reaper's
                    # claim does: a leftover 'drain' would exempt a LATER crash of this
                    # row from the poison cap.
                    interrupt_reason=None,
                    lease_expires_at=lease_until,
                    updated_at=now,
                    heartbeat_at=now,
                    **_rerun_accounting(existing, now),
                )
                .returning(CompileJob.job_id)
            )
        ).scalar_one_or_none()
        if claimed is None:
            # Another concurrent resend won the claim → dedup, do NOT spawn.
            return AcceptResult(job_id=existing.job_id, is_new=False, action="dedup")
        return AcceptResult(job_id=existing.job_id, is_new=False, action="rerun")

    # Both attempts raced a vanishing conflict row (pathological). Fail loudly
    # rather than silently double-render or dedup to nothing.
    raise RuntimeError("compile_job_store.accept: idempotency row kept vanishing")


def _owner_conds(job_id: str, instance_id: str) -> list[ColumnElement[bool]]:
    """WHERE conditions for a terminal write: still-``processing`` AND owned by THIS pod.

    The status guard alone gives exactly-once at replicas:1 (a superseded attempt hits 0
    rows). At replicas>1 it is NOT enough: a reclaiming pod flips the row back to
    ``processing`` under ITS instance_id, so the displaced ORIGINAL render (whose lease
    lapsed under a DB stall) would still see ``processing`` and could clobber the
    reclaimer's in-flight re-run. The ``instance_id`` fence makes only the CURRENT owner
    win. ``instance_id == ""`` (legacy/test callers that pass no id) keeps the pre-fence
    status-only behaviour.
    """
    conds: list[ColumnElement[bool]] = [
        CompileJob.job_id == job_id,
        CompileJob.status == "processing",
    ]
    if instance_id:
        conds.append(CompileJob.instance_id == instance_id)
    return conds


async def mark_completed(
    session: AsyncSession, job_id: str, result: dict[str, object], instance_id: str = ""
) -> bool:
    # Guard on status='processing' (+ instance_id when set, see _owner_conds): a stale
    # terminal write from a superseded attempt (a timed-out original racing a reclaimed
    # re-run of the same job_id) must NOT clobber a row since re-run/interrupted/finished
    # or owned by another pod. Returns whether THIS write flipped the row (the exactly-one
    # winner); the caller gates cost metering on it. Also NULLs the lease so a completed
    # row is never reaper-reclaimable.
    now = _now()
    won = (
        await session.execute(
            update(CompileJob)
            .where(*_owner_conds(job_id, instance_id))
            .values(
                status="completed",
                result=result,
                error=None,
                lease_expires_at=None,
                updated_at=now,
                heartbeat_at=now,
            )
            .returning(CompileJob.job_id)
        )
    ).scalar_one_or_none()
    return won is not None


async def mark_failed(
    session: AsyncSession,
    job_id: str,
    error: str,
    instance_id: str = "",
    *,
    result: dict[str, object] | None = None,
) -> bool:
    # See mark_completed: only a still-``processing`` row THIS pod owns accepts a terminal
    # write. Returns the win-bool so the caller can gate side effects. NULLs the lease.
    #
    # VC-W14/V2: ``result`` is the STRUCTURED verdict for a failure that HAS one — an
    # armed mastering gate carries ``violations`` (stage / metric / measured value /
    # breached limit / tier), and until now a failing master delivered only the joined
    # ``detail`` prose in ``error``. ``get_job`` already projects ``result["mastering"]``
    # onto the poll response, so writing it here is what makes the structured contract
    # live on the path that most needs it. Omitted ⇒ the column is left untouched (NOT
    # overwritten with NULL), so every other failure keeps its existing behaviour.
    now = _now()
    values: dict[str, object] = {
        "status": "failed",
        "error": error,
        "lease_expires_at": None,
        "updated_at": now,
        "heartbeat_at": now,
    }
    if result is not None:
        values["result"] = result
    won = (
        await session.execute(
            update(CompileJob)
            .where(*_owner_conds(job_id, instance_id))
            .values(**values)
            .returning(CompileJob.job_id)
        )
    ).scalar_one_or_none()
    return won is not None


async def mark_interrupted(
    session: AsyncSession, job_id: str, error: str, instance_id: str = "", *, reason: str
) -> None:
    """Best-effort terminal-ish stamp on SIGTERM cancel; a resend re-runs it.

    Only flips a still-``processing`` row THIS pod owns (a genuine terminal write from a
    completing render must win the race; at replicas>1 a draining pod must not interrupt a
    render another pod has already reclaimed).

    ``reason`` is REQUIRED and keyword-only — a default would be fail-open: whichever
    value it defaulted to would be wrong for half the callers, and defaulting to
    ``drain`` would silently exempt genuine render crashes from the poison cap, making a
    crash-looping render unbounded. Use the ``INTERRUPT_REASON_*`` constants.
    """
    now = _now()
    await session.execute(
        update(CompileJob)
        .where(*_owner_conds(job_id, instance_id))
        .values(status="interrupted", error=error, interrupt_reason=reason, updated_at=now)
    )


async def cancel_by_key_prefix(
    session: AsyncSession, *, tenant_id: str | uuid.UUID | None, key_prefix: str
) -> list[str]:
    """Terminally cancel every live job of ONE tenant whose Idempotency-Key starts with
    ``key_prefix``. Returns the ``job_id``s this call flipped.

    Prefix rather than exact key because the caller (the orchestrator) submits one compile
    per run PER STAGE — key ``{run_id}:{stage}``, optionally ``:{discriminator}`` for a
    re-render — so a run-scoped cancel cannot enumerate the keys it must reap. The prefix
    is matched with ``autoescape`` so a ``%``/``_`` in it is a literal, not a wildcard that
    would cancel a wider set than the caller named.

    Tenant-scoped exactly like :func:`get`: a cancel can never reach another tenant's
    render. Only ``processing``/``interrupted`` rows are claimed (see ``_CANCELLABLE``);
    the write NULLs the lease so no reaper can reclaim the row, and clears
    ``interrupt_reason`` so a stale ``drain`` can't leak into a later reclaim's poison
    accounting.

    This write is the DURABLE half of a cancel and MUST land before the in-process task is
    cancelled: the task's ``CancelledError`` handler stamps ``interrupted``, which is
    guarded on ``status='processing'`` — flipping to ``cancelled`` first makes that stamp
    a 0-row no-op, so the render cannot resurrect itself into a re-runnable state.
    """
    now = _now()
    rows = (
        await session.execute(
            update(CompileJob)
            .where(
                CompileJob.tenant_id == require_tenant(tenant_id),
                CompileJob.idempotency_key.startswith(key_prefix, autoescape=True),
                CompileJob.status.in_(_CANCELLABLE),
            )
            .values(
                status="cancelled",
                error=CANCELLED_ERROR,
                interrupt_reason=None,
                lease_expires_at=None,
                updated_at=now,
            )
            .returning(CompileJob.job_id)
        )
    ).scalars()
    return list(rows)


async def get(
    session: AsyncSession, job_id: str, tenant_id: str | uuid.UUID | None
) -> CompileJob | None:
    """Tenant-scoped poll. Returns None for a missing OR cross-tenant job (→ 404)."""
    return (
        await session.execute(
            select(CompileJob).where(
                CompileJob.job_id == job_id,
                CompileJob.tenant_id == require_tenant(tenant_id),
            )
        )
    ).scalar_one_or_none()


async def prune(session: AsyncSession, *, ttl_seconds: int, work_timeout_seconds: int) -> int:
    """Cross-tenant housekeeping: flip zombie ``processing`` + evict aged terminal rows.

    Delegates to the SECURITY DEFINER ``prune_compile_jobs`` so it works ACROSS ALL
    tenants under RLS enforce (a plain UPDATE/DELETE by the restricted runtime role
    would be scoped to the caller's GUC tenant only). A render is bounded by
    ``asyncio.wait_for(work(), work_timeout)``, so a row still ``processing`` past the
    work timeout has lost its task → flip to ``interrupted`` (a resend re-runs it).
    Returns the number of rows flipped + deleted.
    """
    return (
        await session.execute(
            text("SELECT prune_compile_jobs(:ttl, :wt)"),
            {"ttl": ttl_seconds, "wt": work_timeout_seconds},
        )
    ).scalar_one()


async def reclaim_orphans(session: AsyncSession, instance_id: str) -> int:
    """Flip restart-orphaned ``processing`` rows → ``interrupted`` (cross-tenant).

    Calls the SECURITY DEFINER ``reclaim_orphaned_compile_jobs`` function so the
    cross-tenant UPDATE works under RLS enforce (the restricted runtime role can't
    UPDATE across tenants). Since migration ``005_lease_aware_startup_reclaim`` the
    function reclaims another instance's ``processing`` row ONLY if its lease has
    LAPSED (``lease_expires_at IS NULL OR lease_expires_at < now()``) — that lapsed
    lease, not the replica count, is the proof the owning pod is dead. This is safe
    at ``replicas > 1`` (the live deployment runs 2): a LIVE sibling renews its lease
    on every heartbeat, so a booting pod can never interrupt its in-flight renders.
    Flipping the genuinely-dead ones to ``interrupted`` makes the queue-managed ones (with
    ``job_options``) IMMEDIATELY reaper-reclaimable — the fast-recovery path that
    resumes a dead pod's render without waiting the full lease TTL. Returns the
    count for a startup log line.
    """
    return (
        await session.execute(
            text("SELECT reclaim_orphaned_compile_jobs(:iid)"), {"iid": instance_id}
        )
    ).scalar_one()


async def renew_lease(
    session: AsyncSession, job_id: str, instance_id: str, lease_seconds: int
) -> bool:
    """Push this pod's lease on ``job_id`` forward (heartbeat, called mid-render).

    Scoped to a row THIS pod still owns and is still ``processing`` (``_owner_conds``, the
    same fence the terminal writes use) — so a renew can never resurrect a lease on a job
    that has already terminated or been reclaimed by another pod. Runs as a plain UPDATE
    (not SECURITY DEFINER): the render task binds the job's own tenant GUC, so RLS admits
    the write to its own row.

    Returns whether the row was still ours to renew. ``False`` is a DEFINITIVE statement
    that this render no longer owns its job — it was cancelled by a caller (status
    ``cancelled``), terminalized, or reclaimed by another pod — and is the cross-replica
    kill switch: the heartbeat aborts the render on it (see
    ``durable_dispatch._heartbeat``). A DB *error* is NOT this signal; it raises and the
    caller retries on the next beat.
    """
    now = _now()
    won = (
        await session.execute(
            update(CompileJob)
            .where(*_owner_conds(job_id, instance_id))
            .values(lease_expires_at=now + timedelta(seconds=lease_seconds), heartbeat_at=now)
            .returning(CompileJob.job_id)
        )
    ).scalar_one_or_none()
    return won is not None


async def reclaim_next(
    session: AsyncSession,
    *,
    worker_id: str,
    lease_seconds: int,
    max_attempts: int,
    max_lifetime_seconds: int,
    active_job_ids: list[str] | None = None,
) -> ClaimedJob | _PoisonSkipped | None:
    """Claim ONE crash-orphaned queue-managed job for re-dispatch on this pod.

    Delegates to the SECURITY DEFINER ``reclaim_next_compile_job`` (``FOR UPDATE SKIP
    LOCKED``, cross-tenant) so concurrent reapers never claim the same row and the
    claim works under RLS enforce. Returns the reconstruction bundle; ``None`` when the
    queue is genuinely empty; or the ``POISON_SKIPPED`` sentinel when the claim CONSUMED
    a poison / over-lifetime job (hard-failed inside the function). The sentinel is
    distinct from ``None`` so the reaper CONTINUES its sweep past a head-of-queue poison
    job instead of breaking — otherwise a healthy orphan behind it waits a full reaper
    interval. Stamps a fresh lease on the claim and increments ``attempts`` — except when
    the row was flipped by a graceful drain (``interrupt_reason='drain'``), which is a
    rollout rather than a fault and so must not erode the poison budget.

    ``active_job_ids`` is the caller pod's in-process live-render set: those job_ids are
    excluded from selection so the reaper never claims (and mutates) a job it is still
    rendering itself. ``None``/empty ⇒ no exclusion.
    """
    row = (
        await session.execute(
            text(
                "SELECT job_id, tenant_id, job_options "
                "FROM reclaim_next_compile_job(:w, :lease, :maxa, :maxl, :active)"
            ),
            {
                "w": worker_id,
                "lease": lease_seconds,
                "maxa": max_attempts,
                "maxl": max_lifetime_seconds,
                "active": list(active_job_ids or []),
            },
        )
    ).one_or_none()
    if row is None:
        return None
    # The SQL function emits an all-NULL sentinel row after hard-failing a poison /
    # over-lifetime job, distinguishing "poison consumed" from "queue empty" (zero rows).
    if row.job_id is None:
        return POISON_SKIPPED
    return ClaimedJob(job_id=row.job_id, tenant_id=row.tenant_id, options=row.job_options or {})
