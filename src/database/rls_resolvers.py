"""SECURITY DEFINER cross-tenant reclaim function for restart orphan recovery.

At startup the service must flip every ``processing`` ``compile_jobs`` row left
behind by a now-dead pod to ``interrupted``, so the next same-key submit RE-RUNS it under
the same ``job_id`` instead of dead-ending on a job whose in-process render task
died with the pod. That UPDATE is cross-tenant, but under ``TENANT_RLS_MODE=enforce``
the runtime is a non-bypass role with no ``app.current_tenant`` bound, so a plain
UPDATE fail-closes to 0 rows. It therefore runs through this SECURITY DEFINER
function — owned by the table owner, which bypasses the ENABLE'd policies.

EXECUTE is REVOKEd from PUBLIC then GRANTed to the restricted app role
``w2v_video_compiler_app`` via a ``pg_roles``-guarded DO block, so the runtime can
call it in prod while CI — where the role is absent — skips the grant cleanly.
Single-sourced so BOTH the migration (prod) and the ``create_all`` ``after_create``
event (tests) run an identical CREATE + REVOKE + guarded-GRANT sequence.

Topology note: this service is ``replicas=2`` + ``RollingUpdate`` in prod
(verified 2026-08-21). Several docstrings here and in the 003/004 migrations
still say ``replicas: 1`` + ``Recreate``; that was true when they were written
and is not true now. The reclaim does not depend on it — see the LEASE-AWARE
comment on ``RECLAIM_FN_SQL`` below, which is the binding contract.

Residual, documented deliberately rather than "fixed" blind: the lease predicate
also admits ``lease_expires_at IS NULL``, and ``submit`` leaves the lease NULL for
UNMANAGED rows (``job_options IS NULL`` — tests and non-render callers). So a
booting sibling WOULD flip a live unmanaged ``processing`` row. Measured on prod
2026-08-21 across all four tenants: 0 rows with ``job_options IS NULL``, 0 with
``interrupt_reason='crash_orphan'``, so no such path exists in practice. Do not
narrow the predicate without a reproducible caller — the NULL branch is what
makes a genuinely dead pod's rows immediately reaper-reclaimable, and that
fast-recovery guarantee is load-bearing.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

# The restricted runtime role the app connects as under TENANT_RLS_MODE=enforce.
APP_ROLE = "w2v_video_compiler_app"

# Flip restart-orphaned processing rows (owned by a now-dead instance) to
# interrupted, cross-tenant. Returns the count for a startup log line. Excludes
# the caller's OWN instance_id so it is a no-op replay for the live pod.
#
# LEASE-AWARE (required for replicas>1): only flip a row whose lease has ALSO lapsed
# (or was never leased — an unmanaged row). At replicas:1+Recreate the prior pod is
# provably gone, so any other-instance processing row is an orphan; but at replicas>1 a
# booting pod must NOT interrupt a LIVE sibling's in-flight render (whose lease is still
# in the future because the sibling is heartbeating it). The lease disambiguates
# "dead pod" from "live-but-different pod". A hard-crashed pod's rows lapse within
# ``job_lease_seconds`` and the reaper (also lease-gated) then reclaims them; a graceful
# drain flips its own rows via mark_interrupted before this ever runs, so a row reaching
# here is an unannounced death, not a rollout — hence ``interrupt_reason='crash_orphan'``,
# which the reclaim claim charges an attempt for (only ``'drain'`` is exempt).
RECLAIM_FN_SQL = """
CREATE OR REPLACE FUNCTION reclaim_orphaned_compile_jobs(p_instance_id text)
  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_count integer;
  BEGIN
    UPDATE compile_jobs
       SET status = 'interrupted', interrupt_reason = 'crash_orphan', updated_at = now()
     WHERE status = 'processing'
       AND (p_instance_id IS NULL OR instance_id IS DISTINCT FROM p_instance_id)
       AND (lease_expires_at IS NULL OR lease_expires_at < now());
    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
  END
  $fn$
"""

# Reaper claim (crash-resumption): atomically claim ONE reclaimable job across all
# tenants and return the bundle needed to re-run it on THIS pod. SECURITY DEFINER for
# the same reason as reclaim/prune — a plain UPDATE under RLS enforce is scoped to the
# caller's GUC tenant, but a reaper must claim any tenant's orphan.
#
# Reclaimable = job_options IS NOT NULL (queue-managed) AND either:
#   * status='interrupted'  — the startup reclaim / shutdown drain already declared it
#     orphaned; claim it immediately regardless of the stale lease; OR
#   * status='processing' AND lease_expires_at < now() — its owning pod stopped
#     heartbeating (died), so the lease lapsed. A live render renews the lease, so this
#     never grabs a job still running on a healthy pod.
# ...AND NOT a job the CALLING pod is actively rendering right now (p_active_job_ids =
# the caller's in-process live-render set). A multi-beat DB stall can lapse the lease of
# a render that is STILL running on this very pod; without this exclusion the reaper would
# claim its own live job — bumping attempts, re-stamping the lease, and (at the attempt
# cap) POISON-FAILING a render that is about to complete. The post-claim veto in
# redispatch_reclaimed cannot undo those side-effects because they already happened inside
# this function; the only correct place to skip a live job is BEFORE the claim, here.
# FOR UPDATE SKIP LOCKED so concurrent reapers never claim the same row. Poison cap
# (attempts >= p_max_attempts → failed) and a wall-clock lifetime cap anchored on
# started_at (a job that has RUN longer than p_max_lifetime_seconds is failed, not
# re-run — so a genuinely wedged render can't be reclaimed + re-spent forever). A
# claim flips the row → processing, stamps this worker, a fresh lease, and started_at on
# first claim. Returns (job_id, tenant_id, job_options).
#
# ATTEMPT ACCOUNTING: the claim charges attempts+1 for every reclaim EXCEPT one whose row
# was flipped by a graceful shutdown drain (status='interrupted' AND
# interrupt_reason='drain'). A drain is a rollout, not a fault: a sibling replica reclaims
# the drained row within ~30s, so without the exemption ``reclaim_max_attempts`` rollouts
# poison-fail a perfectly healthy PAID render. The test is an ALLOWLIST OF ONE — a NULL
# reason (legacy rows, an older pod that doesn't write the column) compares UNKNOWN and
# falls to ELSE 1, so anything unrecognised still costs an attempt. ``interrupt_reason`` is
# CLEARED on the claim: a stale 'drain' would otherwise exempt every later CRASH of that
# same row and make a crash-loop unbounded. The exemption covers the INCREMENT only — the
# poison gate above still hard-fails a drained row that is ALREADY at the cap — and
# ``started_at`` is anchored UNCONDITIONALLY, so the wall-clock lifetime cap remains the
# backstop against an endless drain→rerun loop.
# ``tenant_id`` is ``uuid`` (migration 010). Postgres has NO assignment cast
# uuid→varchar, so the old ``RETURNS TABLE(... tenant_id varchar)`` + ``v_tenant varchar``
# broke this SECURITY DEFINER reaper at RUNTIME after the column flipped — ``ALTER TABLE``
# does not revalidate plpgsql function bodies, so nothing said so until a reclaim ran.
# The return type CHANGED, so this must be DROP + CREATE: ``CREATE OR REPLACE`` cannot
# change a function's return type (``cannot change return type of existing function``).
RECLAIM_NEXT_FN_SQL = """
CREATE OR REPLACE FUNCTION reclaim_next_compile_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int, p_max_lifetime_seconds int,
    p_active_job_ids text[]
) RETURNS TABLE(job_id varchar, tenant_id uuid, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant uuid; v_options jsonb;
    v_attempts int; v_started timestamptz; v_status varchar; v_reason varchar;
  BEGIN
    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at,
           j.status, j.interrupt_reason
      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started,
           v_status, v_reason
    FROM compile_jobs j
    WHERE j.job_options IS NOT NULL
      AND (p_active_job_ids IS NULL OR j.job_id <> ALL(p_active_job_ids))
      AND (
        j.status = 'interrupted'
        OR (j.status = 'processing' AND j.lease_expires_at IS NOT NULL
            AND j.lease_expires_at < now())
      )
    ORDER BY j.created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NULL THEN
      RETURN;
    END IF;

    IF (p_max_attempts > 0 AND COALESCE(v_attempts, 0) >= p_max_attempts)
       OR (p_max_lifetime_seconds > 0 AND v_started IS NOT NULL
           AND v_started < now() - make_interval(secs => p_max_lifetime_seconds)) THEN
      UPDATE compile_jobs
        SET status = 'failed',
            error = 'exceeded max reclaim attempts / lifetime — failed as poison job',
            lease_expires_at = NULL,
            updated_at = now()
      WHERE id = v_id;
      -- Emit ONE all-NULL sentinel row so the caller can DISTINGUISH "poison job
      -- consumed here" (a row was hard-failed) from "queue empty" (zero rows). Without
      -- it, a poison job at head-of-queue is indistinguishable from an empty queue, so
      -- the reaper BREAKS its sweep and healthy orphans queued behind the poison job
      -- wait a full reaper interval. The sentinel lets the caller CONTINUE past it.
      RETURN QUERY SELECT NULL::varchar, NULL::uuid, NULL::jsonb;
      RETURN;
    END IF;

    UPDATE compile_jobs
      SET status = 'processing',
          instance_id = p_worker_id,
          lease_expires_at = now() + make_interval(secs => p_lease_seconds),
          attempts = COALESCE(attempts, 0)
                   + CASE WHEN v_status = 'interrupted' AND v_reason = 'drain'
                          THEN 0 ELSE 1 END,
          interrupt_reason = NULL,
          started_at = COALESCE(started_at, now()),
          updated_at = now(),
          heartbeat_at = now()
    WHERE id = v_id;

    RETURN QUERY SELECT v_job_id, v_tenant, v_options;
  END
  $fn$
"""

# Housekeeping (flip zombie processing → interrupted; evict aged terminal rows)
# ACROSS ALL TENANTS. Like reclaim it must be SECURITY DEFINER: a plain UPDATE/DELETE
# under enforce is RLS-scoped to the caller's GUC tenant, so it would only ever prune
# the submitting tenant's rows (unbounded growth + a zombie never flipped for a silent
# tenant). A render is bounded by asyncio.wait_for(work(), work_timeout), so a row
# still processing past work_timeout has lost its task. Returns the flipped+deleted count.
# The DELETE list must name EVERY terminal status: a status missing from it is never
# evicted and grows without bound (this is why 'cancelled' is here — see migration 009).
PRUNE_FN_SQL = """
CREATE OR REPLACE FUNCTION prune_compile_jobs(p_ttl_seconds int, p_work_timeout_seconds int)
  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_flipped integer; v_deleted integer;
  BEGIN
    UPDATE compile_jobs
       SET status = 'interrupted', error = 'render task lost (zombie prune)',
           interrupt_reason = 'crash_zombie', updated_at = now()
     WHERE status = 'processing'
       AND heartbeat_at < now() - make_interval(secs => p_work_timeout_seconds);
    GET DIAGNOSTICS v_flipped = ROW_COUNT;
    DELETE FROM compile_jobs
     WHERE status IN ('completed', 'failed', 'interrupted', 'cancelled')
       AND updated_at < now() - make_interval(secs => p_ttl_seconds);
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_flipped + v_deleted;
  END
  $fn$
"""

_FNS = (
    "reclaim_orphaned_compile_jobs(text)",
    "reclaim_next_compile_job(text, int, int, int, text[])",
    "prune_compile_jobs(int, int)",
)
_DROP = [f"DROP FUNCTION IF EXISTS {sig}" for sig in _FNS]
_REVOKE = [f"REVOKE EXECUTE ON FUNCTION {sig} FROM PUBLIC" for sig in _FNS]


def _guarded_grant(grants: list[str]) -> str:
    """Wrap EXECUTE grants in a ``pg_roles``-guarded DO block.

    Grants to ``APP_ROLE`` only when the role exists — so the statement runs in
    prod (role present) and is a clean no-op in CI/tests (role absent).
    """
    body = "\n    ".join(grants)
    return (
        "DO $$\nBEGIN\n"  # nosec B608  # noqa: S608 -- APP_ROLE/grants are hardcoded constants
        f"  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN\n"
        f"    {body}\n"
        "  END IF;\nEND $$"
    )


_GRANT = _guarded_grant([f"GRANT EXECUTE ON FUNCTION {sig} TO {APP_ROLE};" for sig in _FNS])


#: ``reclaim_next_compile_job``'s RETURN TYPE changed varchar→uuid, and
#: ``CREATE OR REPLACE`` cannot change a return type. It must be DROPped first; the
#: REVOKE/GRANT below re-establish its EXECUTE ACL, which the DROP discards.
_DROP_RECLAIM_NEXT = "DROP FUNCTION IF EXISTS reclaim_next_compile_job(text, int, int, int, text[])"


def upgrade_statements() -> list[str]:
    return [
        RECLAIM_FN_SQL,
        _DROP_RECLAIM_NEXT,
        RECLAIM_NEXT_FN_SQL,
        PRUNE_FN_SQL,
        *_REVOKE,
        _GRANT,
    ]


def downgrade_statements() -> list[str]:
    return list(_DROP)


def register_reclaim_fns(metadata: MetaData) -> None:
    """Create + grant the reclaim resolver after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for stmt in upgrade_statements():
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
