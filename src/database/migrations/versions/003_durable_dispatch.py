"""Durable crash-resumption: claim-lease columns + reaper reclaim function.

Adds the columns the per-pod reaper needs on ``compile_jobs`` so a render whose pod
died mid-flight can be reclaimed and re-run on another pod from its persisted request:

- ``job_options``       JSONB reconstruction bundle (kind + request payload + identity +
                        work_timeout). ALSO the reaper-ownership flag: NULL ⇒ unmanaged.
- ``lease_expires_at``  when the current claim expires; a ``processing`` row past this
                        (its owning pod stopped heartbeating) is reclaimable.
- ``attempts``          reclaim count, for the poison-job cap.
- ``started_at``        first-claim wall-clock anchor for the lifetime bound.

Plus the SECURITY DEFINER ``reclaim_next_compile_job(text,int,int,int)`` the reaper uses
to atomically claim ONE orphan across tenants (``FOR UPDATE SKIP LOCKED``) under RLS
enforce. The columns are added FIRST so the function references only existing columns.

This service is ``replicas: 1`` + ``Recreate`` — the old pod is torn down before this
migration's initContainer runs, so there is NO concurrently-live old image calling an
old function signature. Hence a single clean 4-arg reclaim function (no rolling-update
compat shim, unlike the multi-replica ai_content_generation_service 009→010 split).

Additive + backward-compatible: every column is nullable-or-defaulted with no backfill,
and ``reclaim_orphaned_compile_jobs`` / ``prune_compile_jobs`` (owned by 002) are
re-created byte-identically via ``CREATE OR REPLACE``. Fully reversible.

Revision ID: 003_durable_dispatch
Revises: 002_rls
"""

from alembic import op


revision = "003_durable_dispatch"
down_revision = "002_rls"
branch_labels = None
depends_on = None

# ⚠️ PINNED LITERAL DDL — DO NOT re-derive this from ``src.database.rls_resolvers``
# (or any other live module). A migration that renders DDL from live CODE breaks
# fresh-database creation the moment that code changes type: at THIS old revision
# the ``compile_jobs.tenant_id`` column is still ``varchar``, so a resolver rendered
# from post-010 code (``RETURNS TABLE(... tenant_id uuid ...)``) / an RLS predicate
# rendered from post-010 model metadata (``tenant_id = ...::uuid``) fails with
#     operator does not exist: character varying = uuid
# A migration must import nothing but alembic/sqlalchemy. Era-correct as of head 009.
_PINNED_RESOLVER_DDL: list[str] = [
    "\nCREATE OR REPLACE FUNCTION reclaim_orphaned_compile_jobs(p_instance_id text)\n  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_count integer;\n  BEGIN\n    UPDATE compile_jobs\n       SET status = 'interrupted', interrupt_reason = 'crash_orphan', updated_at = now()\n     WHERE status = 'processing'\n       AND (p_instance_id IS NULL OR instance_id IS DISTINCT FROM p_instance_id)\n       AND (lease_expires_at IS NULL OR lease_expires_at < now());\n    GET DIAGNOSTICS v_count = ROW_COUNT;\n    RETURN v_count;\n  END\n  $fn$\n",
    "\nCREATE OR REPLACE FUNCTION reclaim_next_compile_job(\n    p_worker_id text, p_lease_seconds int, p_max_attempts int, p_max_lifetime_seconds int,\n    p_active_job_ids text[]\n) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)\n  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb;\n    v_attempts int; v_started timestamptz; v_status varchar; v_reason varchar;\n  BEGIN\n    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at,\n           j.status, j.interrupt_reason\n      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started,\n           v_status, v_reason\n    FROM compile_jobs j\n    WHERE j.job_options IS NOT NULL\n      AND (p_active_job_ids IS NULL OR j.job_id <> ALL(p_active_job_ids))\n      AND (\n        j.status = 'interrupted'\n        OR (j.status = 'processing' AND j.lease_expires_at IS NOT NULL\n            AND j.lease_expires_at < now())\n      )\n    ORDER BY j.created_at\n    LIMIT 1\n    FOR UPDATE SKIP LOCKED;\n\n    IF v_id IS NULL THEN\n      RETURN;\n    END IF;\n\n    IF (p_max_attempts > 0 AND COALESCE(v_attempts, 0) >= p_max_attempts)\n       OR (p_max_lifetime_seconds > 0 AND v_started IS NOT NULL\n           AND v_started < now() - make_interval(secs => p_max_lifetime_seconds)) THEN\n      UPDATE compile_jobs\n        SET status = 'failed',\n            error = 'exceeded max reclaim attempts / lifetime — failed as poison job',\n            lease_expires_at = NULL,\n            updated_at = now()\n      WHERE id = v_id;\n      -- Emit ONE all-NULL sentinel row so the caller can DISTINGUISH \"poison job\n      -- consumed here\" (a row was hard-failed) from \"queue empty\" (zero rows). Without\n      -- it, a poison job at head-of-queue is indistinguishable from an empty queue, so\n      -- the reaper BREAKS its sweep and healthy orphans queued behind the poison job\n      -- wait a full reaper interval. The sentinel lets the caller CONTINUE past it.\n      RETURN QUERY SELECT NULL::varchar, NULL::varchar, NULL::jsonb;\n      RETURN;\n    END IF;\n\n    UPDATE compile_jobs\n      SET status = 'processing',\n          instance_id = p_worker_id,\n          lease_expires_at = now() + make_interval(secs => p_lease_seconds),\n          attempts = COALESCE(attempts, 0)\n                   + CASE WHEN v_status = 'interrupted' AND v_reason = 'drain'\n                          THEN 0 ELSE 1 END,\n          interrupt_reason = NULL,\n          started_at = COALESCE(started_at, now()),\n          updated_at = now(),\n          heartbeat_at = now()\n    WHERE id = v_id;\n\n    RETURN QUERY SELECT v_job_id, v_tenant, v_options;\n  END\n  $fn$\n",
    "\nCREATE OR REPLACE FUNCTION prune_compile_jobs(p_ttl_seconds int, p_work_timeout_seconds int)\n  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_flipped integer; v_deleted integer;\n  BEGIN\n    UPDATE compile_jobs\n       SET status = 'interrupted', error = 'render task lost (zombie prune)',\n           interrupt_reason = 'crash_zombie', updated_at = now()\n     WHERE status = 'processing'\n       AND heartbeat_at < now() - make_interval(secs => p_work_timeout_seconds);\n    GET DIAGNOSTICS v_flipped = ROW_COUNT;\n    DELETE FROM compile_jobs\n     WHERE status IN ('completed', 'failed', 'interrupted', 'cancelled')\n       AND updated_at < now() - make_interval(secs => p_ttl_seconds);\n    GET DIAGNOSTICS v_deleted = ROW_COUNT;\n    RETURN v_flipped + v_deleted;\n  END\n  $fn$\n",
    "REVOKE EXECUTE ON FUNCTION reclaim_orphaned_compile_jobs(text) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION reclaim_next_compile_job(text, int, int, int, text[]) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION prune_compile_jobs(int, int) FROM PUBLIC",
    "DO $$\nBEGIN\n  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_compiler_app') THEN\n    GRANT EXECUTE ON FUNCTION reclaim_orphaned_compile_jobs(text) TO w2v_video_compiler_app;\n    GRANT EXECUTE ON FUNCTION reclaim_next_compile_job(text, int, int, int, text[]) TO w2v_video_compiler_app;\n    GRANT EXECUTE ON FUNCTION prune_compile_jobs(int, int) TO w2v_video_compiler_app;\n  END IF;\nEND $$",
]


def upgrade() -> None:
    # Columns FIRST so the reclaim function below references only existing columns.
    op.execute("ALTER TABLE compile_jobs ADD COLUMN IF NOT EXISTS job_options JSONB")
    op.execute("ALTER TABLE compile_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ")
    op.execute(
        "ALTER TABLE compile_jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE compile_jobs ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ")
    # Partial claim index: only queue-owned rows, keyed by (status, lease) so the
    # FOR UPDATE SKIP LOCKED reclaim stays cheap as completed rows accumulate.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_compile_jobs_claimable "
        "ON compile_jobs (status, lease_expires_at) WHERE job_options IS NOT NULL"
    )
    # Re-run the single-source function set (idempotent CREATE OR REPLACE): re-creates
    # 002's reclaim/prune byte-identically and adds reclaim_next_compile_job + its
    # REVOKE-from-PUBLIC + pg_roles-guarded app-role GRANT. Same pattern 002_rls uses,
    # so the migration (prod) and the create_all after_create event (tests) stay in sync.
    for stmt in _PINNED_RESOLVER_DDL:
        op.execute(stmt)


def downgrade() -> None:
    # Drop ONLY what this revision added — leave 002's reclaim/prune functions intact.
    op.execute("DROP FUNCTION IF EXISTS reclaim_next_compile_job(text, int, int, int)")
    op.execute("DROP INDEX IF EXISTS ix_compile_jobs_claimable")
    op.execute("ALTER TABLE compile_jobs DROP COLUMN IF EXISTS started_at")
    op.execute("ALTER TABLE compile_jobs DROP COLUMN IF EXISTS attempts")
    op.execute("ALTER TABLE compile_jobs DROP COLUMN IF EXISTS lease_expires_at")
    op.execute("ALTER TABLE compile_jobs DROP COLUMN IF EXISTS job_options")
