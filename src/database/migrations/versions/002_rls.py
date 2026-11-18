"""Phase-D RLS — per-tenant policy on compile_jobs + SECURITY DEFINER reclaim fn.

(1) Strict per-table RLS policy (w2v_common) on the single ``tenant_id`` table
    ``compile_jobs``.
(2) SECURITY DEFINER ``reclaim_orphaned_compile_jobs(text)`` so the startup
    orphan-reclaim can flip restart-orphaned ``processing`` rows to ``interrupted``
    across all tenants under enforce (a non-bypass role with no GUC would touch 0
    rows). EXECUTE is REVOKEd from PUBLIC and GRANTed to the app role via a
    pg_roles-guarded DO block (no-op in CI where the role is absent).

ENABLE-not-FORCE → inert until the runtime connects as the restricted
``w2v_video_compiler_app`` role with ``TENANT_RLS_MODE=enforce``. Fully reversible.

Revision ID: 002_rls
Revises: 001_compile_jobs
"""

from alembic import op

revision = "002_rls"
down_revision = "001_compile_jobs"
branch_labels = None
depends_on = None

# ⚠️ PINNED LITERAL DDL — DO NOT re-derive from ``w2v_common.database.rls`` /
# ``Base.metadata`` / ``src.database.rls_resolvers``. A migration that renders DDL from
# live CODE breaks fresh-database creation the moment that code changes type: at THIS
# revision ``compile_jobs.tenant_id`` is still ``varchar``, so the post-010 helper would
# render the uuid predicate (``tenant_id = NULLIF(...)::uuid``) and CREATE POLICY fails:
#     operator does not exist: character varying = uuid
# ...and the post-010 resolver would declare ``RETURNS TABLE(... tenant_id uuid ...)``
# against a varchar column. A migration must import nothing but alembic/sqlalchemy.
# Era-correct as of head 009. Revision 010 flips the column to uuid and re-emits both.
_PINNED_RLS_UP: list[str] = [
    "ALTER TABLE compile_jobs ENABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON compile_jobs",
    "CREATE POLICY tenant_isolation ON compile_jobs USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), ''))",
]

# Drops EVERY policy name the helper could ever have emitted (strict + the four
# global-or-own variants a NULLABLE tenant_id would have produced), so the downgrade is
# total regardless of which shape created the table.
_PINNED_RLS_DOWN: list[str] = [
    "DROP POLICY IF EXISTS tenant_isolation ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_read ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_ins ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_upd ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_del ON compile_jobs",
    "ALTER TABLE compile_jobs DISABLE ROW LEVEL SECURITY",
]

_PINNED_RESOLVER_UP: list[str] = [
    "\nCREATE OR REPLACE FUNCTION reclaim_orphaned_compile_jobs(p_instance_id text)\n  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_count integer;\n  BEGIN\n    UPDATE compile_jobs\n       SET status = 'interrupted', interrupt_reason = 'crash_orphan', updated_at = now()\n     WHERE status = 'processing'\n       AND (p_instance_id IS NULL OR instance_id IS DISTINCT FROM p_instance_id)\n       AND (lease_expires_at IS NULL OR lease_expires_at < now());\n    GET DIAGNOSTICS v_count = ROW_COUNT;\n    RETURN v_count;\n  END\n  $fn$\n",
    "\nCREATE OR REPLACE FUNCTION reclaim_next_compile_job(\n    p_worker_id text, p_lease_seconds int, p_max_attempts int, p_max_lifetime_seconds int,\n    p_active_job_ids text[]\n) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)\n  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb;\n    v_attempts int; v_started timestamptz; v_status varchar; v_reason varchar;\n  BEGIN\n    SELECT j.id, j.job_id, j.tenant_id, j.job_options, j.attempts, j.started_at,\n           j.status, j.interrupt_reason\n      INTO v_id, v_job_id, v_tenant, v_options, v_attempts, v_started,\n           v_status, v_reason\n    FROM compile_jobs j\n    WHERE j.job_options IS NOT NULL\n      AND (p_active_job_ids IS NULL OR j.job_id <> ALL(p_active_job_ids))\n      AND (\n        j.status = 'interrupted'\n        OR (j.status = 'processing' AND j.lease_expires_at IS NOT NULL\n            AND j.lease_expires_at < now())\n      )\n    ORDER BY j.created_at\n    LIMIT 1\n    FOR UPDATE SKIP LOCKED;\n\n    IF v_id IS NULL THEN\n      RETURN;\n    END IF;\n\n    IF (p_max_attempts > 0 AND COALESCE(v_attempts, 0) >= p_max_attempts)\n       OR (p_max_lifetime_seconds > 0 AND v_started IS NOT NULL\n           AND v_started < now() - make_interval(secs => p_max_lifetime_seconds)) THEN\n      UPDATE compile_jobs\n        SET status = 'failed',\n            error = 'exceeded max reclaim attempts / lifetime — failed as poison job',\n            lease_expires_at = NULL,\n            updated_at = now()\n      WHERE id = v_id;\n      -- Emit ONE all-NULL sentinel row so the caller can DISTINGUISH \"poison job\n      -- consumed here\" (a row was hard-failed) from \"queue empty\" (zero rows). Without\n      -- it, a poison job at head-of-queue is indistinguishable from an empty queue, so\n      -- the reaper BREAKS its sweep and healthy orphans queued behind the poison job\n      -- wait a full reaper interval. The sentinel lets the caller CONTINUE past it.\n      RETURN QUERY SELECT NULL::varchar, NULL::varchar, NULL::jsonb;\n      RETURN;\n    END IF;\n\n    UPDATE compile_jobs\n      SET status = 'processing',\n          instance_id = p_worker_id,\n          lease_expires_at = now() + make_interval(secs => p_lease_seconds),\n          attempts = COALESCE(attempts, 0)\n                   + CASE WHEN v_status = 'interrupted' AND v_reason = 'drain'\n                          THEN 0 ELSE 1 END,\n          interrupt_reason = NULL,\n          started_at = COALESCE(started_at, now()),\n          updated_at = now(),\n          heartbeat_at = now()\n    WHERE id = v_id;\n\n    RETURN QUERY SELECT v_job_id, v_tenant, v_options;\n  END\n  $fn$\n",
    "\nCREATE OR REPLACE FUNCTION prune_compile_jobs(p_ttl_seconds int, p_work_timeout_seconds int)\n  RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$\n  DECLARE\n    v_flipped integer; v_deleted integer;\n  BEGIN\n    UPDATE compile_jobs\n       SET status = 'interrupted', error = 'render task lost (zombie prune)',\n           interrupt_reason = 'crash_zombie', updated_at = now()\n     WHERE status = 'processing'\n       AND heartbeat_at < now() - make_interval(secs => p_work_timeout_seconds);\n    GET DIAGNOSTICS v_flipped = ROW_COUNT;\n    DELETE FROM compile_jobs\n     WHERE status IN ('completed', 'failed', 'interrupted', 'cancelled')\n       AND updated_at < now() - make_interval(secs => p_ttl_seconds);\n    GET DIAGNOSTICS v_deleted = ROW_COUNT;\n    RETURN v_flipped + v_deleted;\n  END\n  $fn$\n",
    "REVOKE EXECUTE ON FUNCTION reclaim_orphaned_compile_jobs(text) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION reclaim_next_compile_job(text, int, int, int, text[]) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION prune_compile_jobs(int, int) FROM PUBLIC",
    "DO $$\nBEGIN\n  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_compiler_app') THEN\n    GRANT EXECUTE ON FUNCTION reclaim_orphaned_compile_jobs(text) TO w2v_video_compiler_app;\n    GRANT EXECUTE ON FUNCTION reclaim_next_compile_job(text, int, int, int, text[]) TO w2v_video_compiler_app;\n    GRANT EXECUTE ON FUNCTION prune_compile_jobs(int, int) TO w2v_video_compiler_app;\n  END IF;\nEND $$",
]

_PINNED_RESOLVER_DOWN: list[str] = [
    "DROP FUNCTION IF EXISTS reclaim_orphaned_compile_jobs(text)",
    "DROP FUNCTION IF EXISTS reclaim_next_compile_job(text, int, int, int, text[])",
    "DROP FUNCTION IF EXISTS prune_compile_jobs(int, int)",
]


def upgrade() -> None:
    for stmt in _PINNED_RLS_UP:
        op.execute(stmt)
    for stmt in _PINNED_RESOLVER_UP:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _PINNED_RESOLVER_DOWN:
        op.execute(stmt)
    for stmt in _PINNED_RLS_DOWN:
        op.execute(stmt)
