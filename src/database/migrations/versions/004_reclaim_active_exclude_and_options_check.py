"""Reaper self-reclaim guard at the claim + job_options JSONB-null CHECK.

Two additive, forward-safe changes on ``compile_jobs``:

1. ``reclaim_next_compile_job`` gains a ``p_active_job_ids text[]`` argument so the
   reaper can exclude jobs the CALLING pod is actively rendering. A multi-beat DB stall
   can lapse the lease of a render STILL running on this very pod; without the exclusion
   the same-pod reaper claims its own live job, bumps ``attempts``, re-stamps the lease,
   and at the attempt cap POISON-FAILS a render about to complete (lost work + no
   billing). The post-claim veto could not undo those side-effects — they happen inside
   the function — so the skip must be BEFORE the claim. The 4-arg signature (created by
   003 under the old code) is DROPped and replaced by the single-sourced 5-arg set.

   This service is ``replicas: 1`` + ``Recreate`` — the initContainer applies this
   migration after the old pod is torn down and before the new (5-arg-calling) pod
   starts, so there is no version skew and no rolling-update compat shim is needed
   (same reasoning as 003).

2. A table CHECK makes the ``none_as_null`` invariant unrepresentable at the storage
   layer: ``job_options`` may be SQL NULL (unmanaged) or a real object, never the JSONB
   scalar ``'null'`` (which would read as IS NOT NULL and mis-mark a row reaper-managed
   with no reconstruction bundle). Any pre-existing ``'null'`` row is sanitized to SQL
   NULL first so ADD CONSTRAINT can never fail on legacy data.

Revision ID: 004_reclaim_active_exclude_and_options_check
Revises: 003_durable_dispatch
"""

from alembic import op


revision = "004_active_exclude_options_ck"
down_revision = "003_durable_dispatch"
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

_CK_NAME = "ck_compile_jobs_job_options_not_json_null"


def upgrade() -> None:
    # Drop the OLD 4-arg reclaim function (003 created it under the pre-fix code); a
    # different argument list would otherwise coexist as a dead overload. The 5-arg set
    # is (re-)created idempotently from the single source below.
    op.execute("DROP FUNCTION IF EXISTS reclaim_next_compile_job(text, int, int, int)")
    for stmt in _PINNED_RESOLVER_DDL:
        op.execute(stmt)

    # Sanitize any legacy JSONB-scalar-'null' bundle to SQL NULL (unmanaged) so the
    # constraint cannot fail on pre-existing rows, then forbid it going forward.
    op.execute(
        "UPDATE compile_jobs SET job_options = NULL WHERE jsonb_typeof(job_options) = 'null'"
    )
    op.execute(
        f"ALTER TABLE compile_jobs ADD CONSTRAINT {_CK_NAME} "
        "CHECK (job_options IS NULL OR jsonb_typeof(job_options) <> 'null')"
    )


def downgrade() -> None:
    # Forward-only service: reverse only the CHECK. The reclaim function stays 5-arg —
    # that is exactly what 003 produces under the current single-sourced code, so a
    # downgrade to 003 legitimately leaves the 5-arg set in place.
    op.execute(f"ALTER TABLE compile_jobs DROP CONSTRAINT IF EXISTS {_CK_NAME}")
