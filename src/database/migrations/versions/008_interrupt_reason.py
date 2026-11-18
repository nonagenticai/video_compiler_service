"""A rollout drain no longer burns a poison-attempt life: record WHY a job was interrupted.

``reclaim_next_compile_job`` charged ``attempts + 1`` on EVERY claim, and a graceful
shutdown drain writes ``status='interrupted'`` — which a sibling replica reclaims within
~30s. With ``reclaim_max_attempts=3`` that meant FOUR rollouts poison-failed a perfectly
healthy PAID render, with no fault of its own. This adds ``compile_jobs.interrupt_reason``
(the producer of each interrupt stamps why) and makes the claim's increment conditional:
``interrupted`` + ``'drain'`` costs zero attempts, everything else — including a NULL from
a legacy row or an older pod mid-rollout — still costs one. The column is CLEARED on the
claim so a stale ``'drain'`` can't exempt a later genuine crash of the same row, and
``started_at`` stays anchored unconditionally so the wall-clock lifetime cap remains the
backstop against an unbounded drain→rerun loop.

No CHECK constraint on the column: unknown values already fail safe (they charge an
attempt), while a DB CHECK widened out of step with the Python literal is a known trap.
The function SIGNATURE is unchanged, so this is a plain ``CREATE OR REPLACE`` (no DROP →
no EXECUTE-ACL loss and no 500s from old pods mid-rollout). Single-sourced: replays
``rls_resolvers.upgrade_statements()``. The ADD COLUMN must run FIRST — the new function
body reads ``j.interrupt_reason``.

Revision ID: 008_interrupt_reason
Revises: 007_reclaim_poison_sentinel
"""

from alembic import op


revision = "008_interrupt_reason"
down_revision = "007_reclaim_poison_sentinel"
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
    # IF NOT EXISTS: the test/create_all path builds the column from the ORM model, and a
    # re-run of this migration against an already-migrated DB must be a no-op, not an error.
    op.execute("ALTER TABLE compile_jobs ADD COLUMN IF NOT EXISTS interrupt_reason varchar(32)")
    for stmt in _PINNED_RESOLVER_DDL:
        op.execute(stmt)


def downgrade() -> None:
    # Deliberately a NO-OP. Dropping ``interrupt_reason`` would break the LIVE function
    # body (which SELECTs it) for every subsequent claim — a downgrade of the column
    # without a simultaneous downgrade of the function is strictly worse than leaving an
    # unused nullable column in place. Forward-only service.
    pass
