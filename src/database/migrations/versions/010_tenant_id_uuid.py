"""tenant_id: varchar(128) + empty-string server_default -> uuid NOT NULL, no default.

Product-owner ruling: ``tenant_id`` may NEVER hold a sentinel. Every row carries a real
registered tenant forwarded from the requestor. The sentinel here lived in the SCHEMA
DEFAULT (``server_default=''``), not the data — the live table is empty, so it would have
been re-minted on the first write after any migration.

What this does, in the only order Postgres allows:
  1. DROP the ``server_default`` (the sentinel factory).
  2. FAIL LOUD if any existing row is not a UUID — it names the count and the first
     offender and refuses to guess. Nothing is dropped or skipped.
  3. DROP the RLS policies. Postgres refuses ``ALTER COLUMN ... TYPE`` while a policy
     references the column (``cannot alter type of a column used in a policy definition``).
  4. ``ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid``. Stays **NOT NULL**:
     ``w2v_common.database.rls`` switches policy shape on nullability — a NULLABLE tenant
     column gets the "global-or-own" set under which NULL-tenant rows are readable by
     EVERY tenant. Making the column nullable to let the sentinel cast would convert it
     into a cross-tenant data leak. Never do that.
  5. Re-CREATE the strict policy with ``::uuid`` on the GUC.
  6. DROP + CREATE ``reclaim_next_compile_job`` with ``tenant_id uuid`` in its RETURNS
     TABLE and a ``v_tenant uuid`` local. ``CREATE OR REPLACE`` cannot change a return
     type, and ``ALTER TABLE`` does not revalidate plpgsql bodies — so without this the
     SECURITY DEFINER reaper would keep its varchar signature and break at RUNTIME
     (no assignment cast uuid -> varchar) with nothing saying so until a reclaim ran.
     The DROP discards the function's EXECUTE ACL, so the REVOKE + guarded GRANT are
     re-applied with it.

Downgrade restores varchar(128) + the legacy TEXT policy predicate and the varchar
resolver, all spelled out LITERALLY. A downgrade can never call the w2v_common /
rls_resolvers helpers: those render from POST-flip code and would re-emit the uuid form
into a varchar column (``operator does not exist: character varying = uuid``).
The old empty-string ``server_default`` is deliberately NOT restored — it is the sentinel
factory this revision exists to remove.

Revision ID: 010_tenant_id_uuid
Revises: 009_prune_cancelled_jobs
"""

from alembic import op

revision = "010_tenant_id_uuid"
down_revision = "009_prune_cancelled_jobs"
branch_labels = None
depends_on = None

_SIG = "reclaim_next_compile_job(text, int, int, int, text[])"

_GUARD_NON_UUID_ROWS = """
DO $guard$
DECLARE
  v_bad bigint;
  v_first text;
BEGIN
  SELECT count(*), min(tenant_id)
    INTO v_bad, v_first
    FROM compile_jobs
   WHERE tenant_id !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';
  IF v_bad > 0 THEN
    RAISE EXCEPTION
      'compile_jobs.tenant_id: % row(s) hold a value that is not a tenant UUID (first: %). '
      'NOTHING was dropped, skipped, or rewritten - this migration refuses to guess an '
      'identity for a row. tenant_id may never hold a sentinel (the old empty-string '
      'server_default or the nil UUID): every row must carry a REAL registered tenant. '
      'Resolve each offending row by mapping its tenant NAME to the tenant uuid in the '
      'licenses service table licenses.tenants, UPDATE compile_jobs.tenant_id to that '
      'uuid, then re-run this migration.',
      v_bad, coalesce(v_first, '<null>');
  END IF;
END
$guard$
"""

# ⚠️ PINNED LITERAL DDL — never re-derive from ``w2v_common.database.rls`` /
# ``Base.metadata`` / ``src.database.rls_resolvers``, and never by string-replacing the
# live module (``sql.replace("tenant_id uuid", "tenant_id varchar")``). A migration that
# renders DDL from live CODE breaks fresh-database creation the moment that code changes
# shape again. A migration must import nothing but alembic/sqlalchemy.
_POLICY_UUID: list[str] = [
    "ALTER TABLE compile_jobs ENABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON compile_jobs",
    "CREATE POLICY tenant_isolation ON compile_jobs USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')::uuid)",
]

_POLICY_VARCHAR_LEGACY: list[str] = [
    "ALTER TABLE compile_jobs ENABLE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_isolation ON compile_jobs",
    "CREATE POLICY tenant_isolation ON compile_jobs USING (tenant_id = NULLIF(current_setting('app.current_tenant', true), '')) WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant', true), ''))",
]

# Drops every policy name the helper could ever have emitted (strict + the four
# global-or-own variants), so the drop is total whichever shape created the table.
_POLICY_DROP_ALL: list[str] = [
    "DROP POLICY IF EXISTS tenant_isolation ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_read ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_ins ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_upd ON compile_jobs",
    "DROP POLICY IF EXISTS tenant_isolation_del ON compile_jobs",
    "ALTER TABLE compile_jobs DISABLE ROW LEVEL SECURITY",
]

_RECLAIM_NEXT_UUID = """
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

_RECLAIM_NEXT_VARCHAR_LEGACY = """
CREATE OR REPLACE FUNCTION reclaim_next_compile_job(
    p_worker_id text, p_lease_seconds int, p_max_attempts int, p_max_lifetime_seconds int,
    p_active_job_ids text[]
) RETURNS TABLE(job_id varchar, tenant_id varchar, job_options jsonb)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
  DECLARE
    v_id bigint; v_job_id varchar; v_tenant varchar; v_options jsonb;
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
      RETURN QUERY SELECT NULL::varchar, NULL::varchar, NULL::jsonb;
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

_REVOKE_RECLAIM_NEXT = f"REVOKE EXECUTE ON FUNCTION {_SIG} FROM PUBLIC"

# Guarded so it runs in prod (role present) and is a clean no-op in CI (role absent).
_GRANT_RECLAIM_NEXT = (
    "DO $grant$\nBEGIN\n"  # nosec B608  # noqa: S608 -- fixed literal sig + role name
    "  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_compiler_app') THEN\n"
    f"    GRANT EXECUTE ON FUNCTION {_SIG} TO w2v_video_compiler_app;\n"
    "  END IF;\nEND $grant$"
)


def upgrade() -> None:
    op.execute("ALTER TABLE compile_jobs ALTER COLUMN tenant_id DROP DEFAULT")
    op.execute(_GUARD_NON_UUID_ROWS)
    for stmt in _POLICY_DROP_ALL:
        op.execute(stmt)
    op.execute("ALTER TABLE compile_jobs ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")
    op.execute("ALTER TABLE compile_jobs ALTER COLUMN tenant_id SET NOT NULL")
    for stmt in _POLICY_UUID:
        op.execute(stmt)
    op.execute(f"DROP FUNCTION IF EXISTS {_SIG}")
    op.execute(_RECLAIM_NEXT_UUID)
    op.execute(_REVOKE_RECLAIM_NEXT)
    op.execute(_GRANT_RECLAIM_NEXT)


def downgrade() -> None:
    op.execute(f"DROP FUNCTION IF EXISTS {_SIG}")
    for stmt in _POLICY_DROP_ALL:
        op.execute(stmt)
    op.execute(
        "ALTER TABLE compile_jobs ALTER COLUMN tenant_id TYPE varchar(128) USING tenant_id::text"
    )
    op.execute("ALTER TABLE compile_jobs ALTER COLUMN tenant_id SET NOT NULL")
    for stmt in _POLICY_VARCHAR_LEGACY:
        op.execute(stmt)
    op.execute(_RECLAIM_NEXT_VARCHAR_LEGACY)
    op.execute(_REVOKE_RECLAIM_NEXT)
    op.execute(_GRANT_RECLAIM_NEXT)
