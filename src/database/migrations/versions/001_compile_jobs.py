"""Initial schema: durable compile_jobs store + Idempotency-Key dedup index.

Replaces the former per-pod in-memory job store + idempotency dict so a
re-submitted render dedupes across a pod restart instead of firing a second
PAID compile.

Revision ID: 001_compile_jobs
Revises: None
Create Date: 2026-07-22
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001_compile_jobs"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compile_jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False, server_default=""),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="processing"),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("instance_id", sa.String(64), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_compile_jobs_tenant_id", "compile_jobs", ["tenant_id"])
    op.create_index("ix_compile_jobs_status", "compile_jobs", ["status"])
    op.create_index("ix_compile_jobs_status_updated_at", "compile_jobs", ["status", "updated_at"])
    # Tenant-scoped Idempotency-Key dedup. NULLS NOT DISTINCT (PG15+) so two
    # NULL-tenant submits of the same key still collide → dedupe. Partial: a
    # keyless submit (idempotency_key IS NULL) never dedupes.
    op.create_index(
        "uq_compile_jobs_tenant_idempotency_key",
        "compile_jobs",
        ["tenant_id", "idempotency_key"],
        unique=True,
        postgresql_nulls_not_distinct=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_compile_jobs_tenant_idempotency_key", table_name="compile_jobs")
    op.drop_index("ix_compile_jobs_status_updated_at", table_name="compile_jobs")
    op.drop_index("ix_compile_jobs_status", table_name="compile_jobs")
    op.drop_index("ix_compile_jobs_tenant_id", table_name="compile_jobs")
    op.drop_table("compile_jobs")
