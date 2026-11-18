"""SQLAlchemy ORM models for video_compiler_service.

A single ``compile_jobs`` table backs BOTH the async job store (``job_id`` →
status/result) and the Idempotency-Key dedup index (``(tenant_id,
idempotency_key)``). Persisting it (vs the former per-pod in-memory dicts) is
what makes a re-submitted render dedupe across a pod restart instead of firing
a second PAID compile.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import CheckConstraint, DateTime, Index, Integer, String, Text, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import BigInteger
from w2v_common.database.rls import register_rls_ddl


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


class CompileJob(Base):
    """A durable compile / concat-audio job.

    ``status`` lifecycle: ``processing`` → ``completed`` | ``failed``; a
    ``processing`` row orphaned by a pod restart is flipped to ``interrupted``
    at startup (see ``reclaim_orphaned_compile_jobs``) and re-run under the SAME
    ``job_id`` on the next same-key submit. ``result`` holds the whole poll
    payload (artifact_url, dims, sha256, …) as JSONB.
    """

    __tablename__ = "compile_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: uuid.uuid7().hex
    )
    # ``uuid`` NOT NULL with NO server_default. Two invariants, both load-bearing:
    #  * NOT NULL — w2v_common then emits the STRICT per-tenant RLS policy
    #    (tenant_id = GUC::uuid). A NULLABLE tenant column gets the "global-or-own"
    #    policy set under which a NULL-tenant row is readable by EVERY tenant, which
    #    would turn a missing tenant into a cross-tenant data leak. Never relax this.
    #  * NO server_default — the old ``server_default=''`` re-created an empty-string
    #    sentinel on the first write after any migration. Every row must carry a REAL
    #    registered tenant forwarded from the requestor; an absent one raises
    #    ``MissingTenantError`` at the boundary rather than being substituted.
    tenant_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="processing", server_default="processing"
    )
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The pod boot id that owns the in-flight render task. A row still
    # ``processing`` at startup with a DIFFERENT instance_id is a restart orphan.
    instance_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Durable-dispatch / crash-resumption (reaper) columns ---
    # ``job_options`` is the serialized submit bundle (kind + request payload +
    # identity + work_timeout) a DIFFERENT pod needs to reconstruct and re-run the
    # render after a crash. It is ALSO the reaper-ownership flag: NULL ⇒ this row is
    # not managed by the claim-lease queue (legacy/unmanaged) and the reaper ignores
    # it. Only http(s)-addressed inputs reconstruct cross-pod; a ``file://`` input
    # or an expired presign fail-fasts the re-dispatch (see the reaper).
    # ``none_as_null=True`` is load-bearing: the reclaim scan gates on
    # ``job_options IS NOT NULL``, so a Python ``None`` MUST persist as SQL NULL, not
    # JSONB ``'null'`` (SQLAlchemy's default for JSON None) — otherwise an unmanaged row
    # would look managed and the reaper would try to reclaim a job with no request.
    job_options: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    # When the current claim expires. A ``processing`` row past this (its owning pod
    # stopped heartbeating ⇒ died) is reclaimable by any pod's reaper; a live render
    # renews it every ``lease_seconds/3``. NULL once the job reaches a terminal state.
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Reaper reclaim count, for the poison-job cap. Bumped by the CLAIM inside
    # ``reclaim_next_compile_job`` — which is what a graceful drain also ends up going
    # through (a drain writes ``status='interrupted'`` and a sibling replica then reclaims
    # that row), so the claim charges an attempt UNLESS ``interrupt_reason='drain'`` marks
    # the interruption as a rollout rather than a fault. Every other reason — and NULL —
    # charges one.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # WHY the row was last flipped to ``interrupted``, written by the producer of that
    # flip (drain / render_failure / crash_zombie / crash_orphan — see
    # ``compile_job_store.InterruptReason``). Read by the reclaim claim to decide whether
    # to charge an attempt, then CLEARED on that claim: a stale ``'drain'`` left behind
    # would exempt every LATER crash of the same row from the poison cap. Deliberately a
    # plain varchar with no CHECK — an unknown/legacy value must fail SAFE (charge an
    # attempt), and a DB CHECK widened out of step with the Python literal is a known
    # fleet trap.
    interrupt_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # First-claim wall-clock anchor for the lifetime cap; NULL = never claimed by the
    # reaper. Never overwritten, so a job legitimately queued for hours is not aged out.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        # Tenant-scoped Idempotency-Key dedup. NULLS NOT DISTINCT (PG15+) so two
        # NULL-tenant submits of the same key still collide → dedupe. Partial:
        # keyless submits (idempotency_key IS NULL) never dedupe.
        Index(
            "uq_compile_jobs_tenant_idempotency_key",
            "tenant_id",
            "idempotency_key",
            unique=True,
            postgresql_nulls_not_distinct=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        # Backpressure/reclaim/prune scans walk by status.
        Index("ix_compile_jobs_status", "status"),
        Index("ix_compile_jobs_status_updated_at", "status", "updated_at"),
        # Reaper claim scan: only queue-owned rows (job_options IS NOT NULL), keyed
        # by (status, lease_expires_at) so the FOR UPDATE SKIP LOCKED reclaim stays
        # cheap as completed rows accumulate. Partial → completed/keyless rows are
        # not indexed here.
        Index(
            "ix_compile_jobs_claimable",
            "status",
            "lease_expires_at",
            postgresql_where=text("job_options IS NOT NULL"),
        ),
        # Make the ``none_as_null`` invariant unrepresentable: ``job_options`` is either
        # SQL NULL (unmanaged) or a real object, never the JSONB scalar ``'null'`` (which
        # reads as IS NOT NULL and would mis-mark a row reaper-managed with no bundle).
        # Mirrors migration 004; keeps create_all (tests) in sync with the migration.
        CheckConstraint(
            "job_options IS NULL OR jsonb_typeof(job_options) <> 'null'",
            name="ck_compile_jobs_job_options_not_json_null",
        ),
    )


# Phase-D RLS: per-tenant_id-table policy set (w2v_common) registered as an
# after_create event so tests' create_all matches the 002_rls migration. Inert
# until the runtime connects as restricted w2v_video_compiler_app with
# TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

from src.database.rls_resolvers import register_reclaim_fns  # noqa: E402

# SECURITY DEFINER cross-tenant reclaim function (owner bypasses RLS ENABLE),
# called once at startup to flip restart-orphaned processing rows to interrupted.
register_reclaim_fns(Base.metadata)
