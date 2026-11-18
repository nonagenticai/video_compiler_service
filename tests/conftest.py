"""Test fixtures — bypass AuthMiddleware so unit tests don't need a license."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Capture the durable-store test DB URL at module load. CI sets DATABASE_URL to
# the per-job ephemeral pgvector sidecar (see .github/workflows/ci-cd.yaml,
# has_postgres: true → DB video_compiler_service_ci); locally falls back to the
# compose port below (5478, NOT 5432 — a 5432 default resolves to whatever
# postgres the dev is running, answering the TCP connect then failing auth,
# disguising a config error as a connection error). Seed it at module import so
# Settings() (which now carries a required DatabaseSettings) constructs during
# collection.
_TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test@127.0.0.1:5478/video_compiler_service_ci",  # pragma: allowlist secret
)
os.environ.setdefault("DATABASE_URL", _TEST_DATABASE_URL)

# ---------------------------------------------------------------------------
# Tenant identity for tests.
#
# ``compile_jobs.tenant_id`` is ``uuid NOT NULL`` with NO server_default (migration
# 010). There is no ``''`` sentinel and no nil-uuid ``DEFAULT_TENANT_ID`` fallback any
# more, so a test tenant must be a REAL uuid — the old ``"t1"``/``"tenant-a"`` string
# literals are not merely stylistically wrong, they raise ``MissingTenantError``.
# Shared here so the whole suite scopes against the same two identities and a
# cross-tenant assertion is unambiguously cross-tenant.
# ---------------------------------------------------------------------------
TENANT_A = uuid.UUID("aaaaaaaa-0000-4000-8000-0000000000a1")
TENANT_B = uuid.UUID("bbbbbbbb-0000-4000-8000-0000000000b2")


@pytest.fixture(autouse=True)
def _isolate_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin ``work_dir`` to a fresh tmp per test so the suite stays hermetic.

    The persistent per-shot render cache lives at ``work_dir/shot-render-cache``
    and — by design — survives across ``POST /api/compile`` calls (it is a
    sibling of the throwaway ``job-*`` scratch). With the default shared
    ``work_dir`` that persistence leaks cached segments BETWEEN local test runs:
    a hash hit skips the (faked) render, so command-shape assertions see a stale
    or absent filter node. CI starts each run with an empty dir, so it only bites
    locally — pin ``work_dir`` to a fresh ``tmp_path`` (and clear the cached
    ``get_settings``) so every test gets an empty cache.
    """
    from src.config import get_settings

    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
async def _db_schema() -> AsyncGenerator[None]:
    """Create the durable-store schema + TRUNCATE per test (real PG18).

    ``create_all`` fires the ``after_create`` listeners registered in
    ``src.database.models`` (RLS policies via w2v_common + the SECURITY DEFINER
    ``reclaim_orphaned_compile_jobs`` function), so tests exercise the real schema.
    TENANT_RLS_MODE is unset here, so RLS is ENABLE-not-FORCE + the owner role
    bypasses it — the store's own ``WHERE tenant_id`` filter is what enforces the
    cross-tenant 404 in tests. Start the DB with
    ``docker compose up -d video_compiler_service_test_postgres``.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()
    yield


@pytest.fixture
async def db_session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession]]:
    """A sessionmaker on a fresh engine to the test DB (for durable-store unit tests).

    The engine is DISPOSED on teardown. Previously it was created and returned without
    disposal, so every test leaked a QueuePool of open connections that lingered
    server-side (finalized only via GC on an already-closed event loop) — across a full
    run they accumulate against the CI Postgres ``max_connections``, which manifests as
    the first tests of a subsequent run erroring until the DB is restarted.
    """
    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    try:
        yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession]:
    async with db_session_factory() as session:
        yield session


@pytest.fixture
async def app() -> FastAPI:
    """A FastAPI app without AuthMiddleware (unit tests skip the license-token loop).

    asgi.create_app installs AuthMiddleware globally; calling it would force
    every test to mint a fake license. The route-level tests below mount the
    routes directly onto a bare app. ``instance_id`` (normally set by the ASGI
    lifespan) is stamped here so the durable store can attribute in-flight rows.
    """
    from fastapi import Request

    from src.api.routes import compile as compile_routes
    from src.api.routes import health as health_routes
    from src.api.routes import thumbnail as thumbnail_routes

    application = FastAPI()
    application.state.instance_id = uuid.uuid7().hex

    # AuthMiddleware is skipped here, but its OUTPUT — ``request.state.tenant_id`` — is
    # not optional any more: ``_caller_tenant`` raises ``MissingTenantError`` → HTTP 400
    # when no real tenant uuid was forwarded. So this fixture stamps one, standing in for
    # an authenticated caller; without it every route test would assert 400 and prove
    # nothing about the route. The 400 boundary ITSELF is proven separately, against apps
    # that deliberately forward no/blank/garbage tenants:
    # ``tests/test_compile_hardening.py::test_cross_tenant_poll_is_404`` and
    # ``tests/unit/test_metering.py::test_unattributed_submit_is_rejected_and_emits_no_cost_event``.
    # Do NOT make the src side tolerate an absent tenant to "simplify" this fixture.
    @application.middleware("http")
    async def _stamp_tenant(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.tenant_id = TENANT_A
        return await call_next(request)

    application.include_router(health_routes.router)
    application.include_router(compile_routes.router)
    application.include_router(thumbnail_routes.router)
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def is_delivered_master_measure(cmd: list[str]) -> bool:
    """True for ``renderer._measure_delivered_lufs`` — the read-only measure of a
    DELIVERED master that ``renderer._correct_master_loudness_miss`` runs after
    every master pass to decide whether the master missed ``master_lufs``.

    Fakes of ``renderer._run`` that assert the master's command SEQUENCE (pass 1,
    pass 2, ``cmds[-1]``) return ``""`` for it without recording it: an empty
    stderr is "unmeasurable", so no correction runs and those tests keep
    describing exactly the chain they were written for. The correction itself is
    covered by ``tests/test_master_loudness_miss_correction.py``."""
    from src.services import renderer

    return renderer._DELIVERED_LOUDNESS_FILTER in cmd
