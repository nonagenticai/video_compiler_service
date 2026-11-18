"""The entitlement gate on the tenant-scoped routers actually refuses an
unlicensed tenant.

Until w2v_common 0.75.0 this service authenticated the licence token for
tenant IDENTITY and stopped there: ``AuthMiddleware`` answers "who are you",
never "may you". A tenant whose licence was ``expired`` or ``revoked`` still
presented a syntactically valid token, so every compile/concat-audio/cancel/
poll/thumbnail route served them — including the billing-bearing render
dispatch. ``LICENSE_GATE`` is mounted on both ``compile.router`` and
``thumbnail.router`` (``src/api/routes/compile.py`` /
``src/api/routes/thumbnail.py``) to close that.

Every test here drives a REAL HTTP request through the REAL routers, because
the interesting question is not whether ``require_license`` works — w2v_common
owns that — but whether THIS service's routes actually carry it. This FastAPI
build includes sub-routers lazily (``_IncludedRouter``), so a parent router's
``dependencies=`` are applied when a request is matched, not baked into the
route objects at import time: static introspection of ``route.dependant``
would report the routes as ungated and would be a *false negative*. A request
is the only honest probe.

The paired 200s (entitled tenant, flag off) are what stop this file from
passing vacuously: if the gate were mounted wrongly (or the fixture were
broken) those cases would not be 200s either.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache, is_license_exempt_path
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.routes import compile as compile_routes
from src.api.routes import health as health_routes
from src.api.routes import thumbnail as thumbnail_routes
from tests.conftest import TENANT_A

SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route on the gated compile router: it
#: needs no DB session and no render — ``cancel_by_key_prefix`` is patched to a
#: no-op, so a non-200 here can only be the gate.
GATED_PATH = "/api/jobs/cancel"

#: The thumbnail router's sole route, gated separately from compile.router —
#: proves BOTH routers carry the gate, not just one of them.
THUMBNAIL_GATED_PATH = "/api/thumbnails"

#: Startup-probe path unique to this service (in addition to the fleet-common
#: /health/live and /health/ready).
STARTUP_PROBE_PATH = "/healthz"
LIVE_PROBE_PATH = "/health/live"
READY_PROBE_PATH = "/health/ready"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=str(TENANT_A), status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=str(TENANT_A),
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=str(TENANT_A),
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _no_real_work(monkeypatch):
    """Make every gated handler's own work free, so only the gate can fail it.

    ``GATED_PATH`` (``/api/jobs/cancel``) opens a real DB session via the
    module-level ``get_db_session`` imported into ``compile.py`` — stub it out
    with a fake async context manager so this file needs no Postgres at all;
    only the gate (which runs BEFORE the handler body) can turn a request into
    a non-200 here.
    """

    class _FakeSessionCtx:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(compile_routes, "get_db_session", lambda: _FakeSessionCtx())

    async def _cancel_noop(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(compile_routes.job_store, "cancel_by_key_prefix", _cancel_noop)

    async def _fake_fetch(*_args: Any, **_kwargs: Any) -> str:
        return "/tmp/fake-video.mp4"

    async def _fake_extract_candidates(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(thumbnail_routes, "_fetch", _fake_fetch)
    monkeypatch.setattr(thumbnail_routes, "extract_candidates", _fake_extract_candidates)


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = str(TENANT_A)
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health_routes.router)
    application.include_router(compile_routes.router)
    application.include_router(thumbnail_routes.router)
    application.state.instance_id = "test-license-gate"
    return application


async def _post(app: FastAPI, path: str, json: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=json)


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


CANCEL_BODY = {"idempotency_key_prefix": "x"}
THUMBNAIL_BODY = {"video_url": "https://example.com/video.mp4", "count": 1}


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _post(app, GATED_PATH, CANCEL_BODY)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.asyncio
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _post(app, GATED_PATH, CANCEL_BODY)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair) — both routers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant_on_compile_router(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _post(app, GATED_PATH, CANCEL_BODY)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant_on_thumbnail_router(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _post(app, THUMBNAIL_GATED_PATH, THUMBNAIL_BODY)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(app, GATED_PATH, CANCEL_BODY)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(app, GATED_PATH, CANCEL_BODY)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated — including this service's unusual THIRD probe
# (startup probe on /healthz, alongside liveness /health/live and readiness
# /health/ready).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [LIVE_PROBE_PATH, READY_PROBE_PATH, STARTUP_PROBE_PATH],
    ids=["live", "ready", "startup"],
)
async def test_probe_is_served_for_a_revoked_tenant(app, monkeypatch, path):
    """A gated probe path CrashLoops the pod. All three probes must stay 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, path)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
    assert not is_license_exempt_path(THUMBNAIL_GATED_PATH)
