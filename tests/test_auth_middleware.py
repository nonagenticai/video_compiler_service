"""Guardrail tests for the auth middleware integration.

video_compiler_service delegates auth to ``w2v_common.auth.middleware.AuthMiddleware``.
This test exercises that the middleware is **wired and registered** on the
asgi app (not merely importable), and that it behaves correctly for the
standard public / protected / invalid-token matrix.

The intent is to catch the "false-merge" pattern: a PR that says it added
auth but never called ``app.add_middleware(AuthMiddleware)``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.middleware import AuthMiddleware

from tests.conftest import TENANT_A


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    cache = w2v_auth._get_license_cache()
    cache.clear()


def _make_minimal_app() -> FastAPI:
    """Tiny app for behaviour tests — independent of asgi lifespan side-effects."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/compile/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "license_present": getattr(request.state, "license", None) is not None,
            }
        )

    return app


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Wiring check — the regression-against-false-merge assertion
# ---------------------------------------------------------------------------


def test_auth_middleware_is_actually_registered_on_asgi_app() -> None:
    """The asgi app must register AuthMiddleware on its middleware stack.

    We verify both:
    (1) the import path resolves to the shared library, and
    (2) the app built by the ``create_app`` factory actually has
        AuthMiddleware in ``user_middleware`` (catches a false-merge that
        claims auth but never wires it — the factory registers it from the
        service ``Settings``, so a bare swap that disabled auth would fail here).
    """
    # (1) AuthMiddleware comes from the shared lib (not a local stub).
    from w2v_common.auth.middleware import AuthMiddleware as SharedAuthMiddleware

    assert AuthMiddleware is SharedAuthMiddleware

    # (2) The asgi app registers AuthMiddleware on its middleware stack.
    from src.asgi import app

    assert any(m.cls is AuthMiddleware for m in app.user_middleware), (
        "asgi app must register AuthMiddleware (via the create_app factory)"
    )


# ---------------------------------------------------------------------------
# Public bypass
# ---------------------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_metrics_is_reachable_without_trailing_slash() -> None:
    """``GET /metrics`` (no trailing slash — the ``prometheus.io/path`` scrape
    convention) must return 200 with the Prometheus text body, not a redirect
    or a 404 — AND must negotiate content type off the ``Accept`` header the
    same way ``prometheus_client``'s own ASGI app does, not a fixed format.

    Exercises the REAL ``src.asgi:app`` (not ``_make_minimal_app``): the 404 bug
    is specific to how that app is built — ``redirect_slashes=False`` disables
    Starlette's usual trailing-slash redirect, and ``app.mount("/metrics",
    make_asgi_app())`` only matches paths with a literal "/" after the mount
    prefix (Starlette's ``Mount`` compiles its regex as
    ``<path>/{path:path}``), so the bare path "/metrics" never matched any
    route and fell through to a 404 — permanently failing an
    annotation-driven scrape target.

    A fixed-format fix (a route that always calls ``generate_latest()``) closes
    the 404 but silently downgrades every REAL Prometheus scrape — which sends
    an OpenMetrics ``Accept`` by default — from OpenMetrics (with ``# EOF``
    truncation detection + exemplar support) to the legacy 0.0.4 text format,
    on exactly the path that gets scraped, while ``/metrics/`` (the mount) kept
    negotiating correctly. Assert both Accept variants explicitly so that
    regression can't sneak back in.
    """
    # Registers the reaper's Gauge/Counter on the default REGISTRY at import
    # (normally done by ``_startup``'s lazy `from src.services.reaper_service import
    # ReaperService`, which this test doesn't trigger since it drives the ASGI app
    # without running the lifespan) so the assertion below proves this route reads
    # the SAME default registry the reaper's own metrics live on, not an empty one.
    import src.services.reaper_service as _reaper_service  # noqa: F401
    from src.asgi import app as real_app

    transport = httpx.ASGITransport(app=real_app)

    # Default Accept (no header, or a plain "*/*") -> legacy text/plain 0.0.4.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics")
    assert resp.status_code == 200, resp.text
    assert "w2v_reaper_consecutive_sweep_failures" in resp.text
    assert resp.headers["content-type"].startswith("text/plain")

    # Real Prometheus' default scrape Accept -> OpenMetrics, WITH negotiation,
    # not silently coerced to text/plain. `# EOF` is OpenMetrics' truncation
    # marker; its presence is the proof the negotiated encoder actually ran.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp_om = await ac.get(
            "/metrics",
            headers={"Accept": "application/openmetrics-text; version=1.0.0; charset=utf-8"},
        )
    assert resp_om.status_code == 200, resp_om.text
    assert resp_om.headers["content-type"].startswith("application/openmetrics-text")
    assert resp_om.text.rstrip().endswith("# EOF")

    # HEAD must succeed (same headers as GET, empty body) — a bare APIRoute
    # registered GET-only 405s HEAD, which a real scraper's readiness probe of
    # the metrics endpoint could hit.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp_head = await ac.head("/metrics")
    assert resp_head.status_code == 200, resp_head.text
    assert resp_head.content == b""

    # "/metrics/" (the mount's own path) must keep negotiating identically.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp_slash = await ac.get("/metrics/")
        resp_slash_om = await ac.get(
            "/metrics/",
            headers={"Accept": "application/openmetrics-text; version=1.0.0; charset=utf-8"},
        )
    assert resp_slash.status_code == 200, resp_slash.text
    assert resp_slash_om.headers["content-type"].startswith("application/openmetrics-text")
    assert resp_slash_om.text.rstrip().endswith("# EOF")


# ---------------------------------------------------------------------------
# Protected paths
# ---------------------------------------------------------------------------


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": str(TENANT_A),
        "subject_id": "subj-1",
        "license_id": "lic-1",
        "roles": ["user"],
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/v1/compile/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == str(TENANT_A)
    assert body["license_present"] is True


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False, "reason": "expired"}

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/v1/compile/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "INVALID_TOKEN"
    assert body["error"]["status"] == 401
    assert "Invalid or expired license token" in body["error"]["message"]


async def test_protected_route_no_token_untrusted_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted source IP + no token must 401, regardless of TRUST_INCLUSTER_IDENTITY."""
    monkeypatch.setattr(w2v_auth, "_trust_incluster_identity", lambda: False)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/compile/echo")
    assert resp.status_code == 401
