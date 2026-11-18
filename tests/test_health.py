"""Health/readiness probe tests.

Liveness is static; readiness gates on POSTGRES ONLY — the durable job store is the
only dependency the request path (submit/poll) actually needs. MinIO is probed and
reported in the readiness body but never gates: it is touched only later, by the
background renderer at upload time, and gating on it de-endpointed BOTH replicas at
once on a shared 3s blip.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


class _FakeStorage:
    """Stand-in for the cached facade StorageService the readiness probe uses.

    ``reachable=False`` mirrors the facade's ``check_health`` returning False on a
    transport error (it catches OSError/S3Error internally rather than raising)."""

    def __init__(self, *, reachable: bool) -> None:
        self._reachable = reachable

    async def check_health(self) -> bool:
        return self._reachable


def _fake_get_storage(*, reachable: bool):
    async def _get_storage(_settings, *, bucket):  # type: ignore[no-untyped-def]
        return _FakeStorage(reachable=reachable)

    return _get_storage


@pytest.mark.parametrize("path", ["/health", "/health/live", "/healthz"])
async def test_liveness_is_static(client: AsyncClient, path: str) -> None:
    r = await client.get(path)
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "live")


@pytest.mark.parametrize("path", ["/health", "/health/live", "/healthz"])
async def test_liveness_body_shape_is_pinned(client: AsyncClient, path: str) -> None:
    """The liveness body's EXACT key set and 200 status, pinned.

    These handlers are annotated ``dict[str, object]`` (they carry an int count and a
    nullable timestamp alongside the string status), which publishes
    ``additionalProperties: true`` in the OpenAPI schema — i.e. the schema itself no longer
    constrains the body at all. That is the correct annotation for a mixed-type body, but
    it means the response shape is only as pinned as the tests make it. This is that pin:
    a field silently dropped, renamed, or added now fails here instead of silently
    breaking whatever reads the probe.
    """
    r = await client.get(path)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "status",
        "reaper_state",
        "reaper_consecutive_failures",
        "reaper_last_failure",
        "reaper_last_failure_at",
    }
    assert body["status"] in ("ok", "live")
    # No reaper is installed on the bare test app, and reaper_enabled defaults on: that is
    # reported as `absent` with a NULL count, never as a clean zero (see _reaper_status).
    assert body["reaper_state"] == "absent"
    assert body["reaper_consecutive_failures"] is None
    assert body["reaper_last_failure"] is None
    assert body["reaper_last_failure_at"] is None


@pytest.mark.parametrize("path", ["/health", "/health/live", "/healthz"])
async def test_absent_reaper_is_degraded_not_fatal(client: AsyncClient, path: str) -> None:
    """An absent reaper is REPORTED, never fatal: liveness must still be 200.

    Failing liveness on a background-housekeeping fault would restart-loop a pod that is
    serving submits/polls perfectly — so the signal is the body, not the status code.
    """
    r = await client.get(path)
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "live")


async def test_disabled_reaper_is_not_reported_as_a_fault(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reaper_enabled=false`` is intentional, so it reads `disabled`, not `absent`."""
    from src.api.routes import health

    monkeypatch.setattr(health, "get_settings", lambda: SimpleNamespace(reaper_enabled=False))

    r = await client.get("/health/live")
    assert r.status_code == 200
    body = r.json()
    assert body["reaper_state"] == "disabled"
    assert body["reaper_consecutive_failures"] is None


class _FakeReaper:
    """Stand-in exposing only what the probe reads off ReaperService."""

    def __init__(self, *, running: bool, consecutive_failures: int = 0) -> None:
        self.running = running
        self.consecutive_failures = consecutive_failures
        self.last_failure: str | None = None
        self.last_failure_at = None


async def test_running_reaper_reports_its_sweep_failure_count(
    client: AsyncClient, app: FastAPI
) -> None:
    app.state.reaper = _FakeReaper(running=True, consecutive_failures=3)
    body = (await client.get("/health/live")).json()
    assert body["reaper_state"] == "running"
    assert body["reaper_consecutive_failures"] == 3


async def test_installed_but_stopped_reaper_is_not_healthy(
    client: AsyncClient, app: FastAPI
) -> None:
    """A reaper whose loop task is gone reclaims nothing — its 0 count must not be trusted."""
    app.state.reaper = _FakeReaper(running=False)
    body = (await client.get("/health/live")).json()
    assert body["reaper_state"] == "stopped"
    assert body["reaper_consecutive_failures"] is None


@pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
async def test_readiness_ok_when_minio_reachable(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from src.api.routes import health

    monkeypatch.setattr(health, "get_storage", _fake_get_storage(reachable=True))
    r = await client.get(path)
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


@pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
async def test_readiness_stays_ready_when_minio_down_but_reports_it(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """MinIO down must be REPORTED, not fatal — 200, with `minio: unreachable` in the body.

    This test previously asserted 503, and the contract was CHANGED deliberately: do not
    "fix" it back. Nothing in the request path touches MinIO — `submit_compile` runs
    `_guard_local_file_inputs` (string checks) → `_accept` (Postgres) → `create_task`, and
    the poll endpoint is DB-only; MinIO is first contacted by the background renderer at
    upload time. Both replicas share that one MinIO, so gating readiness on it turned a
    3s latency blip into an empty endpoint list for the whole Service. A real outage
    surfaces where it belongs: the individual render fails and is reported per job.
    """
    from src.api.routes import health

    monkeypatch.setattr(health, "get_storage", _fake_get_storage(reachable=False))
    r = await client.get(path)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["minio"] == "unreachable"  # postgres is reachable in tests (real DB)
    assert body["postgres"] == "reachable"


def _unreachable_db_session():
    @contextlib.asynccontextmanager
    async def _get_db_session() -> AsyncGenerator[None]:
        raise OSError("connection refused")
        yield None  # pragma: no cover — unreachable, keeps this an async generator

    return _get_db_session


@pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
async def test_readiness_503_when_postgres_down(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Postgres DOES still gate: `_accept` writes the job row, so no DB = cannot accept."""
    from src.api.routes import health

    monkeypatch.setattr(health, "get_storage", _fake_get_storage(reachable=True))
    monkeypatch.setattr(health, "get_db_session", _unreachable_db_session())
    r = await client.get(path)
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unavailable"
    assert body["postgres"] == "unreachable"
    assert body["minio"] == "reachable"


@pytest.mark.parametrize("reachable, expected", [(True, "reachable"), (False, "unreachable")])
async def test_readiness_body_reports_minio_state_both_ways(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, reachable: bool, expected: str
) -> None:
    """MinIO is non-fatal but must stay VISIBLE: both states surface in the 200 body.

    Losing the field is how a non-fatal dependency becomes an invisible one — this is the
    pin that keeps `kubectl get --raw /readyz` able to show the truth either way.
    """
    from src.api.routes import health

    monkeypatch.setattr(health, "get_storage", _fake_get_storage(reachable=reachable))
    r = await client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["minio"] == expected
