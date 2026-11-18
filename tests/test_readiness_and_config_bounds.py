"""Readiness gates on BOTH hard deps (MinIO + Postgres); render/expiry config bounded."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import Response
from pydantic import ValidationError

from src.api.routes import health
from src.config import Settings


async def _ready_status(monkeypatch: pytest.MonkeyPatch, *, minio: bool, pg: bool) -> int:
    async def _m() -> bool:
        return minio

    async def _p() -> bool:
        return pg

    monkeypatch.setattr(health, "_minio_reachable", _m)
    monkeypatch.setattr(health, "_postgres_reachable", _p)
    resp = Response()
    await health.health_ready(resp)
    return resp.status_code


async def test_ready_gates_on_postgres_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres gates readiness. MinIO is reported but MUST NOT gate it.

    This test used to assert 503 whenever MinIO was unreachable. That contract
    was wrong and is deliberately reversed here — do not "fix" it back.

    Nothing in the request path touches MinIO: ``submit_compile`` runs
    ``_guard_local_file_inputs`` (pure string checks), then ``_accept`` (a
    Postgres row), then hands off to a background task; the poll endpoint is
    DB-only. MinIO is first contacted by the renderer when it uploads a result.
    Gating readiness on it therefore made MinIO a *serving* dependency that the
    serving path does not actually have — and because both replicas probe the
    same MinIO, a 3s blip emptied the Service endpoint list for the whole
    deployment at once. A real MinIO outage still surfaces, per job, as a failed
    render; that is the correct blast radius.

    Postgres is different: ``_accept`` cannot write its job row without it, so
    an unreachable DB genuinely means the pod cannot accept work.
    """
    assert await _ready_status(monkeypatch, minio=True, pg=True) == 200
    assert await _ready_status(monkeypatch, minio=False, pg=True) == 200
    # Postgres is a hard dependency — an unreachable DB must 503, not falsely-ready.
    assert await _ready_status(monkeypatch, minio=True, pg=False) == 503
    assert await _ready_status(monkeypatch, minio=False, pg=False) == 503


def test_render_fps_and_dims_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # fps=0 (env-set) would divide-by-zero in the cadence math — rejected at load.
    monkeypatch.setenv("RENDER_FPS", "0")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.delenv("RENDER_FPS", raising=False)
    monkeypatch.setenv("RENDER_WIDTH", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_reclaim_lifetime_cap_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 (= 'lifetime bound off') is rejected at load — it is no longer a safe value.

    A reclaim of a drain-interrupted row is charged ZERO poison attempts, so the attempt
    cap cannot bound a drain->rerun loop; this wall clock is the ONLY backstop left. At 0
    a rollout-flapping PAID render is reclaimed and re-billed forever with attempts stuck
    at 0. Narrowed ge=0 -> ge=1; no GitOps overlay sets this key, so nothing regresses.
    """
    monkeypatch.setenv("RECLAIM_MAX_LIFETIME_SECONDS", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_minio_expiry_capped_at_sigv4_max(monkeypatch: pytest.MonkeyPatch) -> None:
    # > 7 days yields URLs MinIO 403s at use — rejected at load.
    monkeypatch.setenv("MINIO_URL_EXPIRY_SECONDS", "999999999")
    with pytest.raises(ValidationError):
        Settings()


async def test_minio_timeout_logs_a_non_empty_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The readiness warning must identify WHY, not log an empty string.

    Live symptom this locks down: both video-compiler replicas emitted
    ``{"error": "", "event": "readiness.minio_unreachable"}`` several times a day.
    ``StorageService.check_health`` already swallows (OSError, S3Error, MinioException)
    and returns False, so the only exception reaching the handler in steady state is the
    ``asyncio.TimeoutError`` raised by the 3s ``wait_for`` — and ``str(TimeoutError())``
    is ``""``. The flap was therefore undiagnosable: a 3-second timeout and a config
    error rendered identically.

    No test covered this branch at all — the sibling above monkeypatches
    ``_minio_reachable`` wholesale, so the except/logging path never executed.
    (structlog writes to stdout, so this reads capsys, not caplog.)
    """

    async def _hang(*args: object, **kwargs: object) -> object:
        await asyncio.sleep(5)
        return True

    monkeypatch.setattr(health, "get_storage", _hang)
    monkeypatch.setattr(health, "_READINESS_DEP_TIMEOUT_S", 0.01)

    assert await health._minio_reachable() is False

    out = capsys.readouterr().out
    assert "readiness.minio_unreachable" in out, out
    # Renderer-agnostic: structlog emits key=value on a console and JSON in CI, so
    # assert on substrings present in BOTH ("error_type": "TimeoutError" vs
    # error_type=TimeoutError). Matching one renderer's punctuation made this pass
    # locally and fail in CI.
    assert "error_type" in out and "TimeoutError" in out, (
        f"readiness warning must name the failure type; got: {out!r}"
    )
    # The repr, not the str. `str(TimeoutError())` is "" — the empty field that
    # shipped. Only repr() renders the trailing "()".
    assert "TimeoutError()" in out, (
        f"error must be repr(), not the empty str() that shipped; got: {out!r}"
    )
