"""``_try_fetch`` must retry a TRANSIENT shot-asset fetch failure (timeout,
connection error, 5xx) with a small bounded backoff before degrading the
shot — but fail fast, on the FIRST attempt, when the failure is definitive
(401/403/404/410, an expired/forged presigned MinIO signature).

Why this matters most for NARRATION: ``_resolve_shot_assets`` turns a bare
``None`` from ``_fetch_asset`` into a compile-failing
``UnfetchableNarrationError`` (see ``test_narration_unfetchable_fails_loud.py``).
Before this fix, a single passing network blip on the narration fetch was
indistinguishable from a genuinely dead URL and failed an otherwise-good, paid
compile outright. video_compiler#232 finding 2.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.services import renderer

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries must not actually block the test suite — the backoff MATH is not
    what these tests are about (its magnitude is fixed/small already)."""

    async def _instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(renderer.asyncio, "sleep", _instant_sleep)


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://minio/x")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


class _FlakyThenOK:
    """Raises ``exc`` ``fail_times`` times, then returns ``dest``."""

    def __init__(self, exc: BaseException, fail_times: int) -> None:
        self.exc = exc
        self.fail_times = fail_times
        self.calls = 0

    async def __call__(self, url: str, dest: Path, http, *, local_root: Path | None = None) -> Path:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return dest


class _AlwaysFails:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    async def __call__(self, url: str, dest: Path, http, *, local_root: Path | None = None) -> Path:
        self.calls += 1
        raise self.exc


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.TimeoutException("timed out"),
        _status_error(500),
        _status_error(503),
    ],
)
async def test_transient_failure_then_success_degrades_nothing(
    exc: BaseException, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FlakyThenOK(exc, fail_times=1)
    monkeypatch.setattr(renderer, "_fetch", fake)
    async with httpx.AsyncClient() as http:
        result = await renderer._try_fetch("http://minio/x", tmp_path / "asset", http)
    assert result == tmp_path / "asset"
    assert fake.calls == 2, "must have retried exactly once before succeeding"


@pytest.mark.parametrize("status", [401, 403, 404, 410])
async def test_definitive_client_error_fails_fast_no_retry(
    status: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _AlwaysFails(_status_error(status))
    monkeypatch.setattr(renderer, "_fetch", fake)
    async with httpx.AsyncClient() as http:
        result = await renderer._try_fetch("http://minio/x", tmp_path / "asset", http)
    assert result is None
    assert fake.calls == 1, f"a {status} is definitive — must not be retried"


async def test_transient_failure_exhausts_retries_then_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _AlwaysFails(httpx.ConnectError("refused"))
    monkeypatch.setattr(renderer, "_fetch", fake)
    async with httpx.AsyncClient() as http:
        result = await renderer._try_fetch("http://minio/x", tmp_path / "asset", http)
    assert result is None
    assert fake.calls == renderer._TRANSIENT_FETCH_RETRIES + 1


async def test_non_http_bad_asset_error_fails_fast_no_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed-URL ``ValueError`` (safe_get's pre-I/O validation) is not
    network flakiness — retrying it feeds the exact same bad URL to the exact
    same check and cannot ever succeed."""
    fake = _AlwaysFails(ValueError("Cannot extract hostname"))
    monkeypatch.setattr(renderer, "_fetch", fake)
    async with httpx.AsyncClient() as http:
        result = await renderer._try_fetch("http://minio/x", tmp_path / "asset", http)
    assert result is None
    assert fake.calls == 1


async def test_resolve_shot_assets_narration_survives_a_transient_blip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: a narration fetch that times out once and then succeeds must
    NOT raise ``UnfetchableNarrationError`` — the whole point of the retry."""
    real_audio = tmp_path / "line.wav"
    real_audio.write_bytes(b"not-really-a-wav")

    fake = _FlakyThenOK(httpx.TimeoutException("timed out"), fail_times=1)

    async def _fake_fetch(url: str, dest: Path, http, *, local_root: Path | None = None) -> Path:
        await fake(url, dest, http, local_root=local_root)
        return real_audio

    monkeypatch.setattr(renderer, "_fetch", _fake_fetch)

    async def _fake_duration(path: Path) -> float:
        return 3.0

    monkeypatch.setattr(renderer, "_ffprobe_duration", _fake_duration)

    shot = Shot(id="s1", seconds=3.0, audio_url="http://minio/line.wav")
    async with httpx.AsyncClient() as http:
        audio_path, *_rest = await renderer._resolve_shot_assets(shot, tmp_path, http)
    assert audio_path == real_audio


async def test_resolve_shot_assets_narration_still_fails_loud_on_definitive_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-regression for Defect A (video_compiler#232): a genuinely dead
    narration URL (403 — expired presigned signature) still raises
    ``UnfetchableNarrationError`` promptly, not after burning the retry budget
    pointlessly."""
    fake = _AlwaysFails(_status_error(403))
    monkeypatch.setattr(renderer, "_fetch", fake)
    shot = Shot(id="s2", seconds=3.0, audio_url="http://minio/line.wav")
    async with httpx.AsyncClient() as http:
        with pytest.raises(renderer.UnfetchableNarrationError, match="s2"):
            await renderer._resolve_shot_assets(shot, tmp_path, http)
    assert fake.calls == 1
