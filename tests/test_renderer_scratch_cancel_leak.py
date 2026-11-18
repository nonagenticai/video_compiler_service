"""Scratch-dir must be torn down on CANCELLATION, not only on Exception.

The two common non-success terminations of a render coroutine — the per-job
work-timeout (``asyncio.wait_for`` cancels the coroutine INSIDE it, then re-raises
``TimeoutError`` to its caller) and the SIGTERM drain (``drain_background_tasks``
cancels in-flight tasks) — both raise ``asyncio.CancelledError`` HERE, inside the
render coroutine whose scratch dir this is about, a ``BaseException`` that ``except Exception`` does
NOT catch. Before the fix the ``job-*`` / ``audio-*`` scratch dir (hundreds of
MB of footage + segments + mp4) leaked on every cancel; the caller's
``_cleanup_job_dir(out_path=None)`` is a no-op, so nothing reclaimed it → disk
fills → pod eviction. The cleanup handlers must catch ``BaseException`` so a
cancel also runs ``shutil.rmtree`` — while STILL re-raising (cancellation
semantics unchanged).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from src.api.schemas import AudioClip, Shot
from src.config import get_settings
from src.services import renderer


def _sole_scratch(work_dir: Path, prefix: str) -> Path:
    """The single freshly-created ``<prefix>-*`` scratch dir under ``work_dir``."""
    matches = [p for p in work_dir.glob(f"{prefix}-*") if p.is_dir()]
    assert len(matches) == 1, f"expected exactly one {prefix}-* scratch, found {matches}"
    return matches[0]


@pytest.mark.asyncio
async def test_compile_shots_removes_scratch_dir_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CancelledError raised mid-render must leave NO ``job-*`` scratch behind,
    and the CancelledError must still propagate."""
    settings = get_settings()
    work_dir = Path(settings.work_dir)
    captured: dict[str, Path] = {}

    def _cancel_midway(shots: list[Shot]) -> list[float]:
        # Runs after work_root.mkdir — capture the live scratch dir, prove it
        # exists, then inject the cancellation the timeout/drain would raise.
        scratch = _sole_scratch(work_dir, "job")
        assert scratch.exists()
        captured["path"] = scratch
        raise asyncio.CancelledError

    monkeypatch.setattr(renderer, "_clip_seek_fractions", _cancel_midway)

    with pytest.raises(asyncio.CancelledError):
        await renderer.compile_shots([Shot(id="s0", seconds=1.0)], settings)

    assert not captured["path"].exists(), f"scratch dir leaked on cancellation: {captured['path']}"


@pytest.mark.asyncio
async def test_concat_audio_removes_scratch_dir_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same guarantee for the audio-stitch path's ``audio-*`` scratch dir."""
    settings = get_settings()
    work_dir = Path(settings.work_dir)
    captured: dict[str, Path] = {}

    async def _cancel_fetch(url: str, dest: Path, http: httpx.AsyncClient) -> Path:
        scratch = _sole_scratch(work_dir, "audio")
        assert scratch.exists()
        captured["path"] = scratch
        raise asyncio.CancelledError

    monkeypatch.setattr(renderer, "_fetch", _cancel_fetch)

    with pytest.raises(asyncio.CancelledError):
        await renderer.concat_audio(
            [AudioClip(id="c0", audio_url="file:///nonexistent.wav")],
            settings,
        )

    assert not captured["path"].exists(), (
        f"audio scratch dir leaked on cancellation: {captured['path']}"
    )
