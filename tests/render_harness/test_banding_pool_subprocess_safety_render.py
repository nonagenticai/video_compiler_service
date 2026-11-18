"""Regression (2026-10-21, #255): ffmpeg spawned while the banding settle's
``InterpreterPoolExecutor`` lives must not die with signal 11.

Root cause, reproduced in the production image: uvloop's subprocess spawn forks
and runs CPython's after-fork-child hook in the child, which tears down the
other interpreters and segfaults before ``exec``. Every master's streamed
banding pass (and narration-presence decodes racing a pool teardown) died
``ffmpeg killed by signal 11`` with empty stderr. The stdlib loop spawns via
``_posixsubprocess`` and is safe; ``asgi.main`` pins it, and the executor
refuses to start under any other loop.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from src import asgi
from src.services import mastering, measure, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _clean_film(tmp_path: Path) -> Path:
    out = tmp_path / "clean.mp4"
    subprocess.run(  # noqa: S603 — fixed argv, test-only
        [  # noqa: S607
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "nullsrc=s=192x108:r=30:d=4",
            "-f", "lavfi", "-i", "sine=d=4",
            "-vf", "geq=lum='16+220*X/W':cb=128:cr=128,noise=alls=4:allf=t",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


async def _settle_then_decode(path: Path, tmp_path: Path) -> tuple[dict[float, float], int]:
    timed = await measure.measure_shot_frames_timed(path)
    assert timed
    factors = await mastering._settle_banding_spatial_factors(path, timed, None, workers=4)
    # The narration-presence shape: a decode spawned right after the settle,
    # while the pool may still be tearing down (``shutdown(wait=False)``).
    await renderer._run(
        [
            "ffmpeg", "-y", "-hide_banner", "-nostdin", "-i", str(path),
            "-vn", "-ac", "1", "-ar", "8000", "-sample_fmt", "s16", "-f", "wav",
            str(tmp_path / "master.wav"),
        ]
    )  # fmt: skip
    candidates = sum(
        1
        for pts, m in timed
        if pts is not None and mastering.banding_score(m) >= mastering.BANDING_SPATIAL_PREFILTER
    )
    return factors, candidates


async def test_stdlib_loop_ffmpeg_survives_live_pool(tmp_path: Path) -> None:
    path = _clean_film(tmp_path)
    factors, candidates = await _settle_then_decode(path, tmp_path)
    assert candidates > 0
    assert len(factors) >= candidates  # the streamed pass ran: nothing segfaulted


def test_uvloop_settle_falls_back_to_serial_and_ffmpeg_survives(tmp_path: Path) -> None:
    uvloop = pytest.importorskip("uvloop")
    path = _clean_film(tmp_path)
    assert uvloop.run(_loop_is_unsafe())
    factors, candidates = uvloop.run(_settle_then_decode(path, tmp_path))
    assert candidates > 0
    assert len(factors) >= candidates


async def _loop_is_unsafe() -> bool:
    return not mastering._loop_spawns_safely_with_subinterpreters() and (
        mastering._banding_executor(4) is None
    )


def test_asgi_main_pins_the_stdlib_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    asgi.main()
    assert seen.get("loop") == "asyncio"


def test_stdlib_loop_is_safe() -> None:
    assert asyncio.run(_safe()) is True


async def _safe() -> bool:
    return mastering._loop_spawns_safely_with_subinterpreters()
