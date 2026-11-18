"""The banding spatial pass never builds an ffmpeg ``select`` over the term bound.

Live 2026-10-21 (generic 2560x1440 30s master, compiler 6d09054): pass 2 of the
settle wanted 787 scattered frames, the plan merged them to 101 ``between()``
terms, and ffmpeg 7.1 refused the graph — ``exit=244 ... Error initializing
filters ... Cannot allocate memory`` — so the reported banding score stayed the
RAW upper bound (0.3552, a false advisory). Measured in the prod image: 100
terms init, 101 fail (expression-tree depth, not memory). The fix splits the
plan into ffmpeg invocations of at most ``_SPATIAL_MAX_TERMS_PER_PASS`` terms.

This drives ~1000 scattered wanted frames through the REAL streamed pass and
checks (a) no single invocation exceeds the bound, (b) it took several, and (c)
every factor equals the serial reference: one select-free decode of every frame
through the same scale/gray filter, measured by the same detector.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from src.services import banding_spatial, measure, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

W, H, N = 320, 180, 1400
# ffmpeg 7.1.5's own limit (measured 2026-10-21 in the prod image).
FFMPEG_SELECT_TERM_LIMIT = 100


def _row(i: int) -> bytes:
    """Frame i: a posterized ramp on every 3rd frame, a smooth ramp otherwise —
    so a one-frame attribution shift changes the factor."""
    if i % 3 == 0:
        bands = 8
        return bytes(round(min(bands - 1, x * bands // W) * 255 / (bands - 1)) for x in range(W))
    return bytes(min(255, int(80 + 96 * x / W + 0.5)) for x in range(W))


def _encode(path: Path) -> None:
    subprocess.run(  # noqa: S603 — fixed argv, test-only
        [  # noqa: S607
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "30",
            "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(path),
        ],
        input=b"".join(_row(i) * H for i in range(N)),
        check=True,
    )  # fmt: skip


def _reference(path: Path) -> list[float]:
    """Serial reference: EVERY frame, no select, same scale/gray filter."""
    out = subprocess.run(  # noqa: S603 — fixed argv, test-only
        [  # noqa: S607
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an",
            "-vf", f"scale=w='min(iw,{measure._SPATIAL_FRAME_MAX_WIDTH})':h=-2:flags=neighbor,format=gray",
            "-fps_mode:v", "passthrough", "-c:v", "pgm", "-f", "image2pipe", "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout  # fmt: skip
    factors: list[float] = []
    while out:
        parsed = measure.parse_pgm(out)
        assert parsed is not None
        pixels, w, h = parsed
        factors.append(banding_spatial.spatial_banding_factor(pixels, w, h))
        out = out[_pgm_len(out, w, h) :]
    return factors


def _pgm_len(raw: bytes, w: int, h: int) -> int:
    m = measure._PGM_HEADER.match(raw)
    assert m is not None
    return m.end() + w * h


async def test_scattered_candidates_stay_under_the_select_term_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scattered.mp4"
    _encode(path)
    timed = await measure.measure_shot_frames_timed(path)
    pts = [p for p, _ in timed if p is not None]
    assert len(pts) == N
    # ~1000 scattered frames: gaps every 5th and 13th frame -> ~300 runs, far
    # past both the per-pass bound and ffmpeg's own limit before merging.
    wanted = {p for i, p in enumerate(pts) if i % 5 != 0 and i % 13 != 0}
    assert 950 <= len(wanted) <= 1100

    terms: list[int] = []
    real = renderer._stream_raw_stdout

    async def spy(
        cmd: list[str],
        consume: Callable[[asyncio.StreamReader], Awaitable[None]],
        *,
        budget_seconds: float,
    ) -> bool:
        vf = cmd[cmd.index("-vf") + 1]
        terms.append(vf.split(",scale=")[0].count("between("))
        return await real(cmd, consume, budget_seconds=budget_seconds)

    monkeypatch.setattr(renderer, "_stream_raw_stdout", spy)
    factors = await measure.measure_spatial_factors_streamed(path, pts, wanted, budget_seconds=120)

    assert len(terms) > 1, terms  # split, not one oversized graph
    assert max(terms) <= measure._SPATIAL_MAX_TERMS_PER_PASS < FFMPEG_SELECT_TERM_LIMIT, terms
    assert wanted <= set(factors)  # every wanted frame measured (count checks held)
    ref = await asyncio.to_thread(_reference, path)
    assert len(ref) == N
    by_pts = dict(zip(pts, ref, strict=True))
    mismatched = {p: (f, by_pts[p]) for p, f in factors.items() if f != by_pts[p]}
    assert not mismatched, list(mismatched.items())[:5]
    # The content alternates, so the comparison above would catch a shift.
    assert {round(by_pts[p], 1) >= 0.5 for p in wanted} == {True, False}
