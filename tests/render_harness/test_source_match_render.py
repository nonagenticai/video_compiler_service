"""Real-ffmpeg render-harness test for T9 — per-shot source-matching (±6/255 AC).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_drift_pad_render.py``. This drives ACTUAL encodes: three FLAT clips at
deliberately different luma stand in for mixed sources (browser recording / AI
image / slide) whose exposure drifts shot-to-shot. ``_apply_source_match``
measures each, takes the median as the "one camera" target, and gamma-nudges the
outliers toward it — the objective gate is every shot's measured YAVG landing
within ±6/255 of the median. The off / match_sources=False paths must leave the
segment bytes untouched.

MinIO/storage is NOT exercised: the pass operates on local segment paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import FinishProfile, GradeSpec
from src.config import get_settings
from src.services import measure, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}


async def _flat_clip(path: Path, *, luma_hex: str, duration: float = 2.0) -> Path:
    """A flat single-colour clip at ``0x<luma_hex>``, held for ``duration``.

    A uniform field so its measured YAVG is fully determined by the fill colour —
    the cleanest possible exposure-drift fixture for the source-match pass."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x{luma_hex}:s={_W}x{_H}:d={duration}:r={_FPS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


def _finish(*, match_sources: bool) -> FinishProfile:
    """A finish whose grade opts into (or out of) source-matching."""
    return FinishProfile(grade=GradeSpec(enabled=True, match_sources=match_sources))


async def _measure_yavg(path: Path) -> float:
    stats = await measure.measure_shot_stats(path)
    return stats["YAVG"]


async def _measure_satavg(path: Path) -> float:
    stats = await measure.measure_shot_stats(path)
    return stats["SATAVG"]


async def test_source_match_tightens_luma_spread(tmp_path: Path) -> None:
    """Three flat shots at different luma (dark / mid / bright) are pulled toward
    their median: after the pass every shot's YAVG sits within ±tolerance of the
    ORIGINAL (pre-correction) median — the AC's fixed target, NOT a median
    recomputed from the corrected outputs (which would move under any overshoot
    and prove a weaker property). Enabled via settings + finish.grade.match_sources."""
    p1 = await _flat_clip(tmp_path / "seg-0000.mp4", luma_hex="606060")
    p2 = await _flat_clip(tmp_path / "seg-0001.mp4", luma_hex="808080")
    p3 = await _flat_clip(tmp_path / "seg-0002.mp4", luma_hex="a0a0a0")
    segs = [p1, p2, p3]

    before = [await _measure_yavg(p) for p in segs]
    # Sanity: the fixture really is spread far wider than the ±6 AC to begin with.
    assert max(before) - min(before) > 12
    # The AC target is the median the algorithm actually corrects toward, captured
    # BEFORE any segment is re-encoded. The median shot (p2) is this target.
    target = renderer._median(before)

    settings = get_settings().model_copy(update={**_SMALL, "render_source_match": "on"})
    tol = settings.source_match_tolerance
    mid_before = p2.read_bytes()
    await renderer._apply_source_match(segs, settings, _finish(match_sources=True))

    after = [await _measure_yavg(p) for p in segs]
    # AC (07-color-finish.md:139): every shot within ±tolerance of the FIXED target.
    for yavg in after:
        assert abs(yavg - target) <= tol, f"YAVG {yavg} not within {tol} of target {target}"
    # The in-tolerance median shot is left byte-identical (returns "" → not re-encoded) — D5.
    assert p2.read_bytes() == mid_before


async def test_source_match_extreme_spread_reduces_but_clamps(tmp_path: Path) -> None:
    """An EXTREME dark-vs-bright spread cannot be fully equalised without blowing a
    shot out, so the bounded gamma converges only as far as the clamp allows. The
    contract for that case is monotonic improvement — the post-pass spread is
    strictly smaller than the original — NOT necessarily within ±6. This covers the
    protective clamp path the milder fixture above deliberately stays inside."""
    p1 = await _flat_clip(tmp_path / "seg-0000.mp4", luma_hex="303030")
    p2 = await _flat_clip(tmp_path / "seg-0001.mp4", luma_hex="808080")
    p3 = await _flat_clip(tmp_path / "seg-0002.mp4", luma_hex="c0c0c0")
    segs = [p1, p2, p3]

    before = [await _measure_yavg(p) for p in segs]
    spread_before = max(before) - min(before)

    settings = get_settings().model_copy(update={**_SMALL, "render_source_match": "on"})
    await renderer._apply_source_match(segs, settings, _finish(match_sources=True))

    after = [await _measure_yavg(p) for p in segs]
    spread_after = max(after) - min(after)
    # The clamp protects the extremes, but the pass must still REDUCE the spread.
    assert spread_after < spread_before, f"spread {spread_after} not < {spread_before}"


async def test_source_match_tightens_saturation_spread(tmp_path: Path) -> None:
    """Three flat clips with CLOSE luma but different CHROMA (low / mid / high
    saturation) are pulled toward their median saturation. The colours are chosen so
    every YAVG sits within ±tolerance of the median luma (the luma gamma term is a
    near-no-op) while their SATAVG differs widely — isolating the SAT term. Because
    the gamma and saturation terms interact through eq, the contract asserted here is
    MONOTONIC improvement: the post-pass SATAVG spread is strictly smaller than the
    original — NOT an exact ±tolerance landing. Mirrors the extreme-luma-spread test."""
    # Y≈126-127 for all three (within the ±6 luma tolerance), SATAVG low→high.
    p1 = await _flat_clip(tmp_path / "seg-0000.mp4", luma_hex="7C8086")
    p2 = await _flat_clip(tmp_path / "seg-0001.mp4", luma_hex="748094")
    p3 = await _flat_clip(tmp_path / "seg-0002.mp4", luma_hex="6880AC")
    segs = [p1, p2, p3]

    yavg_before = [await _measure_yavg(p) for p in segs]
    sat_before = [await _measure_satavg(p) for p in segs]
    # Sanity: luma is tight (so the SAT term is what moves) but chroma is spread wide.
    assert max(yavg_before) - min(yavg_before) < 12
    assert max(sat_before) - min(sat_before) > 12
    spread_before = max(sat_before) - min(sat_before)

    settings = get_settings().model_copy(update={**_SMALL, "render_source_match": "on"})
    await renderer._apply_source_match(segs, settings, _finish(match_sources=True))

    sat_after = [await _measure_satavg(p) for p in segs]
    spread_after = max(sat_after) - min(sat_after)
    # The saturation term must REDUCE the chroma spread across the mixed sources.
    assert spread_after < spread_before, f"sat spread {spread_after} not < {spread_before}"


async def test_source_match_off_is_byte_identical(tmp_path: Path) -> None:
    """``render_source_match='off'`` skips the pass entirely — every segment's
    bytes are unchanged."""
    p1 = await _flat_clip(tmp_path / "seg-0000.mp4", luma_hex="606060")
    p2 = await _flat_clip(tmp_path / "seg-0001.mp4", luma_hex="808080")
    p3 = await _flat_clip(tmp_path / "seg-0002.mp4", luma_hex="a0a0a0")
    segs = [p1, p2, p3]
    before = [p.read_bytes() for p in segs]

    settings = get_settings().model_copy(update={**_SMALL, "render_source_match": "off"})
    await renderer._apply_source_match(segs, settings, _finish(match_sources=True))

    assert [p.read_bytes() for p in segs] == before


async def test_source_match_disabled_grade_is_byte_identical(tmp_path: Path) -> None:
    """``finish.grade.match_sources=False`` leaves the segment bytes untouched
    even with the render policy on — the per-request opt-in is authoritative."""
    p1 = await _flat_clip(tmp_path / "seg-0000.mp4", luma_hex="606060")
    p2 = await _flat_clip(tmp_path / "seg-0001.mp4", luma_hex="808080")
    p3 = await _flat_clip(tmp_path / "seg-0002.mp4", luma_hex="a0a0a0")
    segs = [p1, p2, p3]
    before = [p.read_bytes() for p in segs]

    settings = get_settings().model_copy(update={**_SMALL, "render_source_match": "on"})
    await renderer._apply_source_match(segs, settings, _finish(match_sources=False))

    assert [p.read_bytes() for p in segs] == before
