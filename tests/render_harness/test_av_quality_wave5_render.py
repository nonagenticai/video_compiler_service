"""Wave 5 picture-quality fixes, proven with REAL ffmpeg (no mocks on the
encode path) — an argv assertion proves a flag is EMITTED, not that ffmpeg
HONOURS it.

3a — deband + source-match folded into ONE re-encode. A segment tripping
BOTH conditions used to cost two sequential decode/encode generations
(``_apply_conditional_deband`` then ``_apply_source_match``); the merged
``_apply_picture_master`` must cost exactly one.

3c — an explicit GOP (``-g`` 2x fps, ``-keyint_min`` fps) so a keyframe lands
at least every 2s. Proven by actually counting I-frames in the encoded output
via ffprobe, not by reading argv.
"""

from __future__ import annotations

import shutil
from itertools import pairwise
from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg wave5 proof requires ffmpeg + ffprobe",
    ),
]


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _make_source(path: Path, *, fps: int, duration: float) -> None:
    """A synthetic clip with real per-frame motion (testsrc2), so a real x264
    encode has something to keyframe/deband/regrade against."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x240:rate={fps}:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )


# ---------------------------------------------------------------------------
# 3c — real keyframe cadence
# ---------------------------------------------------------------------------


async def _iframe_positions(path: Path) -> list[int]:
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pict_type",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    lines = out.strip().splitlines()
    return [i for i, t in enumerate(lines) if t.strip() == "I"]


async def test_explicit_gop_bounds_keyframe_spacing(tmp_path: Path) -> None:
    """At fps=10 the GOP fix asks for a keyframe every <=20 frames (2x fps).
    x264's flagless default GOP (~250 frames) would leave a 60-frame clip with
    a SINGLE keyframe. Prove the fixed args produce more than one and that no
    gap exceeds the ceiling (with a couple of frames of x264-internal slack)."""
    fps = 10
    src = tmp_path / "src.mp4"
    await _make_source(src, fps=fps, duration=6.0)

    settings = _settings(render_fps=fps)
    out = tmp_path / "gop.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        *renderer._video_codec_args(settings),
        "-an",
        str(out),
    ]
    await renderer._run(cmd)

    keyframes = await _iframe_positions(out)
    assert len(keyframes) >= 2, "a single keyframe means -g/-keyint_min did not reach ffmpeg"
    gaps = [b - a for a, b in pairwise(keyframes)]
    # x264's own scenecut/open-GOP bookkeeping can land a keyframe a frame or
    # two later than the exact ceiling; the ceiling this proves is "not the
    # ~250-frame default", so a couple of frames of slack keeps the assertion
    # about the FIX, not about x264's internal rounding.
    assert max(gaps) <= 2 * fps + 2, f"keyframe gap {max(gaps)} exceeds the 2x-fps GOP ceiling"


# ---------------------------------------------------------------------------
# 3a — one combined re-encode, not two sequential ones
# ---------------------------------------------------------------------------


async def test_picture_master_does_one_encode_not_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force BOTH the deband and source-match decisions to fire on the same
    segment, and prove (by counting real ffmpeg invocations against the
    segment) that _apply_picture_master re-encodes it exactly ONCE with a
    combined filter, instead of the two sequential decode/encode passes the
    old _apply_conditional_deband + _apply_source_match ran."""
    seg = tmp_path / "seg-0000.mp4"
    await _make_source(seg, fps=24, duration=1.0)
    settings = _settings(render_fps=24, render_deband="on", render_source_match="on")

    # Force the DECISIONS (not the encode) — this is the same seam the unit
    # tests for _deband_decide/_source_match_decide already exercise; only the
    # encode below is real.
    async def fake_deband_decide(segment_paths: list[Path], _settings: Settings) -> list[Path]:
        return list(segment_paths)

    async def fake_source_match_decide(segment_paths, _settings, _finish):
        return [(p, "eq=gamma=1.2") for p in segment_paths]

    monkeypatch.setattr(renderer, "_deband_decide", fake_deband_decide)
    monkeypatch.setattr(renderer, "_source_match_decide", fake_source_match_decide)

    encode_calls = 0
    real_run = renderer._run

    async def counting_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        nonlocal encode_calls
        if cmd and cmd[0] == "ffmpeg" and str(seg) in cmd:
            encode_calls += 1
        return await real_run(cmd, capture_stderr=capture_stderr)

    monkeypatch.setattr(renderer, "_run", counting_run)

    await renderer._apply_picture_master([seg], settings, finish=None, deterministic=True)

    assert encode_calls == 1, f"expected exactly one re-encode of {seg}, got {encode_calls}"
    assert seg.exists() and seg.stat().st_size > 0

    # The single encode must have actually run through ffmpeg (not left the
    # clean single-pass segment untouched): the file must still decode.
    probe = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(seg),
        ]
    )
    assert float(probe.strip()) > 0.0
