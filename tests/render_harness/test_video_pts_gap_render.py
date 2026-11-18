"""A real one-frame hole in a stream-copy concat must read ``cfr_ok is False``.

Delivered master 01a11032 (2026-10-06): 952 frames in 953 slots, the empty slot at
the keyframe opening the last ``concat -c copy`` segment, and ``cfr_ok: true``.
This builds that file's SHAPE with real ffmpeg — the concat demuxer is told (a
``duration`` directive) that the first segment lasts one frame longer than its
picture, which is what a segment whose container outlasts its video track tells it
on its own, so the next segment starts one frame late — and checks both readings:

* the old one (``framerate_coherence``: headers, then declared durations of the
  head) still says coherent — the blindness, pinned so it is not mistaken for a fix;
* ``video_pts_gap_report`` counts exactly one gap, at the join;
* ``post_mux_verify`` therefore reports ``cfr_ok is False``.

The pure function is covered in ``tests/unit/test_video_pts_gaps.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="video pts gap round trip requires ffmpeg + ffprobe",
    ),
]

# 25 fps so a frame is a whole 40 ms: ffmpeg 4.4's concat demuxer advances by the
# MP4 header duration, stored in milliseconds, and would smear a 30 fps join by a
# fraction of a millisecond (see test_audio_timestamp_holes_render.py).
_FPS = 25
_FRAMES = 50  # per segment: 2.000 s of picture


async def _segment(path: Path) -> None:
    await renderer._run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=160x120:r={_FPS}",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:r=48000",
            "-t",
            str(_FRAMES / _FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(_FPS),
            "-c:a",
            "alac",
            "-ac",
            "2",
            str(path),
        ]
    )


async def _concat(
    first: Path, second: Path, out: Path, *, first_lasts: float | None = None
) -> Path:
    """Stream-copy join. ``first_lasts`` overrides how long the demuxer believes
    ``first`` is — i.e. where it places ``second``."""
    listing = out.with_suffix(".txt")
    claim = f"duration {first_lasts:.6f}\n" if first_lasts is not None else ""
    listing.write_text(f"file '{first}'\n{claim}file '{second}'\n")
    await renderer._run(
        [
            *["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i"],
            str(listing),
            "-c",
            "copy",
            str(out),
        ]
    )
    return out


@pytest.mark.asyncio
async def test_a_one_frame_hole_at_a_copy_join_is_not_cfr(tmp_path: Path) -> None:
    frame = 1.0 / _FPS
    picture = _FRAMES * frame
    head, tail = tmp_path / "head.mp4", tmp_path / "tail.mp4"
    await _segment(head)
    await _segment(tail)

    # ONE FRAME past the picture: 51 slots claimed for 50 frames.
    holed = await _concat(head, tail, tmp_path / "holed.mp4", first_lasts=picture + frame)
    clean = await _concat(head, tail, tmp_path / "clean.mp4")
    settings = get_settings().model_copy(update={"render_fps": _FPS, "render_colorspace": ""})

    # The control first: the same two segments joined at the picture's own end.
    assert await mastering.video_pts_gap_report(clean) == (0, [])

    count, positions = await mastering.video_pts_gap_report(holed)
    assert count == 1
    # The last frame of segment 1 (pts 49/25) is followed by two frame periods.
    assert positions == [
        {"at_seconds": pytest.approx((_FRAMES - 1) * frame, abs=1e-3), "interval_ms": 80.0}
    ]

    # THE BLINDNESS, on a real file: the pre-existing reading passes the holed one.
    assert await mastering.framerate_coherence(holed, _FPS) is True

    report = await mastering.post_mux_verify(holed, settings, loudness={"integrated_lufs": -14.0})
    assert report["video_pts_gap_count"] == 1
    assert report["cfr_ok"] is False
    assert "cfr" in report["rerun_stages"]

    control = await mastering.post_mux_verify(clean, settings, loudness={"integrated_lufs": -14.0})
    assert control["video_pts_gap_count"] == 0
    assert control["cfr_ok"] is True
