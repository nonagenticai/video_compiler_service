"""Real-ffmpeg render-harness tests: a footage segment must deliver EXACTLY the
frame count its planned span implies.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_clip_seek_duration.py``.

THE DEFECT. The footage branch of ``render_shot`` bounds a segment with an
OUTPUT-side ``-t`` while ``_clip_seek_args`` emits an INPUT-side ``-ss``. An
output ``-t`` measures its window from the first OUTPUT TIMESTAMP, not from
zero. When the seek lands mid-frame — the normal case, because the cut planner's
in-points are musical times and not source-frame multiples — the first decoded
frame sits after the requested seek point and the segment's video ``start_time``
is non-zero. Measured here on a 23.976 fps source (frame period 0.041708s)
seeked to ``-ss 1.002``: ``start_time = 0.041016``, all but a whole frame period.
The ``-t`` window then runs ``[0.041016, 0.041016 + t)``, the last frame of the
span falls outside it, and the segment ships one frame short.

WHAT IT COST. A delivered ``action_reel`` carried 685 video frames against a plan
of 690 — five two-frame PTS gaps, one per affected shot seam — walking the cuts
up to 56.8 ms off the music beat against a one-frame (41.7 ms) tolerance.

These tests drive ACTUAL encodes and count frames with ``-count_frames``,
because the CONTAINER duration hides this completely: the silent branch bounds
an infinite ``anullsrc`` with the same ``-t``, so the container always reports
the full ``-t`` on the strength of the audio alone even when the video stream is
a frame short. An argv-level assertion on the filter string cannot prove a video
renders; only a decode can.

NOTE on what actually discriminates. It is the PHASE OF THE SEEK, not the
"integrality" of the span: under a mid-frame seek a 1.75s (42-frame) span loses a
frame exactly as a 2.25s (54-frame) one does, and under a frame-aligned seek
neither does. So the control below is a seek that lands exactly on a source
frame boundary (``-ss 1.001`` == 24 source frames at 24000/1001), which was
correct before the fix and must stay correct after it.

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer
from tests.render_harness import probe_video_frame_count

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_W, _H = 320, 240
# The master cadence under test. 24 is the reel's delivery rate; the SOURCE is
# 23.976 (24000/1001), which is what the real footage was and what puts the seek
# off the output grid.
_FPS = 24
_SOURCE_RATE = "24000/1001"
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}
# A seek that lands MID-FRAME on a 23.976 source: 1.002 / (1001/24000) = 24.024
# source frames, so the first readable frame is #25 at 1.043375 and the segment
# starts 0.041016 late (probed, not derived).
_MID_FRAME_SEEK = 1.002
# A seek that lands exactly ON a source frame boundary: 1.001 == 24 * 1001/24000.
_ALIGNED_SEEK = 1.001


async def _make_source(path: Path, *, duration: float) -> Path:
    """A real MP4 at 23.976 fps with NO audio stream.

    ``make_test_clip`` cannot produce this: its ``fps`` is an ``int`` (so it
    cannot express 24000/1001) and it always muxes a sine tone. The silence
    matters — it routes ``render_shot`` down the ``anullsrc`` branch, which keeps
    ``trim`` True and so bounds the segment with the output ``-t`` under test.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={_W}x{_H}:rate={_SOURCE_RATE}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


async def _render_window(
    tmp_path: Path, source: Path, *, lead: float, span: float, name: str, speed: float = 1.0
) -> int:
    """Render one footage shot as an explicit source window and return its exact
    delivered video frame count."""
    settings = get_settings().model_copy(update=_SMALL)
    seg = tmp_path / f"{name}.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id=name,
                seconds=span,
                video_url=str(source),
                clip_lead_seconds=lead,
                clip_out_seconds=lead + span,
                speed=speed,
            ),
            seg,
            settings,
            tmp_path,
            http,
        )
    return await probe_video_frame_count(seg)


async def test_mid_frame_seek_delivers_every_planned_frame(tmp_path: Path) -> None:
    """THE regression. A 2.25s span at 24 fps is 54 frames. Under a mid-frame
    seek into a 23.976 fps source this delivered 53 — the exact shape of the
    685-vs-690 reel. ``setpts=PTS-STARTPTS`` after the cadence ``fps=`` rebases
    the segment to a true zero so the output ``-t`` measures the planned window.
    """
    source = await _make_source(tmp_path / "src.mp4", duration=20.0)
    frames = await _render_window(
        tmp_path, source, lead=_MID_FRAME_SEEK, span=2.25, name="mid-frame"
    )
    assert frames == 54, f"delivered {frames} frames for a 54-frame span"


async def test_mid_frame_seek_is_exact_for_a_whole_second_span_too(tmp_path: Path) -> None:
    """The shortfall is one frame per SEGMENT, not per second: a 3.0s span (72
    frames) lost exactly one frame the same way. Pins that the fix is not tuned
    to one particular span length."""
    source = await _make_source(tmp_path / "src.mp4", duration=20.0)
    frames = await _render_window(
        tmp_path, source, lead=_MID_FRAME_SEEK, span=3.0, name="mid-frame-3s"
    )
    assert frames == 72, f"delivered {frames} frames for a 72-frame span"


async def test_frame_aligned_seek_integral_span_is_unchanged(tmp_path: Path) -> None:
    """Regression guard on the FIX. A 1.75s span (42 frames) behind a seek that
    lands exactly on a source frame boundary already delivered 42 frames with
    ``start_time`` 0.000000 — ``setpts=PTS-STARTPTS`` subtracts zero there and
    must leave it at 42. Proves the rebase adds nothing and removes nothing on
    the already-correct path (i.e. it cannot itself drop or duplicate a frame).
    """
    source = await _make_source(tmp_path / "src.mp4", duration=20.0)
    frames = await _render_window(tmp_path, source, lead=_ALIGNED_SEEK, span=1.75, name="aligned")
    assert frames == 42, f"delivered {frames} frames for a 42-frame span"


async def test_speed_retime_still_changes_the_frame_count(tmp_path: Path) -> None:
    """The rebase must not fight ``_apply_shot_speed``.

    ``_apply_speed_to_vf`` PREPENDS ``setpts=PTS/speed`` to the whole body, so the
    chain is ``setpts=PTS/2 , scale , pad , setsar , fps=24 ,
    setpts=PTS-STARTPTS , format``. The speed retime must still halve the
    delivered frames (a trailing rebase is a constant OFFSET, not a rate change,
    so it can neither undo nor double-apply it) — and the result must be exact
    under the same mid-frame seek: 3.0s at 2x is 1.5s == 36 frames.
    """
    source = await _make_source(tmp_path / "src.mp4", duration=20.0)
    frames = await _render_window(
        tmp_path, source, lead=_MID_FRAME_SEEK, span=3.0, name="fast", speed=2.0
    )
    assert frames == 36, f"delivered {frames} frames for a 2x-retimed 72-frame span"
