"""Real-ffmpeg render-harness tests: a window whose OUT-point is at EOF must
still deliver every planned frame.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render``, the
same shape as ``test_frame_exact_segments.py`` — read that file first; this is
the INPUT-side twin of the defect it fixed.

THE DEFECT. ``_clip_seek_args`` emitted ``-ss`` at the requested time with no
snap to the SOURCE frame grid, and derived the input ``-t`` span from that same
requested time. ffmpeg's accurate seek starts at the first frame AT OR AFTER the
request, so a mid-frame in-point — the normal case, since the cut planner's
in-points are musical times, not source-frame multiples — loses up to one source
frame at the HEAD while the ``-t`` still stops at the requested out-point.
``setpts=PTS-STARTPTS`` (#174) correctly rebases the OUTPUT window onto that late
first frame, but it cannot conjure the lost frame back.

WHY IT NORMALLY HIDES. The output ``-t`` usually has enough tail slack to absorb
the head loss. It is only fatal with ZERO slack, i.e. when the window ends at
EOF. Shot 12 of the live 19-shot reel was exactly that: ``out_seconds 10.010 ==
source duration 10.010``, ``-ss 8.010`` landing 0.0397s (0.95 frame periods)
before the next real frame at 8.04972 (source frame 193). It delivered 42 frames
where the plan said 43 — the whole remaining deficit after #174 took the reel
from -5 to -1.

Reproduced here at 320x240 with the same rates: pre-fix 47 frames against a
48-frame plan, post-fix 48. The frame count is measured with ``-count_frames``
because the CONTAINER duration hides this completely (the silent branch bounds an
infinite ``anullsrc`` with the same ``-t``, so the container reports the full
length on the strength of the audio alone).

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
# 24 is the delivery cadence; the SOURCE is 23.976 (24000/1001) — the rate the
# live reel's footage carried and what puts the in-point off the frame grid.
_FPS = 24
_SOURCE_RATE = "24000/1001"
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}
# 240 source frames exactly. An out-point of 10.010 therefore sits AT EOF: no
# tail slack whatsoever, which is the condition that makes the head loss fatal.
_SOURCE_SECONDS = 10.010
# 8.010 == 192.048 source frames: 0.048 of a frame past boundary 192 (8.008), so
# the accurate seek skips to frame 193 at 8.04972 and the head loses 0.0397s.
_MID_FRAME_IN = 8.010


async def _make_source(path: Path, *, duration: float) -> Path:
    """A real MP4 at 23.976 fps with NO audio stream (``make_test_clip`` can
    express neither: its ``fps`` is an ``int`` and it always muxes a tone). The
    silence routes ``render_shot`` down the ``anullsrc`` branch, which keeps the
    output ``-t`` in play — the same chain the reel rendered through."""
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
    tmp_path: Path,
    source: Path,
    *,
    lead: float,
    out: float,
    seconds: float,
    name: str,
    target_fps: float | None = None,
) -> int:
    """Render one footage shot as an explicit source window; return its exact
    delivered video frame count."""
    settings = get_settings().model_copy(update=_SMALL)
    seg = tmp_path / f"{name}.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id=name,
                seconds=seconds,
                video_url=str(source),
                clip_lead_seconds=lead,
                clip_out_seconds=out,
                target_fps=target_fps,
            ),
            seg,
            settings,
            tmp_path,
            http,
        )
    return await probe_video_frame_count(seg)


async def test_eof_window_with_a_mid_frame_in_point_delivers_every_planned_frame(
    tmp_path: Path,
) -> None:
    """THE regression, end to end. 2.0s at 24 fps is 48 frames; the window ends
    at EOF so there is no tail to absorb the head loss. Pre-fix this rendered 47.

    ``target_fps`` is deliberately UNSET — the production shape, where the cut
    planner states musical times and leaves the rate to the file. This therefore
    also proves the ``r_frame_rate`` probe fallback is live; without it the fix
    would be inert on exactly the renders it was written for."""
    source = await _make_source(tmp_path / "src.mp4", duration=_SOURCE_SECONDS)
    frames = await _render_window(
        tmp_path,
        source,
        lead=_MID_FRAME_IN,
        out=_SOURCE_SECONDS,
        seconds=2.0,
        name="eof-window",
    )
    assert frames == 48, f"delivered {frames} frames for a 48-frame span ending at EOF"


async def test_declared_target_fps_drives_the_snap_too(tmp_path: Path) -> None:
    """Same window with the rate DECLARED on the shot. The declared rate is
    preferred over the probe, so both wirings must land on the same 48 frames —
    otherwise the fix would silently depend on which caller shape ran."""
    source = await _make_source(tmp_path / "src.mp4", duration=_SOURCE_SECONDS)
    frames = await _render_window(
        tmp_path,
        source,
        lead=_MID_FRAME_IN,
        out=_SOURCE_SECONDS,
        seconds=2.0,
        name="eof-declared",
        target_fps=24000.0 / 1001.0,
    )
    assert frames == 48, f"delivered {frames} frames with a declared source rate"


async def test_zero_in_point_window_is_unchanged(tmp_path: Path) -> None:
    """The byte-identical guard. ``in == 0.0`` is below the 0.1s noise floor, so
    no ``-ss`` is emitted, there is nothing to snap, and the span must be
    untouched. Asserted at BOTH levels: the exact args, and a real encode."""
    source = await _make_source(tmp_path / "src.mp4", duration=_SOURCE_SECONDS)
    args, offset, span = await renderer._clip_seek_args(
        source, 0.0, 2.0, clip_lead_seconds=0.0, clip_out_seconds=2.0
    )
    assert args == ["-t", "2.000"]
    assert (offset, span) == (0.0, 2.0)
    frames = await _render_window(tmp_path, source, lead=0.0, out=2.0, seconds=2.0, name="zero-in")
    assert frames == 48, f"delivered {frames} frames for an unseeked 48-frame span"


async def test_mid_window_with_tail_slack_is_not_regressed(tmp_path: Path) -> None:
    """The other 18 shots of the reel: a mid-frame in-point whose out-point is
    well short of EOF already had slack and already delivered its plan. Moving the
    in-point EARLIER by up to one source frame (<=41.7ms) is a sub-frame CONTENT
    shift there, not a count change — this pins that it stays a count of 48."""
    source = await _make_source(tmp_path / "src.mp4", duration=_SOURCE_SECONDS)
    frames = await _render_window(
        tmp_path, source, lead=3.010, out=5.010, seconds=2.0, name="mid-window"
    )
    assert frames == 48, f"delivered {frames} frames for a 48-frame span with tail slack"


async def test_a_short_seeked_shot_does_not_run_long(tmp_path: Path) -> None:
    """The trap the 19-shot sweep caught, at the encode level.

    Recovering the head frame by WIDENING the span (leaving the out-point fixed
    and measuring from the snapped in-point) makes the widened span the segment's
    own output ``-t``, so the shot ships one frame OVER plan — #174's defect with
    the sign flipped, walking the cuts LATE off the beat. Eight of eighteen
    slack-bearing pseudo-shots did exactly that; this one (0.344s == 8 frames at
    24 fps, in-point 0.637 which floors to source frame 15 at 0.625) delivered 9.
    The span is therefore measured from the REQUESTED in-point and this must stay
    at 8."""
    source = await _make_source(tmp_path / "src.mp4", duration=_SOURCE_SECONDS)
    frames = await _render_window(
        tmp_path, source, lead=0.637, out=0.981, seconds=0.344, name="short-seeked"
    )
    assert frames == 8, f"delivered {frames} frames for an 8-frame span"
