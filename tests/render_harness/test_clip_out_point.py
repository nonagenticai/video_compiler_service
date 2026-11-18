"""Real-ffmpeg proof that ``Shot.clip_out_seconds`` actually bounds the segment.

The unit test (``tests/test_clip_out_point.py``) proves the ARGS are right. This
one proves the delivered mp4 is, because the two are not the same claim: the
out-point is emitted as an INPUT-side ``-t`` (before ``-i``), and whether an
input option really caps what the decoder reads — as opposed to an output ``-t``
that merely truncates a stream ffmpeg already decoded — is a property of ffmpeg,
not of our string building. A mocked ffprobe would agree with whatever the
renderer computed and prove nothing.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_clip_seek_duration.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip, probe_duration, probe_video_frame_count

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}
_FRAME = 1.0 / _FPS
_TOL = 3 * _FRAME


async def test_out_point_bounds_the_delivered_segment(tmp_path: Path) -> None:
    """A 20s source, window 4.0s-7.5s ⇒ a ~3.5s segment, not the 16s tail.

    The clip carries its own audio and the shot has no narration, so the
    own-audio branch emits NO output ``-t``: the segment length is decided
    ENTIRELY by the input-side window. That makes this the sharpest possible
    assertion — without the input ``-t`` the delivered file is the whole 16s
    post-seek tail.
    """
    settings = get_settings().model_copy(update=_SMALL)
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=20.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "windowed.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(
                id="a",
                seconds=10.0,
                video_url=str(footage),
                clip_lead_seconds=4.0,
                clip_out_seconds=7.5,
            ),
            seg,
            settings,
            tmp_path,
            http,
        )

    delivered = await probe_duration(seg)
    assert delivered == pytest.approx(3.5, abs=0.2), (
        f"window 4.0-7.5s delivered {delivered:.3f}s — the input -t did not bound the read"
    )
    assert out_dur == pytest.approx(delivered, abs=_TOL)


async def test_no_out_point_still_delivers_the_whole_tail(tmp_path: Path) -> None:
    """The same shot WITHOUT ``clip_out_seconds`` is unchanged: the full tail.

    This is the no-op fence. It is the control for the test above — if both
    delivered ~3.5s the bound would be coming from somewhere else entirely.
    """
    settings = get_settings().model_copy(update=_SMALL)
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=20.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "unwindowed.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=10.0, video_url=str(footage), clip_lead_seconds=4.0),
            seg,
            settings,
            tmp_path,
            http,
        )

    delivered = await probe_duration(seg)
    assert delivered == pytest.approx(16.0, abs=0.3), (
        f"unwindowed shot delivered {delivered:.3f}s — expected the full post-seek tail"
    )
    assert out_dur == pytest.approx(delivered, abs=_TOL)


async def test_window_is_measured_on_the_source_timeline_under_speed(tmp_path: Path) -> None:
    """``clip_out_seconds`` is SOURCE seconds; ``speed`` retimes what it selected.

    The same 3.5s window at ``speed=2.0`` must deliver ~1.75s. If the window were
    (wrongly) scaled by ``1/speed`` before being handed to the decoder, the
    selected span itself would move — a different 1.75s of footage would be
    shown, and the delivered length would be ~0.875s.
    """
    settings = get_settings().model_copy(update=_SMALL)
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=20.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "windowed-2x.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(
                id="a",
                seconds=10.0,
                video_url=str(footage),
                clip_lead_seconds=4.0,
                clip_out_seconds=7.5,
                speed=2.0,
            ),
            seg,
            settings,
            tmp_path,
            http,
        )

    video_dur = await probe_video_frame_count(seg) / _FPS
    assert out_dur == pytest.approx(1.75, abs=0.2)
    assert video_dur == pytest.approx(out_dur, abs=_TOL), (
        f"video stream {video_dur:.3f}s != reported {out_dur:.3f}s"
    )
