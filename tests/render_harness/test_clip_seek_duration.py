"""Real-ffmpeg render-harness tests: the REPORTED duration of a SEEKED footage
shot must equal the duration actually delivered.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_freeze_pad_speed.py``.

``_clip_seek_args`` emits ``-ss`` BEFORE ``-i`` (an INPUT seek), so output
timestamps restart at the seek point and the rendered span is only
``clip_dur - offset`` seconds. The own-audio branch of ``render_shot``
(``audio_path is None`` and the clip carries its own audio) emits NO ``-t``, yet
it used to report the clip's FULL pre-seek length — a 20s ``clip_anchor='end'``
clip under a 6s shot rendered ~6.1s and claimed 20.0s, over-stating by exactly
the offset. That nominal duration feeds the API response and the music/ambient
bed length, so it has to be the real one.

These tests drive ACTUAL encodes and probe the delivered PER-STREAM lengths,
because a mocked ffprobe would happily agree with whatever the renderer computed
— and because the CONTAINER duration actively HIDES the silent-clip defect. The
silent branch bounds an infinite ``anullsrc`` with ``-t``, so the container
always reaches the ``-t`` value on the strength of the audio alone. Measured
before the fix, a 20s clip under a 6s ``clip_anchor='end'`` shot produced:

    container 20.000s | AUDIO 20.000s | VIDEO 6.080s

i.e. ~13.9s of audio running past the end of the video. A container-only
assertion passes happily on that.

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
from tests.render_harness import (
    make_test_clip,
    probe_duration,
    probe_has_audio,
    probe_video_frame_count,
)

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
# A seeked shot's reported length must land within a couple of frames of the
# delivered container — anything looser would hide a whole-offset error.
_TOL = 2 * _FRAME


async def _make_silent_clip(path: Path, *, duration: float) -> Path:
    """A real MP4 with a video stream and NO audio stream (``-an``), which
    ``make_test_clip`` cannot produce — it always muxes a sine tone."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={_W}x{_H}:rate={_FPS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


async def test_own_audio_end_anchored_clip_reports_the_post_seek_length(
    tmp_path: Path,
) -> None:
    """A 20s clip WITH its own audio, no narration, ``clip_anchor='end'`` under a
    6s shot: the input seek lands at 20 - 6 - 0.10 = 13.9s, so ~6.1s renders. The
    returned duration must be that, not the 20.0s full clip length (pre-fix: 20.0
    returned for a ~6.1s file — off by the entire 13.9s offset)."""
    settings = get_settings().model_copy(update=_SMALL)
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=20.0, size=f"{_W}x{_H}", fps=_FPS
    )
    assert await probe_has_audio(footage)
    seg = tmp_path / "own-audio-end.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=6.0, video_url=str(footage), clip_anchor="end"),
            seg,
            settings,
            tmp_path,
            http,
        )

    delivered = await probe_duration(seg)
    # The seek really happened: the segment is the settled tail, not the whole clip.
    assert delivered == pytest.approx(6.1, abs=0.2)
    assert out_dur == pytest.approx(delivered, abs=_TOL)


async def test_narration_driven_seeked_clip_still_reports_its_bounded_length(
    tmp_path: Path,
) -> None:
    """Regression fence for the branch that was already correct: with narration
    (``audio_url``) the same end-anchored seek is bounded by ``-t duration``, so
    the reported length is the narration length. Must hold before AND after the
    own-audio fix."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=6.0, size=f"{_W}x{_H}", fps=_FPS
    )
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=20.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "narrated-end.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(
                id="a",
                seconds=6.0,
                video_url=str(footage),
                audio_url=str(narration),
                clip_anchor="end",
            ),
            seg,
            settings,
            tmp_path,
            http,
        )

    delivered = await probe_duration(seg)
    assert out_dur == pytest.approx(6.0, abs=0.2)
    assert out_dur == pytest.approx(delivered, abs=_TOL)


async def test_silent_clip_seeked_delivers_matching_audio_and_video(tmp_path: Path) -> None:
    """The ``anullsrc`` branch (clip with NO audio stream): ``trim`` stays True so
    ``-t`` bounds the segment against the infinite silent source.

    The bound must be the POST-SEEK playable length. Using the unseeked file
    length made the AUDIO run ~13.9s past the end of the VIDEO — a real A/V
    mismatch that the container duration hides completely, because the container
    simply follows the longer (audio) stream. Measured pre-fix on exactly this
    input: container 20.000 | audio 20.000 | video 6.080.

    So this asserts per-STREAM lengths, not the container.
    """
    settings = get_settings().model_copy(update=_SMALL)
    footage = await _make_silent_clip(tmp_path / "silent.mp4", duration=20.0)
    assert not await probe_has_audio(footage)
    seg = tmp_path / "silent-end.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=6.0, video_url=str(footage), clip_anchor="end"),
            seg,
            settings,
            tmp_path,
            http,
        )

    # anchor='end' on a 20s clip under a 6s shot seeks to ~13.9s, leaving ~6.1s.
    assert out_dur == pytest.approx(6.1, abs=0.2)
    video_dur = await probe_video_frame_count(seg) / _FPS
    assert video_dur == pytest.approx(out_dur, abs=_TOL), (
        f"video stream {video_dur:.3f}s != reported {out_dur:.3f}s"
    )
    # The container may still round up by a frame; what must NOT happen is the
    # audio outliving the video.
    assert await probe_duration(seg) == pytest.approx(out_dur, abs=_TOL)


async def test_silent_clip_unseeked_is_unchanged(tmp_path: Path) -> None:
    """Regression guard on the fix itself: with NO seek the offset is 0.0, so the
    reported length is still the full clip. Proves the subtraction is applied
    once and only when a seek actually happened (no double-subtract)."""
    settings = get_settings().model_copy(update=_SMALL)
    footage = await _make_silent_clip(tmp_path / "silent-plain.mp4", duration=6.0)
    seg = tmp_path / "silent-plain-seg.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=6.0, video_url=str(footage)),
            seg,
            settings,
            tmp_path,
            http,
        )

    assert out_dur == pytest.approx(6.0, abs=0.2)
    assert await probe_video_frame_count(seg) / _FPS == pytest.approx(out_dur, abs=_TOL)
