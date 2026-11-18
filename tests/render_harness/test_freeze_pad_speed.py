"""Real-ffmpeg render-harness tests for the freeze-pad x speed interaction.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_drift_pad_render.py`` and ``test_speed_xfade_render.py``.

The freeze-pad ``tpad`` chain is emitted UPSTREAM of the ``setpts=PTS/speed``
retime that ``_apply_shot_speed`` prepends, so its ``stop_duration`` is measured
on the POST-retime timeline while the ``-t`` bound is rescaled to
``duration/speed``. Before the fix the tail was sized in PRE-speed seconds: at
``speed<1.0`` (slow-mo) the video stream EOF'd BEFORE the larger ``-t`` bound, so
the delivered video was SHORTER than its audio (an A/V mismatch). These tests
drive ACTUAL encodes and probe the per-stream durations to prove the delivered
video now covers the audio at every speed.

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
from tests.render_harness import make_test_clip, probe_video_frame_count

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


async def _video_duration(path: Path) -> float:
    """Delivered VIDEO-stream duration in seconds, derived from the exact decoded
    frame count / fps. Robust (frame count never reports N/A) and — crucially —
    isolates the video stream so a shorter-than-audio video (the pre-fix bug) is
    visible even though the container/format duration equals the longer stream."""
    return await probe_video_frame_count(path) / _FPS


async def _audio_duration(path: Path) -> float:
    """Delivered AUDIO-stream duration in seconds (ffprobe stream=duration)."""
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(out.strip())


async def test_slowmo_underrun_video_covers_the_retimed_audio(tmp_path: Path) -> None:
    """A ~3s footage clip under a ~6s narration at ``speed=0.5`` retimes the
    segment to ``duration/speed`` = 12s. The freeze-pad must fill the video to that
    retimed length so the delivered VIDEO stream matches the AUDIO stream (~12s,
    within one frame). Pre-fix the tail was sized in pre-speed seconds, so the
    video EOF'd at ~9s while the audio ran to 12s — an A/V mismatch this asserts
    against (pre-fix: video ~9s ≠ audio ~12s)."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=6.0, size=f"{_W}x{_H}", fps=_FPS
    )
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=3.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "slowmo.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(footage), audio_url=str(narration), speed=0.5),
            seg,
            settings,
            tmp_path,
            http,
        )

    video_dur = await _video_duration(seg)
    audio_dur = await _audio_duration(seg)
    # render_shot reports the retimed (post-speed) segment length: 6.0 / 0.5 = 12.0.
    assert out_dur == pytest.approx(12.0, abs=0.2)
    # The delivered video now reaches the retimed audio length — no short-video EOF.
    assert video_dur == pytest.approx(12.0, abs=0.2)
    assert audio_dur == pytest.approx(12.0, abs=0.2)
    # A/V coherent within one frame (the pre-fix bug: video ~9s, ~3s = 75 frames short).
    assert video_dur == pytest.approx(audio_dur, abs=2 * _FRAME)


async def test_speed_1x_underrun_pads_to_full_narration(tmp_path: Path) -> None:
    """The default path (speed=1.0) is unchanged: a ~3s clip under a ~6s narration
    is frozen-padded to the full narration, so video ≈ audio ≈ 6s (within one
    frame). This is the pre-fix behaviour the byte-identity invariant preserves —
    the string-level identity is proved in ``test_renderer_unit.py``."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=6.0, size=f"{_W}x{_H}", fps=_FPS
    )
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=3.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "onex.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(footage), audio_url=str(narration)),
            seg,
            settings,
            tmp_path,
            http,
        )

    video_dur = await _video_duration(seg)
    audio_dur = await _audio_duration(seg)
    assert out_dur == pytest.approx(6.0, abs=0.2)
    assert video_dur == pytest.approx(6.0, abs=0.2)
    assert video_dur == pytest.approx(audio_dur, abs=2 * _FRAME)


async def test_speed_2x_underrun_video_matches_audio_no_overrun(tmp_path: Path) -> None:
    """At ``speed=2.0`` the segment retimes to ``duration/speed`` = 3s. The freeze
    tail is sized in the post-retime timeline so the video ends WITH the audio at
    the ``-t`` = 3s bound — video ≈ audio ≈ 3s, with no frozen frames produced
    beyond ``-t`` (the tidy path: over-production would merely be truncated)."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=6.0, size=f"{_W}x{_H}", fps=_FPS
    )
    footage = await make_test_clip(
        tmp_path / "footage.mp4", duration=3.0, size=f"{_W}x{_H}", fps=_FPS
    )
    seg = tmp_path / "twox.mp4"
    async with httpx.AsyncClient() as http:
        out_dur = await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(footage), audio_url=str(narration), speed=2.0),
            seg,
            settings,
            tmp_path,
            http,
        )

    video_dur = await _video_duration(seg)
    audio_dur = await _audio_duration(seg)
    assert out_dur == pytest.approx(3.0, abs=0.2)
    assert video_dur == pytest.approx(3.0, abs=0.2)
    assert video_dur == pytest.approx(audio_dur, abs=2 * _FRAME)
    # No over-run past the -t bound: the delivered video never exceeds the retimed
    # segment length by more than a frame.
    assert video_dur <= out_dur + 2 * _FRAME
