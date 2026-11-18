"""A short cover clip + separate narration audio must never livelock a segment
render (#232).

THE DEFECT, reproduced here with real ffmpeg. ``_freeze_pad_suffix`` used to
append an UNCONDITIONAL ``tpad=stop_mode=clone:stop=-1`` (infinite last-frame
clone) whenever the segment carried an output-side ``-t`` — including on the
sub-one-output-frame under-run this file reproduces. Under ``-fps_mode:v cfr`` with
a SECOND, finite input (the narration audio) in the same filtergraph, that
infinite clone livelocks: ffmpeg keeps synthesising duplicate frames forever,
every one of them dropped against the shorter audio stream, and the encode
never reaches ``-t``. Measured on production inputs (h264 1440x2560 30fps,
1.933333s / 58 real frames, against a 1.95s narration WAV — a ~17ms gap, well
under one output frame at 30fps): ffmpeg froze at a fixed byte count with
``drop_frames`` climbing without bound and did not respond to SIGTERM — only
SIGKILL stopped it.

THE FIX (``renderer._bounded_cover_pad``) replaces ``stop=-1`` with a FINITE,
computed ``stop_duration`` at every emit site in ``_freeze_pad_suffix``.

This test reproduces the exact shape at small scale: a real (honest, not
truncated/lying) cover clip genuinely shorter than its segment target by
LESS than one output frame, and a real, separate narration audio input of
exactly the target length. Guarded with an outer ``asyncio.wait_for`` timeout
so a regression FAILS FAST rather than hanging the CI runner forever the way
the real bug did.

⚠️ The literal livelock is ffmpeg-BUILD-dependent (observed on the production
box's ffmpeg; NOT reproduced on this repo's dev-machine ffmpeg 4.4, which
completes this exact shape in ~1s even with ``stop=-1`` reinstated by hand).
So this test also pins the invariant that actually prevents the livelock
directly, at the emitted-command level: the real ffmpeg command line built for
this shape must NEVER contain ``stop=-1`` — mutation-proven by monkeypatching
``renderer._bounded_cover_pad`` back to the old unconditional-infinite-clone
shape and confirming the assertion goes red. That check does not depend on any
particular ffmpeg build's scheduling behaviour the way an observed hang would.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg livelock reproduction requires ffmpeg + ffprobe",
    ),
]

_W, _H, _FPS = 320, 240, 30
# The production shape: 58 real frames at 30fps = 1.933333s, ~17ms short of a
# 1.95s narration line — under one output frame (1/30 = 33.3ms).
_CLIP_SECONDS = 58 / 30
_NARRATION_SECONDS = 1.95
_RENDER_TIMEOUT_SECONDS = 30.0


def _settings():
    return get_settings().model_copy(
        update={"render_width": _W, "render_height": _H, "render_fps": _FPS}
    )


async def _clip(tmp_path: Path, seconds: float) -> Path:
    dest = tmp_path / "clip.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_W}x{_H}:rate={_FPS}:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    return dest


async def _narration(tmp_path: Path, seconds: float) -> Path:
    dest = tmp_path / "nar.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(dest),
        ]
    )
    return dest


async def test_sub_frame_undercut_cover_clip_does_not_livelock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact production shape: a real (not truncated/lying) clip honestly
    shorter than its narration by less than one output frame, rendered against
    a separate finite audio input. Must complete well inside the timeout and
    deliver a segment whose duration matches the narration target — on the OLD
    (``stop=-1``) code this hangs forever on the production ffmpeg build, which
    is why the render is wrapped in ``asyncio.wait_for`` rather than left to
    hang the test suite. It also pins the finite-clone invariant directly on
    the REAL command ffmpeg was handed."""
    captured: list[list[str]] = []
    real_run = renderer._run

    async def _spy_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured.append(cmd)
        return await real_run(cmd, capture_stderr=capture_stderr)

    monkeypatch.setattr(renderer, "_run", _spy_run)

    clip = await _clip(tmp_path, _CLIP_SECONDS)
    narration = await _narration(tmp_path, _NARRATION_SECONDS)

    # Preconditions: the clip must genuinely under-run by LESS than one output
    # frame, or this does not exercise the sub-frame branch this test targets.
    probed_clip = await renderer._ffprobe_duration(clip)
    probed_audio = await renderer._ffprobe_duration(narration)
    assert probed_clip is not None and probed_audio is not None
    gap = probed_audio - probed_clip
    assert 0.0 < gap < 1.0 / _FPS, f"gap {gap} must be positive and sub-frame at {_FPS}fps"

    out = tmp_path / "out.mp4"
    try:
        async with httpx.AsyncClient() as http:
            dur = await asyncio.wait_for(
                renderer.render_shot(
                    Shot(id="cover", seconds=1.0, video_url=str(clip), audio_url=str(narration)),
                    out,
                    _settings(),
                    tmp_path,
                    http,
                ),
                timeout=_RENDER_TIMEOUT_SECONDS,
            )
    except TimeoutError:
        pytest.fail(
            f"render_shot livelocked past the {_RENDER_TIMEOUT_SECONDS}s guard — "
            "this is the #232 stop=-1 livelock regressing"
        )

    assert dur == pytest.approx(_NARRATION_SECONDS, abs=0.05)
    video, audio = await mastering.av_stream_seconds(out)
    assert video is not None and audio is not None
    assert audio == pytest.approx(_NARRATION_SECONDS, abs=0.25)
    assert video >= audio - mastering.AV_COVERAGE_TOLERANCE_SECONDS, (
        f"picture {video:.3f}s under {audio:.3f}s of audio"
    )

    # The encode command actually handed to ffmpeg for this exact sub-frame
    # shape must never carry the infinite clone — this is the invariant that
    # removes the livelock class regardless of ffmpeg build/version.
    encode_cmds = [c for c in captured if "-filter_complex" in c]
    assert encode_cmds, "expected the compositing ffmpeg invocation to be captured"
    joined = " ".join(encode_cmds[-1])
    assert "stop=-1" not in joined, f"the encode command still carries an infinite clone: {joined}"
    assert "tpad=stop_mode=clone:stop_duration=" in joined, (
        f"expected a FINITE terminal cover clone in: {joined}"
    )
