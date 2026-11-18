"""A narrated shot's ``seconds`` is a picture FLOOR, end to end (real ffmpeg).

THE DEFECT: ``_resolve_shot_assets`` set a narrated shot's duration to the PROBED
narration length and read ``shot.seconds`` only when the probe failed. The
orchestrator clamps ``seconds`` into reconcile_timeline's ``[min, max]`` window and
pins a footage beat's ``min`` to its measured clip, so the compiler threw the plan
away. Orchestrator run ``01a0ab23-6c47-76a2-9d53-f16220e4792d`` (a 30.0s order):

    beat    footage  narration  delivered
    clip-1  10.96    10.185     10.200
    clip-4  11.92     7.371      7.367
    clip-5   8.40     5.105      5.100

22.67s delivered, 8.61s of recorded footage discarded, and every downstream duration
check agreed with the short film. Those three beats are the cases below.

Measured PER STREAM with an explicit ``ffprobe -select_streams v:0`` / ``a:0`` —
never the container ``format=duration``, which is the max over streams and cannot
see an audio leg that stops before the picture.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import Settings, get_settings
from src.services import renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg narrated picture-floor round trip requires ffmpeg + ffprobe",
    ),
]

_W, _H, _FPS = 320, 240, 25
# One output frame: the segment -t lands on the frame grid, so a floor honoured to
# within a frame is honoured. 8.61s of lost footage is ~215 frames of error.
_FRAME = 1.0 / _FPS

# (beat, measured footage seconds, measured narration seconds) — run 01a0ab23.
_MEASURED_BEATS = [
    ("clip-1", 10.96, 10.185),
    ("clip-4", 11.92, 7.371),
    ("clip-5", 8.40, 5.105),
]


def _settings() -> Settings:
    return get_settings().model_copy(
        update={"render_width": _W, "render_height": _H, "render_fps": _FPS}
    )


async def _narration(tmp_path: Path, seconds: float) -> Path:
    dest = tmp_path / "nar.wav"
    await renderer._run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"sine=frequency=440:sample_rate=24000:duration={seconds}",
            "-ac", "1", str(dest),
        ]
    )  # fmt: skip
    return dest


async def _clip(tmp_path: Path, seconds: float, *, with_audio: bool = False) -> Path:
    dest = tmp_path / ("clip-a.mp4" if with_audio else "clip.mp4")
    audio_in = (
        ["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=48000:duration={seconds}"]
        if with_audio
        else []
    )
    audio_out = ["-c:a", "aac", "-shortest"] if with_audio else []
    await renderer._run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", f"testsrc=size={_W}x{_H}:rate={_FPS}:duration={seconds}",
            *audio_in,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", *audio_out, str(dest),
        ]
    )  # fmt: skip
    return dest


async def _stream_seconds(path: Path, selector: str) -> float:
    """``stream=duration`` of ONE stream (``v:0`` / ``a:0``). Fails, never falls back
    to the container duration, when the stream or its duration is missing."""
    out = await renderer._run(
        [
            "ffprobe", "-v", "error", "-select_streams", selector,
            "-show_entries", "stream=duration", "-of", "json", str(path),
        ]
    )  # fmt: skip
    streams = json.loads(out).get("streams") or []
    assert streams, f"{path.name}: no {selector} stream"
    raw = streams[0].get("duration")
    assert raw not in (None, "N/A"), f"{path.name}: {selector} carries no duration"
    return float(raw)


async def _audio_mean_db(path: Path, start: float, seconds: float) -> float:
    """``volumedetect`` ``mean_volume`` (dBFS) of the ``a:0`` window
    ``[start, start + seconds]``. A digitally silent window reads ``-91 dB`` (the
    floor volumedetect reports); a sine reads within a few dB of full scale."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "info", "-nostats", "-ss", f"{start}", "-t", f"{seconds}",
        "-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    _, err = await proc.communicate()
    assert proc.returncode == 0, err.decode(errors="replace")[-400:]
    match = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", err.decode(errors="replace"))
    assert match, f"{path.name}: volumedetect printed no mean_volume"
    raw = match.group(1)
    return float("-inf") if raw == "-inf" else float(raw)


async def _render(tmp_path: Path, shot: Shot) -> tuple[float, float]:
    out = tmp_path / f"{shot.id}-out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(shot, out, _settings(), tmp_path, http)
    video = await _stream_seconds(out, "v:0")
    audio = await _stream_seconds(out, "a:0")
    print(f"{shot.id}: v:0={video:.3f}s a:0={audio:.3f}s (seconds={shot.seconds})")
    return video, audio


@pytest.mark.parametrize(("beat", "footage", "narration"), _MEASURED_BEATS)
async def test_measured_footage_beat_keeps_its_footage(
    tmp_path: Path, beat: str, footage: float, narration: float
) -> None:
    """The orchestrator sends ``seconds`` = the measured footage; the segment's
    picture AND its audio both reach it, and the narration is not cut."""
    nar = await _narration(tmp_path, narration)
    clip = await _clip(tmp_path, footage)
    video, audio = await _render(
        tmp_path,
        Shot(
            id=beat,
            seconds=footage,
            max_seconds=footage,
            video_url=str(clip),
            audio_url=str(nar),
        ),
    )
    assert abs(video - footage) <= _FRAME, (
        f"{beat}: picture must reach the {footage}s floor, got {video:.3f}s "
        f"(narration is {narration}s — the defect delivers that)"
    )
    assert abs(audio - footage) <= _FRAME, (
        f"{beat}: audio must be padded to span the {video:.3f}s picture, got {audio:.3f}s"
    )


async def test_seconds_below_narration_never_cuts_the_narration(tmp_path: Path) -> None:
    """A FLOOR, not a cap: 7.371s of narration under ``seconds=5.0`` still delivers
    7.371s of picture and audio."""
    nar = await _narration(tmp_path, 7.371)
    clip = await _clip(tmp_path, 11.92)
    video, audio = await _render(
        tmp_path,
        Shot(id="cap", seconds=5.0, max_seconds=5.0, video_url=str(clip), audio_url=str(nar)),
    )
    assert abs(audio - 7.371) <= _FRAME, f"narration must not be cut, got {audio:.3f}s"
    assert abs(video - 7.371) <= _FRAME, f"picture must follow the narration, got {video:.3f}s"


async def test_sub_frame_floor_is_ignored(tmp_path: Path) -> None:
    """``seconds`` within one frame of the narration is metadata rounding, not a
    planned hold: the segment stays narration-length."""
    nar = await _narration(tmp_path, 5.105)
    clip = await _clip(tmp_path, 8.40)
    video, audio = await _render(
        tmp_path,
        Shot(
            id="subframe",
            seconds=5.105 + _FRAME / 2,
            max_seconds=5.105 + _FRAME / 2,
            video_url=str(clip),
            audio_url=str(nar),
        ),
    )
    assert abs(video - 5.105) <= _FRAME, f"got {video:.3f}s"
    assert abs(audio - 5.105) <= _FRAME, f"got {audio:.3f}s"


async def test_unreconciled_seconds_is_not_a_floor(tmp_path: Path) -> None:
    """No ``max_seconds`` ⇒ ``seconds`` is NOT a reconciled plan value, so it is not a
    floor. When TTS cannot measure a clip, narrate stores ``seconds=None`` and the
    orchestrator's ``_reconcile_seconds`` sends the STORYBOARD guess alongside a valid
    ``audio_url``; ``max_seconds`` rides only on a shot reconcile_timeline bounded
    (``shot_assembly._freeze_pad_ceiling_patch``). lesson / app_review / audit_video /
    website_review never send bounds — honouring the guess there would freeze the
    picture over silence. The segment stays narration-length on BOTH streams."""
    nar = await _narration(tmp_path, 5.105)
    clip = await _clip(tmp_path, 12.0)
    video, audio = await _render(
        tmp_path,
        Shot(id="unreconciled", seconds=12.0, video_url=str(clip), audio_url=str(nar)),
    )
    assert abs(video - 5.105) <= _FRAME, (
        f"an un-reconciled storyboard seconds=12.0 must not hold the picture, got {video:.3f}s"
    )
    assert abs(audio - 5.105) <= _FRAME, (
        f"an un-reconciled storyboard seconds=12.0 must not pad silence, got {audio:.3f}s"
    )


async def test_silent_shot_keeps_its_seconds(tmp_path: Path) -> None:
    """No narration ⇒ ``seconds`` is the length, exactly as before."""
    video, audio = await _render(tmp_path, Shot(id="card", seconds=3.0, description="card"))
    assert abs(video - 3.0) <= _FRAME, f"got {video:.3f}s"
    assert abs(audio - 3.0) <= _FRAME, f"got {audio:.3f}s"


async def test_own_audio_clip_keeps_its_natural_length(tmp_path: Path) -> None:
    """A clip carrying its OWN audio (no ``audio_url``) with a placeholder
    ``seconds=1.0`` still plays its full 6.0s — the floor never touches it.

    ⛔ Durations alone prove nothing here. Every shot gets an edge fade, anchored at
    the length ``_primary_audio_frag`` is told the narration has. If ``render_shot``
    re-derived ``narration_padded`` from ``duration`` (reassigned to the clip length
    by the own-audio branch) instead of latching it, the fade-out would land at the
    1.0s placeholder and the audio would be SILENT from there — while both streams
    still report 6.0s. So the LEVEL after 1.0s is asserted too."""
    clip = await _clip(tmp_path, 6.0, with_audio=True)
    out_shot = Shot(id="own", seconds=1.0, video_url=str(clip))
    video, audio = await _render(tmp_path, out_shot)
    assert abs(video - 6.0) <= _FRAME, f"own-audio clip must keep 6.0s, got {video:.3f}s"
    assert abs(audio - 6.0) <= 2 * _FRAME, f"own-audio clip must keep 6.0s, got {audio:.3f}s"
    tail_db = await _audio_mean_db(tmp_path / "own-out.mp4", 1.5, 4.0)
    print(f"own: a:0 mean_volume over [1.5s, 5.5s] = {tail_db} dB")
    assert tail_db > -40.0, (
        f"own-audio clip must stay audible after the 1.0s placeholder, got {tail_db} dB "
        "over [1.5s, 5.5s] — its fade-out was anchored at seconds=1.0"
    )
