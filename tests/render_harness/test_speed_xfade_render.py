"""Real-ffmpeg render-harness tests for per-clip speed + xfade (WS-2).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group. These drive ACTUAL encodes
and interrogate the output with ffprobe — the string tests in
``tests/test_renderer_speed_xfade.py`` prove the command shape; these prove the
delivered pixels/timestamps.

MinIO/storage is NOT exercised here: ``render_shot`` + ``xfade_concat_shots``
operate on local paths (a bare filesystem path is used as the shot's
``video_url``), so no bucket/presign is needed. The full ``compile_shots`` route
(submit→poll→upload) is covered by the live E2E, not this local harness.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot, Transition
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip, probe_duration, probe_has_audio
from tests.render_harness.harness import probe_video_frame_count

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


async def test_speed_2x_halves_duration_and_frames(tmp_path: Path) -> None:
    """A shot rendered at speed=2.0 lasts ~half as long with ~half the frames of
    the same shot at speed=1.0."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    settings = get_settings().model_copy(update={"render_fps": 25})

    seg_1x = tmp_path / "seg_1x.mp4"
    seg_2x = tmp_path / "seg_2x.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=2.0, video_url=str(src)), seg_1x, settings, tmp_path, http
        )
        out_2x = await renderer.render_shot(
            Shot(id="b", seconds=2.0, video_url=str(src), speed=2.0),
            seg_2x,
            settings,
            tmp_path,
            http,
        )

    dur_1x = await probe_duration(seg_1x)
    dur_2x = await probe_duration(seg_2x)
    assert dur_2x == pytest.approx(dur_1x / 2, rel=0.3)
    # render_shot reports the retimed (post-speed) segment length.
    assert out_2x == pytest.approx(dur_1x / 2, rel=0.3)

    frames_1x = await probe_video_frame_count(seg_1x)
    frames_2x = await probe_video_frame_count(seg_2x)
    assert frames_2x < frames_1x * 0.7  # materially fewer frames, not just re-stamped


async def test_xfade_two_segments_overlap_and_keep_audio(tmp_path: Path) -> None:
    """Two ~1s segments joined by a 0.5s crossfade yield ~1.5s total with audio."""
    s0 = await make_test_clip(tmp_path / "s0.mp4", duration=1.0, fps=25, frequency=440)
    s1 = await make_test_clip(tmp_path / "s1.mp4", duration=1.0, fps=25, frequency=660)
    settings = get_settings().model_copy(update={"render_fps": 25})
    out = tmp_path / "joined.mp4"

    total = await renderer.xfade_concat_shots(
        [s0, s1],
        [None, Transition(type="fade", duration=0.5)],
        out,
        settings,
    )

    # sum(lengths) - overlap = 1.0 + 1.0 - 0.5 = 1.5
    assert total == pytest.approx(1.5, rel=0.1)
    assert await probe_duration(out) == pytest.approx(1.5, abs=0.25)
    assert await probe_has_audio(out) is True


async def test_xfade_preserves_hard_cut_between_groups(tmp_path: Path) -> None:
    """A None boundary is a hard cut: three 1s clips with a crossfade only on the
    3rd (0.5s) yield 1.0 + (1.0 + 1.0 - 0.5) = 2.5s total."""
    # The inputs must be REAL segments (``render_shot`` output): the hard-cut
    # singleton h0 is stream-copied next to the re-encoded crossfade group, and
    # ``concat_shots`` refuses a join whose audio differs. Raw ``make_test_clip``
    # files (AAC 44.1 kHz mono) next to an ALAC group decoded as noise here.
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, fps=25)
    settings = get_settings().model_copy(update={"render_fps": 25})
    s0, s1, s2 = (tmp_path / f"h{i}.mp4" for i in range(3))
    async with httpx.AsyncClient() as http:
        for i, seg in enumerate((s0, s1, s2)):
            await renderer.render_shot(
                Shot(id=f"h{i}", seconds=1.0, video_url=str(src)), seg, settings, tmp_path, http
            )
    out = tmp_path / "mixed.mp4"

    total = await renderer.xfade_concat_shots(
        [s0, s1, s2],
        [None, None, Transition(type="fade", duration=0.5)],
        out,
        settings,
    )

    assert total == pytest.approx(2.5, rel=0.1)
    assert await probe_duration(out) == pytest.approx(2.5, abs=0.3)
    assert await probe_has_audio(out) is True


async def _streams(path: Path) -> str:
    """``ffprobe -show_streams`` summary, for failure messages."""
    return await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,sample_fmt,bits_per_raw_sample,sample_rate,"
            "channels,duration,nb_frames",
            "-of",
            "compact",
            str(path),
        ]
    )


async def _decoded_audio_samples(path: Path) -> int:
    raw = path.with_name(path.name + ".raw")
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a?",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "48000",
            str(raw),
        ]
    )
    n = raw.stat().st_size // 2 if raw.exists() else 0
    raw.unlink(missing_ok=True)
    return n


async def test_xfade_group_file_always_carries_its_audio(tmp_path: Path) -> None:
    """Every crossfaded group file must carry ALL of its audio, EVERY time.

    ``acrossfade`` truncated or dropped the group's audio depending on thread
    timing: PR CI (ffmpeg 6.1.1) produced 10/10 group files with NO audio stream;
    the prod image (7.1.5) at 1 CPU produced 576..21056-sample stubs of a
    72000-sample crossfade. Repeated renders because it was a race; the DECODED
    sample count is checked, not the stream's presence or its header duration.
    """
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, fps=25)
    settings = get_settings().model_copy(
        update={"render_fps": 25, "render_width": 320, "render_height": 240}
    )
    segs = [tmp_path / f"g{i}.mp4" for i in range(2)]
    async with httpx.AsyncClient() as http:
        for i, seg in enumerate(segs):
            await renderer.render_shot(
                Shot(id=f"g{i}", seconds=1.0, video_url=str(src)), seg, settings, tmp_path, http
            )
    lengths = [await probe_duration(seg) for seg in segs]
    # The overlap lands on the output frame grid (0.5 s is 12.5 frames at 25 fps),
    # so the group's audio is exactly as long as its picture.
    overlap = renderer._nearest_frame(0.5, 25)
    expected = round((lengths[0] + lengths[1] - overlap) * 48000)
    bad: list[str] = []
    for n in range(10):
        out = tmp_path / f"group-{n}.mp4"
        await renderer._render_xfade_group(
            segs, lengths, [None, Transition(type="fade", duration=0.5)], out, settings
        )
        got = await _decoded_audio_samples(out)
        if abs(got - expected) > 48:  # 1 ms
            bad.append(f"render {n}: {got} samples (expected {expected}); {await _streams(out)}")
    if bad:
        inputs = "; ".join([f"{p.name}: {await _streams(p)}" for p in segs])
        pytest.fail(f"{len(bad)}/10 group files lost audio: {bad} | inputs: {inputs}")
