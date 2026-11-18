"""A copy-concat of an SFX shot and a plain narration shot must still DECODE to
the narration (real ffmpeg, no mocks).

THE DEFECT (prod run 01a09f82, website_review, 2026-09-14): the delivered master
had no narration — silence, plus ~10 kHz broadband noise bursts where the lines
had been.

Mechanism: segments carry lossless ALAC (``_SEGMENT_AUDIO_CODEC_ARGS``), and the
ALAC encoder picks its BIT DEPTH from the sample format the filter graph hands it.
A plain narration shot (24 kHz ``pcm_s16le`` WAV) negotiates ``s16p`` -> 16-bit;
a shot whose audio runs through ``amix``/``alimiter`` (any transition/accent SFX)
is float -> ``s32p`` -> 24-bit. ``concat_shots`` is a ``-c copy`` join, so the
film's audio stream carries the FIRST segment's codec header and every segment of
the other depth decodes as garbage. Measured on the prod image (ffmpeg 7.1.5)
with real speech at -22.7 dBFS: the 16-bit line after a 24-bit segment decoded at
-8.2 dBFS with spectral flatness 0.91 (noise), another at -124.6 dBFS (gone).

WHY THESE TESTS DECODE. The container still reports the right codec, rate,
channel count and duration for the broken film — only the decoded SAMPLES show
it. Every assertion compares a shot's samples INSIDE the joined film to the same
shot's samples decoded from its OWN segment file (which is always correct, since
it carries its own header): level (RMS) and zero-crossing rate (a cheap spectral
proxy — a 440 Hz tone crosses ~880 times a second, broadband noise ~10-20x more).
"""

from __future__ import annotations

import array
import itertools
import math
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import AccentCue, Shot, SoundCue
from src.config import Settings, get_settings
from src.services import renderer
from tests.render_harness import make_bed_tone

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="decoded concat round trip requires ffmpeg + ffprobe",
    ),
]

_SR = 48000
_W, _H = 320, 240
_SHOT_SECONDS = 2.0
_TONE_HZ = 440
# Shot edges carry short fades; measure the steady interior only.
_EDGE = 0.15
# Level must match to within this many dB. A correct join is sample-identical
# (measured delta 0.00 dB); the defect misses by >= 14 dB in either direction.
_MAX_RMS_DELTA_DB = 1.0
# Zero-crossing rate must match to within this ratio. Correct: 1.00; the defect's
# noise reads many times the tone's rate (or ~0 when the line decodes to silence).
_MAX_ZCR_RATIO = 1.25


def _settings() -> Settings:
    return get_settings().model_copy(update={"render_width": _W, "render_height": _H})


async def _make_narration(path: Path) -> Path:
    """Narration exactly as the orchestrator premaster ships it: 24 kHz MONO
    ``pcm_s16le`` — the s16 input that negotiates 16-bit ALAC."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={_TONE_HZ}:sample_rate=24000:duration={_SHOT_SECONDS}",
            "-af",
            "volume=-6dB",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(path),
        ]
    )
    return path


async def _make_still(path: Path) -> Path:
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={_W}x{_H}",
            "-frames:v",
            "1",
            str(path),
        ]
    )
    return path


async def _decode(path: Path) -> array.array[int]:
    raw = path.with_name(path.name + ".decoded.raw")
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(_SR),
            str(raw),
        ]
    )
    samples: array.array[int] = array.array("h")
    samples.frombytes(raw.read_bytes())
    raw.unlink(missing_ok=True)
    return samples


def _rms_db(samples: array.array[int]) -> float:
    if not samples:
        return -math.inf
    mean_sq = sum(s * s for s in samples) / len(samples)
    return -math.inf if mean_sq == 0 else 10 * math.log10(mean_sq / (32768.0 * 32768.0))


def _zcr(samples: array.array[int]) -> float:
    """Zero crossings per second (0.0 for no samples — a join that decoded SHORT
    is itself the defect, and must fail the comparison, not crash it)."""
    if len(samples) < 2:
        return 0.0
    crossings = sum(1 for a, b in itertools.pairwise(samples) if (a < 0) != (b < 0))
    return crossings / (len(samples) / _SR)


def _interior(samples: array.array[int], start: float, length: float) -> array.array[int]:
    lo = round((start + _EDGE) * _SR)
    hi = round((start + length - _EDGE) * _SR)
    return samples[lo:hi]


async def _render_segments(tmp_path: Path, sfx_flags: list[bool]) -> list[Path]:
    """One REAL ``render_shot`` per flag: narration over a still, plus a quiet
    transition + accent SFX when the flag is set (the float mix path)."""
    still = await _make_still(tmp_path / "still.png")
    vo = await _make_narration(tmp_path / "vo.wav")
    # A 44.1 kHz cue, as library SFX are; quiet so it barely moves the level.
    sfx = await make_bed_tone(
        tmp_path / "sfx.wav", duration=0.2, frequency=_TONE_HZ, gain_db=-40.0, sample_rate=44100
    )
    settings = _settings()
    segments: list[Path] = []
    async with httpx.AsyncClient() as http:
        for i, with_sfx in enumerate(sfx_flags):
            seg = tmp_path / f"seg-{i:04d}{renderer._SEGMENT_EXT}"
            shot = Shot(
                id=f"s{i}",
                seconds=_SHOT_SECONDS,
                image_url=str(still),
                audio_url=str(vo),
                transition_sfx=SoundCue(url=str(sfx), gain_db=0.0) if with_sfx else None,
                sfx=[AccentCue(url=str(sfx), at_seconds=0.5)] if with_sfx else [],
            )
            await renderer.render_shot(shot, seg, settings, tmp_path, http)
            segments.append(seg)
    return segments


async def _assert_every_shot_survives_the_join(tmp_path: Path, sfx_flags: list[bool]) -> None:
    segments = await _render_segments(tmp_path, sfx_flags)
    joined = tmp_path / f"compiled{renderer._SEGMENT_EXT}"
    await renderer.concat_shots(segments, joined)
    film = await _decode(joined)
    start = 0.0
    failures: list[str] = []
    for i, seg in enumerate(segments):
        own = await _decode(seg)
        length = len(own) / _SR
        ref = _interior(own, 0.0, length)
        got = _interior(film, start, length)
        ref_db, got_db = _rms_db(ref), _rms_db(got)
        ref_zcr, got_zcr = _zcr(ref), _zcr(got)
        ratio = got_zcr / ref_zcr if ref_zcr else math.inf
        if not (
            abs(got_db - ref_db) <= _MAX_RMS_DELTA_DB
            and 1 / _MAX_ZCR_RATIO <= ratio <= _MAX_ZCR_RATIO
        ):
            failures.append(
                f"shot {i} (sfx={sfx_flags[i]}): own segment {ref_db:.1f} dBFS / "
                f"{ref_zcr:.0f} zc/s, in joined film {got_db:.1f} dBFS / {got_zcr:.0f} zc/s"
            )
        start += length
    assert not failures, "narration did not survive the copy-concat: " + "; ".join(failures)


async def test_sfx_shot_then_narration_shot_decodes_the_narration(tmp_path: Path) -> None:
    """The prod shape: the film opens on a shot with SFX (24-bit before the fix),
    then plain narration lines (16-bit). Before the fix the narration lines
    decoded as silence / broadband noise."""
    await _assert_every_shot_survives_the_join(tmp_path, [True, False, False, True])


async def test_narration_shot_then_sfx_shot_decodes_both(tmp_path: Path) -> None:
    """The reverse order: the header is the narration's, the SFX shot is the one
    decoded wrong. Pins that the fix is not order-dependent."""
    await _assert_every_shot_survives_the_join(tmp_path, [False, True, False])


async def test_every_segment_carries_the_same_audio_signature(tmp_path: Path) -> None:
    """Both render paths must produce an IDENTICAL audio codec signature — the
    property the join depends on, read from the files the encoder wrote."""
    segments = await _render_segments(tmp_path, [False, True])
    sigs = []
    for seg in segments:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=" + ",".join(renderer._CONCAT_AUDIO_FIELDS),
                "-of",
                "json",
                str(seg),
            ]
        )
        sigs.append(renderer._segment_audio_signature(out))
    assert sigs[0] is not None
    assert sigs[0] == sigs[1], f"segment audio differs between render paths: {sigs}"


async def _alac_segment(path: Path, sample_fmt: str) -> Path:
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={_TONE_HZ}:sample_rate={_SR}:duration=1",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s={_W}x{_H}:r=25:d=1",
            "-map",
            "1:v",
            "-map",
            "0:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "alac",
            "-sample_fmt",
            sample_fmt,
            "-ar",
            str(_SR),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(path),
        ]
    )
    return path


async def test_concat_refuses_segments_whose_audio_differs(tmp_path: Path) -> None:
    """The tripwire: hand ``concat_shots`` a 16-bit and a 24-bit ALAC segment and
    it must RAISE rather than deliver a film that decodes as noise."""
    a = await _alac_segment(tmp_path / "a.mp4", "s16p")
    b = await _alac_segment(tmp_path / "b.mp4", "s32p")
    out = tmp_path / "joined.mp4"
    with pytest.raises(renderer.FfmpegError, match="segment audio parameters differ"):
        await renderer.concat_shots([a, b], out)
    assert not out.exists()


async def test_concat_accepts_uniform_segments(tmp_path: Path) -> None:
    """Negative control for the tripwire: identical segments still join."""
    a = await _alac_segment(tmp_path / "a.mp4", "s32p")
    b = await _alac_segment(tmp_path / "b.mp4", "s32p")
    out = tmp_path / "joined.mp4"
    await renderer.concat_shots([a, b], out)
    assert out.exists()
