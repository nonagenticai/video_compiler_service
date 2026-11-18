"""Real-ffmpeg proof that the opt-in ``deterministic`` audio-concat is reproducible.

Task-1 companion to ``test_deterministic_encode.py``: the audiobook stitch path
(``concat_audio`` → ``_concat_audio_copy`` / ``_concat_audio_filter``) grew a
``deterministic`` flag that threads ``-fflags +bitexact`` before ``-i`` (strips the
container ``creation_time``/``encoder`` tag on the WAV stream-copy path) and, on the
MP3/libmp3lame path, ``-flags:a +bitexact`` (drops the LAME version stamp). This
drives each stitch TWICE with ``deterministic=True`` on a FIXED input set and
asserts the two whole-file md5s are identical.

CRITICAL: both runs use the SAME local ffmpeg; the assertion is WITHIN-binary
equality of the two runs, never a comparison against a pre-baked hash (mp3/WAV
bytes differ across ffmpeg builds). Gated at MODULE level (skipped without
ffmpeg/ffprobe) and tagged ``render`` like the rest of the real-render suite.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.services import renderer
from tests.render_harness import make_bed_tone, probe_duration

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _file_md5(path: Path) -> str:
    """Whole-file MD5 of the delivered audio file (read in chunks)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _fixed_segments(tmp_path: Path) -> list[Path]:
    """Two FIXED, format-uniform WAV clips (same rate/channels, distinct tone) —
    a valid input set for both the stream-copy demuxer and the concat filter."""
    seg0 = await make_bed_tone(tmp_path / "seg0.wav", duration=1.0, frequency=440)
    seg1 = await make_bed_tone(tmp_path / "seg1.wav", duration=1.0, frequency=660)
    return [seg0, seg1]


async def test_concat_audio_copy_deterministic_is_byte_reproducible(tmp_path: Path) -> None:
    """Two ``deterministic=True`` WAV stream-copy stitches of the SAME fixed clip
    set produce a byte-identical file (the container tag is stripped)."""
    segments = await _fixed_segments(tmp_path)

    out1 = tmp_path / "book1.wav"
    out2 = tmp_path / "book2.wav"
    await renderer._concat_audio_copy(segments, out1, deterministic=True)
    await renderer._concat_audio_copy(segments, out2, deterministic=True)

    assert await probe_duration(out1) > 0.0
    assert await probe_duration(out2) > 0.0
    assert _file_md5(out1) == _file_md5(out2)


async def test_concat_audio_filter_mp3_deterministic_is_byte_reproducible(tmp_path: Path) -> None:
    """Two ``deterministic=True`` MP3 filter stitches of the SAME fixed clip set
    produce a byte-identical file (the LAME version stamp is dropped)."""
    segments = await _fixed_segments(tmp_path)

    out1 = tmp_path / "book1.mp3"
    out2 = tmp_path / "book2.mp3"
    await renderer._concat_audio_filter(segments, out1, "mp3", 0, deterministic=True)
    await renderer._concat_audio_filter(segments, out2, "mp3", 0, deterministic=True)

    assert await probe_duration(out1) > 0.0
    assert await probe_duration(out2) > 0.0
    assert _file_md5(out1) == _file_md5(out2)


async def test_concat_audio_copy_off_path_still_produces_valid_file(tmp_path: Path) -> None:
    """``deterministic=False`` (the prod default) still stitches a valid file —
    today's behaviour is unchanged; cross-run identity is NOT asserted for off."""
    segments = await _fixed_segments(tmp_path)

    out = tmp_path / "book-off.wav"
    await renderer._concat_audio_copy(segments, out, deterministic=False)

    assert out.exists()
    assert await probe_duration(out) > 0.0


# ---------------------------------------------------------------------------
# Heterogeneous-input guard (_assert_uniform_audio) — real ffprobe over real WAVs
# ---------------------------------------------------------------------------


async def _make_wav(path: Path, *, sample_rate: int, channels: int, frequency: int) -> Path:
    """A real WAV at an EXACT sample rate + channel count via ffmpeg lavfi."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration=1.0:sample_rate={sample_rate}",
            "-ac",
            str(channels),
            str(path),
        ]
    )
    return path


async def test_concat_rejects_heterogeneous(tmp_path: Path) -> None:
    """A 44.1 kHz stereo clip stitched with a 24 kHz mono clip must be rejected
    with the ffmpeg error BEFORE any concat command runs — the demuxer would
    otherwise silently corrupt the join, and the filter path would size its gap
    silence to the first clip only."""
    a = await _make_wav(tmp_path / "a.wav", sample_rate=44100, channels=2, frequency=440)
    b = await _make_wav(tmp_path / "b.wav", sample_rate=24000, channels=1, frequency=660)

    with pytest.raises(renderer.FfmpegError, match="heterogeneous audio"):
        await renderer._concat_audio_copy([a, b], tmp_path / "copy.wav")
    with pytest.raises(renderer.FfmpegError, match="heterogeneous audio"):
        await renderer._concat_audio_filter([a, b], tmp_path / "filter.mp3", "mp3", 0)


async def test_concat_uniform_unchanged(tmp_path: Path) -> None:
    """Two format-uniform clips still concat to a valid file — the guard is a
    no-op on the prod (uniform) input and never changes the emitted command."""
    a = await _make_wav(tmp_path / "u0.wav", sample_rate=24000, channels=1, frequency=440)
    b = await _make_wav(tmp_path / "u1.wav", sample_rate=24000, channels=1, frequency=660)

    out = tmp_path / "uniform.wav"
    await renderer._concat_audio_copy([a, b], out)
    assert out.exists()
    assert await probe_duration(out) > 0.0
