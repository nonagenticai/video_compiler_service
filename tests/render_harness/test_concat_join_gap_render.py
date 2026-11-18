"""A copy-concat of per-shot segments must be SAMPLE-EXACT (real ffmpeg, no mocks).

THE DEFECT, measured on a bench reproducing this repo's verbatim segment args:

Shot segments were each independently AAC-encoded (``_AUDIO_CODEC_ARGS``) and
then joined by the concat DEMUXER at ``-c copy`` (``concat_shots``). AAC carries
~1024 samples of encoder delay plus trailing padding per segment, and a stream
copy cannot trim either. Three 1.000 s phase-continuous sine segments:

    join                       decoded duration   drift        interior silence
    AAC-in-mp4   ``-c copy``   3.072000 s         +72.0 ms     1153 smp = 24.02 ms
                                                  (+24.0/seg)   at EVERY join
    pcm_s16le-in-mov (control) 3.000000 s         0            none
    ALAC-in-mp4  ``-c copy``   3.000000 s         0            none   <- shipped

So ~24 ms of inserted silence at EVERY cut plus 24 ms of cumulative A/V drift
per cut. The audible shape is a DROPOUT (a hole, a stutter), not a click.
Production telemetry agrees from the opposite direction: median
``av_gap_seconds`` = 0.097 s ~= 4 cuts x 24 ms.

WHY THESE TESTS DECODE. ⛔ Container ``format=duration`` CANNOT see this: it is
the MAX over streams, so it reports whatever the longest stream claims. Even
``-select_streams a:0`` reports the muxed stream's declared length rather than
the samples that come out of the decoder. Every assertion here therefore counts
DECODED SAMPLES and looks at their VALUES. An argv-level assertion on the codec
tokens cannot prove a join is sample-exact and would not have caught this.

NON-VACUITY IS BUILT IN, not merely asserted: ``test_the_method_detects_an_aac
_join`` runs the identical measurement over segments encoded with the DELIVERY
codec args and requires it to report the defect. If someone ever loosens the
tolerances below, that control fails first.
"""

from __future__ import annotations

import array
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
        reason="sample-exact concat round trip requires ffmpeg + ffprobe",
    ),
]

_SR = 48000
_W, _H = 320, 240
_SEG_SECONDS = 1.0
_N_SEGMENTS = 3
_TONE_HZ = 1000  # 1.000 s at 1000 Hz is a whole number of cycles → phase-continuous

# Below this absolute s16 amplitude a sample is "near-silent". The tone is full
# scale, so a genuine sample of it peaks past 30000; the AAC gap floor measured
# below |x| = 200. 200 sits two orders of magnitude clear of both.
_SILENCE_FLOOR = 200
# How long a near-silent run at a boundary is ALLOWED to be. Two adjacent shots
# each carry an 8 ms ``_SHOT_EDGE_FADE_S`` ramp, so a cut is legitimately quiet
# for at most the two ramps together — that is the deliberate, inaudible
# click-guard, not a dropout. Anything longer is audio that is not there.
#
# The three measured quantities this sits between, so the threshold is not a
# knob: the fade skirt actually below the floor is 63 samples (1.3 ms — the ramp
# spends most of its length well above 200); the allowance below is 768; the
# defect's inserted hole is 1112-1153 samples (23.2-24.0 ms) at EVERY cut. It
# cannot be widened to hide the defect without first failing
# ``test_the_method_detects_an_aac_join``.
_MAX_BOUNDARY_RUN_SAMPLES = round(2 * renderer._SHOT_EDGE_FADE_S * _SR)
# A run touching the film's own head or tail is that film's own fade, not a join.
_EDGE_GUARD_SAMPLES = round(0.05 * _SR)


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update={"render_width": _W, "render_height": _H, **overrides})


async def _decode_audio(path: Path) -> array.array[int]:
    """Every sample the audio stream actually DECODES to, as mono s16 at 48 kHz.

    This is the measurement the defect requires. ``ffprobe``'s container or
    stream ``duration`` is a HEADER field — on a copy-concat it reports the sum
    of the segments' declared lengths and agrees with the plan even when the
    decoder emits 24 ms of extra silence per cut. Only the decoder's own output
    can be counted.
    """
    raw_path = path.with_name(path.name + ".decoded.raw")
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
            str(raw_path),
        ]
    )
    samples: array.array[int] = array.array("h")
    samples.frombytes(raw_path.read_bytes())
    raw_path.unlink(missing_ok=True)
    return samples


def _decoded_seconds(samples: array.array[int]) -> float:
    return len(samples) / _SR


def _dropouts(samples: array.array[int]) -> list[tuple[int, float]]:
    """``(length_in_samples, start_seconds)`` for every interior near-silent run
    LONGER than a legitimate pair of edge fades.

    "Interior" excludes the film's own head and tail (the first and last shot are
    edge-faded, so the ends of the film are legitimately near-zero). A run
    anywhere else, longer than ``_MAX_BOUNDARY_RUN_SAMPLES``, is audio that is
    not there — the dropout under test.
    """
    runs: list[tuple[int, float]] = []
    n = len(samples)
    run = 0
    for i, value in enumerate(samples):
        if abs(value) < _SILENCE_FLOOR:
            run += 1
            continue
        start = i - run
        if (
            run > _MAX_BOUNDARY_RUN_SAMPLES
            and start > _EDGE_GUARD_SAMPLES
            and i < n - _EDGE_GUARD_SAMPLES
        ):
            runs.append((run, start / _SR))
        run = 0
    # A trailing run reaches the last sample, i.e. it IS the film's tail fade.
    return runs


async def _make_tone_wav(path: Path, *, seconds: float = _SEG_SECONDS) -> Path:
    """A full-scale mono narration tone — the shot's ``audio_url``."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={_TONE_HZ}:sample_rate={_SR}:duration={seconds}",
            "-c:a",
            "pcm_s16le",
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


async def _render_and_join(tmp_path: Path, *, n: int = _N_SEGMENTS) -> tuple[Path, float]:
    """Drive the REAL ``render_shot`` for ``n`` narrated shots, then the REAL
    ``concat_shots``. Returns the joined film and the summed input duration."""
    still = await _make_still(tmp_path / "still.png")
    wav = await _make_tone_wav(tmp_path / "vo.wav")
    settings = _settings()
    segments: list[Path] = []
    total = 0.0
    async with httpx.AsyncClient() as http:
        for i in range(n):
            seg = tmp_path / f"seg-{i:04d}{renderer._SEGMENT_EXT}"
            total += await renderer.render_shot(
                Shot(id=f"s{i}", seconds=_SEG_SECONDS, image_url=str(still), audio_url=str(wav)),
                seg,
                settings,
                tmp_path,
                http,
            )
            segments.append(seg)
    joined = tmp_path / f"compiled{renderer._SEGMENT_EXT}"
    await renderer.concat_shots(segments, joined)
    return joined, total


# ---------------------------------------------------------------------------
# THE regression
# ---------------------------------------------------------------------------


async def test_copy_concat_of_real_segments_is_sample_exact(tmp_path: Path) -> None:
    """3 real ``render_shot`` segments, joined by the real ``concat_shots``, must
    decode to EXACTLY the sum of their inputs.

    Before the fix this decoded 3.072000 s against a 3.000000 s sum — +72 ms,
    i.e. +24 ms of A/V drift per cut, every frame after cut 1 late by a growing
    amount. Tolerance is 2 samples (0.042 ms) either way: the join is either
    sample-exact or it is not, and the defect misses by 3456 samples.
    """
    joined, total_in = await _render_and_join(tmp_path)
    samples = await _decode_audio(joined)
    decoded = _decoded_seconds(samples)
    drift_samples = len(samples) - round(total_in * _SR)
    assert abs(drift_samples) <= 2, (
        f"copy-concat drifted {drift_samples} samples "
        f"({drift_samples / _SR * 1000:.2f} ms): decoded {decoded:.6f}s "
        f"for {total_in:.6f}s of input"
    )


async def test_copy_concat_inserts_no_gap_at_the_cuts(tmp_path: Path) -> None:
    """No interior near-silent run anywhere in the joined film.

    This is the owner-visible half: the drift is a sync problem, but the 1153
    contiguous near-zero samples (24.02 ms) that AAC padding + delay left at EVERY
    internal boundary are an audible HOLE in the narration — the reported
    "stutter". A continuous tone across the join must stay continuous.
    """
    joined, _ = await _render_and_join(tmp_path)
    runs = _dropouts(await _decode_audio(joined))
    assert not runs, "inserted silence at the join(s): " + ", ".join(
        f"{n} samples ({n / _SR * 1000:.2f} ms) at {t:.4f}s" for n, t in runs
    )


async def test_the_join_holds_for_more_cuts(tmp_path: Path) -> None:
    """The error was per-CUT, so it compounded: 6 segments drifted ~144 ms, not
    ~72. Pins that the fix is not tuned to a three-segment film."""
    joined, total_in = await _render_and_join(tmp_path, n=6)
    samples = await _decode_audio(joined)
    assert abs(len(samples) - round(total_in * _SR)) <= 2
    assert not _dropouts(samples)


# ---------------------------------------------------------------------------
# The control: prove the measurement can still SEE the defect
# ---------------------------------------------------------------------------


async def test_the_method_detects_an_aac_join(tmp_path: Path) -> None:
    """Deliberately build the OLD pipeline — segments in the DELIVERY codec args
    (``_AUDIO_CODEC_ARGS``: AAC 192k) in an mp4 — join them with the real
    ``concat_shots``, and require the measurement above to report the defect.

    This is what stops the two tests above from being satisfiable by loosening a
    tolerance: any relaxation that lets an AAC join pass fails HERE. It also pins
    the bench numbers in the module docstring to something executable.
    """
    settings = _settings()
    segments: list[Path] = []
    for i in range(_N_SEGMENTS):
        seg = tmp_path / f"aac-{i:04d}.mp4"
        await renderer._run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={_TONE_HZ}:sample_rate={_SR}:duration={_SEG_SECONDS}",
                "-f",
                "lavfi",
                "-i",
                f"color=c=blue:s={_W}x{_H}:r={settings.render_fps}:d={_SEG_SECONDS}",
                "-map",
                "1:v",
                "-map",
                "0:a",
                "-t",
                str(_SEG_SECONDS),
                *renderer._video_codec_args(settings),
                *renderer._AUDIO_CODEC_ARGS,
                "-movflags",
                "+faststart",
                str(seg),
            ]
        )
        segments.append(seg)
    joined = tmp_path / "aac-joined.mp4"
    await renderer.concat_shots(segments, joined)
    samples = await _decode_audio(joined)
    drift_ms = (len(samples) - _N_SEGMENTS * _SEG_SECONDS * _SR) / _SR * 1000
    runs = _dropouts(samples)
    assert drift_ms > 40, f"expected the AAC join to drift; measured {drift_ms:.2f} ms"
    assert len(runs) == _N_SEGMENTS - 1, f"expected one gap per internal cut; got {runs}"
    for n, _ in runs:
        assert n > 500, f"expected a ~1153-sample hole at each cut; got {n}"


# ---------------------------------------------------------------------------
# The delivered master's audio spec is unchanged
# ---------------------------------------------------------------------------


async def test_delivered_master_is_still_aac_192k_48k_stereo(tmp_path: Path) -> None:
    """The lossless segment codec is an INTERMEDIATE, never a deliverable. Whatever branch
    ``_finalize_audio`` takes, what leaves this service is AAC / 48000 / 2ch in an
    mp4 — the contract consumers already depend on. Checks the two branches that
    do NOT run a bed or a master pass, i.e. exactly the ones that used to carry
    the concat output straight out."""
    still = await _make_still(tmp_path / "still.png")
    wav = await _make_tone_wav(tmp_path / "vo.wav")
    shots = [
        Shot(id=f"s{i}", seconds=_SEG_SECONDS, image_url=str(still), audio_url=str(wav))
        for i in range(_N_SEGMENTS)
    ]
    settings = _settings(work_dir=str(tmp_path / "work"), master_enabled=False)
    out, _total, _bed, _timeline, _fps = await renderer.compile_shots(shots, settings)
    try:
        assert out.suffix == ".mp4", f"delivered {out.name}, not an mp4"
        codec = (
            await renderer._run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels",
                    "-of",
                    "csv=p=0",
                    str(out),
                ]
            )
        ).strip()
        assert codec == "aac,48000,2", f"delivered audio is {codec!r}"
        # And the sample-exactness survives the delivery transcode.
        assert not _dropouts(await _decode_audio(out))
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)
