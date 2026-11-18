"""Measured proof that mastering the SAME audio twice destroys loudness range.

Why this exists
---------------
Course videos are assembled in two compiler passes: every lesson is compiled (and
mastered) in its own run, then ``course_stitch`` concatenates those finished lesson
MP4s and the bed-less ``_finalize_audio`` path masters the result a SECOND time.
``master_audio`` has no idempotency guard — its only skip is "no audio stream" — so
the full ``acompressor -> alimiter -> loudnorm`` chain (``renderer._master_filter``)
runs twice over the same programme.

This went unnoticed for a long time because the metric everyone checks does not move:
**integrated LUFS is unchanged** by the second pass. What collapses is the loudness
RANGE. Measured here, over three independent renders:

    source (narration-shaped, real macro-dynamics)   LRA 7.7 - 7.9
    A, mastered once                                 LRA 4.5 - 4.6   I -12.4
    B, mastered twice                                LRA 1.5 - 1.6   I -12.4
    A -> B delta                                     -2.90 .. -3.00 LU
    control: A re-encoded with NO master             +0.00 .. +0.10 LU

So the second pass throws away ~2/3 of the range that survived the first, at zero cost
to the number anybody was watching.

Two things this test is NOT
---------------------------
* It is not a claim about how it *sounds*. It measures a dynamics metric. Ear-checking
  a real render is an operator action and is not asserted here.
* It does not assert crest factor. Crest was measured first and rejected: it moved only
  -0.3 dB (inside run-to-run noise) because the limiter pins the peak and loudnorm
  re-pins the mean, so both ends move together. LRA is the metric that actually tracks
  the damage.

If someone ever makes ``master_audio`` idempotent, this test SHOULD fail — that is a
deliberate signal to come back here and delete it, not a flake to loosen.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import get_settings
from src.services import renderer
from src.services.loudness import final_mux_loudness

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

# Stepped passage levels (start, end, gain_db) with 0.5 s gaps between them.
_PASSAGES = [
    (0.0, 2.5, 0.0),
    (3.0, 5.5, -14.0),
    (6.0, 8.5, -4.0),
    (9.0, 11.5, -20.0),
    (12.0, 14.5, -2.0),
    (15.0, 17.5, -10.0),
    (18.0, 20.0, -6.0),
]
_DURATION = 20.0

# Bounds below are MEASURED (see the docstring), not estimated. The observed A->B drop
# is 2.9-3.0 LU; 2.0 leaves headroom for encoder differences between the macOS dev box
# and the Linux CI runner without being loose enough to pass on a no-op.
_MIN_RANGE_LOSS_LU = 2.0
_MAX_LOUDNESS_DRIFT_LU = 0.5
_MAX_CONTROL_DRIFT_LU = 0.5


async def _make_dynamic_narration(path: Path, *, lufs: float = -18.0) -> Path:
    """Synthesise an MP4 whose audio has NARRATION-SHAPED macro-dynamics.

    ``make_premastered_narration_clip`` in the shared harness is deliberately flat
    (steady speech-band noise, LRA ~0.3) because it exists to key a sidechain. That
    makes it useless here: a compressor cannot squash range that is not there, and it
    measures a ~0 LU delta whether the defect exists or not.

    This builder instead steps the level between passages (0 to -20 dB) with real
    pauses, giving a source LRA of ~7.8 — the shape an actual narrated lesson has to
    an EBU R128 meter — and premasters to ``lufs`` (-18 LUFS), which is what every
    voiceover on the bus already is.
    """
    gate = "+".join(f"between(t,{s},{e})" for s, e, _ in _PASSAGES)
    envelope = "".join(
        f"volume=enable='between(t,{s},{e})':volume={10 ** (g / 20):.4f}," for s, e, g in _PASSAGES
    )
    audio_filter = (
        "highpass=f=300,lowpass=f=3400,"
        f"volume=enable='not({gate})':volume=0,"
        f"{envelope}"
        "aformat=channel_layouts=stereo:sample_rates=48000,"
        f"loudnorm=I={lufs}:TP=-1.5"
    )
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:r=25:d={_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=c=pink:r=48000:d={_DURATION}",
            "-filter_complex",
            f"[1:a]{audio_filter}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(path),
        ]
    )
    return path


async def _reencode_without_master(src: Path, dst: Path) -> Path:
    """Re-encode the audio at the SAME codec/rate/layout but with no master filter.

    The negative control: it costs the signal one extra AAC generation, exactly like a
    second master pass does, but applies no compressor and no loudnorm.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )
    return dst


async def _loudness(path: Path) -> tuple[float, float]:
    """(lra, integrated_lufs) of the delivered file, via the production measurer."""
    measured = await final_mux_loudness(str(path))
    assert measured is not None, f"loudness measurement failed for {path}"
    lra, integrated = measured["lra"], measured["integrated_lufs"]
    assert lra is not None and integrated is not None, f"incomplete loudness: {measured}"
    return lra, integrated


async def _master(src: Path, dst: Path, tmp_path: Path) -> Path:
    settings = get_settings().model_copy(update={"work_dir": str(tmp_path)})
    assert settings.master_enabled, "this test measures the master; it must be enabled"
    return await renderer.master_audio(src, dst, settings)


async def test_second_master_pass_collapses_loudness_range(tmp_path: Path) -> None:
    """The defect: mastering an already-mastered programme costs ~3 LU of range."""
    source = await _make_dynamic_narration(tmp_path / "vo.mp4")
    once = await _master(source, tmp_path / "once.mp4", tmp_path)
    twice = await _master(once, tmp_path / "twice.mp4", tmp_path)

    source_lra, _ = await _loudness(source)
    once_lra, _ = await _loudness(once)
    twice_lra, _ = await _loudness(twice)

    # Guard the stimulus itself: if the source ever stops having real dynamics, the
    # rest of this module silently stops testing anything.
    assert source_lra > 5.0, f"stimulus has no macro-dynamics to lose (LRA {source_lra})"
    assert once_lra - twice_lra >= _MIN_RANGE_LOSS_LU, (
        f"second master pass cost only {once_lra - twice_lra:.2f} LU of range "
        f"(once={once_lra}, twice={twice_lra}); expected >= {_MIN_RANGE_LOSS_LU}"
    )


async def test_second_master_pass_leaves_integrated_loudness_unchanged(
    tmp_path: Path,
) -> None:
    """Why nobody caught it: the headline number does not move.

    Also proves the range collapse above is a DYNAMICS change and not simply a quieter
    render — a gain drop would show up here.
    """
    source = await _make_dynamic_narration(tmp_path / "vo.mp4")
    once = await _master(source, tmp_path / "once.mp4", tmp_path)
    twice = await _master(once, tmp_path / "twice.mp4", tmp_path)

    _, once_i = await _loudness(once)
    _, twice_i = await _loudness(twice)

    assert abs(twice_i - once_i) <= _MAX_LOUDNESS_DRIFT_LU, (
        f"integrated loudness moved {twice_i - once_i:+.2f} LU "
        f"(once={once_i}, twice={twice_i}); the double master is supposed to be "
        "invisible on THIS metric"
    )


async def test_extra_encode_generation_alone_does_not_move_range(tmp_path: Path) -> None:
    """Negative control — without it the measurement above is unfalsifiable.

    Re-encodes the mastered file once more at identical codec settings but with no
    master chain. If merely passing through AAC again moved LRA, the -3 LU result would
    be an artefact of the extra generation rather than of the compressor and limiter.
    """
    source = await _make_dynamic_narration(tmp_path / "vo.mp4")
    once = await _master(source, tmp_path / "once.mp4", tmp_path)
    control = await _reencode_without_master(once, tmp_path / "control.mp4")

    once_lra, _ = await _loudness(once)
    control_lra, _ = await _loudness(control)

    assert abs(control_lra - once_lra) <= _MAX_CONTROL_DRIFT_LU, (
        f"a plain re-encode moved LRA by {control_lra - once_lra:+.2f} LU "
        f"(once={once_lra}, control={control_lra}); the rig manufactures deltas and the "
        "double-master result cannot be trusted"
    )
