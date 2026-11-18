"""Wave 10, proven with REAL ffmpeg — an argv assertion proves a flag is
EMITTED, never that ffmpeg HONOURS it. The argv/filtergraph side lives in
``tests/unit/test_av_quality_wave10.py``; these three are the claims that are
about what ffmpeg actually DOES.

3a — the colour triplet must land ON THE DELIVERED FILE. Measured on the
     PRODUCTION ffmpeg build (7.1.5, ``python:3.14-slim`` + Debian trixie
     ``ffmpeg``) with this repo's verbatim ``_video_codec_args``::

         before: color_space=bt709  color_transfer=unknown  color_primaries=unknown
         after:  color_space=bt709  color_transfer=bt709    color_primaries=bt709

     ⚠️ On ffmpeg **4.4** the declarative ``-color_trc``/``-color_primaries``
     already work, so on a 4.4 box this test passes either way — it is a
     HONOURED-not-just-emitted proof, and the fail-without-the-change gate for
     3a is the argv test. On any ffmpeg >= 5 (CI, prod) it is also a regression
     gate. The copy-propagation leg below is version-independent.

3b — the bed-loop seam must not DIP. This one fails without the change on every
     ffmpeg version: it measures the delivered level at the seam centre.

3d — ``print_format=summary`` must actually make loudnorm print the line
     ``_NORMALIZATION_TYPE_RE`` parses, and must not corrupt the output.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import renderer
from tests.render_harness import probe_duration

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg wave10 proof requires ffmpeg + ffprobe",
    ),
]


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _probe_colour(path: Path) -> dict[str, str]:
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_space,color_transfer,color_primaries",
            "-of",
            "default=nw=1",
            str(path),
        ],
    )
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    return fields


# ---------------------------------------------------------------------------
# 3a — all THREE colour fields on the delivered file
# ---------------------------------------------------------------------------


async def test_encode_delivers_all_three_colour_fields(tmp_path: Path) -> None:
    """``colorspace_ok`` reads all three. Only the matrix was landing on prod."""
    settings = _settings()
    out = tmp_path / "tagged.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=1",
            *renderer._video_codec_args(settings),
            str(out),
        ]
    )
    colour = await _probe_colour(out)
    assert colour == {
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }, colour


async def test_colour_tags_survive_the_terminal_audio_only_copy_pass(tmp_path: Path) -> None:
    """The terminal passes (`master_audio` / `apply_postclip_audio` /
    `_encode_delivery_audio`) all stream-copy video alongside an audio re-encode.
    Once the ENCODE tags correctly, those passes carry the triplet through — which
    is why the fix belongs at the encode and the copy branches need nothing."""
    settings = _settings()
    src = tmp_path / "src.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            *renderer._video_codec_args(settings),
            # ALAC, the repo's own `_SEGMENT_AUDIO_CODEC_ARGS` codec — this is the
            # lossless bus `_encode_delivery_audio` exists to convert to AAC.
            "-c:a",
            "alac",
            "-shortest",
            str(src),
        ]
    )
    assert (await _probe_colour(src))["color_primaries"] == "bt709"
    out = tmp_path / "delivery.mp4"
    await renderer._encode_delivery_audio(src, out, settings)
    colour = await _probe_colour(out)
    assert colour == {
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }, colour


# ---------------------------------------------------------------------------
# 3b — the bed-loop seam must not dip
# ---------------------------------------------------------------------------


async def _rms_window(path: Path, start: float, end: float) -> float:
    """Overall RMS (dBFS) of ``[start, end)``.

    ``astats``, not ``volumedetect``: volumedetect reports the mean of |x|
    quantised to 0.1 dB, which is both the wrong statistic for a power question
    and too coarse for a ~0.3 dB claim. A -3 dB seam dip is an RMS/POWER effect.
    """
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"amovie={path},atrim=start={start:.6f}:end={end:.6f},astats=metadata=1:reset=0",
            "-show_entries",
            "frame_tags=lavfi.astats.Overall.RMS_level",
            "-of",
            "csv=p=0:nk=1",
        ]
    )
    values = [line.strip() for line in out.splitlines() if line.strip()]
    assert values, f"astats produced no RMS for {path} [{start}, {end})"
    return float(values[-1])


_SR = 48000
_ASSET = 4.0
_XFADE = 0.25


async def _join_bed_loop(tmp_path: Path, curve: str, out: Path) -> None:
    """Two DECORRELATED sides of a loop seam, joined exactly the way
    ``_bed_loop_filter`` builds one.

    Two sine tones at DIFFERENT frequencies, not two noise seeds: decorrelation is
    the property under test, and orthogonal tones give it exactly while holding a
    rock-steady RMS on both sides of the seam. Pink noise is decorrelated too but
    its own window-to-window level wander (measured ~1.4 dB over 0.5 s windows)
    is the same size as the effect, so it cannot referee a 0.3 dB claim.
    """
    step = _ASSET - _XFADE
    delay = round(step * _SR)
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1000:duration={_ASSET}:sample_rate={_SR},"
            "aformat=channel_layouts=stereo",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1600:duration={_ASSET}:sample_rate={_SR},"
            "aformat=channel_layouts=stereo",
            "-filter_complex",
            f"[0:a]atrim=end={_ASSET:.6f},apad=whole_dur={_ASSET:.6f},"
            f"afade=t=out:curve={curve}:st={step:.6f}:d={_XFADE:.6f}[p0];"
            f"[1:a]atrim=end={_ASSET:.6f},apad=whole_dur={_ASSET:.6f},"
            f"afade=t=in:curve={curve}:st=0:d={_XFADE:.6f},"
            f"adelay=delays={delay}S:all=1[p1];"
            "[p0][p1]amix=inputs=2:duration=longest:normalize=0[out]",
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )


async def test_equal_gain_tri_dips_at_a_loop_seam_and_qsin_does_not(tmp_path: Path) -> None:
    """THE MEASUREMENT. Both sides of a loop seam are the same asset at different
    offsets — decorrelated — so an equal-GAIN (``tri``) sum is -3 dB at the seam
    centre: a periodic duck under the music once per ``asset_seconds``."""
    seam_centre = (_ASSET - _XFADE) + _XFADE / 2
    dips: dict[str, float] = {}
    for curve in ("tri", "qsin"):
        out = tmp_path / f"seam_{curve}.wav"
        await _join_bed_loop(tmp_path, curve, out)
        # Reference = the mean of the single-source steady level on BOTH sides of
        # the seam, so any level difference between the two sides cancels instead
        # of being charged to the curve.
        pre = await _rms_window(out, 1.00, 1.50)
        post = await _rms_window(out, 5.00, 5.50)
        seam = await _rms_window(out, seam_centre - 0.075, seam_centre + 0.075)
        dips[curve] = (pre + post) / 2.0 - seam
    # Equal gain sums two decorrelated sides to 0.707 => -3 dB at the seam centre.
    # Equal power holds the level flat.
    # Measured: tri -> 2.52 dB of dip, qsin -> 0.003 dB. (tri is short of the
    # theoretical 3.01 dB only because the 0.15 s window straddles the ramp rather
    # than sampling its exact centre.)
    assert dips["tri"] > 2.0, dips
    assert dips["qsin"] < 0.5, dips
    assert dips["tri"] - dips["qsin"] > 1.5, dips


async def test_bed_loop_filter_delivers_a_seam_that_does_not_dip(tmp_path: Path) -> None:
    """The same claim through the PRODUCTION filter builder, not a hand-written
    graph: render ``_bed_loop_filter``'s own output and measure its seams.

    ⚠️ The BED FIXTURE is load-bearing and was got wrong once: a plain 1 kHz tone
    looped against itself is CORRELATED at the seam, so equal-gain shows no dip
    and this test passed with ``tri`` — vacuous. A linear chirp (500 -> 1700 Hz
    over the asset) is what a real music bed is like at a loop point: the two
    sides are 3.75 s apart in the sweep, i.e. ~1.1 kHz apart in frequency, hence
    decorrelated — while holding a constant amplitude so the RMS reference does
    not wander the way noise does.
    """
    src = tmp_path / "bed.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"aevalsrc=0.5*sin(2*PI*(500*t+150*t*t)):d={_ASSET}:s={_SR},"
            "aformat=channel_layouts=stereo",
            str(src),
        ]
    )
    graph = renderer._bed_loop_filter(0, 3, _XFADE, "bed", _ASSET)
    out = tmp_path / "looped.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *[arg for _ in range(3) for arg in ("-i", str(src))],
            "-filter_complex",
            graph,
            "-map",
            "[bed]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    step = _ASSET - _XFADE
    assert await probe_duration(out) == pytest.approx(2 * step + _ASSET, abs=0.02)
    steady = await _rms_window(out, 1.00, 1.50)
    for k in (1, 2):
        centre = k * step + _XFADE / 2
        seam = await _rms_window(out, centre - 0.075, centre + 0.075)
        assert steady - seam < 1.2, f"seam {k} dipped {steady - seam:.2f} dB below steady"


# ---------------------------------------------------------------------------
# 3d — the summary must actually print, and must not corrupt the output
# ---------------------------------------------------------------------------


async def test_pass2_loudnorm_summary_is_printed_and_parseable(tmp_path: Path) -> None:
    """``master_normalization_type`` was None on EVERY row ever recorded because
    af_loudnorm defaults to ``print_format=none``. Run the real pass-2 chain and
    prove the line the read-back parses now exists."""
    src = tmp_path / "src.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration=5:sample_rate={_SR},aformat=channel_layouts=stereo",
            str(src),
        ]
    )
    settings = _settings()
    measured = {
        "input_i": "-23.00",
        "input_tp": "-6.00",
        "input_lra": "3.00",
        "input_thresh": "-33.00",
    }
    out = tmp_path / "mastered.wav"
    stderr = await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            f"[0:a]{renderer._loudnorm_filter(settings, measured)}[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        capture_stderr=True,
    )
    assert renderer._NORMALIZATION_TYPE_RE.search(stderr) is not None, stderr[-2000:]
    # The summary goes to STDERR only — the output file is intact.
    assert out.exists() and out.stat().st_size > 0
    assert await probe_duration(out) == pytest.approx(5.0, abs=0.05)


async def test_master_audio_records_the_normalization_type(tmp_path: Path) -> None:
    """End to end through the production pass: the field that was permanently
    None must now carry a real mode."""
    from tests.render_harness import make_test_clip

    src = await make_test_clip(tmp_path / "in.mp4", duration=3.0)
    renderer.reset_master_normalization_type()
    await renderer.master_audio(src, tmp_path / "mastered.mp4", _settings())
    assert renderer.get_master_normalization_type() in {"linear", "dynamic"}
