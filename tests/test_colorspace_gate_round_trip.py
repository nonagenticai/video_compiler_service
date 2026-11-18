"""The colorspace gate must be satisfiable by the tag the encoder actually writes.

REGRESSION (real ffmpeg, no mocks). ``post_mux_verify`` used to compare the
``render_colorspace`` NAME against ffprobe's ``color_space`` field, which carries
the AVColorSpace MATRIX name. Those two vocabularies coincide for exactly one of
the four legal values:

    render_colorspace  matrix encoded  ffprobe color_space  old gate
    bt709              bt709           bt709                pass
    bt601              smpte170m       smpte170m            FAIL (wrongly)
    bt2020             bt2020nc        bt2020nc             FAIL (wrongly)
    srgb               bt709           bt709                FAIL (wrongly)

So a deck compiled with ``tags={"colorspace": "bt2020"}`` was tagged CORRECTLY and
then reported ``colorspace_ok=False, rerun_stages=["colorspace"], gate_pass=False``
on a compliant file — and re-running re-emitted the identical tag, so it could
never clear. These tests encode a real file with ``renderer._video_codec_args`` —
the ONE function every production video encode goes through — and drive the real
gate over it, so they measure the round trip rather than a hand-written
expectation.

⛔ They used to encode with ``mastering.colorspace_tag_args`` ALONE. That is not
what production runs, and on production's ffmpeg it does not tag the file.
Measured on 7.1.5 (the deployed image)::

    ffmpeg -f lavfi -i testsrc=... -c:v libx264 -pix_fmt yuv420p \
        -color_primaries bt709 -color_trc bt709 -colorspace bt709 a.mp4
    => color_space=bt709  color_transfer=unknown  color_primaries=unknown

    ... same, plus -bsf:v h264_metadata=colour_primaries=1:\
        transfer_characteristics=1:matrix_coefficients=1 b.mp4
    => bt709 / bt709 / bt709

ffmpeg >= 5 lets the (unspecified) filtergraph frame properties override the
output ``-color_primaries``/``-color_trc`` options; only the matrix survives. On
4.4 and 6.1.1 the bare options tag all three, so the stale shape passed locally
and in CI and failed 8 tests on 7.1.5 while production — whose
``_video_codec_args`` also emits ``colorspace_bsf_args`` — delivered correctly
tagged masters. Encoding through ``_video_codec_args`` makes these tests fail the
day the production encode stops tagging, on every ffmpeg, and not before.

They must NOT be satisfiable by loosening the comparison: the last two tests prove
a genuinely mis-tagged master and an untagged master still fail.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from src.config import Settings, get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg round trip requires ffmpeg + ffprobe",
    ),
]

# Every legal value of ``render_colorspace``. bt709 is the fleet default; the other
# three are reachable today via ``tags={"colorspace": ...}`` / MasteringProfile.
LEGAL_COLORSPACES = ("bt709", "bt601", "bt2020", "srgb")


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _encode(dest: Path, colorspace: str) -> None:
    """Encode a tiny textured clip with PRODUCTION's encode args for ``colorspace``.

    ``colorspace="none"`` is the untagged control: ``_video_codec_args`` then emits
    neither the ``-color_*`` options nor the ``h264_metadata`` stamp.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=1",
            *renderer._video_codec_args(_settings(render_colorspace=colorspace)),
            str(dest),
        ]
    )


# ---------------------------------------------------------------------------
# The round trip: every legal value must be able to PASS its own gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("colorspace", LEGAL_COLORSPACES)
async def test_probe_round_trips_the_encoded_triplet(colorspace: str, tmp_path: Path) -> None:
    """ffprobe reports back exactly the (primaries, transfer, matrix) that was written.

    This is the ground truth the gate rests on. Before the fix ``_probe_colorspace``
    returned only the matrix string, so this assertion could not even be expressed.
    """
    dest = tmp_path / f"{colorspace}.mp4"
    await _encode(dest, colorspace)

    assert await mastering._probe_colorspace(dest) == mastering._COLORSPACE_TRIPLETS[colorspace]


@pytest.mark.asyncio
@pytest.mark.parametrize("colorspace", LEGAL_COLORSPACES)
async def test_gate_passes_a_correctly_tagged_master(colorspace: str, tmp_path: Path) -> None:
    """A master tagged with its own render_colorspace clears the colorspace gate.

    Before the fix this passed for bt709 ONLY; bt601/bt2020/srgb each reported
    ``colorspace_ok=False`` on a correctly tagged file.
    """
    settings = _settings(render_colorspace=colorspace)
    dest = tmp_path / f"{colorspace}.mp4"
    await _encode(dest, colorspace)

    report = await mastering.post_mux_verify(
        dest, settings, loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0}
    )
    assert report["colorspace_ok"] is True
    assert "colorspace" not in report["rerun_stages"]
    # This master is video-only: the audio packet-gap advisory is UNMEASURED
    # (``None``), never "zero gaps" — and probing it must not raise or warn.
    assert report["audio_pts_gap_count"] is None
    assert report["audio_pts_gap_ms_total"] is None
    with capture_logs() as logs:
        assert await mastering.audio_pts_gap_report(dest) == (None, None)
    assert "mastering.audio_pts_gaps.failed" not in [entry["event"] for entry in logs]


# ---------------------------------------------------------------------------
# NOT VACUOUS: the gate must still go red on a real mis-tag.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "tagged_as"),
    [
        ("bt709", "bt2020"),
        ("bt2020", "bt709"),
        ("bt601", "bt2020"),
        ("bt2020", "bt601"),
    ],
)
async def test_gate_fails_a_genuinely_mistagged_master(
    requested: str, tagged_as: str, tmp_path: Path
) -> None:
    """A file carrying a DIFFERENT colorspace's triplet than the one requested fails.

    This is the anti-vacuity control: the fix widens WHAT is compared, never
    WHETHER a mismatch counts.
    """
    settings = _settings(render_colorspace=requested)
    dest = tmp_path / f"{requested}-as-{tagged_as}.mp4"
    await _encode(dest, tagged_as)
    # The file really carries the OTHER triplet in full — not a half-tagged file
    # that fails the gate for the wrong reason.
    assert await mastering._probe_colorspace(dest) == mastering._COLORSPACE_TRIPLETS[tagged_as]

    report = await mastering.post_mux_verify(dest, settings, loudness={})
    assert report["colorspace_ok"] is False
    assert "colorspace" in report["rerun_stages"]
    assert report["gate_pass"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("colorspace", LEGAL_COLORSPACES)
async def test_gate_fails_an_untagged_master(colorspace: str, tmp_path: Path) -> None:
    """An untagged master (ffprobe reports "unknown") fails for EVERY legal value.

    The pre-master behaviour the tagging step exists to replace must never read as
    compliant — including for bt709, where the old name-vs-matrix comparison
    happened to work.
    """
    settings = _settings(render_colorspace=colorspace)
    dest = tmp_path / f"untagged-{colorspace}.mp4"
    await _encode(dest, "none")
    assert await mastering._probe_colorspace(dest) == ("unknown", "unknown", "unknown")

    report = await mastering.post_mux_verify(dest, settings, loudness={})
    assert report["colorspace_ok"] is False
    assert "colorspace" in report["rerun_stages"]


# ---------------------------------------------------------------------------
# The srgb / bt709 collision, decided EXPLICITLY.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_srgb_and_bt709_are_distinguished_by_transfer(tmp_path: Path) -> None:
    """srgb and bt709 share matrix AND primaries; only the transfer separates them.

    A matrix-only comparison cannot tell an sRGB deck from a Rec.709 one, so
    "compare the matrix" would have made srgb pass by ALIASING it to bt709 — a
    silently vacuous check for one of the four legal values. Comparing the full
    triplet keeps them apart: an srgb-tagged master requested as bt709 (and the
    reverse) is a real mismatch and is reported as one.
    """
    srgb = mastering._COLORSPACE_TRIPLETS["srgb"]
    bt709 = mastering._COLORSPACE_TRIPLETS["bt709"]
    # The premise: identical primaries + matrix, different transfer.
    assert (srgb[0], srgb[2]) == (bt709[0], bt709[2])
    assert srgb[1] != bt709[1]

    srgb_file = tmp_path / "srgb.mp4"
    await _encode(srgb_file, "srgb")

    # Real files, real probe: the matrix alone is identical…
    srgb_probed = await mastering._probe_colorspace(srgb_file)
    assert srgb_probed == srgb
    # …but the gate still separates them, in both directions.
    as_bt709 = await mastering.post_mux_verify(
        srgb_file, _settings(render_colorspace="bt709"), loudness={}
    )
    assert as_bt709["colorspace_ok"] is False

    bt709_file = tmp_path / "bt709.mp4"
    await _encode(bt709_file, "bt709")
    as_srgb = await mastering.post_mux_verify(
        bt709_file, _settings(render_colorspace="srgb"), loudness={}
    )
    assert as_srgb["colorspace_ok"] is False


# ---------------------------------------------------------------------------
# The passes downstream of the encode must not lose the tag.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("colorspace", LEGAL_COLORSPACES)
async def test_stream_copy_passes_keep_the_encoded_triplet(colorspace: str, tmp_path: Path) -> None:
    """The triplet survives every stream-copy shape between encode and delivery.

    After the per-shot encode, video is only ever COPIED: the ``-f concat -c copy``
    join, the audio-master pass (``_grade_video_args`` with no grade: ``-c:v copy``
    riding with an audio ``-filter_complex``), and the plain ``-c copy`` remux the
    no-audio / caption branches use. If any of them dropped primaries/transfer the
    delivered master would fail the gate however well the encode tagged it.
    """
    settings = _settings(render_colorspace=colorspace)
    expected = mastering._COLORSPACE_TRIPLETS[colorspace]
    shot = tmp_path / "shot.mp4"
    await _encode(shot, colorspace)

    listing = tmp_path / "list.txt"
    listing.write_text(f"file '{shot}'\nfile '{shot}'\n")
    joined = tmp_path / "joined.mp4"
    ff = ["ffmpeg", "-y", "-v", "error"]
    faststart = ["-c", "copy", "-movflags", "+faststart"]
    await renderer._run(
        [*ff, "-f", "concat", "-safe", "0", "-i", str(listing), *faststart, str(joined)]
    )
    assert await mastering._probe_colorspace(joined) == expected

    mastered = tmp_path / "mastered.mp4"
    await renderer._run(
        [
            *ff,
            *("-i", str(joined), "-f", "lavfi", "-i", "sine=d=2"),
            *("-filter_complex", "[1:a]volume=0.5[a]"),
            *renderer._grade_video_args("", settings),
            *("-map", "[a]", "-c:a", "aac", "-shortest", str(mastered)),
        ]
    )
    assert await mastering._probe_colorspace(mastered) == expected

    remuxed = tmp_path / "remuxed.mp4"
    await renderer._run([*ff, "-i", str(mastered), *faststart, str(remuxed)])
    assert await mastering._probe_colorspace(remuxed) == expected
