"""A delivered master must not outlive its picture (real ffmpeg, no mocks).

THE DEFECT, measured on delivered artifacts 2026-09-10 (ffprobe, per stream):

    artifact       video      audio      missing picture
    lesson         738.567s   778.196s   -39.6s
    lesson         574.800s   592.327s   -17.5s
    course master  1352.996s  1370.544s  -17.5s
    lesson         135.513s   144.293s    -8.8s
    demo_video      23.800s    36.300s   -12.5s
    promo           58.200s    58.361s   intact (AAC tail padding only)

Every one of those reached ``completed`` and shipped. Two independent causes,
one test file:

PART 1 — the picture ends early. ``_ffprobe_duration`` feeds the freeze-pad
guard, which pads a clip only when ``0.0 < clip_dur < duration``. PR #202 made
the ``v:0`` stream duration win over the container (which is the MAX over all
streams, i.e. the audio). It left one shape open: a v:0 stream that EXISTS but
carries NO ``duration`` tag — every Matroska/WebM source, plus several
fragmented-MP4 writers. On that shape there is no stream number to win with, the
container's audio length comes back as the picture length, an audio-padded source
reads as "already fills its narration", and NO ``tpad`` clone is emitted.

PART 2 — nothing looked. ``probe_output_duration`` reduces to ONE scalar via
``min(stream, container)``, and on a truncated master that scalar AGREES with the
plan. The audio stream length was never probed and never compared, so no gate,
report or consumer could see the gap.

These tests encode real files with real ffmpeg and drive the real functions, so
they measure the round trip rather than a hand-written expectation. They are NOT
satisfiable by loosening the comparison: ``test_aac_tail_padding_is_not_rejected``
proves a legitimate 0.161s-padded master still passes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg A/V coverage round trip requires ffmpeg + ffprobe",
    ),
]


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _encode_av(
    dest: Path,
    *,
    video_seconds: float,
    audio_seconds: float,
    container: str,
    rate: int = 25,
) -> None:
    """A clip whose AUDIO is longer than its PICTURE — the shape every row of the
    table above has. ``-c:v copy`` is not used here; the length asymmetry comes
    from the two lavfi sources, exactly as it does in production where the audio
    bed is generated to the planned total over a stream-copied short video."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate={rate}:duration={video_seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={audio_seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-t",
            f"{audio_seconds}",
            "-f",
            container,
            str(dest),
        ]
    )


# ---------------------------------------------------------------------------
# PART 1 — the picture length must come from the PICTURE
# ---------------------------------------------------------------------------


async def test_untagged_video_stream_probes_as_the_picture_not_the_audio(
    tmp_path: Path,
) -> None:
    """A 6s picture under a 12s narration must probe as ~6s, not ~12s.

    Matroska writes no stream-level ``duration``, so this file reports
    ``streams[0] == {}`` and ``format.duration == 12.02``. Before the packet-
    timestamp fallback, ``_ffprobe_duration`` returned that 12.02 — the AUDIO
    length — as the picture length.
    """
    clip = tmp_path / "audio_longer.mkv"
    await _encode_av(clip, video_seconds=6.0, audio_seconds=12.0, container="matroska")

    # Precondition, asserted rather than assumed: this container really does omit
    # the stream duration tag, otherwise the test would pass for the wrong reason.
    probed = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(clip),
        ]
    )
    assert '"duration"' not in probed.split('"format"')[0], (
        "fixture no longer reproduces the untagged-video-stream shape"
    )

    measured = await renderer._ffprobe_duration(clip)
    assert 5.5 < measured < 7.0, f"expected the ~6s PICTURE length, got {measured}"


async def test_audio_longer_source_emits_the_freeze_pad(tmp_path: Path) -> None:
    """The regression that produced every row of the table.

    A 6s picture under a 12s narration line must read as "picture is short" and
    get its ``tpad=stop_mode=clone`` extension. Reading the container instead made
    it read as "already fills its narration", emitting nothing, and the output-side
    ``-t`` then held the segment at the planned length over a picture that stopped
    at 6s.
    """
    clip = tmp_path / "audio_longer.mkv"
    await _encode_av(clip, video_seconds=6.0, audio_seconds=12.0, container="matroska")

    narration_seconds = 12.0
    clip_dur = await renderer._playable_clip_duration(clip, 0.0, None)
    assert clip_dur < narration_seconds, (
        f"playable picture {clip_dur} must under-run the {narration_seconds}s narration "
        "for the freeze-pad guard to have anything to decide"
    )

    suffix = renderer._freeze_pad_suffix(
        narration_driven=True,
        clip_dur=clip_dur,
        duration=narration_seconds,
        settings=_settings(),
    )
    assert "tpad=stop_mode=clone" in suffix, (
        "an audio-padded source must still emit the clone-pad; got an EMPTY suffix, "
        "which is the exact silent truncation this fixes"
    )


async def test_vfr_source_is_measured_from_timestamps_not_frame_count(
    tmp_path: Path,
) -> None:
    """A VFR picture must measure by TIMESTAMP, never ``nb_frames / fps``.

    This fixture holds 36 packets spanning 5.866s while reporting
    ``r_frame_rate=30`` — a frame-count derivation yields 36/30 = 1.2s, off by 4.9
    seconds, which would under-report every picture and over-pad every shot. The
    fleet really is VFR (an observed capture: ``avg_frame_rate`` 5.96 against
    ``r_frame_rate`` 30).
    """
    clip = tmp_path / "vfr.mkv"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=6",
            "-vf",
            "select='not(mod(n,5))'",
            "-fps_mode:v",
            "passthrough",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(clip),
        ]
    )

    # Drive the TIMESTAMP measurement itself, not `_ffprobe_duration`. This
    # container's `format.duration` happens to equal the true picture span (there is
    # no audio stream to inflate it), so probing through the resolver would pass on
    # the container fallback alone and prove nothing about how the picture is
    # measured. `_video_stream_packet_end` is the function under test here.
    packet_measured = await renderer._video_stream_packet_end(clip, 0.0)
    assert 5.5 < packet_measured < 6.2, f"expected the ~5.87s timestamp span, got {packet_measured}"
    # The number a frame-count derivation would have produced (36 packets at the
    # declared r_frame_rate of 30 = 1.2s), named so a future "optimisation" back to
    # nb_frames/fps fails here loudly instead of silently under-reporting.
    assert packet_measured > 3.0, (
        f"{packet_measured} is in frame-count-over-rate territory (36/30 = 1.2s), "
        "not a timestamp measurement"
    )
    # And the whole-file probe agrees.
    assert 5.5 < await renderer._ffprobe_duration(clip) < 6.2


# ---------------------------------------------------------------------------
# PART 2 — a master must not outlive its picture
# ---------------------------------------------------------------------------


async def test_truncated_master_is_rejected_in_advisory_mode(tmp_path: Path) -> None:
    """A master whose audio outruns its picture must FAIL the compile, and must do
    so under the DEFAULT ``mastering_gate_mode='advisory'``.

    This is the assertion that makes the gate a gate. ``post_mux_verify`` runs in
    ``compile.py`` immediately BEFORE the upload stage, so raising is what actually
    prevents delivery; a check that only cleared ``gate_pass`` would be discarded in
    advisory mode and the ragged master would ship exactly as it has been shipping.
    """
    master = tmp_path / "truncated.mp4"
    await _encode_av(master, video_seconds=6.0, audio_seconds=12.0, container="mp4")

    settings = _settings(mastering_gate_mode="advisory")
    assert not mastering.gate_enforced(settings), "this test is only meaningful in advisory mode"

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(master, settings)

    stages = [v.stage for v in excinfo.value.violations]
    assert "av_coverage" in stages, f"expected an av_coverage violation, got {stages}"
    violation = next(v for v in excinfo.value.violations if v.stage == "av_coverage")
    assert violation.value is not None and violation.value > 5.0, (
        f"the reported gap {violation.value} should be ~6s of missing picture"
    )
    assert violation.limit == mastering.AV_COVERAGE_TOLERANCE_SECONDS


async def test_aac_tail_padding_is_not_rejected(tmp_path: Path) -> None:
    """THE FALSE-POSITIVE GUARD, and the one that decides whether this gate is
    usable at all.

    A correct master really does end a little audio-long: AAC pads its final
    1024-sample frame and the mux rounds to the container timebase. The delivered
    promo measured 58.200s video against 58.361s audio — 0.161s — with nothing
    wrong with it. This fixture reproduces that gap. Gating at 0.0 would reject
    every AAC delivery in the fleet.
    """
    master = tmp_path / "promo_like.mp4"
    await _encode_av(master, video_seconds=6.0, audio_seconds=6.161, container="mp4")

    video_seconds, audio_seconds = await mastering.av_stream_seconds(master)
    assert video_seconds is not None and audio_seconds is not None
    gap = audio_seconds - video_seconds
    assert 0.0 < gap < mastering.AV_COVERAGE_TOLERANCE_SECONDS, (
        f"fixture must reproduce a SMALL legitimate tail, measured {gap}"
    )

    # No raise, and no av_coverage verdict on the report.
    report = await mastering.post_mux_verify(master, _settings(mastering_gate_mode="advisory"))
    assert "av_coverage" not in report["rerun_stages"], (
        f"a legitimate {gap:.3f}s AAC tail was rejected as a truncated master"
    )
    assert report["av_gap_seconds"] is not None
    assert "av_gap_seconds" not in report["unmeasured"]
