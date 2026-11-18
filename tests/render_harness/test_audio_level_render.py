"""Real-ffmpeg render-harness tests for per-clip audio level (WS-C).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group. These drive ACTUAL encodes
and interrogate the delivered loudness with ffmpeg ``volumedetect`` — the string
tests in ``tests/test_renderer_sfx.py`` prove the command shape; these prove the
delivered samples.

Per-clip GAIN/MUTE now ride a POST-master, time-ranged ``volume`` pass keyed to
each shot's absolute bus offset (the default-ON loudness master would otherwise
flatten a pre-master gain back to target — the exact defect proven live). So the
level assertions run over the FINAL ``compile_shots`` bus, not the per-shot
segment, and probe the edited shot's window vs its neighbours.

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths (a bare
filesystem path is used as the shot's ``video_url``), so no bucket/presign is
needed.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot, ShotAudio, Transition
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip
from tests.render_harness.harness import (
    probe_mean_volume,
    probe_mean_volume_window,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _compile(shots: list[Shot], tmp_path: Path, *, master: bool = True) -> Path:
    """Run a full compile into ``tmp_path`` and return the final bus."""
    settings = get_settings().model_copy(
        update={"render_fps": 25, "master_enabled": master, "work_dir": str(tmp_path)}
    )
    final_path, _, _, _, _ = await renderer.compile_shots(shots, settings)
    return final_path


async def test_gain_db_lowers_mean_volume(tmp_path: Path) -> None:
    """gain_db=-20 survives the master: on the final bus the edited shot's window
    reads ~20 dB below an unedited neighbour's window (pre-master it would be
    flattened back to target by loudnorm)."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="a", seconds=2.0, video_url=str(src)),
            Shot(id="b", seconds=2.0, video_url=str(src), audio=ShotAudio(gain_db=-20)),
        ],
        tmp_path,
    )
    # Shot a ≈ [0,2), shot b ≈ [2,4); probe insets clear of segment boundaries.
    base_mean = await probe_mean_volume_window(final, 0.3, 1.7)
    gain_mean = await probe_mean_volume_window(final, 2.3, 3.7)
    assert gain_mean == pytest.approx(base_mean - 20, abs=3.0)


async def test_mute_silences_audio(tmp_path: Path) -> None:
    """mute=True drops the edited shot's window on the final bus to the silence
    floor, far below an unedited neighbour."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="a", seconds=2.0, video_url=str(src)),
            Shot(id="b", seconds=2.0, video_url=str(src), audio=ShotAudio(mute=True)),
        ],
        tmp_path,
    )
    base_mean = await probe_mean_volume_window(final, 0.3, 1.7)
    mute_mean = await probe_mean_volume_window(final, 2.3, 3.7)
    assert mute_mean < base_mean - 40


async def test_gain_db_minus_20_on_middle_shot_of_three(tmp_path: Path) -> None:
    """The live repro: a 3-shot compile with gain_db=-20 on shot 2 ONLY. On the
    final mastered bus shot 2's window sits ~20 dB below BOTH neighbours — the
    exact thing that was flattened by the pre-master master chain in production."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="s0", seconds=2.0, video_url=str(src)),
            Shot(id="s1", seconds=2.0, video_url=str(src), audio=ShotAudio(gain_db=-20)),
            Shot(id="s2", seconds=2.0, video_url=str(src)),
        ],
        tmp_path,
    )
    left = await probe_mean_volume_window(final, 0.3, 1.7)  # shot 0 ≈ [0,2)
    mid = await probe_mean_volume_window(final, 2.3, 3.7)  # shot 1 ≈ [2,4)
    right = await probe_mean_volume_window(final, 4.3, 5.7)  # shot 2 ≈ [4,6)
    assert mid == pytest.approx(left - 20, abs=3.0)
    assert mid == pytest.approx(right - 20, abs=3.0)
    # Neighbours were untouched, so they track each other.
    assert left == pytest.approx(right, abs=3.0)


async def test_mute_middle_shot_of_three_is_silent_in_range(tmp_path: Path) -> None:
    """mute on the middle shot of a 3-shot compile silences ONLY its window; the
    neighbours stay audible on the final bus."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="s0", seconds=2.0, video_url=str(src)),
            Shot(id="s1", seconds=2.0, video_url=str(src), audio=ShotAudio(mute=True)),
            Shot(id="s2", seconds=2.0, video_url=str(src)),
        ],
        tmp_path,
    )
    left = await probe_mean_volume_window(final, 0.3, 1.7)
    mid = await probe_mean_volume_window(final, 2.3, 3.7)
    right = await probe_mean_volume_window(final, 4.3, 5.7)
    assert mid < left - 40
    assert mid < right - 40


# --------------------------------------------------------------------------
# Crossfade: the edit must land on the edited clip and NOT damage its neighbours
# --------------------------------------------------------------------------
# 3 shots x 3.0s joined by 1.0s crossfades. Probed segments run ~3.02s, so the
# raw bus is ≈ [(0,3.02), (2.02,5.04), (4.04,7.06)] and the SOLO regions are
# s0 [0,2.02)  s1 [3.02,4.04)  s2 [5.04,7.06). The 1s blends are [2.02,3.02) and
# [4.04,5.04) — a naive raw-offset enable would attenuate BOTH of them, damaging
# s0's fade-out tail and s2's fade-in, which the user never edited.
_S0_SOLO = (0.3, 1.7)
_S1_SOLO = (3.3, 3.8)
_S2_SOLO = (5.4, 6.9)
_BLEND_S0_S1 = (2.2, 2.9)  # s0's fade-out tail — MUST be untouched by an s1 edit


def _xfade_shots(src: Path, middle: ShotAudio | None) -> list[Shot]:
    return [
        Shot(id="s0", seconds=3.0, video_url=str(src)),
        Shot(
            id="s1",
            seconds=3.0,
            video_url=str(src),
            audio=middle,
            transition=Transition(type="fade", duration=1.0),
        ),
        Shot(
            id="s2",
            seconds=3.0,
            video_url=str(src),
            transition=Transition(type="fade", duration=1.0),
        ),
    ]


async def test_xfade_gain_on_middle_shot_does_not_damage_neighbours(tmp_path: Path) -> None:
    """gain_db=-20 on the MIDDLE shot of a crossfaded 3-shot compile lands ~20 dB
    down on that shot's SOLO region, while the un-edited neighbours — INCLUDING
    the crossfade blend that carries s0's fade-out tail — are bit-for-bit level
    with a no-edit control render.

    The blend assertion is the regression test for the overlap-bleed bug: keyed to
    the RAW bus offset (not the solo window), s1's enable would span [2.02,5.04)
    and attenuate s0's tail and s2's onset. It FAILS without ``_solo_windows``.
    """
    src = await make_test_clip(tmp_path / "src.mp4", duration=3.0, fps=25)
    control = await _compile(_xfade_shots(src, None), tmp_path / "ctl")
    edited = await _compile(_xfade_shots(src, ShotAudio(gain_db=-20)), tmp_path / "edt")

    # The edited clip's solo region is ~20 dB down.
    assert await probe_mean_volume_window(edited, *_S1_SOLO) == pytest.approx(
        await probe_mean_volume_window(control, *_S1_SOLO) - 20, abs=3.0
    )
    # Every un-edited region is UNCHANGED vs the control — no bleed.
    for window in (_S0_SOLO, _S2_SOLO, _BLEND_S0_S1):
        assert await probe_mean_volume_window(edited, *window) == pytest.approx(
            await probe_mean_volume_window(control, *window), abs=1.5
        ), f"un-edited region {window} was damaged by the neighbouring shot's gain"


async def test_xfade_mute_on_middle_shot_does_not_damage_neighbours(tmp_path: Path) -> None:
    """As above for mute: the middle shot's solo region goes silent while the
    crossfade blend carrying s0's tail keeps its control level."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=3.0, fps=25)
    control = await _compile(_xfade_shots(src, None), tmp_path / "ctl")
    edited = await _compile(_xfade_shots(src, ShotAudio(mute=True)), tmp_path / "edt")

    assert await probe_mean_volume_window(edited, *_S1_SOLO) < (
        await probe_mean_volume_window(control, *_S1_SOLO) - 40
    )
    for window in (_S0_SOLO, _S2_SOLO, _BLEND_S0_S1):
        assert await probe_mean_volume_window(edited, *window) == pytest.approx(
            await probe_mean_volume_window(control, *window), abs=1.5
        ), f"un-edited region {window} was damaged by the neighbouring shot's mute"


# --------------------------------------------------------------------------
# Master OFF: the per-clip pass must STILL apply
# --------------------------------------------------------------------------


async def test_gain_applies_with_master_disabled(tmp_path: Path) -> None:
    """master_enabled=False → no loudness bus at all, but the per-clip pass still
    runs over the raw concat and the delivered audio still reflects the edit."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="s0", seconds=2.0, video_url=str(src)),
            Shot(id="s1", seconds=2.0, video_url=str(src), audio=ShotAudio(gain_db=-20)),
            Shot(id="s2", seconds=2.0, video_url=str(src)),
        ],
        tmp_path,
        master=False,
    )
    assert final.name == "postclip.mp4"  # the postclip pass ran without a master
    left = await probe_mean_volume_window(final, 0.3, 1.7)
    mid = await probe_mean_volume_window(final, 2.3, 3.7)
    right = await probe_mean_volume_window(final, 4.3, 5.7)
    assert mid == pytest.approx(left - 20, abs=3.0)
    assert mid == pytest.approx(right - 20, abs=3.0)


async def test_mute_applies_with_master_disabled(tmp_path: Path) -> None:
    """mute with the master OFF still silences the shot on the delivered bus."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    final = await _compile(
        [
            Shot(id="s0", seconds=2.0, video_url=str(src)),
            Shot(id="s1", seconds=2.0, video_url=str(src), audio=ShotAudio(mute=True)),
        ],
        tmp_path,
        master=False,
    )
    left = await probe_mean_volume_window(final, 0.3, 1.7)
    mid = await probe_mean_volume_window(final, 2.3, 3.7)
    assert mid < left - 40


async def _channels(path: Path) -> int:
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return int(out.strip())


async def _make_stereo_clip(path: Path, *, duration: float = 1.0, fps: int = 25) -> Path:
    """A stand-in for a real CONCAT output: 48 kHz **stereo**.

    ``make_test_clip`` muxes a MONO sine, which no segment this compiler produces
    ever is — ``_A48`` (``aformat=channel_layouts=stereo``) and the ``-ac 2`` in
    both codec-arg constants make every segment, and therefore every concat
    output, 48 kHz stereo. Feeding a mono file to a pass that emits ``-ac 2``
    measures ffmpeg's mono→stereo power-preserving upmix (-3 dB per channel) and
    reports it as a level change the pass does not actually make on any real bus.
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
            f"testsrc=duration={duration}:size=320x240:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration}",
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


async def _video_frame_md5(path: Path) -> str:
    """Per-frame MD5 body of the VIDEO stream only."""
    out = await renderer._run(["ffmpeg", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"])
    return "\n".join(ln for ln in out.splitlines() if ln and not ln.startswith("#"))


async def test_finalize_audio_noop_applies_only_the_delivery_transcode(tmp_path: Path) -> None:
    """No per-clip edits + master OFF: ``_finalize_audio`` applies NO audio
    PROCESSING — no bed, no loudnorm, no per-clip volume, and the picture is
    stream-copied untouched.

    This test used to assert the stronger ``result == concat`` (identity object).
    That invariant was deliberately retired: the concat output is now the lossless
    ALAC intermediate (see ``_SEGMENT_AUDIO_CODEC_ARGS`` — segments must carry no
    encoder delay or the copy-concat inserts ~24 ms at every cut), and ALAC is
    never a deliverable. The one remaining pass on this branch is the plain
    delivery transcode to the unchanged AAC 192k/48 kHz/stereo contract, which is
    what is asserted instead. "No processing" is the property that mattered;
    "returns the same object" was how it happened to be implemented.
    """
    concat = await _make_stereo_clip(tmp_path / "concat.mp4", duration=1.0, fps=25)
    settings = get_settings().model_copy(
        update={"master_enabled": False, "work_dir": str(tmp_path)}
    )
    before = _sha256(concat)
    result, music_bed = await renderer._finalize_audio(
        concat,
        tmp_path,
        1.0,
        None,
        [],
        settings,
        shots=[Shot(id="a", seconds=1.0, video_url="x", audio=ShotAudio())],
        shot_offsets=[(0.0, 1.0)],
    )
    assert result.name == "delivery.mp4"
    # The input is left exactly as it was — this pass produces a new file.
    assert _sha256(concat) == before
    # Video stream-copied: identical per-frame hashes, no re-encode. Probed on
    # the VIDEO stream only — a whole-file ``framemd5`` also hashes the audio
    # frames, which this pass is supposed to change.
    assert await _video_frame_md5(result) == await _video_frame_md5(concat)
    # Audio untouched in LEVEL and in CHANNEL COUNT — a loudnorm/volume pass
    # would move the first, and an unwanted up/downmix the second. Both are
    # measured on a STEREO input because that is what a real concat output is
    # (see ``_make_stereo_clip``); the delivery pass emits ``-ac 2``, the same
    # count it received, so it is a pure codec change.
    assert abs(await probe_mean_volume(result) - await probe_mean_volume(concat)) < 0.5
    assert await _channels(result) == await _channels(concat) == 2
    # No bed on this path ⇒ nothing measured ⇒ no report.
    assert music_bed is None


async def test_default_audio_is_byte_identical(tmp_path: Path) -> None:
    """A shot with audio=None, audio=ShotAudio() (all-default), and audio omitted
    all render to byte-identical segments — the non-negotiable no-op invariant.
    (Gain/mute no longer touch the segment, so a gain-only shot is byte-identical
    at the segment layer too; the level change is applied post-master downstream.)
    """
    src = await make_test_clip(tmp_path / "src.mp4", duration=2.0, fps=25)
    settings = get_settings().model_copy(update={"render_fps": 25})

    seg_baseline = tmp_path / "baseline.mp4"
    seg_none = tmp_path / "none.mp4"
    seg_default = tmp_path / "default.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=2.0, video_url=str(src)), seg_baseline, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(id="a", seconds=2.0, video_url=str(src), audio=None),
            seg_none,
            settings,
            tmp_path,
            http,
        )
        await renderer.render_shot(
            Shot(id="a", seconds=2.0, video_url=str(src), audio=ShotAudio()),
            seg_default,
            settings,
            tmp_path,
            http,
        )

    baseline = _sha256(seg_baseline)
    assert _sha256(seg_none) == baseline
    assert _sha256(seg_default) == baseline
