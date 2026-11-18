"""Real-ffmpeg render-harness tests for the music-vs-narration duck (A6-v2).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` like
its siblings. ``tests/test_renderer_sfx.py`` proves the filter STRING; this proves
the delivered SAMPLES, through the public ``overlay_audio_bed`` mix.

Acceptance (great-video spec 06-sound-music §T3 / criterion #3): "in any VO gap
> 400 ms, music RMS rises within one release." The old hardcoded duck
(``threshold=0.125:attack=200:release=1000``) never delivered it: 0.125 is an
ABSOLUTE linear key level that the premastered narration bus does not cross, so
the compressor barely opened, and a 1000 ms release kept the bed pinned through
the pauses anyway.

**The duck's key is an absolute threshold, so it is calibrated against a
production invariant: ALL voiceover on this bus is premastered to -18 LUFS** —
TTS premasters to it, and uploaded VO is normalised to the same target. These
tests therefore drive the mix with a -18 LUFS speech-band narration, not a bare
tone: a sine at an arbitrary gain has a detected level unrelated to speech's, and
keying a real sidechain off it proves nothing about production. A VO-RELATIVE
(measured) key that would make the duck level-independent is the documented
follow-up (spec T1); this design does not deliver level-independence and these
tests deliberately do not assert it.

Method: narration is speech-band pink noise (300-3400 Hz) with an 800 ms silent
gap, loudnorm'd to -18 LUFS. The bed is an 8 kHz sine — deliberately OUT of the
speech band — so ``probe_band_mean_volume_window`` (2x 6 kHz high-pass) can
isolate the BED's own delivered level inside the final mix. ``rise`` is the bed's
level in the gap minus its level under speech. The loudness master is OFF for the
render under test: its compressor + loudnorm would squash exactly the dynamics
being measured.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import MusicBed
from src.config import get_settings
from src.services import renderer
from tests.render_harness import (
    make_bed_tone,
    make_premastered_narration_clip,
    probe_band_mean_volume_window,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_DURATION = 5.0
_GAP = (2.0, 2.8)  # 800 ms of silence — comfortably over the spec's 400 ms

# Steady-state duck (the attack has long settled) vs the tail of the gap (one
# 300 ms release has elapsed, so a breathing bed is back at its resting level).
_SPEECH_WINDOW = (1.0, 1.9)
_GAP_WINDOW = (2.45, 2.75)


async def _render_with_bed(tmp_path: Path, bed: MusicBed) -> Path:
    """Render premastered narration-with-a-gap under ``bed`` through the REAL
    public duck path (``overlay_audio_bed``), master OFF."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = await make_premastered_narration_clip(
        tmp_path / "narration.mp4", duration=_DURATION, gap=_GAP
    )
    music = await make_bed_tone(tmp_path / "bed.wav", duration=_DURATION)
    settings = get_settings().model_copy(
        update={"master_enabled": False, "work_dir": str(tmp_path)}
    )
    out = tmp_path / "final.mp4"
    final, _report = await renderer.overlay_audio_bed(
        video, out, _DURATION, (music, bed), [], settings=settings
    )
    return final


async def _bed_rise_db(final: Path) -> float:
    """How far the BED rises in the VO gap vs under narration, in dB."""
    under_speech = await probe_band_mean_volume_window(final, *_SPEECH_WINDOW)
    in_gap = await probe_band_mean_volume_window(final, *_GAP_WINDOW)
    return in_gap - under_speech


async def test_music_bed_breathes_in_a_vo_gap(tmp_path: Path) -> None:
    """Acceptance #3, on real samples: with the SHIPPED defaults (threshold 0.03 /
    attack 30 ms / release 300 ms — no duck knobs sent at all), the bed's level
    inside an 800 ms VO gap swells well back up vs its level under narration.

    Measured on a real render: -44.5 dB under speech, -32.1 dB in the gap -> +12.4
    dB. The 6 dB bar keeps margin so codec/loudnorm jitter can't make it flaky.

    ``gain_db`` and the fades are pinned only to keep the *measurement* honest; no
    duck knob is sent, so the duck runs entirely on its shipped defaults:

    * the default 2.5 s fade-out starts at t=2.5 s — inside the gap window — and would
      swamp the very swell it is meant to expose;
    * the default ``gain_db=-18`` would drop the bed ~8 dB BELOW the narration's own
      out-of-band noise floor, so the probe would measure narration (loud under speech,
      silent in the gap) and report a nonsensical *negative* rise.
    """
    bed = MusicBed(url="bed", gain_db=0.0, fade_in_s=0.1, fade_out_s=0.1)
    final = await _render_with_bed(tmp_path, bed)
    rise = await _bed_rise_db(final)
    assert rise >= 6.0, f"bed rose only {rise:.1f} dB in the VO gap; the duck is not breathing"


async def test_the_old_duck_settings_do_not_breathe(tmp_path: Path) -> None:
    """The pre-fix values, on the same premastered narration, are DEAD: the bed
    barely moves between speech and the gap (measured -33.5 -> -32.1, +1.4 dB).

    Double duty. It is the regression guard on the bug — proof the old
    ``threshold=0.125:attack=200:release=1000`` genuinely never breathed on a
    real -18 LUFS VO bus — and it is the proof that the three new knobs actually
    REACH the filter: they are the only thing that differs from the test above,
    and they turn +12.4 dB into +1.4 dB.
    """
    bed = MusicBed(
        url="bed",
        gain_db=0.0,
        fade_in_s=0.1,
        fade_out_s=0.1,
        duck_threshold=0.125,
        duck_attack_ms=200,
        duck_release_ms=1000,
    )
    final = await _render_with_bed(tmp_path, bed)
    rise = await _bed_rise_db(final)
    assert rise <= 3.0, (
        f"the OLD duck settings moved the bed {rise:.1f} dB in the VO gap — expected a dead "
        "bed (<= 3 dB); either the knobs are not reaching sidechaincompress or the probe "
        "is not measuring the bed"
    )
