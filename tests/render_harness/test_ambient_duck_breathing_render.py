"""Real-ffmpeg render-harness tests for the ambient-only-vs-narration duck.

Sibling of ``test_duck_breathing_render.py``: that one proves the MUSIC bed
breathes; this proves the AMBIENT-ONLY bed (no music) now does too. Same method,
same -18 LUFS premastered narration invariant, same band-isolation probe —
driven through the public ``overlay_audio_bed`` mix with ``music=None`` and a
single ``AmbientCue``.

Acceptance (great-video spec 06-sound-music §T3 / criterion #3): "in any VO gap
> 400 ms, the bed RMS rises within one release." The ambient-only branch used to
fall to a DEAD hardcoded duck (``threshold=0.125:attack=200:release=1000``): 0.125
is an absolute linear key level the premastered -18 LUFS narration bus never
crosses (compressor barely opened) and a 1000 ms release pinned the bed flat
through every pause. The fix gives the ambient branch the SAME breathing shape as
music — HPF'd key, threshold 0.03, attack 30 ms, release 300 ms — driven by the
cue's own knobs.

Method identical to the music sibling: narration is speech-band pink noise
(300-3400 Hz) with an 800 ms silent gap, loudnorm'd to -18 LUFS; the bed is an
8 kHz sine (OUT of the speech band) so ``probe_band_mean_volume_window`` (2x 6 kHz
high-pass) isolates the BED's delivered level in the mix. ``rise`` is the bed's
level in the gap minus its level under speech. The loudness master is OFF for the
render under test — its compressor + loudnorm would squash the dynamics measured.

``gain_db=0`` is pinned (not the -24 default) to keep the measurement honest: the
default would drop the bed below the narration's out-of-band noise floor and the
probe would measure narration, not bed. ``AmbientCue`` has no fades, so — unlike
the music sibling — there is no fade-out to fence off the gap window.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import AmbientCue
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

# Steady-state duck (attack settled) vs the tail of the gap (one 300 ms release
# has elapsed, so a breathing bed is back at its resting level).
_SPEECH_WINDOW = (1.0, 1.9)
_GAP_WINDOW = (2.45, 2.75)


async def _render_with_ambient(tmp_path: Path, cue: AmbientCue) -> Path:
    """Render premastered narration-with-a-gap under a single ambient ``cue``
    through the REAL public duck path (``overlay_audio_bed``, music=None), master
    OFF."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = await make_premastered_narration_clip(
        tmp_path / "narration.mp4", duration=_DURATION, gap=_GAP
    )
    bed = await make_bed_tone(tmp_path / "ambient.wav", duration=_DURATION)
    settings = get_settings().model_copy(
        update={"master_enabled": False, "work_dir": str(tmp_path)}
    )
    out = tmp_path / "final.mp4"
    final, _report = await renderer.overlay_audio_bed(
        video, out, _DURATION, None, [(bed, cue)], settings=settings
    )
    return final


async def _bed_rise_db(final: Path) -> float:
    """How far the ambient bed rises in the VO gap vs under narration, in dB."""
    under_speech = await probe_band_mean_volume_window(final, *_SPEECH_WINDOW)
    in_gap = await probe_band_mean_volume_window(final, *_GAP_WINDOW)
    return in_gap - under_speech


async def test_ambient_bed_breathes_in_a_vo_gap(tmp_path: Path) -> None:
    """Acceptance #3 on real samples: with the SHIPPED ambient defaults (threshold
    0.03 / attack 30 ms / release 300 ms — no duck knobs sent), an ambient-only
    bed swells well back up inside an 800 ms VO gap vs its level under narration.
    The 6 dB bar keeps margin so codec/loudnorm jitter can't make it flaky."""
    cue = AmbientCue(url="ambient", gain_db=0.0)
    final = await _render_with_ambient(tmp_path, cue)
    rise = await _bed_rise_db(final)
    assert rise >= 6.0, (
        f"ambient bed rose only {rise:.1f} dB in the VO gap; the duck is not breathing"
    )


async def test_the_old_ambient_duck_settings_do_not_breathe(tmp_path: Path) -> None:
    """The pre-fix values, on the same premastered narration, are DEAD: the bed
    barely moves between speech and the gap. Double duty — regression guard on the
    dead ``threshold=0.125:attack=200:release=1000`` filter, and proof the new
    ambient knobs actually REACH the sidechain (they are the only difference from
    the test above, and they turn a healthy rise into a near-flat one)."""
    cue = AmbientCue(
        url="ambient",
        gain_db=0.0,
        duck_threshold=0.125,
        duck_attack_ms=200,
        duck_release_ms=1000,
    )
    final = await _render_with_ambient(tmp_path, cue)
    rise = await _bed_rise_db(final)
    assert rise <= 3.0, (
        f"the OLD ambient duck settings moved the bed {rise:.1f} dB in the VO gap — expected a "
        "dead bed (<= 3 dB); either the knobs are not reaching sidechaincompress or the probe "
        "is not measuring the bed"
    )
