"""Real-ffmpeg render-harness tests for the VO-RELATIVE sidechain duck key (T1).

Sibling of ``test_duck_breathing_render.py`` (music bed breathes) and
``test_ambient_duck_breathing_render.py`` (ambient bed breathes). Those two prove
the duck OPENS at all on the production -18 LUFS narration invariant. This one
proves the next thing: that the duck's DEPTH stops depending on how loud the
voiceover happens to be.

The defect being fixed: ``duck_threshold`` is an ABSOLUTE linear key level, so a
loud cast blows through it and gets a deep duck while a quiet cast barely crosses
it and gets almost none. The mix's music-under-speech balance therefore drifts
silently with cast loudness — exactly the drift ``music_offset_lu`` kills for the
RESTING level and ``duck_open_lu_below_vo`` kills for the DUCK.

Casts are -12 / -18 / -24 LUFS: the -18 production invariant plus 6 LU either
side, which is the realistic spread of voices this pipeline actually receives.

MEASURED on this harness at that range, five consecutive runs each, AFTER
e6e414d re-anchored the sidechain ratio 9->4 (de-pumping fix — a shallower
ratio compresses every cast less, which narrows BOTH sides' spread):

* WITHOUT the fix (absolute key), depth still drifts with cast loudness, but
  less dramatically at ratio 4 than at the old ratio 9: 8.6-9.6 / 9.8-10.1 /
  6.9-7.1 dB -> spread **2.8-3.2 dB** (was 7.2-7.6 dB at ratio 9).
* WITH the measured key, depth holds tighter still: 8.1-9.3 / 9.9-10.0 /
  9.7-10.3 dB -> spread **0.9-1.8 dB** (was 1.2-1.6 dB at ratio 9).

The bars follow from those numbers, not the other way round. The headline bar
stays ``<= 3.0 dB`` — comfortably above the observed 0.9-1.8 dB so run-to-run
jitter cannot red it. The negative control's bar is now ``> 2.0 dB`` (was
``> 5.0 dB`` at ratio 9): ~0.8 dB under the smallest spread ever observed at
ratio 4 (2.8) so it is not flaky, and ~0.2 dB over the largest headline spread
ever observed (1.8) so a result cannot satisfy both in practice — the two
distributions do not overlap even though the bars themselves sit closer
together than before. If the ratio anchor moves again, re-measure both sides
before touching either bar.

Note the negative control's honest margin is smaller than the ~10 dB an earlier
draft reported: that draft reached 10.9 dB only by casting -12 / -20 / -28, a
wider range than production sees. The bar is set to what -12 / -18 / -24 actually
delivers. Do NOT widen the casts to buy headroom — if this control needs a bigger
gap, the answer is a better measurement, not a less realistic input.

``duck_open_lu_below_vo=12.46`` is not arbitrary: ``10 ** ((-18 - 12.46) / 20) ==
0.03``, the shipped absolute ``duck_threshold``. So on the -18 LUFS production
invariant this key is a no-op, and every other cast level is re-derived relative
to it — the feature re-centres the key on the measured VO instead of moving it.

Method (inherited wholesale from the two siblings — read them first):

* Narration is speech-band pink noise (300-3400 Hz) with an 800 ms silent gap,
  loudnorm'd to a requested LUFS via ``make_premastered_narration_clip(lufs=...)``.
  The three casts stand in for materially different voices.
* The bed is an 8 kHz sine, deliberately OUT of the speech band, so
  ``probe_band_mean_volume_window`` (2x 6 kHz high-pass) isolates the BED's own
  delivered level inside the summed mix.
* **Duck depth** is the gap-window level minus the speech-window level — the
  ``_bed_rise_db`` gap-minus-speech pattern from
  ``test_duck_breathing_render.py``. Per the project's audio-measurement traps,
  EVERY assertion here is a DELTA (a depth, or a spread across depths); nothing
  asserts an absolute dBFS off a synthetic source, whose generator level (ffmpeg
  ``sine`` runs at -18 dBFS, not full scale) would silently poison the number.
* The loudness master is OFF for every render (``master_enabled=False``): it is a
  dynamic AGC (acompressor -> alimiter -> loudnorm) and would flatten the exact
  dynamics under test.
* Probe windows stay STEREO throughout — a mono downmix reads ~3 LU light.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import AmbientCue, MusicBed
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

# 10 ** ((-18 - 12.46) / 20) == 0.03 == the shipped absolute duck_threshold, so
# this key is a no-op on the -18 LUFS production invariant and a re-centring
# everywhere else.
_OPEN_LU_BELOW_VO = 12.46

_CASTS = [-12.0, -18.0, -24.0]


async def _duck_depth(final: Path) -> float:
    """The bed's level in the VO gap minus its level under narration, in dB —
    i.e. how deep the sidechain pushed the bed down while speech was present."""
    under_speech = await probe_band_mean_volume_window(final, *_SPEECH_WINDOW)
    in_gap = await probe_band_mean_volume_window(final, *_GAP_WINDOW)
    return in_gap - under_speech


async def _render_with_music(tmp_path: Path, tag: str, cast_lufs: float, bed: MusicBed) -> Path:
    """Render a cast at ``cast_lufs`` under ``bed`` through the REAL public duck
    path (``overlay_audio_bed``), master OFF."""
    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    video = await make_premastered_narration_clip(
        work / "narration.mp4", duration=_DURATION, gap=_GAP, lufs=cast_lufs
    )
    music = await make_bed_tone(work / "bed.wav", duration=_DURATION)
    settings = get_settings().model_copy(update={"master_enabled": False, "work_dir": str(work)})
    final, _report = await renderer.overlay_audio_bed(
        video, work / "final.mp4", _DURATION, (music, bed), [], settings=settings
    )
    return final


async def _render_with_ambient(tmp_path: Path, tag: str, cast_lufs: float, cue: AmbientCue) -> Path:
    """Same, but through the AMBIENT-ONLY branch (``music=None``) — the branch
    ``duck_open_lu_below_vo`` was newly threaded into."""
    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    video = await make_premastered_narration_clip(
        work / "narration.mp4", duration=_DURATION, gap=_GAP, lufs=cast_lufs
    )
    bed = await make_bed_tone(work / "ambient.wav", duration=_DURATION)
    settings = get_settings().model_copy(update={"master_enabled": False, "work_dir": str(work)})
    final, _report = await renderer.overlay_audio_bed(
        video, work / "final.mp4", _DURATION, None, [(bed, cue)], settings=settings
    )
    return final


async def test_duck_depth_is_stable_across_casts_with_vo_relative_key(tmp_path: Path) -> None:
    """Headline: three casts at materially different loudness (-12 / -18 / -24
    LUFS), same bed, ``duck_open_lu_below_vo`` set. The delivered duck DEPTH must
    land within 3 dB across all three — the duck no longer drifts with cast level.
    Measured 10.6-10.9 / 11.9 / 12.1-12.2 dB -> spread 1.2-1.6 dB over three runs;
    the 3.0 dB bar is ~2x that, so jitter cannot red it.

    ``gain_db=0`` and short fades are pinned only to keep the measurement honest
    (see the sibling's docstring): the -18 dB default would drop the bed below the
    narration's own out-of-band floor, and the 2.5 s default fade-out lands inside
    the gap window."""
    depths: dict[float, float] = {}
    for cast_lufs in _CASTS:
        bed = MusicBed(
            url="bed",
            gain_db=0.0,
            fade_in_s=0.1,
            fade_out_s=0.1,
            duck_open_lu_below_vo=_OPEN_LU_BELOW_VO,
        )
        final = await _render_with_music(tmp_path, f"rel{cast_lufs:.0f}", cast_lufs, bed)
        depths[cast_lufs] = await _duck_depth(final)

    spread = max(depths.values()) - min(depths.values())
    assert spread <= 3.0, (
        f"duck depth per cast {depths} — spread {spread:.1f} dB exceeds 3.0 dB; the "
        "VO-relative key is not holding the duck constant across cast loudness"
    )


async def test_absolute_key_duck_depth_collapses_across_casts(tmp_path: Path) -> None:
    """Negative control, and the load-bearing half of this proof: the SAME loop
    with ``duck_open_lu_below_vo`` UNSET (genuinely absent from the ``MusicBed``
    kwargs — the field is ``float | None`` and ``_vo_relative_threshold`` tests it
    with ``is None``, so an explicit falsy 0.0 would NOT take the fallback path)
    must show a LARGER spread than the VO-relative key, because an absolute key
    still opens wider for a hot cast than a quiet one. Measured (ratio 4, post
    e6e414d) 8.6-9.6 / 9.8-10.1 / 6.9-7.1 dB -> spread 2.8-3.2 dB over five runs.

    The bar is ``> 2.0`` (was ``> 5.0`` at the pre-e6e414d ratio of 9: a
    shallower compression ratio compresses every cast less, which narrows the
    absolute key's own spread along with the measured key's). 2.0 sits ~0.8 dB
    below the smallest spread observed at ratio 4 (2.8, not flaky) and ~0.2 dB
    above the largest headline spread ever observed (1.8) — the two bars now sit
    closer together than before, but the two underlying distributions (2.8-3.2
    vs 0.9-1.8) still do not overlap in practice.

    If this test ever passes-by-not-spreading, the harness has gone insensitive
    and the headline test above proves nothing — a constant duck depth would then
    be an artefact of the measurement, not of the feature."""
    depths: dict[float, float] = {}
    for cast_lufs in _CASTS:
        bed = MusicBed(url="bed", gain_db=0.0, fade_in_s=0.1, fade_out_s=0.1)
        final = await _render_with_music(tmp_path, f"abs{cast_lufs:.0f}", cast_lufs, bed)
        depths[cast_lufs] = await _duck_depth(final)

    spread = max(depths.values()) - min(depths.values())
    assert spread > 2.0, (
        f"absolute-key duck depth per cast {depths} — spread only {spread:.1f} dB. The "
        "harness is expected to be SENSITIVE to cast loudness on the absolute key; a flat "
        "result here means the headline test's stability is a measurement artefact"
    )


async def test_ambient_only_duck_depth_is_stable_across_casts(tmp_path: Path) -> None:
    """Ambient parity: the ambient-only branch (``music=None``, a single
    ``AmbientCue``) honours ``duck_open_lu_below_vo`` the same way the music
    branch does. This is the branch the fix threads the measured key into — the
    music branch already had it. Uses the endpoints of the same realistic range
    (-12 / -24 LUFS); measured 11.2-11.5 / 12.1-12.2 dB -> spread 0.6-1.0 dB
    against the same 3.0 dB bar.

    ``AmbientCue`` has no fades, so (unlike the music renders) nothing needs
    fencing off the gap window; ``gain_db=0`` is still pinned so the probe reads
    bed rather than the narration's out-of-band floor."""
    casts = [_CASTS[0], _CASTS[-1]]
    depths: dict[float, float] = {}
    for cast_lufs in casts:
        cue = AmbientCue(url="ambient", gain_db=0.0, duck_open_lu_below_vo=_OPEN_LU_BELOW_VO)
        final = await _render_with_ambient(tmp_path, f"amb{cast_lufs:.0f}", cast_lufs, cue)
        depths[cast_lufs] = await _duck_depth(final)

    spread = max(depths.values()) - min(depths.values())
    assert spread <= 3.0, (
        f"ambient-only duck depth per cast {depths} — spread {spread:.1f} dB exceeds 3.0 dB; "
        "duck_open_lu_below_vo is not reaching the ambient-only sidechain"
    )


async def test_unset_key_still_breathes_on_the_production_invariant(tmp_path: Path) -> None:
    """Back-compat: with ``duck_open_lu_below_vo`` UNSET at the -18 LUFS
    production invariant, the shipped breathing behaviour is untouched — the same
    ``rise >= 6.0`` bar ``test_duck_breathing_render.py::
    test_music_bed_breathes_in_a_vo_gap`` holds. The feature is additive; the
    default (``None``) path must render exactly as it does today."""
    bed = MusicBed(url="bed", gain_db=0.0, fade_in_s=0.1, fade_out_s=0.1)
    final = await _render_with_music(tmp_path, "backcompat", -18.0, bed)
    rise = await _duck_depth(final)
    assert rise >= 6.0, (
        f"bed rose only {rise:.1f} dB in the VO gap with duck_open_lu_below_vo unset; the "
        "additive VO-relative key regressed the shipped default duck"
    )
