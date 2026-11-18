"""Real-ffmpeg render proof for the PR #130 bed fixes (multi-cue span + 0 s fades).

``tests/test_audio_bed.py`` covers both fixes at the COMMAND-STRING level only —
its own docstring says "captured ffmpeg cmd, no real render". A filtergraph
assertion proves we WROTE ``amix=...:duration=longest`` and that no ``afade``
token was emitted; it cannot prove ffmpeg DELIVERS audible bed in the late
window, nor that a 0-second-fade bed renders at all. This module closes that gap
on decoded samples, the way every other audio fixture in this repo does.

Two proofs:

* **Multi-cue span** — three ambient cues tile a 30 s timeline (0-10 / 10-20 /
  20-30 s). Under the pre-#130 ``amix=inputs=3:duration=first`` graph the mix ran
  only as long as ``bed_amb0`` (``atrim``'d to its own 10 s window), so
  everything past t=10 s was BED SILENCE — the late window measured here would
  have carried narration leakage alone. With ``duration=longest`` the bed spans
  all three cues.
* **Zero-fade guard** — ``MusicBed.fade_in_s``/``fade_out_s`` are ``ge=0``, so
  0.0 is reachable from a caller; a degenerate ``afade=d=0`` could abort the bed
  pass. The render must simply SUCCEED and still deliver audio.

Method is the ambient-duck sibling's, unchanged: narration is speech-band pink
noise (300-3400 Hz) premastered to -18 LUFS; the bed is an 8 kHz sine, OUT of the
speech band, so ``probe_band_mean_volume_window`` (2x 6 kHz high-pass) isolates
the BED's delivered level inside the summed mix rather than measuring narration.
``gain_db=0`` is pinned on every cue (not the -24 default) so the bed stays well
clear of the narration's out-of-band floor. The loudness master is OFF — its
compressor + loudnorm would smear the levels under test.

Per the calibration trap in this suite: no assertion pins an absolute dB value
read off a synthetic generator with any precision. The headline check is a DELTA
(late window vs an early window rendered by the same graph) plus an absolute
floor that only has to separate "bed present" from "bed silent".

Every bar below is set from a MEASURED render on ffmpeg 4.4, not from theory.
The two states are NOT the ~40 dB apart a naive reading suggests: the bed is
sidechain-ducked under continuous narration in every window under test, so
"bed present" reads -43.4 dBFS and "bed silent" reads -57.1 dBFS (the narration's
own out-of-band leak floor). That is a 13.7 dB separation, and the floor is
parked at its midpoint.
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
    probe_duration,
    probe_has_audio,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_MULTICUE_DURATION = 30.0
_CUE_SPAN = 10.0
# The VO gap is parked in cue[0]'s window and OUTSIDE both probe windows, so the
# early and late windows are both measured under steady narration — i.e. in the
# same duck state — and the delta between them is the bed's span, not the duck's.
_MULTICUE_GAP = (4.0, 4.8)
_EARLY_WINDOW = (6.0, 9.0)  # inside cue[0] — bed is present in BOTH graphs
_LATE_WINDOW = (21.0, 29.0)  # inside cue[2] — bed is present ONLY after #130

# MEASURED on ffmpeg 4.4, gain_db=0 bed under steady narration:
#   bed present  -> -43.4 dBFS (ducked; the un-ducked tone is far hotter)
#   bed silent   -> -57.1 dBFS (narration's own leak above the 2x 6 kHz high-pass;
#                   confirmed by rendering a SINGLE 0-10 s cue and probing 21-29 s,
#                   which is exactly the pre-#130 state)
# -50.0 is the midpoint: 6.6 dB of margin in BOTH directions. An earlier -45.0 was
# only 1.6 dB under the present state and would flip on runner jitter.
_BED_PRESENT_FLOOR_DBFS = -50.0
# The two windows are the same tone at the same gain in the same duck state, so
# they should land within a dB or two; 8 dB is pure slack for a shared runner.
_WINDOW_SPREAD_DB = 8.0

_FADE_DURATION = 6.0
_FADE_GAP = (5.0, 5.5)


async def _render_bed(
    tmp_path: Path,
    tag: str,
    *,
    duration: float,
    gap: tuple[float, float],
    music: tuple[Path, MusicBed] | None = None,
    ambient: list[tuple[Path, AmbientCue]] | None = None,
) -> Path:
    """Render premastered narration under ``music``/``ambient`` through the REAL
    public bed path (``overlay_audio_bed``), loudness master OFF."""
    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    narration = await make_premastered_narration_clip(
        work / "narration.mp4", duration=duration, gap=gap
    )
    settings = get_settings().model_copy(update={"master_enabled": False, "work_dir": str(work)})
    out = work / "final.mp4"
    final, _report = await renderer.overlay_audio_bed(
        narration, out, duration, music, ambient or [], settings=settings
    )
    return final


async def test_multi_cue_ambient_bed_is_audible_in_the_last_cue(tmp_path: Path) -> None:
    """HEADLINE PROOF (#130, on decoded samples): three ambient cues tiling
    0-10 / 10-20 / 20-30 s deliver real bed energy in the LAST cue's window.

    Under the pre-#130 ``amix=inputs=3:duration=first`` graph this window would be
    SILENCE: the mix ended with ``bed_amb0``, which is ``atrim``'d to its own 10 s
    window, so every sample after t=10 s carried narration only. That sentence is
    the regression: the assertion below fails hard on the old graph and passes on
    the shipped one, which the filter-string test in ``tests/test_audio_bed.py``
    cannot demonstrate.
    """
    bed = await make_bed_tone(tmp_path / "ambient.wav", duration=_CUE_SPAN)
    cues = [
        (
            bed,
            AmbientCue(
                url="ambient",
                start_seconds=n * _CUE_SPAN,
                end_seconds=(n + 1) * _CUE_SPAN,
                gain_db=0.0,
            ),
        )
        for n in range(3)
    ]
    final = await _render_bed(
        tmp_path,
        "multicue",
        duration=_MULTICUE_DURATION,
        gap=_MULTICUE_GAP,
        ambient=cues,
    )

    late = await probe_band_mean_volume_window(final, *_LATE_WINDOW)
    early = await probe_band_mean_volume_window(final, *_EARLY_WINDOW)

    assert late > _BED_PRESENT_FLOOR_DBFS, (
        f"the third ambient cue's window ({_LATE_WINDOW[0]}-{_LATE_WINDOW[1]}s) measured "
        f"{late:.1f} dBFS in the bed band — at or under the {_BED_PRESENT_FLOOR_DBFS} dBFS floor, "
        "i.e. the bed was truncated to the FIRST cue (the pre-#130 amix duration=first defect)"
    )
    assert late >= early - _WINDOW_SPREAD_DB, (
        f"bed level collapsed across the timeline: {early:.1f} dBFS in cue[0] vs {late:.1f} dBFS "
        f"in cue[2] (drop > {_WINDOW_SPREAD_DB} dB) — the later cues are not carrying full bed"
    )


async def test_zero_fade_music_bed_renders_and_delivers_audio(tmp_path: Path) -> None:
    """Zero-fade guard, on a real render: ``fade_in_s=0``/``fade_out_s=0`` are
    reachable (both are ``Field(ge=0)``), and a degenerate ``afade=d=0`` could
    abort the bed pass. The pass must complete, and the output must still carry
    audio at the requested length. Omitting the fade IS "no fade"."""
    music = await make_bed_tone(tmp_path / "music.wav", duration=_FADE_DURATION)
    # gain_db=0 for the same reason the ambient cues pin it: the bed is ducked
    # under narration, so the default -6 dB lands at -48.9 dBFS — inside the
    # noise of the -50 dBFS floor rather than clear of it. At 0 dB it reads -43.4.
    bed = MusicBed(url="music", gain_db=0.0, fade_in_s=0.0, fade_out_s=0.0)
    final = await _render_bed(
        tmp_path,
        "zerofade",
        duration=_FADE_DURATION,
        gap=_FADE_GAP,
        music=(music, bed),
    )

    assert await probe_has_audio(final), "zero-fade bed render produced no audio stream"
    duration = await probe_duration(final)
    assert abs(duration - _FADE_DURATION) <= 0.5, (
        f"zero-fade bed render is {duration:.2f}s, expected ~{_FADE_DURATION}s"
    )
    # And the bed is actually there — a pass that silently dropped the bed input
    # would also "render successfully".
    level = await probe_band_mean_volume_window(final, 1.0, 4.0)
    assert level > _BED_PRESENT_FLOOR_DBFS, (
        f"zero-fade bed rendered but measured {level:.1f} dBFS in the bed band — bed missing"
    )


async def test_nonzero_fade_in_attenuates_the_head(tmp_path: Path) -> None:
    """Counterpart to the guard above: a real ``fade_in_s`` still attenuates.
    Measured in the BED band (the narration underneath is un-faded and would
    dominate a plain ``volumedetect``): the first 400 ms of a 3 s fade-in must sit
    well below the settled bed level past the fade.

    The head window does not measure the bed at all — the fade has it ~20 dB down
    there, below the narration's own leak floor — so the MEASURED head reading is
    that floor, -57.2 dBFS, and the delta is capped at ``settled - floor``. At the
    default -6 dB bed gain that cap is 8.4 dB against an 8 dB bar (0.4 dB of
    slack — a flake). At ``gain_db=0`` settled is -43.4 dBFS, so the cap is
    13.8 dB and the 8 dB bar keeps 5.8 dB of real margin."""
    music = await make_bed_tone(tmp_path / "music.wav", duration=_FADE_DURATION)
    bed = MusicBed(url="music", gain_db=0.0, fade_in_s=3.0, fade_out_s=0.5)
    final = await _render_bed(
        tmp_path,
        "fadein",
        duration=_FADE_DURATION,
        gap=_FADE_GAP,
        music=(music, bed),
    )

    head = await probe_band_mean_volume_window(final, 0.0, 0.4)
    settled = await probe_band_mean_volume_window(final, 3.5, 4.5)
    assert head <= settled - 8.0, (
        f"fade_in_s=3.0 left the head at {head:.1f} dBFS vs {settled:.1f} dBFS once settled — "
        "the fade is not attenuating the bed"
    )
