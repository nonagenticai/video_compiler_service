"""Real-ffmpeg render-harness proof that a transition SFX cue's ``gain_db``
reaches the DELIVERED SAMPLES.

Background: orchestrator PR #607 made transition SFX cues carry ``gain_db``
(default -10.0) instead of ffmpeg's 0.0. That was proven only at the
orchestrator level (a function returns the right number), and in this repo
``tests/test_renderer_sfx.py`` only asserts that ``volume=<n>dB`` appears in the
filtergraph STRING. Neither proves the rendered audio changes. These tests do:
they drive a REAL ``compile_shots`` encode and measure the cue in the output.

Method:

* Programme audio is ``make_test_clip``'s 440 Hz sine; the cue is
  ``make_bed_tone``'s 8 kHz sine. The cue sits at t=0 with no ``adelay``
  (``renderer._audio_sfx_graph``), so it occupies the first second of the shot.
* ``probe_band_mean_volume_window`` (2 x 2-pole high-pass at 6 kHz) puts the
  440 Hz programme ~90 dB down, so what is measured inside the SFX window is
  the CUE. The cutoff stays at 6 kHz: the duck tests prove an 8 kHz tone
  survives AAC, a higher cutoff is not proven.
* The loudness master is OFF (``master_enabled=False``): ``loudnorm=I=-12.5``
  would partially re-normalise the very delta under test.
* The mix at ``renderer.py`` uses ``amix=...:normalize=0``, so a dB trim on the
  cue arrives on the bus undivided and measures cleanly.

⚠ ffmpeg ``sine`` is -18 dBFS and a mono downmix reads ~3 LU light, so every
assertion here is a RATIO/DELTA between two windows or two renders — never an
absolute dBFS. Subtraction cancels the offset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import Shot, SoundCue
from src.config import get_settings
from src.services import renderer
from tests.render_harness import (
    make_bed_tone,
    make_test_clip,
    probe_band_mean_volume_window,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_SHOT_SECONDS = 3.0
_SFX_WINDOW = (0.15, 0.85)  # inside the 1.0 s cue
_TAIL_WINDOW = (1.5, 2.8)  # programme only, cue long finished


async def _render(tmp_path: Path, gain_db: float | None) -> Path:
    """Render ONE shot whose transition SFX carries ``gain_db`` (``None`` = no
    cue at all) through the real ``compile_shots`` path, master OFF."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = await make_test_clip(tmp_path / "src.mp4", duration=_SHOT_SECONDS, fps=25)
    cue: SoundCue | None = None
    if gain_db is not None:
        sfx = await make_bed_tone(tmp_path / "sfx.wav", duration=1.0, frequency=8000, gain_db=-6.0)
        cue = SoundCue(url=str(sfx), gain_db=gain_db)
    shot = Shot(id="s0", seconds=_SHOT_SECONDS, video_url=str(src), transition_sfx=cue)
    settings = get_settings().model_copy(
        update={"render_fps": 25, "master_enabled": False, "work_dir": str(tmp_path)}
    )
    final, _, _, _, _ = await renderer.compile_shots([shot], settings)
    return final


async def _band_rise_db(final: Path) -> float:
    """Out-of-band (cue) level in the SFX window minus the same band in the
    programme-only tail. Positive = a cue is actually present in the samples."""
    sfx = await probe_band_mean_volume_window(final, *_SFX_WINDOW)
    tail = await probe_band_mean_volume_window(final, *_TAIL_WINDOW)
    return sfx - tail


async def test_transition_sfx_is_audible(tmp_path: Path) -> None:
    """A transition cue at ``gain_db=0`` lands in the delivered samples: the
    8 kHz band inside the cue window sits far above the same band in the
    programme-only tail.

    Measured on a real render: -32.1 dBFS in the cue window vs -78.0 dBFS in the
    tail -> +45.9 dB. The bar is the measured rise minus ~40% headroom, so
    codec/probe jitter cannot make it flaky; the no-cue control below measures
    only +2.7 dB, so a dropped fetch is still caught with ~24 dB to spare.
    """
    final = await _render(tmp_path, 0.0)
    rise = await _band_rise_db(final)
    assert rise >= 27.0, f"transition SFX rose only {rise:.1f} dB above the tail; cue not delivered"


async def test_transition_gain_db_attenuates_rendered_audio(tmp_path: Path) -> None:
    """THE PROOF. The same shot rendered with ``gain_db=0.0`` and ``gain_db=-10.0``
    differs by ~10 dB in the DELIVERED cue band — the orchestrator's -10.0 default
    is not just a number a function returns, it reaches the samples.

    Measured: cue window -32.1 dBFS at 0.0 vs -42.1 dBFS at -10.0, against an
    identical -78.0 dBFS tail in both -> exactly 10.0 dB. ``abs=3.0`` matches the
    tolerance the sibling per-clip gain tests use for AAC round-trips.
    """
    loud = await _band_rise_db(await _render(tmp_path / "loud", 0.0))
    quiet = await _band_rise_db(await _render(tmp_path / "quiet", -10.0))
    assert loud - quiet == pytest.approx(10.0, abs=3.0), (
        f"gain_db=-10 moved the delivered cue by {loud - quiet:.1f} dB, expected ~10 dB"
    )


async def test_identical_gain_renders_yield_zero_delta(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: two renders of the SAME shot at the SAME gain must agree.

    Without this the A/B above is unfalsifiable — a rig that manufactures dB
    deltas out of encode nondeterminism would 'prove' anything. Measured: both
    renders read -32.1 / -78.0 dBFS, delta 0.0 dB.
    """
    a = await _band_rise_db(await _render(tmp_path / "a", 0.0))
    b = await _band_rise_db(await _render(tmp_path / "b", 0.0))
    assert abs(a - b) < 1.5, f"two identical renders differed by {abs(a - b):.1f} dB; rig is noisy"


async def test_shot_without_transition_sfx_has_no_band_rise(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: with ``transition_sfx=None`` the SFX window and the tail
    read the same band level.

    Proves the rise measured above comes from the CUE and not from the window
    position (e.g. an encoder transient at t=0). Measured -75.3 dBFS vs -78.0
    dBFS -> +2.7 dB: a real but tiny AAC start-up artefact, ~43 dB below the
    cue's rise. The 6 dB bar is that measured artefact with ~2x headroom.
    """
    rise = await _band_rise_db(await _render(tmp_path, None))
    assert abs(rise) < 6.0, f"a shot with NO transition cue still rose {rise:.1f} dB at t=0"
