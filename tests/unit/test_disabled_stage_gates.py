"""The mastering gate must not convict a job of a stage the job DISABLED.

Two instances of the unsatisfiable-gate class that #180 fixed one level up
(there the gate and its trigger read two different measurements; here the named
remediation stage is not even enabled):

1. LOUDNESS. ``renderer._loudnorm_filter`` returns ``""`` and the master pass is
   skipped entirely when ``master_enabled`` is False — which any caller sets per
   job via ``tags={"mastering": "off"}`` (``compile._settings_for_tags``). A raw
   mix lands around -21 LUFS against the -12.5 target, i.e. far outside the 3 LU
   tolerance, so ``post_mux_verify`` compared UNCONDITIONALLY, cleared
   ``gate_pass`` and returned ``rerun_stages=["loudness"]``. Re-running that
   stage normalizes nothing and re-measures the identical number: the gate can
   never be satisfied, and under ``mastering_gate_mode=enforce`` a CORRECT render
   fails permanently.

2. DEBAND — HISTORICAL. Identical shape: ``renderer._apply_conditional_deband``
   skips debanding when ``deband_enabled(settings)`` is False (``tags={"deband":
   "off"}``), yet ``_banding_violation`` convicted on residual banding and named
   ``deband`` as the fix. That guard is now MOOT, and the tests below say so
   rather than being deleted: as of 2026-09-11 the banding score does not gate in
   ANY mode (no usable spatial term; it fired on 89.2% of 120 measured masters),
   so there is no conviction left for the flag to suppress. The end-to-end
   deband-off test is kept as a regression floor — it must keep passing, now for
   a stronger reason than the one it was written for.

The loudness metric stays REPORTED — only the gating is suppressed. And the
MEASUREMENT-GAP verdict is unaffected in both cases: its remediation is to
measure again, which does not need the disabled stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import mastering, renderer

# A textured, moving, clean frame — nothing here should convict anything.
CLEAN_FRAME = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 9.0}
# A genuine banding signature: wide smooth ramp with sub-threshold dither.
BANDED_FRAME = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 0.5}

# What an UNMASTERED (raw) mix measures: ~-21 LUFS, >3 LU off the -12.5 target.
RAW_MIX_LUFS = -21.0


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=dict(overrides))


def _fake_run(dump: str):
    async def run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "mpdecimate" in joined:
            return "frame= 3\n"
        return dump

    return run


def _frames_dump(frames: list[dict[str, float]]) -> str:
    out = []
    for i, f in enumerate(frames):
        out.append(f"frame:{i} pts:{i * 512} pts_time:{i * 0.017:.3f}\n")
        out.append(
            f"lavfi.signalstats.YMIN={f['YMIN']}\n"
            f"lavfi.signalstats.YMAX={f['YMAX']}\n"
            f"lavfi.signalstats.YDIF={f['YDIF']}\n"
        )
    return "".join(out)


# --------------------------------------------------------------- loudness (defect 2)


def test_loudness_gate_is_silent_when_the_job_disabled_mastering() -> None:
    """RED before the fix: an unmastered raw mix at -21 LUFS produced a
    ``loudness`` violation naming a stage the job switched off."""
    off = _settings(master_enabled=False)
    assert mastering._loudness_violation(RAW_MIX_LUFS, off) is None


def test_loudness_gate_still_convicts_when_mastering_is_on() -> None:
    """GUARD: the fix must scope the gate to the enabled case, not delete it."""
    on = _settings(master_enabled=True)
    violation = mastering._loudness_violation(RAW_MIX_LUFS, on)
    assert violation is not None
    assert violation.stage == "loudness"
    # And an on-target master still passes.
    target = float(on.master_lufs)
    assert mastering._loudness_violation(target, on) is None


def test_unmeasurable_loudness_does_not_gate_a_job_with_mastering_off() -> None:
    """The measurement-gap verdict names the SAME disabled stage, so it is just as
    unsatisfiable — it must be suppressed too."""
    assert mastering._loudness_violation(None, _settings(master_enabled=False)) is None
    assert mastering._loudness_violation(None, _settings(master_enabled=True)) is not None


@pytest.mark.asyncio
async def test_a_correct_unmastered_render_passes_the_whole_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end RED: ``tags={"mastering":"off"}`` → a clean master came back
    ``gate_pass=False`` / ``rerun_stages=['loudness']`` forever."""
    monkeypatch.setattr(renderer, "_run", _fake_run(_frames_dump([CLEAN_FRAME] * 3)))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(master_enabled=False),
        loudness={"integrated_lufs": RAW_MIX_LUFS, "true_peak_dbtp": -1.1},
    )
    assert "loudness" not in report["rerun_stages"], report
    assert report["gate_pass"] is True, report
    # The measurement is still REPORTED — suppressed gating, not suppressed data.
    assert report["integrated_lufs"] == pytest.approx(RAW_MIX_LUFS)


# ----------------------------------------------------------------- deband (defect 3)


def test_the_banding_score_never_convicts_whatever_the_deband_flag_says() -> None:
    """SUPERSEDES the two deband-flag tests that used to live here.

    They asserted that a banded frame convicts with ``render_deband='auto'`` and
    is excused with ``'off'``. Banding no longer gates in EITHER mode: the score
    has no usable spatial term and fired on 89.2% of 120 measured masters, so the
    enable-flag guard has nothing left to guard. The frame below is over the
    trigger threshold in both settings and produces no violation in either."""
    for mode in ("auto", "off"):
        settings = _settings(render_deband=mode)
        scores = [mastering.banding_score(BANDED_FRAME)]
        # The frame really is over the threshold — this test is not vacuous.
        assert scores[0] >= mastering.BANDING_GATE_CEILING
        assert mastering.deband_enabled(settings) is (mode == "auto")
        assert mastering._banding_violation([BANDED_FRAME]) is None


def test_a_measurement_gap_still_gates() -> None:
    """The ``banding_measurement`` verdict SURVIVES banding going advisory, and it
    must: with no frames, ``luma_spread`` and ``dead_frame_ratio`` are both None
    and skipped by the no-gate-on-a-gap policy, so without it a dead signalstats
    probe would report ``gate_pass=True`` on zero picture evidence. Its
    remediation is to MEASURE again, which runs no filter, so it never depended on
    the deband flag either."""
    violation = mastering._banding_violation([])
    assert violation is not None
    assert violation.stage == "banding_measurement"


@pytest.mark.asyncio
async def test_a_banded_render_with_deband_off_passes_the_whole_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end RED: ``tags={"deband":"off"}`` → permanent
    ``rerun_stages=['deband']`` for a stage that never runs."""
    monkeypatch.setattr(renderer, "_run", _fake_run(_frames_dump([BANDED_FRAME] * 3)))
    settings = _settings(render_deband="off")
    assert mastering.deband_enabled(settings) is False
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        settings,
        loudness={"integrated_lufs": float(settings.master_lufs), "true_peak_dbtp": -1.1},
    )
    assert "deband" not in report["rerun_stages"], report
    assert report["gate_pass"] is True, report
    # Still measured and reported, just not gating.
    assert max(report["banding_scores"]) >= mastering.BANDING_GATE_CEILING
