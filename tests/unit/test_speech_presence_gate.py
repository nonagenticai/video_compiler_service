"""Unit proofs for the speech-presence gate — the metric, the verdict, the wiring.

A production ``website_review`` master shipped carrying no speech at all, only
faint hiss, and every existing signal passed it. #224 fixed the CAUSE (a sample-
format mismatch surviving a stream-copy concat); this gate is the missing
DETECTION — nothing measured the FINISHED master for the presence of speech.

The numbers asserted below are the ones MEASURED against real ffmpeg on
2026-09-16 and recorded at ``mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB``.
``tests/render_harness/test_speech_presence_render.py`` re-measures them
end-to-end through real encodes; these tests pin the verdict logic that consumes
them, with no ffmpeg in the loop.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.api.schemas import MasteringReport
from src.config import Settings, get_settings
from src.services import mastering, renderer
from src.services.loudness import parse_loudness, program_to_noise_floor_db

#: The exact readings measured on 2026-09-16 (ffmpeg 4.4, astats + loudnorm) over
#: AAC-in-MP4 — the container this pipeline actually delivers.
#: ``(integrated_lufs, noise_floor_db)``.
_HISS_RAW = (-36.71, -31.26)  # white noise @ -34 dBFS        -> gap  -5.45
_HISS_NORMALISED = (-14.74, -8.62)  # the same, lifted to -14 LUFS  -> gap  -6.12
_STEADY_TONE = (-55.05, -51.82)  # worst NO-SPEECH reading of ten   -> gap  -3.23
_NARRATION_WITH_BED = (-14.10, -36.68)  # narration + pink bed      -> gap +22.58
_NARRATION_OVER_LOUD_BED = (-16.97, -21.86)  # hardest legit case   -> gap  +4.89


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------


def test_program_to_noise_floor_is_the_difference() -> None:
    assert program_to_noise_floor_db(-14.0, -50.0) == pytest.approx(36.0)


def test_program_to_noise_floor_is_level_invariant() -> None:
    """THE property the threshold rests on: the metric measures DYNAMICS, not
    level, so lifting a whole file by any gain cannot move it.

    This is why a hiss-only master that the loudness master normalised up to -14
    LUFS is caught just as a raw one is, and why the check needs no
    ``master_enabled`` guard (unlike loudness and true peak, which gate on an
    absolute level only the master pass can put in range)."""
    quiet = program_to_noise_floor_db(-40.0, -70.0)
    lifted = program_to_noise_floor_db(-40.0 + 26.0, -70.0 + 26.0)
    assert quiet == lifted == pytest.approx(30.0)


@pytest.mark.parametrize(("integrated", "floor"), [(None, -50.0), (-14.0, None), (None, None)])
def test_program_to_noise_floor_is_none_when_either_input_is_absent(
    integrated: float | None, floor: float | None
) -> None:
    """An unmeasured input yields NO verdict — never a fabricated 0.0, which would
    read as a measured "steady noise, no speech"."""
    assert program_to_noise_floor_db(integrated, floor) is None


def test_parse_loudness_reads_the_noise_floor_from_astats() -> None:
    """The parser picks the trailing ``Overall`` block, not the first channel."""
    astats = (
        "[Parsed_astats_0] Noise floor dB: -41.111111\n"
        "[Parsed_astats_0] Noise floor dB: -33.969862\n"
    )
    assert parse_loudness("", "", astats)["noise_floor_db"] == pytest.approx(-33.969862)


def test_parse_loudness_noise_floor_absent_without_astats_output() -> None:
    assert parse_loudness('{ "input_i" : "-14.0" }')["noise_floor_db"] is None


# --------------------------------------------------------------------------
# the verdict — fires on hiss, silent on narration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "reading", "expected_gap"),
    [
        ("hiss_raw", _HISS_RAW, -5.45),
        ("hiss_normalised_to_target", _HISS_NORMALISED, -6.12),
        ("steady_tone_worst_no_speech_reading", _STEADY_TONE, -3.23),
    ],
)
def test_it_fires_on_a_master_carrying_only_hiss(
    name: str, reading: tuple[float, float], expected_gap: float
) -> None:
    """THE DEFECT. Both readings are REAL measurements of a hiss-only file, and
    note the second: loudnorm lifted it to -14.74 LUFS, i.e. within a LU of the
    -14 target and comfortably inside the 3.0 LU tolerance. A loudness gate reads
    that as a perfect master."""
    integrated, floor = reading
    violation = mastering._speech_presence_violation(integrated, floor)
    assert violation is not None, name
    assert violation.stage == "speech_presence"
    assert violation.metric == "program_to_noise_floor_db"
    assert violation.tier == "structural"
    assert violation.value == pytest.approx(expected_gap, abs=0.01)
    assert violation.limit == mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB
    # The detail must name WHAT was measured, not merely that something failed.
    assert "noise floor" in violation.detail
    assert f"{integrated:.2f}" in violation.detail
    assert f"{floor:.2f}" in violation.detail


@pytest.mark.parametrize(
    ("name", "reading"),
    [
        ("narration_plus_music_bed", _NARRATION_WITH_BED),
        ("narration_barely_over_a_full_timeline_bed", _NARRATION_OVER_LOUD_BED),
    ],
)
def test_it_stays_silent_on_a_normal_narrated_master(
    name: str, reading: tuple[float, float]
) -> None:
    """NOT VACUOUS — this is the half that stops it being a signal that fires on
    everything. The second case is the hardest legitimate master constructed:
    narration only 3 dB above a pink bed running the WHOLE timeline, i.e. almost
    no pauses for astats to find a floor in. It still clears by 3.39 dB."""
    assert mastering._speech_presence_violation(*reading) is None, name


def test_the_threshold_separates_the_two_measured_populations() -> None:
    """The chosen number must sit strictly INSIDE the measured gap between the
    worst no-speech reading and the worst legitimate one. A threshold outside it
    is either a gate that fires on good masters or one that cannot fire at all."""
    worst_no_speech = program_to_noise_floor_db(*_STEADY_TONE)
    worst_legitimate = program_to_noise_floor_db(*_NARRATION_OVER_LOUD_BED)
    assert worst_no_speech is not None and worst_legitimate is not None
    assert worst_no_speech == pytest.approx(-3.23, abs=0.01)
    assert worst_legitimate == pytest.approx(4.89, abs=0.01)
    assert worst_no_speech < mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB < worst_legitimate


@pytest.mark.parametrize(("integrated", "floor"), [(None, -8.0), (-14.0, None), (None, None)])
def test_an_unmeasured_input_is_not_a_violation(
    integrated: float | None, floor: float | None
) -> None:
    """NO-GATE-ON-A-GAP. An absent measurement is not evidence of a defect, and
    ``_loudness_violation`` already names an unmeasurable ``integrated_lufs`` —
    a second violation over the same missing number reports one probe failure
    twice and names a remediation for a measurement rather than for the master."""
    assert mastering._speech_presence_violation(integrated, floor) is None


def test_no_master_enabled_guard_is_needed() -> None:
    """``mastering: off`` cannot manufacture this violation, so unlike loudness
    and true peak the check takes no ``Settings`` at all. Proven by construction:
    the signature has no settings parameter and the metric is level-invariant."""
    # A raw, UNMASTERED narration mix (quiet, nowhere near the -14 target that
    # would satisfy _loudness_violation) still reads a huge margin.
    assert mastering._speech_presence_violation(-32.29, -120.0) is None


# --------------------------------------------------------------------------
# the wiring — tier, kill switch, arming, and the published report
# --------------------------------------------------------------------------


def test_the_gate_is_structural_and_not_armed_by_default() -> None:
    """⛔ A GATE MUST NEVER DESTROY A PAID DELIVERABLE. The threshold is measured
    over FIXTURES; the fleet distribution is n=0, so it reports and does not
    raise — the same footing as ``colorspace`` and ``bed_calibration``."""
    assert mastering.gate_tier("speech_presence") == "structural"
    assert not mastering.gate_armed(_settings(), "speech_presence")
    assert mastering._GATE_ARMED_DEFAULTS["speech_presence"] is False


def test_its_kill_switch_is_a_real_settings_field_that_can_arm_it() -> None:
    """The back-out path (and the arm path) is config, never a code change.

    Also covers a settings object that predates the field — the module default
    must answer, not a fail-open ``False`` read off a missing attribute."""
    switch = mastering._GATE_KILL_SWITCHES["speech_presence"]
    assert switch == "mastering_gate_speech_presence_enabled"
    assert Settings.model_fields[switch].default is False
    assert mastering.gate_armed(_settings(**{switch: True}), "speech_presence")
    predates_the_field = cast(Settings, SimpleNamespace())
    assert not mastering.gate_armed(predates_the_field, "speech_presence")


@pytest.mark.asyncio
async def test_a_hiss_master_reports_a_violation_and_ships(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """END TO END through ``post_mux_verify``: the hiss reading clears
    ``gate_pass``, publishes a ``violations`` entry that survives the response
    model, and — because the gate is UNARMED — RETURNS rather than raising, so the
    paid deliverable is still delivered."""

    async def probe(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return ""

    monkeypatch.setattr(renderer, "_run", probe)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="off"),
        loudness={
            "integrated_lufs": _HISS_NORMALISED[0],
            "true_peak_dbtp": -1.0,
            "noise_floor_db": _HISS_NORMALISED[1],
        },
    )

    assert raw["gate_pass"] is False
    assert "speech_presence" in raw["rerun_stages"]
    assert raw["noise_floor_db"] == pytest.approx(_HISS_NORMALISED[1])
    assert raw["program_to_noise_floor_db"] == pytest.approx(-6.12, abs=0.01)
    assert raw["speech_presence_min_program_over_floor_db"] == pytest.approx(1.5)

    report = MasteringReport.model_validate(raw)
    entry = next(v for v in report.violations if v.stage == "speech_presence")
    assert entry.tier == "structural"
    assert entry.value == pytest.approx(-6.12, abs=0.01)
    assert entry.limit == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_a_narrated_master_reports_the_numbers_and_no_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NOT VACUOUS: the same call with a narrated reading publishes the metrics
    and NO ``speech_presence`` entry, so the assertion above discriminates."""

    async def probe(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return ""

    monkeypatch.setattr(renderer, "_run", probe)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="off"),
        loudness={
            "integrated_lufs": _NARRATION_WITH_BED[0],
            "true_peak_dbtp": -1.0,
            "noise_floor_db": _NARRATION_WITH_BED[1],
        },
    )

    assert "speech_presence" not in raw["rerun_stages"]
    assert raw["program_to_noise_floor_db"] == pytest.approx(22.58, abs=0.01)
    report = MasteringReport.model_validate(raw)
    assert [v for v in report.violations if v.stage == "speech_presence"] == []


@pytest.mark.asyncio
async def test_arming_it_makes_the_same_hiss_master_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unarmed default is a DECISION, not a dead code path: flipping the one
    kill switch turns the identical reading into a ``MasteringGateError``. Without
    this, "unarmed" would be indistinguishable from "never wired up"."""

    async def probe(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return ""

    monkeypatch.setattr(renderer, "_run", probe)
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(render_colorspace="off", mastering_gate_speech_presence_enabled=True),
            loudness={
                "integrated_lufs": _HISS_NORMALISED[0],
                "true_peak_dbtp": -1.0,
                "noise_floor_db": _HISS_NORMALISED[1],
            },
        )
    assert any(v.stage == "speech_presence" for v in excinfo.value.violations)


def test_the_degraded_report_carries_the_same_keys_as_nulls() -> None:
    """Both report shapes must agree, or a consumer has to branch on which one it
    received. ``None``, never 0.0: nothing was measured."""
    degraded = mastering.degraded_report()
    for key in (
        "noise_floor_db",
        "program_to_noise_floor_db",
        "speech_presence_min_program_over_floor_db",
    ):
        assert key in degraded, key
        assert degraded[key] is None, key
    assert MasteringReport.model_validate(degraded).program_to_noise_floor_db is None
