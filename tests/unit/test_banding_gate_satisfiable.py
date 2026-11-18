"""The banding gate must be SATISFIABLE, and must not pass an UNMEASURED master.

Three defects, one root cause plus one asymmetry.

1. ``banding_score`` read ``YDIF`` — a TEMPORAL (frame-to-frame) difference — as a
   proxy for spatial dither, and added a span-independent ``smoothness * 0.25``
   floor. A DUPLICATE frame has ``YDIF == 0`` (and so does signalstats' frame 0,
   always), so any duplicate/first frame with a normal luma span scored >= 0.25 —
   above the 0.15 ceiling — *regardless of the actual gradient*. Fleet masters run
   ~90% duplicate frames, so the gate essentially could never pass: it is the
   mechanism behind ``gate_pass=false`` on 96.7% of masters.
2. The remediation the gate names is IMPOSSIBLE: ``deband`` is a SPATIAL filter and
   ``YDIF`` is a TEMPORAL metric, so re-running deband cannot move the number the
   gate compared — hence ``rerun_stages=["deband"]`` followed by "no stage was
   re-run; degrading". A gate that demands a remediation which cannot satisfy it is
   the same unsatisfiable-gate class already fixed twice in this service.
3. The same score drives the deband TRIGGER (``banding_risk_frames``, and
   ``render_deband`` defaults to ``auto``), so ONE duplicate frame per segment
   forced a second re-encode of EVERY shot — the documented "clean shots stay a
   byte-identical single pass" behaviour never held.
4. Asymmetry: ``_banding_gate_pass([])`` is ``True``, so a master whose measurement
   pass failed outright PASSED the banding gate — while loudness fails CLOSED on
   exactly the same condition.

Why CI never caught 1-3: every ``YDIF=0`` fixture in the suite pairs it with
``YMAX == YMIN`` (span 0 → score 0), and the only ``post_mux_verify`` happy path
uses a single ``YDIF=9.0`` frame. The production shape — a TEXTURED duplicate
frame — was never exercised. It is exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import mastering, renderer

# THE production shape: a fully textured frame (16..235 luma, span 219) that happens
# to be identical to its predecessor, so signalstats reports no temporal difference.
# This is what ~90% of every fleet master's frames look like.
# ⚠️ YLOW/YHIGH are signalstats' 10th/90th luma percentiles and are part of EVERY
# real frame's stats — the spatial term added 2026-09-02 reads them. Omitting them
# here would not be "a minimal fixture", it would be a frame that cannot occur.
# Textured content concentrates its pixels: a narrow 10-90 band with outliers
# stretching YMIN/YMAX. Measured on `mandelbrot`: ratio 0.390.
TEXTURED_DUPLICATE = {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 90.0, "YHIGH": 175.0, "YDIF": 0.0}
# The same frame carrying real motion — the control.
TEXTURED_MOVING = {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 90.0, "YHIGH": 175.0, "YDIF": 9.0}
# A genuine banding signature: wide smooth ramp, dither present but far below the
# threshold. Must STILL trip, or the fix has simply disabled the gate.
# A ramp spreads pixels EVENLY, so the 10-90 band covers nearly the whole range.
# Measured on a real 8-bit gradient: ratio 0.914.
GENUINE_BANDING = {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 38.0, "YHIGH": 213.0, "YDIF": 0.5}


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


# --------------------------------------------------------------------------- score


def test_a_textured_duplicate_frame_is_not_evidence_of_banding() -> None:
    """RED: scored 1.0 (>= the 0.15 ceiling) on a frame with no gradient at all.

    ``YDIF == 0`` means "no temporal information", not "no dither". Since deband is
    spatial it could never move this number, so scoring on it can only produce an
    unsatisfiable verdict."""
    assert mastering.banding_score(TEXTURED_DUPLICATE) == 0.0
    assert mastering.banding_risk(TEXTURED_DUPLICATE) is False


def test_the_span_floor_no_longer_convicts_a_frame_with_zero_span() -> None:
    """RED: at exactly the minimum luma span the span factor is 0, yet the
    span-INDEPENDENT ``smoothness * 0.25`` term still scored 0.208 — over the
    ceiling — so the verdict did not depend on the gradient it claims to measure.
    The score must be continuous across the span threshold."""
    at_floor = {"YMIN": 100.0, "YMAX": 160.0, "YDIF": 0.5}  # span exactly 60.0
    just_below = {"YMIN": 100.0, "YMAX": 159.9, "YDIF": 0.5}
    assert mastering.banding_score(just_below) == 0.0
    assert mastering.banding_score(at_floor) == 0.0


def test_genuine_banding_is_still_convicted() -> None:
    """GUARD: the fix must narrow the false positives, not disable the gate."""
    assert mastering.banding_score(GENUINE_BANDING) >= mastering.BANDING_GATE_CEILING
    assert mastering.banding_risk(GENUINE_BANDING) is True
    assert mastering.banding_risk(TEXTURED_MOVING) is False


def test_score_is_monotone_in_the_gradient_it_claims_to_measure() -> None:
    """A wider smooth span must never score LOWER than a narrower one.

    Both frames carry the SAME spatial uniformity (a ramp: the 10-90 band covers
    ~90% of the range) so the only thing varying is the span — which is what this
    test is about. Holding uniformity fixed is the point; letting it drift would
    make the comparison meaningless.
    """
    narrow = mastering.banding_score(
        {"YMIN": 100.0, "YMAX": 180.0, "YLOW": 104.0, "YHIGH": 176.0, "YDIF": 0.5}
    )
    wide = mastering.banding_score(
        {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 38.0, "YHIGH": 213.0, "YDIF": 0.5}
    )
    assert wide > narrow


# ------------------------------------------------------------------- deband trigger


def test_one_duplicate_frame_does_not_force_a_second_encode_of_every_shot() -> None:
    """RED: ``banding_risk_frames`` takes the WORST frame, so a single textured
    duplicate anywhere in a segment tripped the trigger — and with
    ``render_deband='auto'`` that is EVERY shot, every render, re-encoded twice."""
    clean_segment_with_a_duplicate = [TEXTURED_MOVING, TEXTURED_DUPLICATE, TEXTURED_MOVING]
    assert mastering.banding_risk_frames(clean_segment_with_a_duplicate) is False


def test_a_genuinely_banded_frame_still_triggers_deband() -> None:
    """GUARD for the trigger side."""
    assert mastering.banding_risk_frames([TEXTURED_MOVING, GENUINE_BANDING]) is True


def test_trigger_and_gate_still_agree_frame_for_frame() -> None:
    """The trigger and the gate MUST stay the same predicate — they disagreed once
    before and that made the gate unsatisfiable (see ``banding_risk_frames``)."""
    for frames in (
        [TEXTURED_DUPLICATE],
        [TEXTURED_MOVING, TEXTURED_DUPLICATE],
        [GENUINE_BANDING],
        [TEXTURED_MOVING, GENUINE_BANDING],
    ):
        worst = max(mastering.banding_score(f) for f in frames)
        assert mastering.banding_risk_frames(frames) is not mastering._banding_gate_pass([worst])


# ------------------------------------------------------------------ whole-file gate


@pytest.mark.asyncio
async def test_a_normal_duplicate_heavy_master_passes_the_banding_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED end-to-end: the production master shape — textured, BT.709, on-target
    loudness, and duplicate-heavy like every fleet master — came back
    ``gate_pass=False`` with ``rerun_stages=['deband']`` and no way to satisfy it."""
    dump = (
        "frame:0 pts:0 pts_time:0\n"
        "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=0.0\n"
        "frame:1 pts:512 pts_time:0.017\n"
        "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"
        "frame:2 pts:1024 pts_time:0.034\n"
        "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=0.0\n"
    )

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "mpdecimate" in joined:
            return "frame= 3\n"
        return dump

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(),
        loudness={"integrated_lufs": -12.4, "true_peak_dbtp": -1.1},
    )
    assert "deband" not in report["rerun_stages"], report
    assert report["banding_scores"] == [0.0], report


@pytest.mark.asyncio
async def test_the_production_deband_call_site_skips_a_duplicate_heavy_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the real call site, not just the helper: ``_apply_conditional_deband``
    (``render_deband`` defaults to ``auto``, so it runs on every render).

    RED before the fix: one textured duplicate frame per segment — i.e. every
    segment of every fleet master — tripped the trigger and re-encoded the shot a
    SECOND time, so the documented "clean shots stay a byte-identical single pass"
    never held."""
    from src.services import measure

    debanded: list[Path] = []

    async def fake_frames(path: Path | str) -> list[dict[str, float]]:
        return [TEXTURED_MOVING, TEXTURED_DUPLICATE, TEXTURED_MOVING]

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        debanded.append(seg)

    monkeypatch.setattr(measure, "measure_shot_frames", fake_frames)
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)
    await renderer._apply_conditional_deband([Path("/tmp/seg-0.mp4")], _settings())
    assert debanded == [], "every shot was re-encoded a second time on a duplicate frame"

    async def banded_frames(path: Path | str) -> list[dict[str, float]]:
        return [TEXTURED_MOVING, GENUINE_BANDING]

    monkeypatch.setattr(measure, "measure_shot_frames", banded_frames)
    await renderer._apply_conditional_deband([Path("/tmp/seg-1.mp4")], _settings())
    assert debanded == [Path("/tmp/seg-1.mp4")], "genuine banding no longer triggers deband"


@pytest.mark.asyncio
async def test_an_unmeasured_master_fails_the_banding_gate_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED: one transient ffmpeg failure made ``measure_shot_frames`` return ``[]``;
    ``_banding_gate_pass([])`` is True and ``luma_spread``/``dead_ratio`` are both
    None, so a master nothing had measured came back ``gate_pass=True``. Loudness
    already fails CLOSED on the identical condition — banding must too."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "mpdecimate" in joined:
            return "frame= 3\n"
        raise renderer.FfmpegError("signalstats pass died")

    monkeypatch.setattr(renderer, "_run", fake_run)
    loudness = {"integrated_lufs": -12.4, "true_peak_dbtp": -1.1}

    # WAVE-13: the fail-closed REPORT stays exactly as it was — this gate is
    # tiered STRUCTURAL but deliberately left DISARMED, because a fire here
    # convicts the PROBE rather than the master and its rate is unmeasured.
    # Armed via its kill switch it raises instead; both directions asserted.
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(mastering_gate_banding_measurement_enabled=True),
            loudness=loudness,
        )
    entry = next(
        v for v in excinfo.value.violations_payload() if v["stage"] == "banding_measurement"
    )
    assert entry["tier"] == "structural"

    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness=loudness)
    assert report["banding_scores"] == []
    assert report["gate_pass"] is False, report
    assert "banding_measurement" in report["rerun_stages"], report
