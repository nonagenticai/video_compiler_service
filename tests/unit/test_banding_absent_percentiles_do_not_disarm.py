"""An absent `YHIGH`/`YLOW` must NOT silently zero the banding score.

The spatial term added 2026-09-02 divides the 10-90 luma band by the full range.
Written as ``stats.get("YHIGH", 0.0) - stats.get("YLOW", 0.0)`` it makes a stats
dict carrying neither percentile score exactly **0.0** — for every frame, on
every segment. That is not "no banding": it is the gate DISARMED, in the
fail-open direction, from a measurement gap. It shipped that way for one CI run
and turned ten existing tests red; the ones it did not turn red are the reason
this file exists, because a real truncated parse would have been silent.

``min_luma_spread`` two functions down already refuses the same trade in as many
words — "a measurement gap is never counted as zero spread" — so the fallback is
the module's own established convention, not a new one.

Pure; no ffmpeg.
"""

from __future__ import annotations

from src.services import mastering

# Wide range + low temporal delta: the shape that scored over the ceiling before
# the spatial term existed, and the shape every pre-2026-09-02 fixture uses.
_LEGACY = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 0.5}


def test_absent_percentiles_keep_the_pre_spatial_term_score() -> None:
    """Unmeasured spatial evidence => uniformity 1.0, i.e. the old formula."""
    score = mastering.banding_score(_LEGACY)

    span = (219.0 - mastering._BANDING_RANGE_MIN) / (255.0 - mastering._BANDING_RANGE_MIN)
    smoothness = (mastering._BANDING_YDIF_MAX - 0.5) / mastering._BANDING_YDIF_MAX
    assert score == round(span * smoothness, 4)
    assert score >= mastering.BANDING_GATE_CEILING, (
        "a stats dict with no percentiles must still be able to trip the gate — "
        "scoring it 0.0 would disarm the gate on a measurement gap"
    )


def test_one_absent_percentile_is_also_unmeasured() -> None:
    """Half the evidence is not evidence: `YHIGH` alone would read as YLOW==0."""
    for partial in ({"YHIGH": 200.0}, {"YLOW": 30.0}):
        assert mastering.banding_score(_LEGACY | partial) == mastering.banding_score(_LEGACY)


def test_present_percentiles_do_suppress_detailed_content() -> None:
    """The control: when the evidence IS there, the spatial term must bite.

    Without this, the two tests above would pass on a build where the spatial
    term had been deleted outright.
    """
    # A narrow 10-90 band inside a wide range == detail with outliers, not a ramp.
    detailed = _LEGACY | {"YLOW": 100.0, "YHIGH": 140.0}
    assert mastering.banding_score(detailed) < mastering.BANDING_GATE_CEILING

    # A band covering nearly the whole range == a ramp; still flagged.
    ramp = _LEGACY | {"YLOW": 26.0, "YHIGH": 225.0}
    assert ramp["YHIGH"] - ramp["YLOW"] > 0.9 * (ramp["YMAX"] - ramp["YMIN"])
    assert mastering.banding_score(ramp) >= mastering.BANDING_GATE_CEILING
