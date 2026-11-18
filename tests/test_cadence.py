"""Unit tests for the frame-rate cadence SELECTION module (VC-3, pure)."""

from __future__ import annotations

from src.services import cadence

# ---------------------------------------------------------------------------
# clamp_cadence
# ---------------------------------------------------------------------------


def test_clamp_cadence_maps_to_the_five_legal_rates() -> None:
    assert cadence.clamp_cadence(23.976) == 24
    assert cadence.clamp_cadence(24) == 24
    assert cadence.clamp_cadence(25) == 25
    assert cadence.clamp_cadence(29.97) == 30
    assert cadence.clamp_cadence(48) == 50
    assert cadence.clamp_cadence(59.94) == 60
    assert cadence.clamp_cadence(1000) == 60


def test_clamp_cadence_degrades_bad_input_to_default() -> None:
    assert cadence.clamp_cadence(None) == 30
    assert cadence.clamp_cadence("garbage") == 30
    assert cadence.clamp_cadence(0) == 30
    assert cadence.clamp_cadence(-5) == 30


def test_clamp_cadence_parses_ffprobe_rational() -> None:
    assert cadence.clamp_cadence("30000/1001") == 30
    assert cadence.clamp_cadence("25/1") == 25


# ---------------------------------------------------------------------------
# retime_plan
# ---------------------------------------------------------------------------


def test_retime_plan_exact_ratio_is_clean_fps_fragment() -> None:
    assert cadence.retime_plan(30, 30) == "fps=30"
    assert cadence.retime_plan(30, 60) == "fps=60"  # integer 2x
    assert cadence.retime_plan(50, 25) == "fps=25"  # integer /2


def test_retime_plan_default_is_fps_30() -> None:
    """The load-bearing byte-identity anchor: default cadence → 'fps=30'."""
    assert cadence.retime_plan(30, 30, high_motion=False, budget=False) == "fps=30"


def test_retime_plan_non_integer_high_motion_is_minterpolate() -> None:
    frag = cadence.retime_plan(25, 30, high_motion=True)
    assert frag == "minterpolate=fps=30:mi_mode=mci"


def test_retime_plan_non_integer_budget_uses_blend_variant() -> None:
    frag = cadence.retime_plan(25, 30, budget=True)
    assert "tblend" in frag
    assert frag.endswith("fps=30")


def test_retime_plan_non_integer_no_flags_accepts_judder_clean() -> None:
    assert cadence.retime_plan(25, 30) == "fps=30"


def test_retime_plan_unknown_source_stays_clean() -> None:
    assert cadence.retime_plan(None, 30, high_motion=True) == "fps=30"


# ---------------------------------------------------------------------------
# judder_risk
# ---------------------------------------------------------------------------


def test_judder_risk_zero_for_integer_ratios() -> None:
    assert cadence.judder_risk(30, 30) == 0.0
    assert cadence.judder_risk(30, 60) == 0.0
    assert cadence.judder_risk(25, 50) == 0.0


def test_judder_risk_positive_for_non_integer_ratio() -> None:
    assert cadence.judder_risk(25, 30) > 0.0
    assert cadence.judder_risk(24, 30) > 0.0


def test_judder_risk_zero_when_rate_unknown() -> None:
    assert cadence.judder_risk(None, 30) == 0.0
    assert cadence.judder_risk(25, 0) == 0.0


# ---------------------------------------------------------------------------
# source_rate_profile + master_cadence
# ---------------------------------------------------------------------------


def test_source_rate_profile_summarizes_rates() -> None:
    prof = cadence.source_rate_profile([25, 25, 30, None, "garbage", "30000/1001"])
    assert prof["count"] == 4  # two 25s, one 30, one ~29.97
    assert prof["dominant"] == 25.0
    assert prof["rates"][25.0] == 2


def test_master_cadence_empty_profiles_defaults_to_30() -> None:
    assert cadence.master_cadence([]) == 30
    assert cadence.master_cadence([cadence.source_rate_profile([])]) == 30


def test_master_cadence_picks_zero_cost_rate() -> None:
    # All-25 sources: 25 is a zero-judder pick over the default 30.
    prof = cadence.source_rate_profile([25, 25, 25])
    assert cadence.master_cadence([prof]) == 25


def test_master_cadence_all_24_picks_24() -> None:
    prof = cadence.source_rate_profile([24, 24])
    assert cadence.master_cadence([prof]) == 24


def test_master_cadence_all_30_stays_30() -> None:
    prof = cadence.source_rate_profile([30, 30, 30])
    assert cadence.master_cadence([prof]) == 30


def test_master_cadence_respects_restricted_candidates() -> None:
    # Only 30/60 allowed: 25 sources snap to the min-cost of those (30, closest).
    prof = cadence.source_rate_profile([25, 25])
    assert cadence.master_cadence([prof], candidates=[30, 60]) == 30


# ---------------------------------------------------------------------------
# master_cadence — the judder cost term must be DECISIVE
#
# Every master_cadence test above is satisfied by the `abs(candidate -
# dominant)` tie-break ALONE: they would pass identically with the judder term
# deleted, so none of them covers `cost()`'s first element.
#
# The three below are each built so some DEGRADED cost function disagrees with
# the real one — the only shape that detects a broken judder term:
#   * the first two make the nearest-rate tie-break and the full cost function
#     pick different rates (catches a destroyed/zeroed judder term);
#   * the third makes the count-WEIGHTED and UNWEIGHTED sums pick different
#     rates (catches a dropped `* w`, which the first two are blind to).
# ---------------------------------------------------------------------------


def _nearest_rate_only(dominant: float) -> int:
    """What master_cadence would pick if the judder term were destroyed —
    i.e. min over the legal rates by (distance-to-dominant, rate) alone."""
    return min(cadence._LEGAL_RATES, key=lambda r: (abs(r - dominant), r))


def test_master_cadence_judder_term_overrides_the_nearest_rate_tiebreak() -> None:
    """All-15fps sources: 30 is an EXACT 2x (zero judder), but 24 is the rate
    NEAREST to the 15fps dominant. The judder term must win.

    Kill the judder term and every candidate costs 0.0, so the tie-break picks
    24 — a 1.6x ratio that stutters. This asserts the disagreement explicitly so
    the test cannot silently degrade into a nearest-rate assertion.
    """
    prof = cadence.source_rate_profile([15, 15, 15])
    assert prof["dominant"] == 15.0
    assert _nearest_rate_only(15.0) == 24, "precondition: nearest-rate alone picks 24"
    assert cadence.judder_risk(15, 30) == 0.0, "15->30 is an exact 2x"
    assert cadence.judder_risk(15, 24) > 0.0, "15->24 is a 1.6x stutter"

    assert cadence.master_cadence([prof]) == 30


def test_master_cadence_mixed_rate_reel_picks_the_judder_clean_master() -> None:
    """A mixed 15/30 reel: 30 is judder-free for BOTH source rates while 24
    judders on both. Dominant is still 15, so the nearest-rate tie-break alone
    would again pick 24.

    This case does NOT cover the `count` weighting — 30 costs 0.0 for every
    source rate present, so dropping the `* w` factor cannot flip the winner
    here. `test_master_cadence_count_weight_is_decisive` below covers that.
    """
    prof = cadence.source_rate_profile([15, 15, 15, 30])
    assert prof["dominant"] == 15.0
    assert prof["rates"] == {15.0: 3, 30.0: 1}
    assert _nearest_rate_only(15.0) == 24, "precondition: nearest-rate alone picks 24"

    assert cadence.master_cadence([prof]) == 30
    # 24 is not merely worse — it judders on every source rate present.
    assert cadence.judder_risk(15, 24) > 0.0
    assert cadence.judder_risk(30, 24) > 0.0


def test_master_cadence_count_weight_is_decisive() -> None:
    """A mostly-24fps film reel with one stray 30 and one stray 60 clip must
    master at 24 — the MAJORITY rate — not at whatever minimises the count of
    judder-y source rates.

    This is the case that covers the `* w` count weighting in `cost()`, which
    the two tests above cannot: drop the weight and the per-rate judder sums
    become 24 -> 1.5 but 30 -> 0.5, so an unweighted cost function masters the
    whole reel at 30 and judders SIX clips to spare one. With the weight, 24
    costs 1.5 against 30's 3.0 and correctly wins.

    Note the nearest-rate tie-break alone ALSO picks 24 here (24 is the
    dominant), so this case is blind to a destroyed judder term — it is the
    complement of the two tests above, not a replacement for them.
    """
    prof = cadence.source_rate_profile([24] * 6 + [30, 60])
    assert prof["rates"] == {24.0: 6, 30.0: 1, 60.0: 1}
    assert prof["dominant"] == 24.0

    assert cadence.master_cadence([prof]) == 24

    # Pin the disagreement the weight creates, so the test cannot silently
    # decay into "the dominant rate wins" if the cost function is reshaped.
    unweighted = {
        c: sum(cadence.judder_risk(r, c) for r in prof["rates"]) for c in (24, 25, 30, 50, 60)
    }
    assert min(unweighted, key=lambda c: unweighted[c]) == 30, (
        "precondition: an UNWEIGHTED cost function would master this reel at 30"
    )
