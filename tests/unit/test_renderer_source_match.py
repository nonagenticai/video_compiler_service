"""Unit tests for T9 source-matching pure helpers (no ffmpeg).

``_median`` (the "one camera" luma/chroma target) and ``_source_match_correction``
(the per-shot combined ``eq=`` builder) are pure and deterministic — this exercises
the luma dark/bright/in-tolerance/absent branches, the saturation branches, and the
clamp bounds of both terms without any encode.
"""

from __future__ import annotations

from src.config import Settings, get_settings
from src.services import renderer

# A SATAVG target far from any single-term test's stats so the SAT term is a no-op
# there (its shot carries no SATAVG); luma-only tests pass this and stay unaffected.
_NO_SAT = 0.0


def _settings(tolerance: float = 6.0) -> Settings:
    """A real Settings with an explicit source-match tolerance."""
    return get_settings().model_copy(update={"source_match_tolerance": tolerance})


def test_median_odd() -> None:
    assert renderer._median([1.0, 2.0, 3.0]) == 2.0


def test_median_even() -> None:
    assert renderer._median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_median_single() -> None:
    assert renderer._median([5.0]) == 5.0


def test_median_empty() -> None:
    assert renderer._median([]) == 0.0


def test_median_unsorted() -> None:
    # median is order-independent
    assert renderer._median([3.0, 1.0, 2.0]) == 2.0


def test_correction_dark_shot_lifts() -> None:
    # A shot darker than the median (80 < 140) is lifted: gamma > 1.
    frag = renderer._source_match_correction({"YAVG": 80.0}, 140.0, _NO_SAT, _settings())
    assert frag.startswith("eq=gamma=")
    bias = float(frag.split("=")[-1])
    assert bias > 1.0


def test_correction_bright_shot_pulls_down() -> None:
    # A shot brighter than the median (200 > 140) is pulled down: gamma < 1.
    frag = renderer._source_match_correction({"YAVG": 200.0}, 140.0, _NO_SAT, _settings())
    assert frag.startswith("eq=gamma=")
    bias = float(frag.split("=")[-1])
    assert bias < 1.0


def test_correction_in_tolerance_is_noop() -> None:
    # Within the ±6 tolerance of the median → byte-identity (empty fragment).
    assert renderer._source_match_correction({"YAVG": 142.0}, 140.0, _NO_SAT, _settings()) == ""


def test_correction_zero_yavg_is_noop() -> None:
    assert renderer._source_match_correction({"YAVG": 0.0}, 140.0, _NO_SAT, _settings()) == ""


def test_correction_absent_yavg_is_noop() -> None:
    assert renderer._source_match_correction({}, 140.0, _NO_SAT, _settings()) == ""


def test_correction_clamps_at_max_for_extreme_dark() -> None:
    # An extremely dark shot against a bright target hits the upper gamma clamp.
    frag = renderer._source_match_correction({"YAVG": 1.0}, 250.0, _NO_SAT, _settings())
    bias = float(frag.split("=")[-1])
    assert bias == renderer._SRC_MATCH_GAMMA_MAX


def test_correction_clamps_at_min_for_extreme_bright() -> None:
    # An extremely bright shot against a dark target hits the lower gamma clamp.
    frag = renderer._source_match_correction({"YAVG": 250.0}, 1.0, _NO_SAT, _settings())
    bias = float(frag.split("=")[-1])
    assert bias == renderer._SRC_MATCH_GAMMA_MIN


# --- Saturation term ------------------------------------------------------------


def _sat_bias(frag: str) -> float:
    """Extract the saturation= value from a ``saturation``-only ``eq=`` fragment."""
    assert frag.startswith("eq=saturation=")
    return float(frag.split("=")[-1])


def test_correction_low_sat_shot_boosts() -> None:
    # Under-saturated shot (40 < 90) with luma on-target → only a saturation>1 term.
    frag = renderer._source_match_correction(
        {"YAVG": 140.0, "SATAVG": 40.0}, 140.0, 90.0, _settings()
    )
    assert "gamma=" not in frag
    assert _sat_bias(frag) > 1.0


def test_correction_high_sat_shot_pulls_in() -> None:
    # Over-saturated shot (140 > 90) with luma on-target → only a saturation<1 term.
    frag = renderer._source_match_correction(
        {"YAVG": 140.0, "SATAVG": 140.0}, 140.0, 90.0, _settings()
    )
    assert "gamma=" not in frag
    assert _sat_bias(frag) < 1.0


def test_correction_sat_clamps_at_max() -> None:
    # A near-grey shot (tiny SATAVG) against a colourful target hits the SAT max clamp.
    frag = renderer._source_match_correction(
        {"YAVG": 140.0, "SATAVG": 1.0}, 140.0, 200.0, _settings()
    )
    assert _sat_bias(frag) == renderer._SRC_MATCH_SAT_MAX


def test_correction_sat_clamps_at_min() -> None:
    # An extremely saturated shot against a near-grey target hits the SAT min clamp.
    frag = renderer._source_match_correction(
        {"YAVG": 140.0, "SATAVG": 250.0}, 140.0, 1.0, _settings()
    )
    assert _sat_bias(frag) == renderer._SRC_MATCH_SAT_MIN


def test_correction_in_sat_tolerance_is_noop() -> None:
    # SATAVG within ±6 of the target (and luma on-target) → no saturation term.
    frag = renderer._source_match_correction(
        {"YAVG": 140.0, "SATAVG": 92.0}, 140.0, 90.0, _settings()
    )
    assert frag == ""


def test_correction_absent_satavg_is_luma_only() -> None:
    # SATAVG absent → luma-only behaviour, byte-identical to the gamma-only era.
    frag = renderer._source_match_correction({"YAVG": 80.0}, 140.0, 90.0, _settings())
    assert frag.startswith("eq=gamma=")
    assert "saturation=" not in frag


def test_correction_zero_satavg_is_luma_only() -> None:
    # A legitimately grey shot (SATAVG 0) contributes no SAT term; luma still fires.
    frag = renderer._source_match_correction(
        {"YAVG": 80.0, "SATAVG": 0.0}, 140.0, 90.0, _settings()
    )
    assert frag.startswith("eq=gamma=")
    assert "saturation=" not in frag


def test_correction_zero_target_sat_is_luma_only() -> None:
    # target_satavg 0.0 (no valid SAT median across the film) → SAT term never fires.
    frag = renderer._source_match_correction(
        {"YAVG": 80.0, "SATAVG": 40.0}, 140.0, 0.0, _settings()
    )
    assert frag.startswith("eq=gamma=")
    assert "saturation=" not in frag


def test_correction_both_terms_combine_single_eq() -> None:
    # Out-of-tolerance on BOTH luma and saturation → one combined eq= (one re-encode).
    frag = renderer._source_match_correction(
        {"YAVG": 80.0, "SATAVG": 40.0}, 140.0, 90.0, _settings()
    )
    assert frag.startswith("eq=")
    assert "gamma=" in frag
    assert "saturation=" in frag
    # A single eq= filter, terms colon-joined — one filter, one re-encode.
    assert frag.count("eq=") == 1
    assert ":" in frag.removeprefix("eq=")


def test_correction_both_in_tolerance_is_noop() -> None:
    # Both luma AND saturation in tolerance → empty fragment (byte-identical shot).
    frag = renderer._source_match_correction(
        {"YAVG": 142.0, "SATAVG": 92.0}, 140.0, 90.0, _settings()
    )
    assert frag == ""
