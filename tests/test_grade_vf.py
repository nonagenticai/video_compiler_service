"""Pure-string tests for the whole-film colour-grade filter builder (VC-5).

These exercise ``renderer._grade_vf`` / ``renderer.grade_report`` as PURE
functions — no ffmpeg, no I/O. They pin the filtergraph SHAPE (each knob → its
expected filter substring, identity → the empty byte-identity gate). The real
picture/cadence/loudness proofs live in the render-harness suite
(``tests/render_harness/test_grade_render.py``), which this module does NOT
touch.
"""

from __future__ import annotations

from src.api.schemas import GradeSpec
from src.services import renderer


def _g(**kw: object) -> GradeSpec:
    """A grade with ``enabled=True`` and every other knob at its identity default,
    overridden by ``kw`` — so a single-knob test isolates that knob's filter."""
    return GradeSpec.model_validate({"enabled": True, **kw})


# --- the byte-identity gate ------------------------------------------------


def test_default_grade_is_noop() -> None:
    # Default GradeSpec is disabled → empty chain (the -c copy fast path).
    assert renderer._grade_vf(GradeSpec()) == ""


def test_enabled_but_all_identity_is_noop() -> None:
    # Enabled but every knob identity → still no-op (nothing to encode).
    assert renderer._grade_vf(_g()) == ""


def test_disabled_grade_ignores_set_knobs() -> None:
    # enabled=False is the master switch: knobs set but not enabled → no-op.
    assert renderer._grade_vf(GradeSpec(enabled=False, contrast=1.4, saturation=1.3)) == ""


def test_zero_intensity_is_noop() -> None:
    # grade_intensity=0 scales every knob back to identity → no-op.
    assert renderer._grade_vf(_g(contrast=1.4, saturation=1.4, gamma=1.5, vignette=1.0)) != ""
    assert renderer._grade_vf(_g(contrast=1.4, saturation=1.4, grade_intensity=0.0)) == ""


# --- eq: look / contrast / saturation / gamma / temperature ----------------


def test_contrast_emits_eq() -> None:
    out = renderer._grade_vf(_g(contrast=1.4))
    assert out.startswith("eq=")
    assert "contrast=1.4" in out


def test_saturation_emits_eq() -> None:
    assert "saturation=1.3" in renderer._grade_vf(_g(saturation=1.3))


def test_gamma_emits_eq() -> None:
    assert "gamma=1.5" in renderer._grade_vf(_g(gamma=1.5))


def test_temperature_emits_per_channel_gamma() -> None:
    out = renderer._grade_vf(_g(temperature=1.0))
    # warm bias: +red gamma, -blue gamma.
    assert "gamma_r=1.12" in out
    assert "gamma_b=0.88" in out


def test_look_preset_supplies_base_grade() -> None:
    # 'vivid' preset = contrast 1.1 + saturation 1.4 (reuses _COLOR_PRESETS).
    out = renderer._grade_vf(_g(look="vivid"))
    assert "contrast=1.1" in out
    assert "saturation=1.4" in out


def test_unknown_look_is_not_an_error() -> None:
    # A free-string look that isn't a preset is a no-op base, never a raise.
    assert renderer._grade_vf(_g(look="teal-and-orange")) == ""


# --- split-tone / vignette / grain / sharpen -------------------------------


def test_split_tone_emits_colorbalance() -> None:
    # Blue shadows against the default white highlights → a colorbalance shift.
    out = renderer._grade_vf(_g(shadow_tint="#0000ff"))
    assert "colorbalance=" in out
    assert "bs=" in out


def test_neutral_tints_emit_no_colorbalance() -> None:
    # A grey shadow tint has no chroma cast → no split-tone (identity default too).
    assert "colorbalance" not in renderer._grade_vf(_g(shadow_tint="#808080", contrast=1.4))


def test_vignette_emits_filter() -> None:
    assert "vignette=a=" in renderer._grade_vf(_g(vignette=0.5))


def test_grain_emits_noise() -> None:
    out = renderer._grade_vf(_g(grain=0.5))
    assert "noise=alls=20" in out
    assert "allf=t+u" in out


def test_sharpen_emits_unsharp() -> None:
    out = renderer._grade_vf(_g(sharpen=0.5))
    assert "unsharp=" in out
    assert "luma_amount=0.75" in out


def test_glow_emits_gblur_blend_bloom() -> None:
    out = renderer._grade_vf(_g(glow=0.5))
    assert "split[gbase][gblur]" in out
    assert "gblur=sigma=8" in out
    assert "blend=all_mode=screen:all_opacity=0.5" in out


# --- broadcast_safe legal clamp --------------------------------------------


def test_broadcast_safe_adds_colorlevels_clamp_last() -> None:
    out = renderer._grade_vf(_g(contrast=1.4, broadcast_safe=True))
    assert "colorlevels=" in out
    assert "235/255" in out
    # The legal clamp is the LAST filter in the chain.
    assert out.rstrip().endswith("bomax=235/255")


def test_broadcast_safe_alone_still_reencodes() -> None:
    # broadcast_safe is a delivery guard, not an artistic knob: it engages a
    # re-encode even with every look knob at identity.
    out = renderer._grade_vf(_g(broadcast_safe=True))
    assert out == renderer._BROADCAST_SAFE_CLAMP


def test_glow_then_broadcast_safe_clamp_is_after_blend() -> None:
    out = renderer._grade_vf(_g(glow=0.5, broadcast_safe=True))
    assert out.index("blend=") < out.index("colorlevels=")
    assert out.rstrip().endswith("bomax=235/255")


# --- grade_intensity scaling -----------------------------------------------


def test_intensity_scales_knob_toward_identity() -> None:
    # contrast 1.4 at 50% intensity → 1.0 + (1.4-1)*0.5 = 1.2.
    assert "contrast=1.2" in renderer._grade_vf(_g(contrast=1.4, grade_intensity=0.5))


# --- match_sources bias ----------------------------------------------------


def test_match_sources_biases_gamma_toward_measured_luma() -> None:
    # A dark film (low YAVG) with match_sources → gamma lifted above identity.
    out = renderer._grade_vf(_g(match_sources=True), stats={"YAVG": 60.0})
    assert "gamma=" in out


def test_stats_ignored_without_match_sources() -> None:
    # Stats present but match_sources off → the film's own metrics are ignored.
    assert renderer._grade_vf(_g(), stats={"YAVG": 60.0}) == ""


# --- grade_report ----------------------------------------------------------


def test_grade_report_none_for_no_grade() -> None:
    assert renderer.grade_report(None) is None


def test_grade_report_reports_reencoded_flag() -> None:
    disabled = renderer.grade_report(GradeSpec())
    assert disabled is not None and disabled["reencoded"] is False
    active = renderer.grade_report(GradeSpec(enabled=True, contrast=1.4))
    assert active is not None and active["reencoded"] is True
    # match_sources ONLY (no static knob): _grade_vf(stats=None) is empty, but the
    # measured bias re-encodes at render time on any real footage — reencoded must
    # be honest (True), not a false "no re-encode" on the cost/QA surface.
    matched = renderer.grade_report(GradeSpec(enabled=True, match_sources=True))
    assert matched is not None and matched["reencoded"] is True
    # Reports the requested knobs, never a fabricated measured grade.
    assert active["contrast"] == 1.4
    assert active["look"] == "none"
