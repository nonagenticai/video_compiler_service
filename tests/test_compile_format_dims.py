"""``_settings_for_format`` — the deliverable-format → output-canvas override.

The renderer reads every output dimension from ``settings.render_{width,height}``,
so retargeting the aspect is a per-job ``model_copy`` of the cached settings. These
prove a 9x16 / 1x1 / 4x5 deliverable retargets the canvas, while an absent / unknown
/ already-default format leaves the settings object untouched (cheap, no copy).
"""

from __future__ import annotations

from src.api.routes.compile import _settings_for_format, _settings_for_tags
from src.config import Settings


def _settings() -> Settings:
    # Defaults are 16:9 (render_width=1920, render_height=1080).
    return Settings()


def test_portrait_9x16_retargets_canvas() -> None:
    out = _settings_for_format({"format": "9x16"}, _settings())
    assert (out.render_width, out.render_height) == (1080, 1920)


def test_square_and_4x5() -> None:
    assert _settings_for_format({"format": "1x1"}, _settings()).render_width == 1080
    s = _settings_for_format({"format": "4x5"}, _settings())
    assert (s.render_width, s.render_height) == (1080, 1350)


def test_default_16x9_returns_same_instance_unchanged() -> None:
    base = _settings()
    # 16x9 already matches the default → no copy needed.
    assert _settings_for_format({"format": "16x9"}, base) is base


def test_absent_or_unknown_format_is_a_noop() -> None:
    base = _settings()
    assert _settings_for_format(None, base) is base
    assert _settings_for_format({}, base) is base
    assert _settings_for_format({"format": "bogus"}, base) is base
    assert _settings_for_format({"format": ""}, base) is base


def test_override_does_not_mutate_the_shared_default() -> None:
    base = _settings()
    _settings_for_format({"format": "9x16"}, base)
    # model_copy must not touch the original (it's the lru_cached singleton).
    assert (base.render_width, base.render_height) == (1920, 1080)


# --------------------------------------------------------------------------
# _settings_for_tags — master_lufs clamp (shared with the CompileRequest field)
# --------------------------------------------------------------------------


def test_master_lufs_tag_in_range_overrides() -> None:
    out = _settings_for_tags(_settings(), {"master_lufs": "-16.0"})
    assert out.master_lufs == -16.0


def test_master_lufs_tag_out_of_range_ignored() -> None:
    base = _settings()  # default master_lufs = -12.5
    # Above the [-30, -5] clamp window → ignored, default kept.
    assert _settings_for_tags(base, {"master_lufs": "-2.0"}).master_lufs == base.master_lufs
    # Below the window → ignored too.
    assert _settings_for_tags(base, {"master_lufs": "-40.0"}).master_lufs == base.master_lufs


def test_master_lufs_tag_non_numeric_ignored() -> None:
    base = _settings()
    assert _settings_for_tags(base, {"master_lufs": "loud"}).master_lufs == base.master_lufs


# --------------------------------------------------------------------------
# 2K/4K resolution TIER contract — the orchestrator's compile_stage.py
# ``resolution_tags()`` (default tier "2k") sends the deliverable ``format``
# PLUS an explicit ``render_width``/``render_height`` pair computed from
# ``motion_slides.format_dims(fmt, tier)``. ``_FORMAT_DIMS`` in this module is
# an ASPECT-ONLY default (1080p canvas shape when no explicit dims arrive);
# the resolution TIER itself is carried by the render_width/render_height
# tags and applied by ``_settings_for_tags`` AFTER ``_settings_for_format``
# runs — reproducing that exact two-step order here so this test proves the
# real request-handling sequence in ``_compile_to_result``, not just one
# helper in isolation.
# --------------------------------------------------------------------------


def test_2k_tier_tags_override_the_1080p_format_default() -> None:
    """A 2K-tier request (16x9 format + explicit 2560x1440 tags) renders at
    true 2K, not the 1080p ``_FORMAT_DIMS`` default for 16x9."""
    tags = {"format": "16x9", "render_width": "2560", "render_height": "1440"}
    settings = _settings_for_format(tags, _settings())
    # Sanity: the format-only step alone still lands on the 1080p default —
    # this is the fallback _FORMAT_DIMS provides absent an explicit tier tag.
    assert (settings.render_width, settings.render_height) == (1920, 1080)
    settings = _settings_for_tags(settings, tags)
    assert (settings.render_width, settings.render_height) == (2560, 1440)


def test_default_1080p_tier_is_unchanged_with_no_resolution_tags() -> None:
    """No render_width/render_height tags at all → stays the 1080p default,
    so a run that never asked for a tier renders exactly as it did before this
    contract existed."""
    tags = {"format": "16x9"}
    settings = _settings_for_format(tags, _settings())
    settings = _settings_for_tags(settings, tags)
    assert (settings.render_width, settings.render_height) == (1920, 1080)


def test_2k_tier_honoured_for_a_non_default_aspect() -> None:
    """9x16 portrait at the 2K tier (720p9x16 base -> 2560/1920 scale, matching
    ``motion_slides._TIER_SCALE['2k'] = 2560/1920``, forced even) renders at the
    tier's true pixel size, not the 1080p portrait default (1080x1920)."""
    # 1080 * (2560/1920) = 1440; 1920 * (2560/1920) = 2560 — matches
    # motion_slides.format_dims("9x16", "2k").
    tags = {"format": "9x16", "render_width": "1440", "render_height": "2560"}
    settings = _settings_for_format(tags, _settings())
    assert (settings.render_width, settings.render_height) == (1080, 1920)
    settings = _settings_for_tags(settings, tags)
    assert (settings.render_width, settings.render_height) == (1440, 2560)
