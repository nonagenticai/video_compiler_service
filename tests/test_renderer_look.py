"""Command/graph-string tests for per-shot colour grade + text overlays.

ffmpeg is NOT invoked: these assert the ``eq``/``drawtext`` fragment SHAPE —
that an all-default grade is a no-op, that presets expand (with explicit fields
overriding), that overlays anchor correctly and route tenant text through an
injection-proof ``textfile=`` (never inline), that an unsafe ``fontcolor`` falls
back to the default, and that ``_apply_shot_look`` leaves the filter untouched
when the shot carries neither grade nor overlay.
Real muxing is proven by ``tests/render_harness/test_look_render.py``.
"""

from __future__ import annotations

from pathlib import Path

from src.api.schemas import ColorAdjust, Shot, TextOverlay
from src.config import get_settings
from src.services.renderer import _apply_shot_look, _eq_filter, _overlay_chain

# --------------------------------------------------------------------------
# _eq_filter: colour grade
# --------------------------------------------------------------------------


def test_eq_filter_none_is_empty() -> None:
    assert _eq_filter(None) == ""


def test_eq_filter_all_default_is_empty() -> None:
    # An all-default eq is a real identity filter, not a no-op — emit nothing so
    # the default render stays byte-identical.
    assert _eq_filter(ColorAdjust()) == ""


def test_eq_filter_changed_component_emits_only_that_component() -> None:
    assert _eq_filter(ColorAdjust(contrast=1.5)) == "eq=contrast=1.5"
    assert _eq_filter(ColorAdjust(saturation=0.0)) == "eq=saturation=0"
    # :g formatting collapses whole factors to integers.
    assert _eq_filter(ColorAdjust(gamma=2.0)) == "eq=gamma=2"


def test_eq_filter_preset_bw() -> None:
    assert _eq_filter(ColorAdjust(preset="bw")) == "eq=saturation=0"


def test_eq_filter_preset_cinematic() -> None:
    assert (
        _eq_filter(ColorAdjust(preset="cinematic")) == "eq=contrast=1.1:saturation=0.9:gamma=0.95"
    )


def test_eq_filter_preset_warm_uses_channel_gamma() -> None:
    assert _eq_filter(ColorAdjust(preset="warm")) == "eq=gamma_r=1.06:gamma_b=0.94"


def test_eq_filter_explicit_overrides_preset() -> None:
    # vivid sets saturation=1.4; an explicit saturation wins, the other preset
    # component (contrast) survives.
    out = _eq_filter(ColorAdjust(preset="vivid", saturation=2.0))
    assert "saturation=2" in out
    assert "saturation=1.4" not in out
    assert "contrast=1.1" in out


# --------------------------------------------------------------------------
# _overlay_chain: text overlays
# --------------------------------------------------------------------------


def test_overlay_chain_empty_is_empty(tmp_path: Path) -> None:
    assert _overlay_chain([], get_settings(), tmp_path, "s") == ""


def test_overlay_chain_positioned_and_textfile(tmp_path: Path) -> None:
    settings = get_settings()
    out = _overlay_chain(
        [TextOverlay(text="Hello: World", position="top-center")], settings, tmp_path, "s"
    )
    assert out.startswith("drawtext=")
    assert f"fontfile={settings.render_font}" in out
    assert "x=(w-text_w)/2" in out
    assert "y=60" in out
    assert "fontsize=48" in out
    assert "fontcolor=white" in out
    # Text goes through textfile= (never inline), so drawtext does no escape
    # interpretation and the raw text with its ``:`` lands verbatim in the file.
    assert f"textfile={tmp_path / 's-overlay-0.txt'}" in out
    assert ":text=" not in out  # no inline text= option that metachars could break out of
    assert (tmp_path / "s-overlay-0.txt").read_text() == "Hello: World"


def test_overlay_chain_text_cannot_inject_filtergraph(tmp_path: Path) -> None:
    # Filtergraph metacharacters (``;`` chainsep, ``,`` filtersep, ``:`` optsep,
    # quotes) in tenant text must NOT reach the graph string — they live only in
    # the textfile, so they can't break out of drawtext.
    nasty = "x[out];crop=1:1,drawtext=text=pwned:'\\evil"
    out = _overlay_chain(
        [TextOverlay(text=nasty, position="bottom-center")], get_settings(), tmp_path, "s"
    )
    assert ";" not in out and "crop=" not in out and "pwned" not in out
    assert (tmp_path / "s-overlay-0.txt").read_text() == nasty


def test_overlay_chain_unsafe_color_falls_back_to_default(tmp_path: Path) -> None:
    settings = get_settings()
    out = _overlay_chain(
        [TextOverlay(text="hi", color="red:x=1;drawtext=y")], settings, tmp_path, "s"
    )
    # The injected colour is rejected; the safe render default is used instead.
    assert f"fontcolor={settings.render_text_color}" in out
    assert "drawtext=y" not in out
    assert ";" not in out


def test_overlay_chain_valid_hex_color_passes(tmp_path: Path) -> None:
    out = _overlay_chain([TextOverlay(text="hi", color="#ff0000")], get_settings(), tmp_path, "s")
    assert "fontcolor=#ff0000" in out


def test_overlay_chain_bottom_center_anchor(tmp_path: Path) -> None:
    out = _overlay_chain(
        [TextOverlay(text="hi", position="bottom-center")], get_settings(), tmp_path, "s"
    )
    assert "x=(w-text_w)/2" in out
    assert "y=h-text_h-60" in out


def test_overlay_chain_explicit_xy_overrides_position(tmp_path: Path) -> None:
    out = _overlay_chain([TextOverlay(text="hi", x=100, y=200)], get_settings(), tmp_path, "s")
    assert "x=100" in out
    assert "y=200" in out


def test_overlay_chain_timed_uses_enable_between(tmp_path: Path) -> None:
    out = _overlay_chain(
        [TextOverlay(text="hi", start=1.0, end=3.0)], get_settings(), tmp_path, "s"
    )
    assert "enable='between(" in out
    assert r"between(t\,1\,3)" in out


def test_overlay_chain_multiple_joined_by_comma(tmp_path: Path) -> None:
    out = _overlay_chain(
        [
            TextOverlay(text="a", position="top-left"),
            TextOverlay(text="b", position="bottom-right"),
        ],
        get_settings(),
        tmp_path,
        "s",
    )
    assert out.count("drawtext=") == 2
    # top-left anchor and bottom-right anchor both present in the joined chain.
    assert "x=60" in out
    assert "x=w-text_w-60" in out
    # Each overlay gets its own textfile.
    assert (tmp_path / "s-overlay-0.txt").read_text() == "a"
    assert (tmp_path / "s-overlay-1.txt").read_text() == "b"


# --------------------------------------------------------------------------
# _apply_shot_look: chain composition + no-op invariant
# --------------------------------------------------------------------------


def test_apply_shot_look_noop_when_unset_returns_input_unchanged(tmp_path: Path) -> None:
    vf = "scale=1920:1080,setsar=1,fps=30,format=yuv420p"
    shot = Shot(id="s", seconds=1.0, image_url="http://m/a.png")
    assert _apply_shot_look(vf, shot, get_settings(), tmp_path) == vf


def test_apply_shot_look_appends_eq_then_overlays(tmp_path: Path) -> None:
    vf = "scale=2:2"
    shot = Shot(
        id="s",
        seconds=1.0,
        image_url="http://m/a.png",
        adjust=ColorAdjust(preset="bw"),
        overlays=[TextOverlay(text="hi", position="top-left")],
    )
    out = _apply_shot_look(vf, shot, get_settings(), tmp_path)
    assert out.startswith("scale=2:2,eq=saturation=0,drawtext=")


def test_apply_shot_look_grade_only(tmp_path: Path) -> None:
    vf = "scale=2:2"
    shot = Shot(id="s", seconds=1.0, image_url="http://m/a.png", adjust=ColorAdjust(contrast=1.2))
    assert _apply_shot_look(vf, shot, get_settings(), tmp_path) == "scale=2:2,eq=contrast=1.2"
