"""Command/graph-string tests for directed footage (ORCH-3.2): the synthetic
cursor + auto-zoom filter builders.

ffmpeg is NOT invoked: these assert the ``zoompan``/``overlay``/``geq`` fragment
SHAPE — that an absent track is a pass-through (the byte-identical no-op the
default render depends on), that a present track emits the expected filter with
a smoothstep-eased expression, and that the ``z=clamp(sqrt(...))`` zoom math is
correct for a known bbox. Real muxing is proven by
``tests/render_harness/test_directed_footage_render.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import Callout, CursorKey, FocusKey, Shot, SplitScreen
from src.config import get_settings
from src.services import renderer
from src.services.renderer import (
    _CAPTURE_HEADROOM,
    _apply_directed_footage,
    _callout_filter,
    _cursor_filter,
    _even_dim,
    _focus_zoom_filter,
    _focus_zoom_target,
    _remap_tracks_to_canvas,
    _smoothstep_expr,
    _split_filter,
)

_SMALL = {"render_width": 320, "render_height": 240, "render_fps": 25}


def _settings():
    return get_settings().model_copy(update=_SMALL)


# --------------------------------------------------------------------------
# _smoothstep_expr
# --------------------------------------------------------------------------


def test_smoothstep_degenerate_window_is_constant_one() -> None:
    assert _smoothstep_expr("on", 5.0, 5.0) == "1"
    assert _smoothstep_expr("on", 5.0, 2.0) == "1"


def test_smoothstep_has_hermite_shape() -> None:
    e = _smoothstep_expr("on", 0.0, 10.0)
    # clamped normalized input, escaped commas, and the p*p*(3-2p) Hermite curve.
    assert "clip((on-0)/10\\,0\\,1)" in e
    assert "(3-2*(" in e


# --------------------------------------------------------------------------
# _focus_zoom_target — the z=clamp(sqrt(...)) math
# --------------------------------------------------------------------------


def test_focus_zoom_target_known_bbox() -> None:
    # 0.6 * 1920*1080 / (768*720) = 2.25 → sqrt = 1.5 (inside the clamp band).
    assert _focus_zoom_target((0.0, 0.0, 768.0, 720.0), 1920, 1080) == pytest.approx(1.5)


def test_focus_zoom_target_tiny_bbox_clamps_to_headroom() -> None:
    # A tiny control needs a huge zoom → clamped to the capture headroom.
    assert _focus_zoom_target((0.0, 0.0, 8.0, 8.0), 1920, 1080) == _CAPTURE_HEADROOM


def test_focus_zoom_target_full_frame_clamps_to_one() -> None:
    # A bbox already larger than the fill target → no punch-in (z == 1.0).
    assert _focus_zoom_target((0.0, 0.0, 1920.0, 1080.0), 1920, 1080) == 1.0


# --------------------------------------------------------------------------
# _focus_zoom_filter
# --------------------------------------------------------------------------


def test_focus_zoom_none_is_passthrough() -> None:
    shot = Shot(id="a", seconds=2.0, focus_track=None)
    assert _focus_zoom_filter("scale=320:240", shot, _settings(), 2.0) == "scale=320:240"


def test_focus_zoom_full_frame_bbox_is_passthrough() -> None:
    # target zoom clamps to 1.0 → no zoompan appended.
    shot = Shot(id="a", seconds=2.0, focus_track=[FocusKey(t=0.0, bbox=(0, 0, 320, 240))])
    assert _focus_zoom_filter("scale=320:240", shot, _settings(), 2.0) == "scale=320:240"


def test_focus_zoom_appends_eased_zoompan() -> None:
    shot = Shot(id="a", seconds=2.0, focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))])
    out = _focus_zoom_filter("scale=320:240", shot, _settings(), 2.0)
    assert out.startswith(
        "scale=320:240," + renderer._zoompan_prescale(320, 240, _settings()) + ",zoompan=z='1+"
    )
    # eased envelope (smoothstep clip + min of ease-in / ease-out)
    assert "clip((on-" in out
    assert "min(" in out
    # Crop window tracks the bbox centre (140+20=160, 100+20=120) — expressed in
    # the PRE-SCALED space, so x2 by _ZOOMPAN_PRESCALE. An unscaled centre here
    # would punch at half the intended position against the doubled iw/ih.
    assert "320-iw/zoom/2" in out
    assert "240-ih/zoom/2" in out
    assert ":d=1:s=320x240:fps=25" in out


def test_focus_zoom_chains_onto_cursor_graph_via_filter_complex() -> None:
    # When a cursor sub-graph (a string containing ';') precedes it, the zoompan
    # must chain on as a labeled filter_complex continuation ([fz];[fz]zoompan…)
    # — NOT a bare ',zoompan' (which is invalid after a filter_complex graph) —
    # so the punch-in magnifies the composited cursor together with the frame.
    cursor_graph = "scale=320:240[cbase];color=…[cursor];[cbase][cursor]overlay=x='10':y='10'"
    shot = Shot(id="a", seconds=2.0, focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))])
    out = _focus_zoom_filter(cursor_graph, shot, _settings(), 2.0)
    assert out.startswith(
        cursor_graph
        + "[fz];[fz]"
        + renderer._zoompan_prescale(320, 240, _settings())
        + ",zoompan=z='1+"
    )
    # The property is that the continuation is reached through a LABEL, never
    # comma-appended onto the previous graph. (A bare `,zoompan` no longer states
    # it: the comma before zoompan is now the lanczos prescale inside that same
    # labelled chain, which is valid.)
    assert "[fz];[fz]" in out
    assert cursor_graph + "," not in out


def test_focus_zoom_passthrough_preserves_cursor_graph_when_bbox_full_frame() -> None:
    # A no-op zoom (full-frame bbox) must leave an upstream cursor graph untouched.
    cursor_graph = "scale=320:240[cbase];x[cursor];[cbase][cursor]overlay=x='1':y='1'"
    shot = Shot(id="a", seconds=2.0, focus_track=[FocusKey(t=0.0, bbox=(0, 0, 320, 240))])
    assert _focus_zoom_filter(cursor_graph, shot, _settings(), 2.0) == cursor_graph


def test_focus_zoom_multi_keyframe_interpolates_center_and_zoom() -> None:
    # N>=2 focus keyframes → the punch centre AND per-key zoom target are
    # smoothstep-interpolated over the zoompan `on` domain (nested if(lt(on\,…))),
    # not frozen on the first click, while still riding the ease envelope.
    shot = Shot(
        id="a",
        seconds=2.0,
        focus_track=[
            FocusKey(t=0.0, bbox=(140, 100, 40, 40)),  # centre (160,120)
            FocusKey(t=1.0, bbox=(20, 20, 40, 40)),  # centre (40,40)
        ],
    )
    out = _focus_zoom_filter("scale=320:240", shot, _settings(), 2.0)
    # nested-if interpolation in the output-frame domain.
    assert "if(lt(on\\," in out
    # both keyframe centres appear (start 160/120, end 40/40).
    assert "160" in out
    assert "40" in out
    # still wrapped in the ease-in / ease-out envelope.
    assert "min(" in out


def test_focus_zoom_multi_all_full_frame_passthrough() -> None:
    # Two keys that BOTH already fill the frame → max target zoom == 1.0 → the
    # whole track is a pass-through (byte-identical no-op).
    shot = Shot(
        id="a",
        seconds=2.0,
        focus_track=[
            FocusKey(t=0.0, bbox=(0, 0, 320, 240)),
            FocusKey(t=1.0, bbox=(0, 0, 320, 240)),
        ],
    )
    assert _focus_zoom_filter("scale=320:240", shot, _settings(), 2.0) == "scale=320:240"


# --------------------------------------------------------------------------
# _apply_directed_footage — speed-retime time-axis scaling (BUG #3)
# --------------------------------------------------------------------------


def test_apply_directed_footage_scales_times_under_speed(tmp_path) -> None:
    # A downstream setpts=PTS/speed compresses the overlay `t` and zoompan `on`
    # ranges by 1/speed, so every authored keyframe/callout time is divided by
    # speed and the zoom envelope runs against the retimed duration/speed.
    shot = Shot(
        id="a",
        seconds=4.0,
        speed=2.0,
        cursor_track=[CursorKey(t=1.0, x=50, y=50)],
        callouts=[Callout(at=1.0, type="badge", bbox=(100, 80, 40, 40), label="1")],
        focus_track=[FocusKey(t=1.0, bbox=(140, 100, 40, 40))],
    )
    out = _apply_directed_footage("scale=320:240", shot, _settings(), 4.0, 2.0, tmp_path)
    # callout onset halved: 1.0 → 0.5 (never the un-scaled 1.0).
    assert "gte(t\\,0.5)" in out
    assert "gte(t\\,1)" not in out
    # zoom ease-out anchors to the RETIMED duration/speed = 2.0s → total=50f,
    # ease=12.5f, ease-out ramp starts at frame 37.5 (not 87.5 for the un-scaled 4s).
    assert "on-37.5" in out
    assert "on-87.5" not in out


def test_apply_directed_footage_speed_one_is_byte_identical(tmp_path) -> None:
    # speed==1.0 must not perturb anything: the output is byte-identical to running
    # the three directed-footage filters directly at the un-scaled duration.
    settings = _settings()
    shot = Shot(
        id="a",
        seconds=4.0,
        cursor_track=[CursorKey(t=1.0, x=50, y=50)],
        callouts=[Callout(at=1.0, type="badge", bbox=(100, 80, 40, 40), label="1")],
        focus_track=[FocusKey(t=1.0, bbox=(140, 100, 40, 40))],
    )
    base = "scale=320:240"
    expected = _cursor_filter(base, shot, settings, 4.0)
    expected = _callout_filter(expected, shot, settings, 4.0, tmp_path)
    expected = _focus_zoom_filter(expected, shot, settings, 4.0)
    assert _apply_directed_footage(base, shot, settings, 4.0, 1.0, tmp_path) == expected


# --------------------------------------------------------------------------
# _cursor_filter
# --------------------------------------------------------------------------


def test_cursor_none_is_passthrough() -> None:
    shot = Shot(id="a", seconds=2.0, cursor_track=None)
    assert _cursor_filter("scale=320:240", shot, _settings(), 2.0) == "scale=320:240"


def test_cursor_emits_synthesized_disc_overlay_graph() -> None:
    shot = Shot(
        id="a",
        seconds=2.0,
        cursor_track=[CursorKey(t=0.0, x=50, y=60), CursorKey(t=1.0, x=200, y=180)],
    )
    out = _cursor_filter("scale=320:240", shot, _settings(), 2.0)
    # multi-chain graph: base label, a synthesized rgba alpha-disc source, overlay.
    assert "scale=320:240[cbase]" in out
    assert "format=rgba" in out
    assert "geq=" in out
    assert "[cbase][cursor]overlay=" in out
    # the base takes [0:v] and the last overlay is label-less so the seam adds [v].
    assert not out.endswith("]")
    # smoothstep interpolation between keyframes, evaluated on the overlay time t.
    assert "if(lt(t\\," in out
    # no click keyframe → no ring layer.
    assert "between(t\\," not in out
    assert "[ring0]" not in out


def test_cursor_click_adds_pulse_ring() -> None:
    shot = Shot(
        id="a",
        seconds=2.0,
        cursor_track=[CursorKey(t=0.5, x=100, y=100, click=True)],
    )
    out = _cursor_filter("scale=320:240", shot, _settings(), 2.0)
    assert "[cov0]" in out
    assert "[ring0]" in out
    # click-pulse window enabled around the click time (0.5-0.05 .. 0.5+0.20).
    assert "enable='between(t\\,0.45\\,0.7)'" in out
    # ring uses the accent colour.
    assert "0x3B82F6" in out


def test_cursor_graph_contains_semicolon_for_filter_complex_routing() -> None:
    # The ';' is the signal _video_audio_map_args uses to avoid the bare -vf path.
    shot = Shot(id="a", seconds=2.0, cursor_track=[CursorKey(t=0.0, x=10, y=10)])
    assert ";" in _cursor_filter("scale=320:240", shot, _settings(), 2.0)


# --------------------------------------------------------------------------
# _remap_tracks_to_canvas — WS-A footage_viewport → render-canvas remap
# --------------------------------------------------------------------------


def _canvas(width: int, height: int, fps: int = 25):
    return get_settings().model_copy(
        update={"render_width": width, "render_height": height, "render_fps": fps}
    )


def test_remap_none_viewport_returns_same_objects() -> None:
    # The legacy/identity path: no footage_viewport → the SAME list objects flow
    # through untouched (no copy), so the render stays byte-identical.
    focus = [FocusKey(t=0.0, bbox=(910, 490, 100, 100))]
    cursor = [CursorKey(t=0.0, x=100, y=100)]
    callouts = [Callout(at=0.0, type="badge", bbox=(910, 490, 100, 100), label="1")]
    rf, rc, rcal = _remap_tracks_to_canvas(focus, cursor, callouts, None, _settings())
    assert rf is focus
    assert rc is cursor
    assert rcal is callouts


def test_remap_viewport_equal_canvas_returns_same_objects() -> None:
    # Recorded viewport already == the render canvas → transform is the identity,
    # originals pass straight through (same objects).
    focus = [FocusKey(t=0.0, bbox=(910, 490, 100, 100))]
    cursor = [CursorKey(t=0.0, x=100, y=100)]
    callouts = [Callout(at=0.0, type="badge", bbox=(910, 490, 100, 100), label="1")]
    rf, rc, rcal = _remap_tracks_to_canvas(
        focus, cursor, callouts, (1920, 1080), _canvas(1920, 1080)
    )
    assert rf is focus
    assert rc is cursor
    assert rcal is callouts


def test_remap_1080p_into_9x16_letterboxes_center() -> None:
    # 1920x1080 source coords into a 1080x1920 (9:16) canvas.
    # s = min(1080/1920, 1920/1080) = 0.5625; pad_x = 0; pad_y = (1920-607.5)/2 = 656.25.
    focus = [FocusKey(t=0.0, bbox=(910, 490, 100, 100))]
    cursor = [CursorKey(t=1.0, x=960, y=540, click=True)]
    callouts = [Callout(at=0.0, type="badge", bbox=(910, 490, 100, 100), label="1")]
    rf, rc, rcal = _remap_tracks_to_canvas(
        focus, cursor, callouts, (1920, 1080), _canvas(1080, 1920)
    )
    assert rf is not None and rc is not None and rcal is not None
    # the callout bbox remaps through the same scale+pad as the focus bbox.
    assert rcal[0].bbox[0] == pytest.approx(511.875)
    assert rcal[0].bbox[1] == pytest.approx(931.875)
    assert rcal[0].bbox[2] == pytest.approx(56.25)
    # non-coordinate fields carry across unchanged.
    assert rcal[0].type == "badge"
    assert rcal[0].label == "1"
    bx, by, bw, bh = rf[0].bbox
    assert bx == pytest.approx(511.875)
    assert by == pytest.approx(931.875)
    assert bw == pytest.approx(56.25)
    assert bh == pytest.approx(56.25)
    # the source-frame centre (960, 540) maps onto the canvas centre (540, 960).
    assert rc[0].x == pytest.approx(540.0)
    assert rc[0].y == pytest.approx(960.0)
    # non-coordinate fields are carried across unchanged.
    assert rc[0].t == 1.0
    assert rc[0].click is True


def test_remap_cover_fit_uses_increase_scale_and_crop_offset() -> None:
    # fit="cover" fills the 1080x1920 canvas from a 1920x1080 source: it scales UP
    # (s = max(1080/1920, 1920/1080) = 1.777…) then centre-crops, so the offset is
    # NEGATIVE (a crop) not a pad. off_x = (1080 - 1920*1.7778)/2 = -1166.67; off_y=0.
    focus = [FocusKey(t=0.0, bbox=(910, 490, 100, 100))]
    cursor = [CursorKey(t=1.0, x=960, y=540, click=True)]
    rf, rc, _rcal = _remap_tracks_to_canvas(
        focus, cursor, None, (1920, 1080), _canvas(1080, 1920), "cover"
    )
    assert rf is not None and rc is not None
    s = 1920 / 1080
    # source centre (960, 540) maps onto the canvas centre (540, 960) under cover too.
    assert rc[0].x == pytest.approx(960 * s + (1080 - 1920 * s) / 2)
    assert rc[0].x == pytest.approx(540.0)
    assert rc[0].y == pytest.approx(960.0)
    bx, _by, bw, _bh = rf[0].bbox
    assert bw == pytest.approx(100 * s)
    assert bx == pytest.approx(910 * s + (1080 - 1920 * s) / 2)


def test_remap_preserves_a_none_track() -> None:
    # Only one of the two tracks present → the absent one stays None.
    rf, rc, rcal = _remap_tracks_to_canvas(
        [FocusKey(t=0.0, bbox=(910, 490, 100, 100))], None, None, (1920, 1080), _canvas(1080, 1920)
    )
    assert rf is not None and rc is None and rcal is None


def test_remapped_coords_drive_the_zoompan_crop_window() -> None:
    # End-to-end wire check at a non-default tier: the remapped bbox centre
    # (the 9:16 canvas centre 540/960) must drive the zoompan crop expressions —
    # proving the remap actually reaches the filter, not just the schema.
    settings = _canvas(1080, 1920)
    rf, _, _ = _remap_tracks_to_canvas(
        [FocusKey(t=0.0, bbox=(910, 490, 100, 100))], None, None, (1920, 1080), settings
    )
    shot = Shot(id="a", seconds=2.0, focus_track=rf)
    out = _focus_zoom_filter("scale=1080:1920", shot, settings, 2.0)
    # x2 by _ZOOMPAN_PRESCALE: the lanczos pre-scale doubles zoompan's input, so
    # the remapped canvas centre (540, 960) must be expressed in THAT space.
    assert "1080-iw/zoom/2" in out
    assert "1920-ih/zoom/2" in out


def test_footage_viewport_none_filter_strings_are_byte_identical() -> None:
    # Regression identity: with footage_viewport defaulted (None) the directed
    # filter strings match a render that never touched the field. The helper
    # returns the ORIGINAL track objects, so render_shot feeds the SAME shot to
    # the filters — the exact same values flow through as before.
    focus = [FocusKey(t=0.0, bbox=(140, 100, 40, 40))]
    cursor = [CursorKey(t=0.0, x=50, y=60), CursorKey(t=1.0, x=200, y=180)]
    settings = _settings()
    shot = Shot(id="a", seconds=2.0, focus_track=focus, cursor_track=cursor)
    rf, rc, _rcal = _remap_tracks_to_canvas(
        shot.focus_track, shot.cursor_track, shot.callouts, shot.footage_viewport, settings
    )
    assert rf is shot.focus_track and rc is shot.cursor_track
    base = _focus_zoom_filter(
        _cursor_filter("scale=320:240", shot, settings, 2.0), shot, settings, 2.0
    )
    remapped = shot.model_copy(update={"focus_track": rf, "cursor_track": rc})
    out = _focus_zoom_filter(
        _cursor_filter("scale=320:240", remapped, settings, 2.0), remapped, settings, 2.0
    )
    assert out == base


# --------------------------------------------------------------------------
# _callout_filter — numbered badge + label callouts
# --------------------------------------------------------------------------


def test_callout_none_is_passthrough(tmp_path) -> None:
    # No callouts → the input filter string is returned UNCHANGED (byte-identical).
    shot = Shot(id="a", seconds=2.0, callouts=None)
    assert _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path) == "scale=320:240"
    shot_empty = Shot(id="a", seconds=2.0, callouts=[])
    assert (
        _callout_filter("scale=320:240", shot_empty, _settings(), 2.0, tmp_path) == "scale=320:240"
    )


def test_callout_badge_emits_overlay_and_drawtext_at_bbox(tmp_path) -> None:
    # A badge composites a disc sprite at the element bbox top-left corner + a
    # centered number drawtext; the multi-chain closes the incoming tail as [co0]
    # and terminates UNLABELED so the focus-zoom seam can append [fz].
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.5, type="badge", bbox=(100, 80, 40, 40), label="1")],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    # incoming single chain closed with a label, at least one ';' for routing.
    assert out.startswith("scale=320:240[co0];")
    assert ";" in out
    # a synthesized rgba disc sprite + the overlay at the (remapped) bbox corner.
    assert "format=rgba" in out
    assert "[co0][badge0]overlay=x=100:y=80" in out
    # a centered number drawtext (disc centre = corner + radius; diameter=28 → r=14).
    assert "drawtext=" in out
    assert "x=114-text_w/2" in out
    # badge appears at its onset and persists.
    assert "enable='gte(t\\,0.5)'" in out
    # UNLABELED tail so _focus_zoom_filter can append [fz].
    assert not out.endswith("]")


def test_callout_bbox_remapped_at_mismatched_tier(tmp_path) -> None:
    # footage_viewport (1920x1080) differing from the 1080x1920 canvas scales the
    # callout bbox through the same fit transform as the focus track — the remapped
    # corner (511.875, 931.875), not the raw (910, 490), reaches the overlay.
    settings = _canvas(1080, 1920)
    _rf, _rc, rcal = _remap_tracks_to_canvas(
        None,
        None,
        [Callout(at=0.0, type="badge", bbox=(910, 490, 100, 100), label="2")],
        (1920, 1080),
        settings,
    )
    shot = Shot(id="a", seconds=2.0, callouts=rcal)
    out = _callout_filter("scale=1080:1920", shot, settings, 2.0, tmp_path)
    assert "overlay=x=511.875:y=931.875" in out


def test_callout_label_emits_lower_third_caption(tmp_path) -> None:
    # A label callout draws a boxed caption anchored to the bbox, enabled over
    # at → at+dwell.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.25, type="label", bbox=(10, 10, 40, 40), label="Click Save")],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert "drawtext=" in out and "box=1:boxcolor=black@0.55" in out
    # dwell = min(2.0, duration) → 2.0; window at → at+dwell.
    assert "enable='between(t\\,0.25\\,2.25)'" in out
    assert not out.endswith("]")


def test_callout_label_anchors_near_bbox_not_bottom_center(tmp_path) -> None:
    # AC3: the label caption is now ANCHORED to the bbox — x centered+clamped via
    # clip(), y below the bbox with an if(gt(...)) flip above when it clips the
    # frame bottom — NOT the old fixed bottom-center numeric anchor.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="label", bbox=(200, 400, 80, 60), label="Click Save")],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    # Safe margin is now a FRACTION of canvas width (identity 100px at 1920px);
    # this test's 320px canvas scales it to round(320 * 100/1920) = 17.
    margin = renderer._safe_margin_px(320)
    assert margin == 17
    # centered-on-bbox x clamped inside the safe margin; commas escaped \,.
    assert f"x=clip(240-text_w/2\\,{margin}\\,w-text_w-{margin})" in out
    # y below the bbox (400+60+24=484) with the flip-above (400-24=376) branch.
    assert f"y=if(gt(484+text_h\\,h-{margin})\\,376-text_h\\,484)" in out
    # the OLD fixed bottom-center numeric anchor is gone.
    assert "y=h-text_h-60" not in out


def test_callout_seam_closes_cursor_multichain_then_focus_zoom(tmp_path) -> None:
    # THE seam contract: after a cursor multi-chain (final node UNLABELED), the
    # callout must (a) close the tail as [co0], (b) end UNLABELED itself, so that
    # _focus_zoom_filter appends a single well-formed [fz];[fz]zoompan reopen — no
    # [cal][fz] double-label, no bare ',zoompan' onto a filter_complex graph.
    cursor_graph = "scale=320:240[cbase];color=x[cursor];[cbase][cursor]overlay=x='10':y='10'"
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="badge", bbox=(100, 80, 40, 40), label="1")],
        focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))],
    )
    after_callout = _callout_filter(cursor_graph, shot, _settings(), 2.0, tmp_path)
    # the incoming cursor tail is closed with [co0], callout ends UNLABELED.
    assert "overlay=x='10':y='10'[co0];" in after_callout
    assert not after_callout.endswith("]")
    final = _focus_zoom_filter(after_callout, shot, _settings(), 2.0)
    # exactly one well-formed zoompan reopen, appended onto the callout's tail.
    assert final.startswith(
        after_callout
        + "[fz];[fz]"
        + renderer._zoompan_prescale(320, 240, _settings())
        + ",zoompan=z='1+"
    )
    assert final.count("[fz];[fz]" + renderer._zoompan_prescale(320, 240, _settings())) == 1
    # Reached through a label, never comma-appended onto the callout graph.
    assert after_callout + "," not in final
    assert "[cal][fz]" not in final  # no double-label at the seam


def test_callout_seam_closes_single_chain_then_focus_zoom(tmp_path) -> None:
    # The no-cursor incoming shape: a simple single chain (no ';'). The callout
    # still labels it [co0], starts a multi-chain, ends UNLABELED, and the focus
    # zoom takes its ';'-branch — one valid filter_complex overall.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="badge", bbox=(100, 80, 40, 40), label="1")],
        focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))],
    )
    after_callout = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert after_callout.startswith("scale=320:240[co0];")
    assert not after_callout.endswith("]")
    final = _focus_zoom_filter(after_callout, shot, _settings(), 2.0)
    assert final.startswith(
        after_callout
        + "[fz];[fz]"
        + renderer._zoompan_prescale(320, 240, _settings())
        + ",zoompan=z='1+"
    )
    assert after_callout + "," not in final
    assert "[cal][fz]" not in final


def test_callout_spotlight_emits_four_drawboxes_gated_to_bbox(tmp_path) -> None:
    # A spotlight dims the COMPLEMENT of the bbox with four drawbox bands
    # (top/bottom/left/right), each t=fill black@0.55 gated over at → at+dwell.
    # No full-frame geq/soft-mask (the OOM class) — plain drawbox only.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="spotlight", bbox=(200, 400, 80, 60), label="")],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert out.startswith("scale=320:240[co0];")
    assert out.count("drawbox=") == 4
    assert "geq=" not in out  # memory-safe: never a full-frame geq
    en = "enable='between(t\\,0\\,2)'"
    # top band above the bbox, bottom band below it (complement in y).
    assert f"drawbox=x=0:y=0:w=iw:h=400:color=black@0.55:t=fill:{en}" in out
    assert f"drawbox=x=0:y=460:w=iw:h=ih-460:color=black@0.55:t=fill:{en}" in out
    # left/right bands within the bbox y-band (complement in x).
    assert f"drawbox=x=0:y=400:w=200:h=60:color=black@0.55:t=fill:{en}" in out
    assert f"drawbox=x=280:y=400:w=iw-280:h=60:color=black@0.55:t=fill:{en}" in out
    assert not out.endswith("]")


def test_callout_arrow_sprite_points_at_bbox(tmp_path) -> None:
    # An arrow synthesizes a small triangle sprite ([arrow0]) and overlays it next
    # to the bbox: default RIGHT-pointing to the LEFT of the bbox; flipped to a
    # LEFT-pointing sprite to the RIGHT when the bbox hugs the safe left margin.
    # render_height=240 → arrow_px = max(24, round(240*0.06)) = 24.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="arrow", bbox=(200, 100, 40, 40), label="")],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert "[arrow0]" in out
    assert "lte(X\\," in out  # right-pointing apex (base on the left edge)
    # placed to the LEFT of the bbox (200-12-24=164), vertically centered (108).
    assert "[co0][arrow0]overlay=x=164:y=108" in out
    assert not out.endswith("]")
    # Flip case: a bbox hugging the left margin → LEFT-pointing sprite to the RIGHT.
    flip = Shot(
        id="b",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="arrow", bbox=(30, 100, 40, 40), label="")],
    )
    fout = _callout_filter("scale=320:240", flip, _settings(), 2.0, tmp_path)
    assert "gte(X\\," in fout  # left-pointing apex (base on the right edge)
    assert "[co0][arrow0]overlay=x=82:y=108" in fout  # 30+40+12=82, to the RIGHT


def test_callout_count_cap_drops_extras(tmp_path) -> None:
    # More than _CALLOUT_MAX_PER_SHOT (=4) callouts → only the first 4 are
    # composited; the 5th/6th never reach the graph.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[
            Callout(at=0.0, type="label", bbox=(10, 10, 40, 40), label=f"L{n}") for n in range(6)
        ],
    )
    out = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert out.count("drawtext=") == 4  # exactly four label captions rendered
    assert "[co3]" in out  # the 4th (last) segment reopens co3
    assert "[co4]" not in out  # a 5th callout label would have appeared
    assert not out.endswith("]")


def test_callout_spotlight_arrow_seam_closes(tmp_path) -> None:
    # A MIXED chain (spotlight + arrow + label) still closes the seam: the incoming
    # tail is labeled [co0], the LAST segment is UNLABELED, and _focus_zoom_filter
    # appends exactly one [fz];[fz]zoompan onto it.
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[
            Callout(at=0.0, type="spotlight", bbox=(120, 90, 40, 40), label=""),
            Callout(at=0.2, type="arrow", bbox=(120, 90, 40, 40), label=""),
            Callout(at=0.4, type="label", bbox=(120, 90, 40, 40), label="Click Save"),
        ],
        focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))],
    )
    after_callout = _callout_filter("scale=320:240", shot, _settings(), 2.0, tmp_path)
    assert after_callout.startswith("scale=320:240[co0];")
    assert not after_callout.endswith("]")
    final = _focus_zoom_filter(after_callout, shot, _settings(), 2.0)
    assert final.startswith(
        after_callout
        + "[fz];[fz]"
        + renderer._zoompan_prescale(320, 240, _settings())
        + ",zoompan=z='1+"
    )
    assert final.count("[fz];[fz]" + renderer._zoompan_prescale(320, 240, _settings())) == 1
    assert after_callout + "," not in final
    assert "[cal][fz]" not in final


# --------------------------------------------------------------------------
# _split_filter (T8 split-screen) — graph-string SHAPE
# --------------------------------------------------------------------------


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, probe_dur: float = 3.0) -> dict[str, Any]:
    """Mock ffmpeg/ffprobe + asset fetch so render_shot builds a cmd without I/O."""
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return probe_dur

    async def fake_has_audio(path: Path) -> bool:
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


def test_even_dim_floors_odd_to_even() -> None:
    assert _even_dim(675) == 674
    assert _even_dim(160) == 160
    assert _even_dim(1) == 0


def test_split_filter_hstack_two_halves() -> None:
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, _settings(), None, None, None, 2.0
    )
    # 320//2 = 160 (even) : full height 240, on BOTH halves.
    assert g.count("scale=160:240:force_original_aspect_ratio=decrease") == 2
    assert "hstack=inputs=2" in g
    assert "vstack" not in g


def test_split_filter_vstack_two_halves() -> None:
    g = _split_filter(
        SplitScreen(layout="v", second_source_url="x"), 2, _settings(), None, None, None, 2.0
    )
    # full width 320 : 240//2 = 120 (even), on BOTH halves.
    assert g.count("scale=320:120:force_original_aspect_ratio=decrease") == 2
    assert "vstack=inputs=2" in g
    assert "hstack" not in g


def test_split_filter_references_second_input_index() -> None:
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 5, _settings(), None, None, None, 2.0
    )
    assert "[5:v]tpad=stop_mode=clone:stop_duration=" in g


def test_split_filter_clones_both_halves_with_a_finite_pad() -> None:
    # ⛔ #232 REGRESSION GUARD: this half used to carry an UNCONDITIONAL, infinite
    # last-frame clone (``tpad=stop_mode=clone:stop`` pinned to the sentinel that
    # never terminates without an external ``-t``) — the exact filter shape that
    # livelocked ffmpeg 7.1.5 in production. It must now be FINITE on both halves,
    # ceiling'd at the caller's own segment duration (see
    # ``renderer._cover_pad_stop_duration``), and the old sentinel must never
    # reappear anywhere in the emitted graph.
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, _settings(), None, None, None, 2.0
    )
    assert g.count("tpad=stop_mode=clone:stop_duration=") == 2
    assert "stop=-1" not in g


def test_split_filter_setsar_and_pixfmt_on_each_half() -> None:
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, _settings(), None, None, None, 2.0
    )
    # each half ends setsar=1,format=yuv420p BEFORE the stack.
    assert g.count("setsar=1,format=yuv420p[sp") == 2
    assert g.count("setsar=1") == 2


def test_split_filter_even_floors_odd_half_dim() -> None:
    # render_height 1350 → //2 = 675 (odd) → even-floored to 674.
    settings = _settings().model_copy(update={"render_height": 1350})
    g = _split_filter(
        SplitScreen(layout="v", second_source_url="x"), 2, settings, None, None, None, 2.0
    )
    assert ":674" in g
    assert ":675" not in g


def test_split_filter_vstack_repads_short_stack_to_canvas() -> None:
    # 4x5 @1080p: canvas 1080x1350, 1350//2 -> 674, vstack yields 1080x1348.
    # The stack MUST be re-padded back to the full 1350 — every non-split shot
    # is 1080x1350 and the final join is a stream copy.
    settings = _settings().model_copy(update={"render_width": 1080, "render_height": 1350})
    g = _split_filter(
        SplitScreen(layout="v", second_source_url="x"), 2, settings, None, None, None, 2.0
    )
    assert "vstack=inputs=2,pad=1080:1350:(ow-iw)/2:(oh-ih)/2:color=" in g


def test_split_filter_exact_canvas_emits_no_restore_pad() -> None:
    # 16x9 @1080p: 1080//2 = 540 (already even) -> vstack is exactly 1920x1080,
    # so no restore pad is emitted (graph byte-identical to the pre-fix shape).
    settings = _settings().model_copy(update={"render_width": 1920, "render_height": 1080})
    g = _split_filter(
        SplitScreen(layout="v", second_source_url="x"), 2, settings, None, None, None, 2.0
    )
    assert "scale=1920:540:force_original_aspect_ratio=decrease" in g
    assert "pad=1920:1080" not in g
    assert g.count(",pad=") == 2  # the two per-half pads only (tpad= is not one)


def test_split_filter_hstack_repads_odd_half_width_to_canvas() -> None:
    # w % 4 == 2: 1350//2 -> 674, hstack yields 1348x1080 against a 1350 canvas.
    settings = _settings().model_copy(update={"render_width": 1350, "render_height": 1080})
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, settings, None, None, None, 2.0
    )
    assert "scale=674:1080:force_original_aspect_ratio=decrease" in g
    assert "hstack=inputs=2,pad=1350:1080:(ow-iw)/2:(oh-ih)/2:color=" in g


def test_split_filter_reapplies_caption_after_stack(tmp_path) -> None:
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, _settings(), tmp_path, "s", "hi", 2.0
    )
    # caption drawtext lands AFTER the stack (the base scale+caption is discarded).
    assert "hstack=inputs=2" in g
    assert "drawtext=" in g
    assert g.index("drawtext=") > g.index("hstack=inputs=2")


def test_split_filter_last_chain_is_labelless() -> None:
    # The wrapper closes the last chain with [v]; it must not carry its own label.
    g = _split_filter(
        SplitScreen(layout="h", second_source_url="x"), 2, _settings(), None, None, None, 2.0
    )
    assert not g.endswith("]")


# --------------------------------------------------------------------------
# render_shot: split disables directed footage + speed, adds a second input
# --------------------------------------------------------------------------


async def test_render_shot_split_composites_and_disables_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=2.0,
        video_url="http://minio/a.mp4",
        speed=2.0,  # would inject setpts/atempo on a non-split shot
        cursor_track=[CursorKey(t=0.0, x=10, y=10)],
        focus_track=[FocusKey(t=0.0, bbox=(10, 10, 40, 40))],
        split=SplitScreen(layout="h", second_source_url="http://minio/b.mp4"),
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "-split.mp4" in cmd  # the SECOND source was injected as an input
    assert "hstack=inputs=2" in cmd  # the two halves composited
    # directed-footage + speed are all suppressed on a split shot.
    assert "zoompan" not in cmd
    assert "overlay=" not in cmd
    # The SPEED retime specifically: ``setpts=PTS/<speed>``. Narrowed from a bare
    # "setpts=" because every chain now also carries the frame-exactness rebase
    # ``setpts=PTS-STARTPTS`` (appended by ``_cadence_fragment``), which is NOT a
    # speed retime and must be present here. The guarantee under test — a split
    # shot suppresses the speed retime — is unchanged and still exact.
    assert "setpts=PTS/" not in cmd
    assert "atempo=" not in cmd


async def test_render_shot_split_none_is_byte_identical_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    plain = Shot(id="s1", seconds=2.0, video_url="http://minio/a.mp4")
    explicit = Shot(id="s1", seconds=2.0, video_url="http://minio/a.mp4", split=None)
    await renderer.render_shot(plain, tmp_path / "p.mp4", get_settings(), tmp_path, http)
    await renderer.render_shot(explicit, tmp_path / "e.mp4", get_settings(), tmp_path, http)
    plain_cmd, explicit_cmd = captured["cmds"][0], captured["cmds"][1]
    # only the output path differs; no extra -i, identical filtergraph.
    assert plain_cmd[:-1] == explicit_cmd[:-1]
    assert " ".join(plain_cmd).count(" -i ") == " ".join(explicit_cmd).count(" -i ")


async def test_render_shot_split_own_audio_clip_is_time_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    # A split composites two FINITE last-frame clones ceiling'd at `duration` (see
    # `_cover_pad_stop_duration`) — the ceiling caps how long tpad is WILLING to
    # clone, not the segment's real length, so it MUST still carry a -t.
    # The own-audio clip branch (fake_has_audio → True, no audio_url) sets trim=False
    # ⇒ trim_args=[]; without the split-bound guard the encode never EOFs (timeout/OOM).
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=2.0,
        video_url="http://minio/a.mp4",  # own audio (fake_has_audio=True), no audio_url
        split=SplitScreen(layout="h", second_source_url="http://minio/b.mp4"),
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    assert "-t" in captured["cmds"][-1]


# --------------------------------------------------------------------------
# caption vs punch-in ORDER (_defers_caption / _deferred_caption_suffix)
# --------------------------------------------------------------------------


def test_caption_is_deferred_only_when_a_focus_zoom_will_magnify_it() -> None:
    # ⛔ THE DEFECT THIS GUARDS. The narration caption was baked into the
    # fit/scale branch, UPSTREAM of the zoompan, so on a shot with a focus track
    # the caption text was magnified and resampled by the punch-in and went soft
    # — while shot titles stayed crisp, because _apply_shot_look already runs
    # downstream of the zoom.
    zoomed = Shot(
        id="a",
        seconds=2.0,
        caption="spoken line",
        focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))],
    )
    assert renderer._defers_caption(zoomed) is True

    # Everything WITHOUT a punch-in must keep the caption exactly where it was,
    # so those filter graphs stay byte-identical.
    assert renderer._defers_caption(Shot(id="b", seconds=2.0, caption="x")) is False
    assert (
        renderer._defers_caption(
            Shot(id="c", seconds=2.0, focus_track=[FocusKey(t=0.0, bbox=(1, 1, 2, 2))])
        )
        is False
    )
    # A split shot skips _apply_directed_footage outright, so there is no zoom to
    # defer past — deferring there would move the caption for no reason.
    assert (
        renderer._defers_caption(
            Shot(
                id="d",
                seconds=2.0,
                caption="x",
                focus_track=[FocusKey(t=0.0, bbox=(1, 1, 2, 2))],
                split=SplitScreen(layout="h", second_source_url="x"),
            )
        )
        is False
    )


def test_deferred_caption_suffix_draws_after_the_zoom(tmp_path: Path) -> None:
    settings = _settings()
    shot = Shot(
        id="a",
        seconds=2.0,
        caption="spoken line",
        focus_track=[FocusKey(t=0.0, bbox=(140, 100, 40, 40))],
    )
    zoomed = _apply_directed_footage("[0:v]scale=320:240", shot, settings, 2.0, 1.0, tmp_path)
    suffix = renderer._deferred_caption_suffix(shot, settings, tmp_path, True)

    assert "zoompan" in zoomed, "precondition: this shot must actually punch in"
    assert suffix.startswith(",drawtext")
    # The caption must land AFTER the zoompan in the chain, not before it.
    chain = zoomed + suffix
    assert chain.index("zoompan") < chain.index("drawtext")

    # Not deferred → contributes nothing, so the untouched paths stay identical.
    assert renderer._deferred_caption_suffix(shot, settings, tmp_path, False) == ""


# --------------------------------------------------------------------------
# render_shot: directed-footage times follow the seek the compiler chose
# --------------------------------------------------------------------------


def _capture_directed(monkeypatch: pytest.MonkeyPatch) -> list[Shot]:
    """Record the shot ``render_shot`` hands to ``_apply_directed_footage``."""
    seen: list[Shot] = []
    real = renderer._apply_directed_footage

    def spy(video_vf: str, shot: Shot, *args: Any, **kwargs: Any) -> str:
        seen.append(shot)
        return real(video_vf, shot, *args, **kwargs)

    monkeypatch.setattr(renderer, "_apply_directed_footage", spy)
    return seen


async def test_render_shot_shifts_tracks_by_the_shared_clip_pan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A clip shared by two shots is panned by ``clip_seek_fraction``; the cursor,
    focus and callout times must move by the same seek.

    Showcase spike 01a0fd4d-9948: clip-2 (33.6s) carried l2 and l3. l3 got
    fraction 1/2, so the decoder started ~14.9s in, but the click ring and the "1"
    badge still fired at the recording's raw 3.075s, over the "Sales" nav of a page
    that was no longer on screen. A key before the window is dropped; a key inside
    it lands at ``t - seek``.
    """
    _install_fakes(monkeypatch, probe_dur=33.6)
    seen = _capture_directed(monkeypatch)
    shot = Shot(
        id="l3",
        seconds=3.77,
        video_url="http://minio/clip-2.mp4",
        cursor_track=[
            CursorKey(t=3.075, x=212, y=864, click=True),  # before the window
            CursorKey(t=16.0, x=918, y=546, click=True),  # inside it
        ],
        focus_track=[
            FocusKey(t=3.075, bbox=(74, 848, 278, 31)),
            FocusKey(t=16.0, bbox=(824, 518, 190, 59)),
        ],
        callouts=[Callout(at=3.075, type="badge", bbox=(74, 848, 278, 31), label="1")],
    )
    await renderer.render_shot(
        shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http, clip_seek_fraction=0.5
    )
    seek = round(0.5 * (33.6 - 3.77), 3)
    directed = seen[-1]
    assert directed.cursor_track is not None
    assert [(round(k.t, 3), k.x) for k in directed.cursor_track] == [(round(16.0 - seek, 3), 918)]
    assert directed.focus_track is not None
    assert [round(k.t, 3) for k in directed.focus_track] == [round(16.0 - seek, 3)]
    # The badge's onset (raw 3.075s) is before the window: no badge on this shot.
    assert directed.callouts is None


async def test_render_shot_lead_rebased_tracks_are_not_shifted_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The caller already rebases track times by ``clip_lead_seconds``; when the
    decoder starts exactly there, the times pass through unchanged."""
    _install_fakes(monkeypatch, probe_dur=33.6)
    seen = _capture_directed(monkeypatch)
    shot = Shot(
        id="l2",
        seconds=4.0,
        video_url="http://minio/clip-2.mp4",
        clip_anchor="start",
        clip_lead_seconds=7.0,
        cursor_track=[CursorKey(t=0.5, x=1352, y=154, click=True)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    assert seen[-1].cursor_track == [CursorKey(t=0.5, x=1352, y=154, click=True)]
