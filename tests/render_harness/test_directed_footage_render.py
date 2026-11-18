"""Real-ffmpeg render-harness tests for directed footage (ORCH-3.2): the
synthetic cursor overlay + auto-zoom punch-in.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_look_render.py``. These drive ACTUAL encodes and compare delivered
pixels: (a) a cursor_track changes the pixels at the keyframed point vs a
no-cursor baseline; (b) a focus_track punch-in makes the focused box grow to
dominate more of the frame at the hold vs the un-zoomed first frame; (c) the
critical regression guard — a shot with focus_track=None,cursor_track=None
renders byte-for-byte identical to a plain render.

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Callout, CursorKey, FocusKey, Shot, SplitScreen
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip
from tests.render_harness.harness import requires_filters

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
    # Presence of the binary is not capability: a Homebrew/macOS ffmpeg is on
    # PATH but built without libass/libfreetype, so this filter is absent and
    # the render dies with a bare `exit=8: Filter not found`. Skip with a reason
    # instead. See tests/render_harness/harness.py::requires_filters.
    requires_filters("drawtext"),
]

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}


async def _frame_rgb_at(path: Path, t: float, out: Path) -> bytes:
    """Decode the frame at time ``t`` of ``path`` to raw rgb24 bytes."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(out),
        ]
    )
    return out.read_bytes()


def _crop_bytes(buf: bytes, cx: int, cy: int, half: int) -> bytes:
    """Extract a (2*half)^2 rgb24 window centred on (cx, cy) from a _Wx_H frame."""
    out = bytearray()
    for y in range(max(0, cy - half), min(_H, cy + half)):
        for x in range(max(0, cx - half), min(_W, cx + half)):
            i = (y * _W + x) * 3
            out += buf[i : i + 3]
    return bytes(out)


def _count_red(buf: bytes) -> int:
    """Number of clearly red-dominant pixels in an rgb24 buffer."""
    n = 0
    for i in range(0, len(buf) - 2, 3):
        r, g, b = buf[i], buf[i + 1], buf[i + 2]
        if r > 110 and r > g + 40 and r > b + 40:
            n += 1
    return n


def _count_bright(buf: bytes) -> int:
    """Number of near-white pixels (the synthesized cursor disc, 0xF5F5F7)."""
    n = 0
    for i in range(0, len(buf) - 2, 3):
        r, g, b = buf[i], buf[i + 1], buf[i + 2]
        if r > 200 and g > 200 and b > 200:
            n += 1
    return n


async def _box_clip(
    path: Path, *, box: tuple[int, int, int, int], color: str = "red", duration: float = 2.0
) -> Path:
    """A black clip with a single filled ``color`` box drawn at ``box`` (x,y,w,h),
    plus a silent audio track — a controlled focus target."""
    bx, by, bw, bh = box
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={_W}x{_H}:d={duration}:r={_FPS}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf",
            f"drawbox=x={bx}:y={by}:w={bw}:h={bh}:color={color}:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def _sized_box_clip(
    path: Path,
    *,
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    color: str = "red",
    duration: float = 2.0,
) -> Path:
    """A black clip at an ARBITRARY ``size`` with one filled ``color`` box — a
    controlled focus target recorded at a resolution != the render canvas."""
    w, h = size
    bx, by, bw, bh = box
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={w}x{h}:d={duration}:r={_FPS}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf",
            f"drawbox=x={bx}:y={by}:w={bw}:h={bh}:color={color}:t=fill",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def test_footage_viewport_remaps_focus_to_on_target_at_mismatched_tier(
    tmp_path: Path,
) -> None:
    """WS-A: focus coords captured in a 640x480 source rendered onto a 320x240
    canvas. With ``footage_viewport=(640,480)`` the remap fits the coords through
    the same decrease-scale the picture gets, so the punch-in still frames the
    centred box (its red pixels grow at the hold). This is the latent-bug fix:
    raw coords interpreted against the smaller canvas would otherwise miss."""
    settings = get_settings().model_copy(update=_SMALL)  # 320x240 canvas
    box = (280, 200, 80, 80)  # centred in the 640x480 source (centre 320,240)
    src = await _sized_box_clip(tmp_path / "box.mp4", size=(640, 480), box=box)
    out = tmp_path / "remapped.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(src),
                focus_track=[FocusKey(t=0.0, bbox=box)],
                footage_viewport=(640, 480),
            ),
            out,
            settings,
            tmp_path,
            http,
        )
    first = _count_red(await _frame_rgb_at(out, 0.0, tmp_path / "first.rgb"))
    hold = _count_red(await _frame_rgb_at(out, 1.0, tmp_path / "hold.rgb"))
    assert first > 0
    assert hold > first * 1.3


async def test_cursor_changes_pixels_at_keyframe(tmp_path: Path) -> None:
    """A cursor_track overlays a synthesized disc at the keyframed point, so the
    delivered pixels there differ from a no-cursor baseline of the same shot."""
    settings = get_settings().model_copy(update=_SMALL)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size=f"{_W}x{_H}", fps=_FPS)
    base = tmp_path / "base.mp4"
    cursor = tmp_path / "cursor.mp4"
    cx, cy = 160, 120
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), base, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(
                id="b",
                seconds=1.0,
                video_url=str(src),
                cursor_track=[CursorKey(t=0.0, x=cx, y=cy)],
            ),
            cursor,
            settings,
            tmp_path,
            http,
        )
    fb = await _frame_rgb_at(base, 0.0, tmp_path / "base.rgb")
    fc = await _frame_rgb_at(cursor, 0.0, tmp_path / "cursor.rgb")
    assert fb != fc
    # the change is localized around the cursor tip.
    assert _crop_bytes(fb, cx, cy, 12) != _crop_bytes(fc, cx, cy, 12)


async def test_focus_zoom_grows_the_focused_box(tmp_path: Path) -> None:
    """A focus_track punch-in eases from z=1 (first frame) up to the zoom target
    (hold), so a box centred in the frame occupies MORE red pixels at the hold
    than un-zoomed — proof the auto-zoom actually magnified the focus region."""
    settings = get_settings().model_copy(update=_SMALL)
    box = (140, 100, 40, 40)  # centred on the frame centre (160,120)
    src = await _box_clip(tmp_path / "box.mp4", box=box, duration=2.0)
    zoomed = tmp_path / "zoomed.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(src),
                focus_track=[FocusKey(t=0.0, bbox=box)],
            ),
            zoomed,
            settings,
            tmp_path,
            http,
        )
    first = _count_red(await _frame_rgb_at(zoomed, 0.0, tmp_path / "first.rgb"))
    hold = _count_red(await _frame_rgb_at(zoomed, 1.0, tmp_path / "hold.rgb"))
    assert first > 0
    assert hold > first * 1.3


async def test_focus_zoom_multi_keyframe_under_speed_grows_at_retimed_moment(
    tmp_path: Path,
) -> None:
    """BUG #2 + #3 together: a 2-keyframe focus_track on a speed=2.0 shot. The
    downstream setpts=PTS/2 halves the output to ~1.0s and compresses the zoompan
    `on` range, so the punch must hold at the RETIMED midpoint (~0.5s output), not
    at the pre-speed 1.0s. The centred box therefore occupies MORE red pixels at
    the retimed hold than at the first frame — proof the multi-key auto-zoom fired
    on the correctly-retimed time axis."""
    settings = get_settings().model_copy(update=_SMALL)
    box = (140, 100, 40, 40)  # centred on the frame centre (160,120)
    src = await _box_clip(tmp_path / "box.mp4", box=box, duration=2.0)
    fast = tmp_path / "fast.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(src),
                speed=2.0,
                focus_track=[
                    FocusKey(t=0.0, bbox=box),
                    FocusKey(t=1.0, bbox=box),
                ],
            ),
            fast,
            settings,
            tmp_path,
            http,
        )
    # Output is ~1.0s (2.0/speed); the eased punch holds around the retimed midpoint.
    first = _count_red(await _frame_rgb_at(fast, 0.0, tmp_path / "mk_first.rgb"))
    hold = _count_red(await _frame_rgb_at(fast, 0.5, tmp_path / "mk_hold.rgb"))
    assert first > 0
    assert hold > first * 1.3


async def test_cursor_and_focus_zoom_compose(tmp_path: Path) -> None:
    """A shot carrying BOTH focus_track and cursor_track must (1) produce a valid
    render — the zoompan chains onto the cursor filter_complex graph as
    ``[fz];[fz]zoompan`` (a bare ``,zoompan`` after a ';' graph is invalid ffmpeg
    and would fail here) — and (2) the cursor, composited BEFORE the zoom, is
    magnified with the frame and lands near the zoomed target centre. This is the
    both-tracks case the adversarial review flagged; it was previously untested."""
    settings = get_settings().model_copy(update=_SMALL)
    box = (140, 100, 40, 40)  # centred on the frame centre (160,120)
    cx, cy = 160, 120  # cursor tip on the focus target
    src = await _box_clip(tmp_path / "box.mp4", box=box, duration=2.0)
    zoom_only = tmp_path / "zoom_only.mp4"
    combined = tmp_path / "combined.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=2.0, video_url=str(src), focus_track=[FocusKey(t=0.0, bbox=box)]),
            zoom_only,
            settings,
            tmp_path,
            http,
        )
        await renderer.render_shot(
            Shot(
                id="b",
                seconds=2.0,
                video_url=str(src),
                focus_track=[FocusKey(t=0.0, bbox=box)],
                cursor_track=[CursorKey(t=0.0, x=cx, y=cy), CursorKey(t=2.0, x=cx, y=cy)],
            ),
            combined,
            settings,
            tmp_path,
            http,
        )
    # (1) the combined render succeeded and is a real, non-trivial MP4.
    assert combined.stat().st_size > 0
    # (2) at the hold the cursor (near-white) shows near the zoomed centre, and
    # the zoom-only render — a red box on black, no cursor — has ~none there.
    hold_combined = await _frame_rgb_at(combined, 1.0, tmp_path / "hc.rgb")
    hold_zoom = await _frame_rgb_at(zoom_only, 1.0, tmp_path / "hz.rgb")
    assert _count_bright(_crop_bytes(hold_combined, cx, cy, 40)) > 0
    assert _count_bright(_crop_bytes(hold_combined, cx, cy, 40)) > _count_bright(
        _crop_bytes(hold_zoom, cx, cy, 40)
    )


async def test_badge_and_label_callouts_render_clean_at_9x16(tmp_path: Path) -> None:
    """A shot carrying BOTH a badge and a label callout renders a valid MP4 at a
    non-default 9:16 tier (1080x1920): the callout multi-chain closes the seam and
    routes onto -filter_complex, and the file ffprobes clean at the requested
    dimensions/duration. Proves the filter graph the string tests assert actually
    encodes end to end (the class of seam bug string tests can't catch)."""
    settings = get_settings().model_copy(
        update={"render_width": 1080, "render_height": 1920, "render_fps": _FPS}
    )
    box = (440, 900, 200, 120)  # a control roughly centred in the 9:16 frame
    src = await _sized_box_clip(tmp_path / "box.mp4", size=(1080, 1920), box=box, duration=2.0)
    out = tmp_path / "callouts.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(src),
                callouts=[
                    Callout(at=0.0, type="badge", bbox=box, label="1"),
                    Callout(at=0.2, type="label", bbox=box, label="Open the settings panel"),
                ],
            ),
            out,
            settings,
            tmp_path,
            http,
        )
    # renders a real, non-trivial MP4 that ffprobes clean at the 9:16 tier.
    assert out.stat().st_size > 0
    assert await renderer.probe_dimensions(out) == (1080, 1920)
    assert await renderer.probe_output_duration(out) > 0


async def test_spotlight_and_arrow_callouts_render_clean_at_9x16(tmp_path: Path) -> None:
    """A shot carrying BOTH a spotlight and an arrow callout renders a valid MP4 at
    a non-default 9:16 tier (1080x1920): the spotlight's four drawbox bands + the
    arrow's synthesized triangle-sprite overlay close the seam, route onto
    -filter_complex, and ffprobe clean at the requested dimensions/duration. Proves
    the AC3 filter graph the string tests assert actually encodes end to end."""
    settings = get_settings().model_copy(
        update={"render_width": 1080, "render_height": 1920, "render_fps": _FPS}
    )
    box = (440, 900, 200, 120)  # a control roughly centred in the 9:16 frame
    src = await _sized_box_clip(tmp_path / "box.mp4", size=(1080, 1920), box=box, duration=2.0)
    out = tmp_path / "callouts.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(src),
                callouts=[
                    Callout(at=0.0, type="spotlight", bbox=box, label=""),
                    Callout(at=0.2, type="arrow", bbox=box, label=""),
                ],
            ),
            out,
            settings,
            tmp_path,
            http,
        )
    # renders a real, non-trivial MP4 that ffprobes clean at the 9:16 tier.
    assert out.stat().st_size > 0
    assert await renderer.probe_dimensions(out) == (1080, 1920)
    assert await renderer.probe_output_duration(out) > 0


async def test_no_direction_is_byte_identical(tmp_path: Path) -> None:
    """A shot with focus_track=None + cursor_track=None renders byte-for-byte
    identical to the same shot with no directed-footage fields at all — the
    critical regression guard end to end."""
    settings = get_settings().model_copy(update=_SMALL)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size=f"{_W}x{_H}", fps=_FPS)
    plain = tmp_path / "plain.mp4"
    explicit = tmp_path / "explicit.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), plain, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src), focus_track=None, cursor_track=None),
            explicit,
            settings,
            tmp_path,
            http,
        )
    assert (
        hashlib.sha256(plain.read_bytes()).hexdigest()
        == hashlib.sha256(explicit.read_bytes()).hexdigest()
    )


async def test_split_screen_renders_clean_at_9x16(tmp_path: Path) -> None:
    """A split-screen shot (hstack of two sources) renders a valid MP4 at a
    non-default 9:16 tier (1080x1920): each half is fitted into an even half-canvas
    (540x1920), setsar/format-normalised, and hstacked back to the full frame — so
    the composite ffprobes clean at the requested dimensions/duration. Proves the T8
    _split_filter graph the string tests assert actually encodes end to end."""
    settings = get_settings().model_copy(
        update={"render_width": 1080, "render_height": 1920, "render_fps": _FPS}
    )
    box = (400, 850, 200, 120)
    a = await _sized_box_clip(
        tmp_path / "a.mp4", size=(1080, 1920), box=box, color="red", duration=2.0
    )
    b = await _sized_box_clip(
        tmp_path / "b.mp4", size=(1080, 1920), box=box, color="blue", duration=2.0
    )
    out = tmp_path / "split.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="a",
                seconds=2.0,
                video_url=str(a),
                split=SplitScreen(layout="h", second_source_url=str(b)),
            ),
            out,
            settings,
            tmp_path,
            http,
        )
    # a real, non-trivial MP4 at the requested 9:16 tier, primary-driven duration.
    assert out.stat().st_size > 0
    assert await renderer.probe_dimensions(out) == (1080, 1920)
    assert await renderer.probe_output_duration(out) == pytest.approx(2.0, abs=0.2)
