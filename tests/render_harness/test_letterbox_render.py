"""Real-ffmpeg render-harness tests for the VC-5 brand letterbox colour.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_look_render.py``. The string tests in ``tests/test_renderer_finish.py``
prove the filtergraph shape; these prove the DELIVERED pixels: a source whose
aspect is wider than the canvas is letterboxed, and the bars must actually carry
the requested brand colour (not the hardcoded service background).

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

# A 4:3 canvas with a 8:3 source ⇒ 'contain' letterboxes: real bars top+bottom.
_CANVAS_W, _CANVAS_H = 320, 240
_SMALL = {"render_width": _CANVAS_W, "render_height": _CANVAS_H, "render_fps": 25}
_SRC_SIZE = "320x120"

# Deliberately far from the service default bg (0x111827 ≈ rgb(17,24,39)) so the
# assertion cannot pass by accident.
_BRAND_HEX = "#C81E64"
_BRAND_RGB = (0xC8, 0x1E, 0x64)
_DEFAULT_BG_RGB = (0x11, 0x18, 0x27)

# The pad colour survives an rgb→yuv420p→rgb round trip (plus chroma subsampling)
# only approximately; a per-channel tolerance keeps the assertion honest without
# being flaky. Still far tighter than the distance between brand and default bg.
# The pad colour is authored in RGB but delivered through limited-range bt709
# yuv420p, so the decoded pixel never round-trips exactly: #C81E64 comes back as
# (213, 48, 100), a per-channel drift of up to ~18. Tolerate the conversion, but
# keep every assertion paired with a `not _close(...)` against the colour it
# replaced — the brand and the default bg are far apart, so a genuinely wrong
# pad colour still fails.
_TOLERANCE = 24


async def _first_frame_rgb(path: Path, out: Path) -> bytes:
    """Decode the first video frame of ``path`` to raw rgb24 bytes."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
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


def _pixel(buf: bytes, x: int, y: int) -> tuple[int, int, int]:
    """The rgb24 pixel at (x, y) of a ``_CANVAS_W``-wide frame."""
    i = (y * _CANVAS_W + x) * 3
    return buf[i], buf[i + 1], buf[i + 2]


def _close(got: tuple[int, int, int], want: tuple[int, int, int]) -> bool:
    return all(abs(g - w) <= _TOLERANCE for g, w in zip(got, want, strict=True))


async def _render_bar_pixel(tmp_path: Path, letterbox_color: str | None) -> tuple[int, int, int]:
    """Render a letterboxed shot and return the colour of a pixel INSIDE the bar.

    The 8:3 source fits to 320x120 centred on the 320x240 canvas, so rows 0..59
    (and 180..239) are pure pad. Row 8 is safely inside the top bar.
    """
    update: dict[str, int | str] = dict(_SMALL)
    if letterbox_color is not None:
        update["render_letterbox_color"] = letterbox_color
    settings = get_settings().model_copy(update=update)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size=_SRC_SIZE, fps=25)
    out = tmp_path / f"out-{letterbox_color or 'default'}.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), out, settings, tmp_path, http
        )
    frame = await _first_frame_rgb(out, tmp_path / "frame.rgb")
    return _pixel(frame, x=_CANVAS_W // 2, y=8)


async def test_letterbox_color_paints_the_bars_in_the_delivered_mp4(tmp_path: Path) -> None:
    """The brand colour reaches the actual pixels of the delivered file — the
    whole point of the feature."""
    got = await _render_bar_pixel(tmp_path, _BRAND_HEX)
    assert _close(got, _BRAND_RGB), f"bar pixel {got} is not the brand colour {_BRAND_RGB}"
    # …and is unmistakably NOT the default background it replaced.
    assert not _close(got, _DEFAULT_BG_RGB)


async def test_no_finish_keeps_the_default_background_bars(tmp_path: Path) -> None:
    """The control: without a finish profile the bars stay the service bg colour.
    Proves the test above is measuring the override, not the source or the codec.
    """
    got = await _render_bar_pixel(tmp_path, None)
    assert _close(got, _DEFAULT_BG_RGB), f"bar pixel {got} is not the default bg {_DEFAULT_BG_RGB}"
    assert not _close(got, _BRAND_RGB)


async def test_the_picture_area_is_not_painted_over(tmp_path: Path) -> None:
    """Sanity: the pad colour fills only the bars — the centre of the frame still
    carries the source picture, so a brand colour can never swallow the video."""
    settings = get_settings().model_copy(update={**_SMALL, "render_letterbox_color": _BRAND_HEX})
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size=_SRC_SIZE, fps=25)
    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), out, settings, tmp_path, http
        )
    frame = await _first_frame_rgb(out, tmp_path / "frame.rgb")
    centre = _pixel(frame, x=_CANVAS_W // 2, y=_CANVAS_H // 2)
    assert not _close(centre, _BRAND_RGB), "the brand colour leaked into the picture area"
