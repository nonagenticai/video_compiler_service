"""Real-ffmpeg render-harness tests for per-shot colour grade + text overlays.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_speed_xfade_render.py``. These drive ACTUAL encodes and compare the
delivered pixels; the string tests in ``tests/test_renderer_look.py`` prove the
filtergraph shape, these prove the grade/overlay actually change the picture and
that the unset (no-look) path is byte-identical.

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths (a bare
filesystem path is the shot's ``video_url``), so no bucket/presign is needed.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import ColorAdjust, Shot, TextOverlay
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

# Small canvas keeps the real encodes fast; a colour source so a saturation
# grade has chroma to remove.
_SMALL = {"render_width": 320, "render_height": 240, "render_fps": 25}


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


def _channel_means(buf: bytes) -> tuple[float, float, float]:
    n = max(1, len(buf) // 3)
    return sum(buf[0::3]) / n, sum(buf[1::3]) / n, sum(buf[2::3]) / n


def _spread(means: tuple[float, float, float]) -> float:
    return max(means) - min(means)


async def test_bw_grade_changes_chroma(tmp_path: Path) -> None:
    """A ``bw`` grade (saturation=0) collapses per-pixel chroma, so the graded
    frame differs from the colour baseline and its channel spread shrinks."""
    settings = get_settings().model_copy(update=_SMALL)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    base = tmp_path / "base.mp4"
    graded = tmp_path / "graded.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), base, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(id="b", seconds=1.0, video_url=str(src), adjust=ColorAdjust(preset="bw")),
            graded,
            settings,
            tmp_path,
            http,
        )
    fb = await _first_frame_rgb(base, tmp_path / "base.rgb")
    fg = await _first_frame_rgb(graded, tmp_path / "graded.rgb")
    assert fb != fg  # the grade changed the delivered pixels
    # Greyscale ⇒ R≈G≈B per pixel, so the channel spread collapses vs the colour
    # baseline.
    assert _spread(_channel_means(fg)) < _spread(_channel_means(fb))


async def test_overlay_changes_pixels(tmp_path: Path) -> None:
    """A burned-in text overlay changes the delivered frame vs a no-overlay render."""
    settings = get_settings().model_copy(update=_SMALL)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    base = tmp_path / "base.mp4"
    titled = tmp_path / "titled.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), base, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(
                id="b",
                seconds=1.0,
                video_url=str(src),
                overlays=[TextOverlay(text="HELLO WORLD", position="middle-center", font_size=48)],
            ),
            titled,
            settings,
            tmp_path,
            http,
        )
    fb = await _first_frame_rgb(base, tmp_path / "base.rgb")
    ft = await _first_frame_rgb(titled, tmp_path / "titled.rgb")
    assert fb != ft


async def test_no_look_is_byte_identical(tmp_path: Path) -> None:
    """A shot with adjust=None + overlays=[] renders byte-for-byte identical to
    the same shot with no look fields at all — the no-op invariant end to end."""
    settings = get_settings().model_copy(update=_SMALL)
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    plain = tmp_path / "plain.mp4"
    explicit = tmp_path / "explicit.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src)), plain, settings, tmp_path, http
        )
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(src), adjust=None, overlays=[]),
            explicit,
            settings,
            tmp_path,
            http,
        )
    assert (
        hashlib.sha256(plain.read_bytes()).hexdigest()
        == hashlib.sha256(explicit.read_bytes()).hexdigest()
    )
