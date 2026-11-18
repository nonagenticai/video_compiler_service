"""Real-encode proof for the banding spatial term's NATIVE-resolution decode.

The spatial decode used to area-downscale each frame to 320px, which averaged
a few-pixel posterization step into a soft ramp — and ``spatial_banding_factor``
scored every smooth 8-bit gradient as a staircase. Now (2026-10-21) ONE streamed
ffmpeg pass (``measure.measure_spatial_factors_streamed``) emits only the wanted
frames as native PGMs and attributes them BY ORDER to the signalstats
``pts_time`` values. This drives the REAL signalstats pass + select command +
PGM stream + detector over a libx264 crf=18 yuv420p encode (this service's
delivery path), which the pure calibration set cannot exercise — in particular
that ``select``'s ``t`` and signalstats' ``pts_time`` name the SAME frames.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services import measure

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

W, H = 1920, 1080


def _encode(
    tmp_path: Path, name: str, rows: list[bytes], w: int = W, h: int = H, *, frames: bool = False
) -> Path:
    """One frame per entry of ``rows`` (each replicated down the frame, or a
    whole ``w*h`` frame when ``frames``), 10 fps."""
    out = tmp_path / f"{name}.mp4"
    subprocess.run(  # noqa: S603 — fixed argv, test-only
        [  # noqa: S607
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "10",
            "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out),
        ],
        input=b"".join(row if frames else row * h for row in rows),
        check=True,
    )  # fmt: skip
    return out


def _smooth_row(levels: int, w: int = W) -> bytes:
    lo = (255 - levels) // 2
    return bytes(min(255, int(lo + levels * x / w + 0.5)) for x in range(w))


def _posterized_row(bands: int, w: int = W) -> bytes:
    return bytes(round(min(bands - 1, x * bands // w) * 255 / (bands - 1)) for x in range(w))


def _posterized_vignette(bands: int, w: int, h: int) -> bytes:
    cx, cy, radius, q = w / 2, h / 2, math.hypot(w / 2, h / 2), 160 / bands

    def val(x: int, y: int) -> int:
        v = 230 - 160 * (math.hypot(x - cx, y - cy) / radius) ** 2
        return max(0, min(255, int(70 + round((v - 70) / q) * q + 0.5)))

    return bytes(val(x, y) for y in range(h) for x in range(w))


async def _all_pts(path: Path) -> list[float]:
    timed = await measure.measure_shot_frames_timed(path)
    return [pts for pts, _ in timed if pts is not None]


@pytest.mark.parametrize("levels", [40, 160])
async def test_smooth_ramp_encoded_scores_zero(tmp_path: Path, levels: int) -> None:
    path = _encode(tmp_path, f"smooth{levels}", [_smooth_row(levels)] * 10)
    pts = await _all_pts(path)
    factors = await measure.measure_spatial_factors_streamed(path, pts, set(pts), budget_seconds=60)
    assert set(factors) == set(pts)  # every frame measured: decode + count check held
    assert max(factors.values()) < 0.05, factors


@pytest.mark.parametrize("bands", [8, 16, 32])
async def test_posterized_ramp_encoded_scores_high(tmp_path: Path, bands: int) -> None:
    path = _encode(tmp_path, f"poster{bands}", [_posterized_row(bands)] * 10)
    pts = await _all_pts(path)
    factors = await measure.measure_spatial_factors_streamed(path, pts, set(pts), budget_seconds=60)
    assert set(factors) == set(pts)
    assert min(factors.values()) >= 0.5, factors


# The prod master size (2560x1440). The detector input is capped at 1920 wide
# with NEAREST-neighbour (``measure._SPATIAL_FRAME_MAX_WIDTH``) for speed: it
# drops columns, never averages a step away (an AREA downscale was the original
# false-positive cause). These prove the cap keeps both verdicts.
PW, PH = 2560, 1440


@pytest.mark.parametrize("levels", [40, 160])
async def test_2560_smooth_ramp_capped_to_1920_still_scores_zero(
    tmp_path: Path, levels: int
) -> None:
    path = _encode(tmp_path, f"smooth{levels}", [_smooth_row(levels, PW)] * 3, PW, PH)
    pts = await _all_pts(path)
    factors = await measure.measure_spatial_factors_streamed(path, pts, set(pts), budget_seconds=60)
    assert set(factors) == set(pts)
    assert max(factors.values()) < 0.05, factors


@pytest.mark.parametrize("bands", [8, 32])
async def test_2560_posterized_ramp_capped_to_1920_still_scores_high(
    tmp_path: Path, bands: int
) -> None:
    path = _encode(tmp_path, f"poster{bands}", [_posterized_row(bands, PW)] * 3, PW, PH)
    pts = await _all_pts(path)
    factors = await measure.measure_spatial_factors_streamed(path, pts, set(pts), budget_seconds=60)
    assert set(factors) == set(pts)
    assert min(factors.values()) >= 0.77, factors


async def test_2560_posterized_vignette_capped_to_1920_still_scores_high(tmp_path: Path) -> None:
    """The weakest calibration positive (2-D, curved bands): 0.773 after x264
    at native 1920; from a 2560 source through the cap it measured 0.881."""
    frame = _posterized_vignette(12, PW, PH)
    path = _encode(tmp_path, "vignette", [frame] * 2, PW, PH, frames=True)
    pts = await _all_pts(path)
    factors = await measure.measure_spatial_factors_streamed(path, pts, set(pts), budget_seconds=60)
    assert set(factors) == set(pts)
    assert min(factors.values()) >= 0.77, factors


async def test_streamed_frames_are_attributed_to_the_right_pts(tmp_path: Path) -> None:
    """Alternate clean/banded frames and ask for a sparse, multi-interval subset:
    every returned factor must match ITS frame's content, so a select/pts or
    order mismatch cannot hide."""
    rows = [_smooth_row(80) if i % 2 == 0 else _posterized_row(16) for i in range(12)]
    path = _encode(tmp_path, "alternating", rows)
    pts = await _all_pts(path)
    assert len(pts) == 12
    wanted = {pts[i] for i in (1, 2, 3, 6, 9, 10)}
    factors = await measure.measure_spatial_factors_streamed(path, pts, wanted, budget_seconds=60)
    assert set(factors) == wanted
    for i, p in enumerate(pts):
        if p in wanted:
            assert (factors[p] >= 0.5) == (i % 2 == 1), (i, p, factors[p])


async def test_unreadable_master_fails_open(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert (
        await measure.measure_spatial_factors_streamed(bad, [0.0], {0.0}, budget_seconds=60) == {}
    )


def test_spatial_stream_cmd_is_native_pgm_not_area_downscale() -> None:
    cmd = measure._spatial_stream_cmd("x.mp4", [(1.0, 2.0)], measure._SPATIAL_FRAME_MAX_WIDTH)
    vf = cmd[cmd.index("-vf") + 1]
    assert "flags=area" not in vf
    assert "flags=neighbor" in vf
    assert "min(iw,1920)" in vf  # detector input cap (speed), nearest-neighbour
    assert vf.startswith("select='between(t,")
    assert cmd[cmd.index("-c:v") + 1] == "pgm"
    assert cmd[cmd.index("-fps_mode:v") + 1] == "passthrough"
