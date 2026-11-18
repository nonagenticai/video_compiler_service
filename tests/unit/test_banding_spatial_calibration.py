"""Labelled calibration set for ``spatial_banding_factor`` (recalibrated 2026-10-21).

The first spatial detector ran on a 320px area-downscaled frame and counted any
plateau chain stepping by >= 1 gray level. Every smooth 8-bit gradient IS such a
chain (a ramp over L levels is L plateaus 1 level apart), so a perfectly smooth
ramp scored 1.000, a smooth vignette 0.892, and posterized 8/16/32-level ramps
also 1.000 — positives and negatives indistinguishable. Three live generic
masters with no visible banding never scored under 0.134 and tripped the 0.15
advisory.

This file is the labelled set the replacement was tuned on. Everything is
generated in-test (deterministic, stdlib only — this service has no numpy) at
1920x1080, plus four NATIVE-resolution crops of those three live masters (sky x2,
bokeh background, blurred pillarbox pad; ~112KB gzipped raw gray8). Each case
asserts BOTH the raw factor and what it does to a whole-frame score that is
itself over the ceiling (the exact false-positive situation the advisory hit).
"""

from __future__ import annotations

import functools
import gzip
import math
import random
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from src.services import mastering
from src.services.mastering import BANDING_GATE_CEILING, banding_score, spatial_banding_factor
from src.services.measure import parse_pgm

W, H = 1920, 1080
FIXTURES = Path(__file__).parent / "fixtures" / "banding"

# A whole-frame stats dict the PIXEL-FREE term scores well over the ceiling:
# full range, near-static, uniform percentiles — what every one of the live
# generic masters' false-positive frames looked like.
OVER_CEILING_STATS = {"YMIN": 4.0, "YMAX": 248.0, "YDIF": 0.5, "YLOW": 30.0, "YHIGH": 220.0}

NEGATIVE_FACTOR_MAX = 0.05
POSITIVE_FACTOR_MIN = 0.5


def _clip(v: float) -> int:
    return max(0, min(255, int(v)))


def _image(fn: Callable[[int, int], float]) -> bytes:
    return bytes(_clip(fn(x, y)) for y in range(H) for x in range(W))


def _row_image(fn: Callable[[int], float]) -> bytes:
    return bytes(_clip(fn(x)) for x in range(W)) * H


def _smooth_ramp(levels: int) -> bytes:
    lo = (255 - levels) // 2
    return _row_image(lambda x: lo + levels * x / W + 0.5)


def _smooth_diagonal(levels: int) -> bytes:
    lo = (255 - levels) // 2
    return _image(lambda x, y: lo + levels * (x + y) / (W + H) + 0.5)


def _posterized_ramp(bands: int, lo: int = 0, hi: int = 255) -> bytes:
    return _row_image(
        lambda x: lo + round(min(bands - 1, x * bands // W) * (hi - lo) / (bands - 1))
    )


def _vignette(bands: int | None = None) -> bytes:
    cx, cy, radius = W / 2, H / 2, math.hypot(W / 2, H / 2)

    def fn(x: int, y: int) -> float:
        val = 230 - 160 * (math.hypot(x - cx, y - cy) / radius) ** 2
        if bands:
            q = 160 / bands
            val = 70 + round((val - 70) / q) * q
        return val + 0.5

    return _image(fn)


def _with_noise(base: bytes, sigma: float, seed: int) -> bytes:
    rng = random.Random(seed)  # noqa: S311 — deterministic test fixture, not crypto
    tile_len = W * 64
    tile = [rng.gauss(0.0, sigma) for _ in range(tile_len)]
    return bytes(_clip(round(b + tile[i % tile_len])) for i, b in enumerate(base))


NEGATIVES: dict[str, Callable[[], bytes]] = {
    "smooth ramp 20 levels": lambda: _smooth_ramp(20),
    "smooth ramp 40 levels": lambda: _smooth_ramp(40),
    "smooth ramp 80 levels": lambda: _smooth_ramp(80),
    "smooth ramp 160 levels": lambda: _smooth_ramp(160),
    "smooth ramp 255 levels": lambda: _smooth_ramp(255),
    "smooth diagonal 40 levels": lambda: _smooth_diagonal(40),
    "smooth diagonal 160 levels": lambda: _smooth_diagonal(160),
    "smooth radial vignette": lambda: _vignette(),
    "ramp 80 levels + gaussian noise sigma 1.5": lambda: _with_noise(_smooth_ramp(80), 1.5, 7),
    "ramp 160 levels + gaussian noise sigma 2.5": lambda: _with_noise(_smooth_ramp(160), 2.5, 11),
}

POSITIVES: dict[str, Callable[[], bytes]] = {
    "posterized ramp 4 bands": lambda: _posterized_ramp(4),
    "posterized ramp 8 bands": lambda: _posterized_ramp(8),
    "posterized ramp 16 bands": lambda: _posterized_ramp(16),
    "posterized ramp 32 bands": lambda: _posterized_ramp(32),
    "posterized vignette 12 bands": lambda: _vignette(bands=12),
    "posterized sky 90..200, 16 bands + noise sigma 0.7": lambda: _with_noise(
        _posterized_ramp(16, 90, 200), 0.7, 3
    ),
    "dark gradient 16..48, 4-level steps": lambda: _posterized_ramp(9, 16, 48),
}


@functools.cache
def _factor(name: str) -> float:
    gen = NEGATIVES.get(name) or POSITIVES[name]
    return spatial_banding_factor(gen(), W, H)


def _live_crops() -> list[Path]:
    return sorted(FIXTURES.glob("*.gray.gz"))


def _load_crop(path: Path) -> tuple[bytes, int, int]:
    m = re.search(r"_(\d+)x(\d+)\.gray\.gz$", path.name)
    assert m, path.name
    w, h = int(m.group(1)), int(m.group(2))
    pixels = gzip.decompress(path.read_bytes())
    assert len(pixels) == w * h, path.name
    return pixels, w, h


@pytest.mark.parametrize("name", list(NEGATIVES))
def test_smooth_gradient_is_not_banding(name: str) -> None:
    factor = _factor(name)
    assert factor < NEGATIVE_FACTOR_MAX, (name, factor)
    assert banding_score(OVER_CEILING_STATS) >= BANDING_GATE_CEILING  # fixture sanity
    assert banding_score(OVER_CEILING_STATS, spatial_factor=factor) < BANDING_GATE_CEILING


@pytest.mark.parametrize("name", list(POSITIVES))
def test_posterized_gradient_is_banding(name: str) -> None:
    factor = _factor(name)
    assert factor >= POSITIVE_FACTOR_MIN, (name, factor)
    # Real staircase evidence saturates the multiplier: the whole-frame verdict
    # passes through untouched, so a true positive still fires the advisory.
    assert banding_score(OVER_CEILING_STATS, spatial_factor=factor) == banding_score(
        OVER_CEILING_STATS
    )


def test_live_master_crops_are_present() -> None:
    """Vacuity floor: deleting the fixtures must fail, not silently pass."""
    assert len(_live_crops()) == 4


@pytest.mark.parametrize("path", _live_crops(), ids=lambda p: p.name)
def test_live_master_crop_is_not_banding(path: Path) -> None:
    """Native-resolution crops of the three live generic masters that tripped
    the advisory (photoreal sky, bokeh, blurred pillarbox pad). No visible
    banding; the old detector called all of them staircases."""
    pixels, w, h = _load_crop(path)
    factor = spatial_banding_factor(pixels, w, h)
    assert factor < NEGATIVE_FACTOR_MAX, (path.name, factor)
    assert banding_score(OVER_CEILING_STATS, spatial_factor=factor) < BANDING_GATE_CEILING


def test_step_height_not_plateau_count_separates_the_classes() -> None:
    """The defining contrast: same span, same plateau COUNT order of magnitude,
    different step height. A 40-level smooth ramp has 40 plateaus stepping by
    1 level (48px each at 1920px); a 20-band posterization of the SAME span has
    20 plateaus stepping by 2. Only the second is banding."""
    smooth = spatial_banding_factor(_smooth_ramp(40), W, H)
    banded = spatial_banding_factor(_posterized_ramp(21, 107, 147), W, H)
    assert smooth == 0.0
    assert banded >= POSITIVE_FACTOR_MIN


def test_single_hard_edge_between_flats_is_not_a_staircase() -> None:
    w, h = 1920, 64
    gray = bytes(250 if x < w // 2 else 40 for x in range(w)) * h
    assert spatial_banding_factor(gray, w, h) == 0.0


def test_vertical_banding_reads_on_columns() -> None:
    w, h = 64, 1080
    gray = b"".join(bytes([min(255, (y * 16 // h) * 16)]) * w for y in range(h))
    assert spatial_banding_factor(gray, w, h) >= POSITIVE_FACTOR_MIN


def test_parse_pgm_round_trip_and_rejects() -> None:
    pixels = bytes(range(12))
    assert parse_pgm(b"P5\n4 3\n255\n" + pixels) == (pixels, 4, 3)
    assert parse_pgm(b"P5\n4 3\n255\n" + pixels[:-1]) is None  # truncated
    assert parse_pgm(b"P6\n4 3\n255\n" + pixels * 3) is None  # colour, not gray
    assert parse_pgm(b"P5\n4 3\n65535\n" + pixels * 2) is None  # 16-bit
    assert parse_pgm(b"") is None


def test_median_line_matches_the_reference_sliding_median() -> None:
    """``_median_line``'s width-5 fast path (zip windows) must be value-identical
    to the plain sliding median it replaced — the calibration above is pinned to
    those values."""

    def reference(line: list[int], width: int) -> list[int]:
        out = list(line)
        r = width // 2
        if r <= 0 or len(out) <= 2 * r:
            return out
        for i in range(r, len(out) - r):
            out[i] = sorted(line[i - r : i + r + 1])[r]
        return out

    rng = random.Random(5)  # noqa: S311 — deterministic test fixture, not crypto
    for _ in range(400):
        n = rng.randint(0, 300)
        line = [rng.randint(0, 255) if rng.random() < 0.3 else 100 + i // 7 for i in range(n)]
        for width in (1, 3, 5, 7):
            assert mastering._median_line(line, width) == reference(line, width), (n, width)
