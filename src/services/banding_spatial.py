"""Banding SPATIAL term — the pure per-frame pixel detector.

STDLIB ONLY and import-light ON PURPOSE: ``mastering._settle_banding_spatial_factors``
runs ``spatial_factor_from_lines`` in worker interpreters
(``concurrent.futures.InterpreterPoolExecutor``), each of which imports THIS module
and nothing else. Importing pydantic/structlog/the app here would make every worker
pay that import (and pydantic-core is not subinterpreter-safe). ``mastering``
re-exports the public names; ``banding_score`` and the multiplier scale stay there.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Banding — SPATIAL term (pure pixel analysis, no ffmpeg)
# ---------------------------------------------------------------------------
#
# ⛔ ``mastering.banding_score``'s whole-frame terms (its ``span``/``smoothness``/
# ``uniformity`` factors) read ONLY whole-frame signalstats aggregates —
# 10th/90th luma PERCENTILES, not pixel NEIGHBOURHOODS — so it cannot tell
# "wide range, static, and actually stepped" from "wide range, static, and
# smooth or a hard edge" (2026-09-18: 6 false-positive shots on two TodoMVC
# masters, a mostly-white page with a logo and a panel/letterbox edge).
# ``spatial_banding_factor`` supplies the missing pixel evidence.
#
# ⛔ RECALIBRATED 2026-10-21 — THE FIRST VERSION OF THIS TERM COULD NOT TELL A
# SMOOTH GRADIENT FROM A BANDED ONE. It ran on a 320px AREA-DOWNSCALED frame and
# counted any chain of plateaus stepping by ``>= 1`` gray level. But EVERY smooth
# 8-bit gradient is, at the pixel level, a chain of plateaus stepping by exactly
# 1 level (a ramp over L levels across S px is L plateaus of S/L px). Measured on
# the old detector: a perfectly smooth ramp scored 1.000 (40/80/160-level spans),
# a smooth radial vignette 0.892 — and 8/16/32-level POSTERIZED ramps also 1.000.
# Positives and negatives were indistinguishable. On three live generic masters
# (photoreal AI stills, blurred pillarbox pads, bokeh; no visible banding) the
# factor never fell below 0.134 over ~800 frames, so the advisory fired
# (0.27/0.36/0.36 vs the 0.15 ceiling) on clean pictures. Simply raising the
# step floor on the DOWNSCALED frame does not work either: area averaging
# smears a 4..8 px posterization step into a soft ramp, so a step_min 3-4 sweep
# killed the false positives AND the 8/16-level positives together.
#
# What banding actually IS, and what this detector now measures: wide FLAT
# plateaus separated by abrupt steps LARGER than the local slope predicts.
# A smooth ramp of slope ``s`` levels/px steps by 1 level every ``1/s`` px; a
# band of the SAME average slope that steps by ``h`` levels has plateaus ``h``x
# wider than that. So "step height >= 2 levels after subtracting the in-plateau
# slope" is exactly "plateau at least 2x wider than a smooth ramp of this slope
# would produce" — the one test separates the two, at any slope.
#
# Method (stdlib only — this service has no numpy/PIL dependency):
#   1. NO averaging (``measure.measure_spatial_factors_streamed`` never
#      area-downscales; frames over 1920 wide are NEAREST-neighbour capped to
#      1920, ``measure._SPATIAL_FRAME_MAX_WIDTH``). Sample
#      ``_BANDING_SPATIAL_LINES`` evenly spaced rows AND columns — cost linear
#      in width + height (~15ms on a 1920x1080 frame locally, ~3x that on the
#      prod nodes).
#   2. Median-filter each line (width ``_BANDING_SPATIAL_MEDIAN``): removes
#      isolated encode/dither noise while preserving a step edge exactly.
#   3. Split into plateaus: maximal runs staying within ``_BANDING_SPATIAL_TOL``
#      of their own running mean; keep runs >= ``_BANDING_SPATIAL_MIN_WIDTH`` px.
#   4. Between two plateaus at most ``_BANDING_SPATIAL_MAX_GAP`` px apart (an
#      ABRUPT transition), the step = (mean of the next plateau's first
#      ``_BANDING_SPATIAL_EDGE_PX`` px) - (mean of this plateau's last ones),
#      MINUS the change the two plateaus' own slopes predict across that span.
#      A step counts when ``_BANDING_SPATIAL_STEP_MIN <= |step| <=
#      _BANDING_SPATIAL_STEP_MAX``. Running-mean segmentation of a smooth ramp
#      produces plateau MEANS several levels apart (that was a second way the old
#      detector manufactured steps); the LOCAL boundary difference stays ~1.
#   5. A staircase is >= ``_BANDING_SPATIAL_MIN_STEPS`` consecutive counted steps
#      in the SAME direction, neighbouring heights within
#      ``_BANDING_SPATIAL_HEIGHT_RATIO`` and neighbouring plateau widths within
#      ``_BANDING_SPATIAL_WIDTH_RATIO`` of each other. One hard edge between two
#      flats is a single step, never a staircase; texture never forms plateaus.
#   6. factor = max(row, column) fraction of SAMPLED pixels inside a staircase's
#      plateaus — horizontal banding reads on rows, vertical on columns.
#
# Calibration set — ``tests/unit/test_banding_spatial_calibration.py`` (1920x1080,
# generated in-test, plus 4 native-resolution crops of the three live generic
# masters). Raw ``spatial_banding_factor`` values:
#
#     NEGATIVES (smooth)                    raw / x264   POSITIVES (banded)           raw / x264
#     linear ramp 20/40/80/160/255 levels   0.000/0.000  posterized ramp 4/8/16/32    1.000/1.000
#     diagonal ramp 40/160 levels           0.000/0.000  posterized vignette (12)     0.999/0.773
#     radial vignette                       0.000/0.000  posterized sky 16 + noise    0.941/1.000
#     ramp 80 + gaussian noise sigma=1.5    0.001/0.000  dark 16..48, 4-level steps   1.000/0.999
#     ramp 160 + gaussian noise sigma=2.5   0.004/0.013  20 bands x 2 levels (40 lv)  1.000/0.000 *
#     live crops (sky x2, bokeh, pillarbox pad): all 0.000
#     (x264 = libx264 crf=18 yuv420p encode/decode round trip, ffmpeg 4.4)
#
#     live masters, native 2560x1440, EVERY frame clearing the pre-filter
#     (2,367 frames over three generic masters + 143 over two showcases): max
#     0.0123, median 0.0025 (old detector: never below 0.134). Through the 1920
#     nearest cap (2026-10-21, 4 generic + 3 showcase masters, 4,143 frames):
#     max 0.0210, median <= 0.0027. Calibration regenerated at 2560x1440 through
#     the cap: negatives <= 0.0066, positives >= 0.881 after x264.
#
#   * KNOWN MISS: a 2-level-step posterization survives raw but libx264 crf=18
#     re-dithers it below detection. 2 levels is the visibility floor, and the
#     encoder's own noise is doing the debanding; accepted, not tuned for.
#
# Constant choices: STEP_MIN 2 is the definition above (a smooth ramp's step is
# 1); 3 dropped the x264-encoded dark 4-level case to 0. MIN_STEPS 3 is the most
# a 4-level posterization can show (4 plateaus) — 4 missed it. TOL 1.5 must stay
# BELOW STEP_MIN (TOL 2 merged 2-level bands into one plateau) yet tolerate +-1
# noise (TOL 1 scored the sigma-0.7 noisy-sky positive 0.000); MEDIAN 5 vs 3
# lifted that positive 0.81 -> 0.94. STEP_MAX 96 admits a 4-level full-range ramp
# (85-level steps); a hard UI edge is excluded by the chain rule, not the height.
_BANDING_SPATIAL_LINES = 32
_BANDING_SPATIAL_MEDIAN = 5
_BANDING_SPATIAL_TOL = 1.5
_BANDING_SPATIAL_MIN_WIDTH = 6
_BANDING_SPATIAL_MAX_GAP = 3
_BANDING_SPATIAL_EDGE_PX = 3
_BANDING_SPATIAL_STEP_MIN = 2.0
_BANDING_SPATIAL_STEP_MAX = 96.0
_BANDING_SPATIAL_MIN_STEPS = 3
_BANDING_SPATIAL_HEIGHT_RATIO = 3.0
_BANDING_SPATIAL_WIDTH_RATIO = 4.0


def _median_line(line: Sequence[int], width: int) -> list[int]:
    """PURE. Sliding median of odd ``width`` (ends copied through). Removes
    isolated noise samples; a monotone step edge passes through unchanged."""
    out = list(line)
    r = width // 2
    n = len(out)
    if r <= 0 or n <= 2 * r:
        return out
    if r == 2:  # the production width: zip-based windows, ~25% faster, same values
        windows = zip(
            out[: n - 4], out[1 : n - 3], out[2 : n - 2], out[3 : n - 1], out[4:], strict=True
        )
        out[2 : n - 2] = [sorted(w)[2] for w in windows]
        return out
    out[r : n - r] = [sorted(out[i - r : i + r + 1])[r] for i in range(r, n - r)]
    return out


def _plateaus(line: Sequence[int], tol: float, min_width: int) -> list[tuple[int, int]]:
    """PURE. Maximal runs whose samples stay within ``tol`` of the run's own
    running mean (no fixed bucket boundary for noise to straddle); returns
    ``[(start, length), ...]`` for runs at least ``min_width`` long."""
    out: list[tuple[int, int]] = []
    n = len(line)
    i = 0
    while i < n:
        start = i
        total = line[i]
        count = 1
        i += 1
        while i < n and abs(line[i] * count - total) <= tol * count:
            total += line[i]
            count += 1
            i += 1
        if count >= min_width:
            out.append((start, count))
    return out


def _boundary_steps(line: Sequence[int], plats: list[tuple[int, int]]) -> list[float | None]:
    """PURE. ``steps[i]`` = the slope-corrected step from ``plats[i]`` to
    ``plats[i + 1]`` (method step 4), or ``None`` when the transition is not
    abrupt or the height is outside ``[STEP_MIN, STEP_MAX]``."""
    k = _BANDING_SPATIAL_EDGE_PX
    steps: list[float | None] = []
    for (s0, c0), (s1, c1) in itertools.pairwise(plats):
        gap = s1 - (s0 + c0)
        if gap > _BANDING_SPATIAL_MAX_GAP:
            steps.append(None)
            continue
        head0 = sum(line[s0 : s0 + k]) / k
        tail0 = sum(line[s0 + c0 - k : s0 + c0]) / k
        head1 = sum(line[s1 : s1 + k]) / k
        tail1 = sum(line[s1 + c1 - k : s1 + c1]) / k
        # The change the plateaus' OWN slopes predict across the transition: a
        # smooth ramp segmented into runs is fully explained by it.
        slope = ((tail0 - head0) / max(1, c0 - k) + (tail1 - head1) / max(1, c1 - k)) / 2.0
        step = (head1 - tail0) - slope * (gap + k)
        ok = _BANDING_SPATIAL_STEP_MIN <= abs(step) <= _BANDING_SPATIAL_STEP_MAX
        steps.append(step if ok else None)
    return steps


def _continues_staircase(a: float, b: float, w0: int, w1: int) -> bool:
    """PURE. Whether step ``b`` (onto a plateau of width ``w1``) continues the
    staircase that step ``a`` (onto width ``w0``) is part of (method step 5)."""
    if (a > 0) != (b > 0):
        return False
    if max(abs(a), abs(b)) > _BANDING_SPATIAL_HEIGHT_RATIO * min(abs(a), abs(b)):
        return False
    return max(w0, w1) <= _BANDING_SPATIAL_WIDTH_RATIO * min(w0, w1)


def _line_staircase_pixels(raw: Sequence[int]) -> int:
    """PURE. Pixels of one sampled line that sit inside a banding staircase
    (see the method block above, steps 2-5)."""
    line = _median_line(raw, _BANDING_SPATIAL_MEDIAN)
    plats = _plateaus(line, _BANDING_SPATIAL_TOL, _BANDING_SPATIAL_MIN_WIDTH)
    steps = _boundary_steps(line, plats)
    counted: set[int] = set()
    i = 0
    while i < len(steps):
        a = steps[i]
        if a is None:
            i += 1
            continue
        j = i
        while j + 1 < len(steps):
            b = steps[j + 1]
            if b is None or not _continues_staircase(a, b, plats[j + 1][1], plats[j + 2][1]):
                break
            a = b
            j += 1
        if j - i + 1 >= _BANDING_SPATIAL_MIN_STEPS:
            counted.update(range(i, j + 2))  # plateaus i..j+1 bound steps i..j
        i = j + 1
    return sum(plats[q][1] for q in counted)


def spatial_banding_factor(gray: bytes, width: int, height: int) -> float:
    """The fraction (0..1) of a NATIVE-resolution grayscale frame's sampled
    row/column pixels that sit inside a banding STAIRCASE — wide flat plateaus
    separated by abrupt steps larger than the local slope predicts. A smooth
    8-bit gradient (1-level steps) scores 0. PURE (no I/O;
    ``measure.measure_spatial_factors_streamed`` owns the ffmpeg decode). See the
    method + calibration block above.

    ``gray`` must be exactly ``width * height`` bytes (one grayscale byte per
    pixel, row-major). Returns ``0.0`` on any shape mismatch rather than
    raising — this is advisory telemetry, never allowed to fail a render."""
    if len(gray) != width * height or width < 3 or height < 3:
        return 0.0
    rows, cols = spatial_banding_lines(gray, width, height)
    return spatial_factor_from_lines(rows, cols, width, height)


def spatial_banding_lines(gray: bytes, width: int, height: int) -> tuple[list[bytes], list[bytes]]:
    """PURE. The ``_BANDING_SPATIAL_LINES`` evenly spaced rows and columns of a
    row-major grayscale frame that ``spatial_banding_factor`` measures (C-level
    slices, ~100 KB for a 1920x1080 frame). Splitting extraction from the
    measure lets the settle pass ship only these lines to a worker, never the
    whole frame. Caller guarantees ``len(gray) == width * height``."""
    n = _BANDING_SPATIAL_LINES
    rows = sorted({int((r + 0.5) * height / n) for r in range(n)})
    cols = sorted({int((c + 0.5) * width / n) for c in range(n)})
    return [gray[y * width : (y + 1) * width] for y in rows], [gray[x::width] for x in cols]


def spatial_factor_from_lines(
    row_lines: Sequence[bytes], col_lines: Sequence[bytes], width: int, height: int
) -> float:
    """PURE. ``spatial_banding_factor`` from the lines ``spatial_banding_lines``
    extracted — the worker-side half, value-identical to the serial path
    (``0.0`` for a frame too small to measure, as there)."""
    if width < 3 or height < 3 or not row_lines or not col_lines:
        return 0.0
    row_px = sum(_line_staircase_pixels(line) for line in row_lines)
    col_px = sum(_line_staircase_pixels(line) for line in col_lines)
    return max(row_px / (len(row_lines) * width), col_px / (len(col_lines) * height))


def measure_lines_in_worker(
    row_lines: Sequence[bytes], col_lines: Sequence[bytes], width: int, height: int
) -> float:
    """Worker-interpreter entrypoint. Self-contained ON PURPOSE: an
    ``InterpreterPoolExecutor`` may run a shared function against its OWN
    ``__main__`` globals rather than this module's, so the body resolves the
    detector by import instead of by module global."""
    from src.services.banding_spatial import spatial_factor_from_lines as measure

    return measure(row_lines, col_lines, width, height)
