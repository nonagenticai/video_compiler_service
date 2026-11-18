"""The banding SPATIAL term (`spatial_banding_factor`) and its combination
with the whole-frame score (`banding_score(..., spatial_factor=...)`).

Everything `banding_score` measured before this term existed reads ONLY
whole-frame signalstats aggregates (10th/90th luma percentiles) — it cannot
tell "wide range, static, and actually stepped" from "wide range, static,
and smooth, or a hard edge". Measured on two real delivered masters (see
`mastering.spatial_banding_factor`'s module comment and the PR description):
6 shots trip the whole-frame score with ZERO visible or measurable banding —
a mostly-white page with a small logo, and the hard edge between a white
panel and a grey letterbox. `spatial_banding_factor` supplies real pixel-
neighbourhood evidence (Sobel edge exclusion + monotonic plateau-chain
detection) so the combined score stops crying wolf on that material while
staying responsive to an actual posterized ramp.

Pure; no ffmpeg. Uses tiny synthetic pixel buffers built in Python, not the
real downscaled-frame fixtures used to derive the module's tuning (those
require ffmpeg + libx264 and are exercised in the PR's own measurement, not
re-run here).
"""

from __future__ import annotations

from src.services.mastering import (
    _BANDING_SPATIAL_SCALE,
    BANDING_GATE_CEILING,
    _banding_spatial_candidates,
    banding_score,
    spatial_banding_factor,
)


def _solid(value: int, w: int, h: int) -> bytes:
    return bytes([value]) * (w * h)


def _horizontal_ramp(w: int, h: int, levels: int, step_width: int) -> bytes:
    """A posterized horizontal ramp: `levels` flat bands, each `step_width`
    pixels wide, stepping ~255/levels gray levels apart. Mirrors the
    synthetic positives measured in the PR (there generated via ffmpeg
    `geq` + libx264; here built directly since the pure function takes raw
    pixels and does not care how they were produced)."""
    row = bytearray(w)
    for x in range(w):
        band = min(levels - 1, x // step_width)
        row[x] = min(255, int(band * 255 / max(1, levels - 1)))
    return bytes(row) * h


def _noise_row(w: int, h: int, seed: int = 1234) -> bytes:
    """Pseudo-random per-pixel noise (no fixed pattern, no plateaus) — the
    stand-in for genuinely detailed/textured content that must NOT read as a
    staircase."""
    state = seed
    out = bytearray(w * h)
    for i in range(w * h):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = state % 256
    return bytes(out)


def test_solid_frame_has_zero_spatial_factor() -> None:
    """A single flat color has no edges and no plateau CHAIN (nothing to
    step to) — the whole frame is one giant run, never counted as a
    staircase."""
    assert spatial_banding_factor(_solid(128, 64, 64), 64, 64) == 0.0


def test_posterized_ramp_scores_high() -> None:
    """A clean posterized ramp (16 bands, evenly spaced, well-separated
    levels) is exactly the material this term exists to catch."""
    gray = _horizontal_ramp(320, 32, levels=16, step_width=20)
    factor = spatial_banding_factor(gray, 320, 32)
    assert factor > 0.5, factor


def test_noise_scores_near_zero() -> None:
    """Random per-pixel noise has no plateau structure — every 'run' is a
    single pixel, well under `_BANDING_SPATIAL_MIN_RUN`."""
    gray = _noise_row(320, 32)
    factor = spatial_banding_factor(gray, 320, 32)
    assert factor < 0.05, factor


def test_hard_edge_between_two_flats_is_not_a_staircase() -> None:
    """The kb3 clip-1 false-positive shape: a hard edge between two large
    flat regions (a white panel against a grey letterbox), not a gradient.
    One edge, two giant flat runs on either side of it — no CHAIN of small
    monotonic steps."""
    w, h = 320, 32
    row = bytearray(w)
    for x in range(w):
        row[x] = 250 if x < w // 2 else 40
    gray = bytes(row) * h
    factor = spatial_banding_factor(gray, w, h)
    assert factor < 0.05, factor


def test_malformed_buffer_returns_zero_not_raise() -> None:
    """Shape mismatch (caller bug / truncated decode) degrades to 0.0 —
    this is advisory telemetry and must never raise."""
    assert spatial_banding_factor(b"\x00" * 10, 320, 180) == 0.0
    assert spatial_banding_factor(b"", 0, 0) == 0.0


def test_banding_score_spatial_factor_none_is_byte_identical() -> None:
    """`spatial_factor=None` (the default — no caller has paid for a frame
    decode) must leave `banding_score` EXACTLY as it was before this term
    existed. This is the backward-compatibility contract every existing
    caller/test relies on."""
    stats = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.5, "YLOW": 20.0, "YHIGH": 230.0}
    assert banding_score(stats) == banding_score(stats, spatial_factor=None)


def test_banding_score_spatial_factor_suppresses_false_positive() -> None:
    """A high whole-frame score with NO spatial evidence (factor near 0,
    e.g. the kb3/kb4 false-positive shapes) must be pulled below the gate
    ceiling by the spatial multiplier."""
    stats = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.5, "YLOW": 20.0, "YHIGH": 230.0}
    whole_frame = banding_score(stats)
    assert whole_frame >= BANDING_GATE_CEILING, (
        "fixture must reproduce a false positive to test suppression"
    )
    suppressed = banding_score(stats, spatial_factor=0.0)
    assert suppressed < BANDING_GATE_CEILING
    assert suppressed == 0.0


def test_banding_score_spatial_factor_at_or_above_scale_is_unchanged() -> None:
    """A spatial factor at/above `_BANDING_SPATIAL_SCALE` (real staircase
    evidence) must leave the whole-frame score UNCHANGED — the multiplier
    saturates at 1.0, it never amplifies past the whole-frame term's own
    verdict."""
    stats = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.5, "YLOW": 20.0, "YHIGH": 230.0}
    whole_frame = banding_score(stats)
    at_scale = banding_score(stats, spatial_factor=_BANDING_SPATIAL_SCALE)
    above_scale = banding_score(stats, spatial_factor=1.0)
    assert at_scale == whole_frame
    assert above_scale == whole_frame


def test_banding_score_spatial_factor_never_raises_score_above_whole_frame() -> None:
    """The multiplier is monotone non-decreasing in `spatial_factor` and
    capped at the whole-frame value — no input can push the combined score
    ABOVE what the whole-frame term alone would have produced."""
    stats = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.5, "YLOW": 20.0, "YHIGH": 230.0}
    whole_frame = banding_score(stats)
    for factor in (0.0, 0.01, 0.05, 0.3, 1.0, 5.0):
        assert banding_score(stats, spatial_factor=factor) <= whole_frame


def test_spatial_candidates_are_bounded_by_prefilter() -> None:
    """`_banding_spatial_candidates` must skip every frame whose whole-frame
    score is already far under the ceiling (nothing there for a spatial
    factor to suppress or confirm), and must return the worst pts_time
    whole-file plus each shot's own worst pts_time when it clears the
    prefilter."""
    clean = {"YMIN": 100.0, "YMAX": 110.0, "YDIF": 5.0}  # scores 0.0, well under prefilter
    banded_a = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.1, "YLOW": 10.0, "YHIGH": 245.0}
    banded_b = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.1, "YLOW": 10.0, "YHIGH": 245.0}
    timed_frames: list[tuple[float | None, dict[str, float]]] = [
        (0.0, clean),
        (1.0, banded_a),
        (2.0, clean),
        (3.0, banded_b),
    ]
    shot_timeline: list[dict[str, object]] = [
        {"line_id": "clip-1", "start": 0.0, "end": 2.0},
        {"line_id": "clip-2", "start": 2.0, "end": 4.0},
    ]
    candidates = _banding_spatial_candidates(timed_frames, shot_timeline)
    # Both banded frames tie for worst-of-file; either may be picked as the
    # whole-file candidate, but BOTH must appear once via their own shot.
    assert 1.0 in candidates
    assert 3.0 in candidates
    assert 0.0 not in candidates
    assert 2.0 not in candidates
    # Deduped: the whole-file candidate and a shot candidate that land on the
    # same pts_time must not be counted twice.
    assert len(candidates) == len(set(candidates))


def test_spatial_candidates_empty_when_nothing_clears_prefilter() -> None:
    clean = {"YMIN": 100.0, "YMAX": 110.0, "YDIF": 5.0}
    timed_frames: list[tuple[float | None, dict[str, float]]] = [(0.0, clean), (1.0, clean)]
    shot_timeline: list[dict[str, object]] = [{"line_id": "clip-1", "start": 0.0, "end": 2.0}]
    assert _banding_spatial_candidates(timed_frames, None) == []
    assert _banding_spatial_candidates(timed_frames, shot_timeline) == []
