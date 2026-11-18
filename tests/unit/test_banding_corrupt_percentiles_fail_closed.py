"""Corrupt luma percentiles must fail CLOSED, and the two accepted misses are pinned.

`banding_score`'s spatial term divides the 10-90 luma band by the full range.
Its absent-key branch deliberately falls back to ``uniformity = 1.0`` — "absent
is UNMEASURED, not zero" — so a measurement gap keeps flagging. The
percentiles-present branch used to disagree with it: ``max(0.0, YHIGH - YLOW)``
clamped an IMPOSSIBLE negative span to 0, uniformity became 0, and the score
became 0.0. Measured on the pre-fix code:

    absent percentiles     {"YMIN":0,"YMAX":255,"YDIF":0.5}                       -> 0.8333  FLAGGED
    YHIGH < YLOW (corrupt) {"YMIN":0,"YMAX":255,"YDIF":0.5,"YLOW":200,"YHIGH":50} -> 0.0000  not flagged

The 90th percentile cannot sit below the 10th, so the second dict cannot come
from a healthy signalstats parse — it means corrupt or transposed values. It
reported a CLEAN master from nonsense, fail-OPEN, on strictly LESS information
than the absent case that fails CLOSED. Both agree now.

The second half of this file pins two MEASURED, ACCEPTED MISSES (see the
`banding_score` docstring). They are documented rather than fixed because no
whole-frame percentile statistic can separate the first from clean dense detail.
They are pinned here so that a future change which catches them shows up as an
INTENTIONAL test update rather than a surprise.

Pure; no ffmpeg.
"""

from __future__ import annotations

import pytest

from src.services import mastering

# Full-range, low temporal delta: the shape that scores well over the ceiling
# once the spatial evidence is unmeasured.
_FULL_RANGE = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 0.5}


def test_corrupt_transposed_percentiles_still_flag() -> None:
    """`YHIGH` < `YLOW` is impossible input, so it is UNMEASURED, not clean."""
    corrupt = _FULL_RANGE | {"YLOW": 200.0, "YHIGH": 50.0}

    score = mastering.banding_score(corrupt)

    assert score == pytest.approx(0.8333), (
        "a negative percentile span must fall back to uniformity 1.0; scoring it "
        "0.0 reports a CLEAN master from corrupt input and disarms the gate"
    )
    assert score >= mastering.BANDING_GATE_CEILING
    assert mastering.banding_risk(corrupt) is True


def test_corrupt_percentiles_agree_with_the_absent_case() -> None:
    """The whole point of the fix: the two UNMEASURED branches must not disagree.

    Absent keys carry strictly LESS information than corrupt values, so any
    build where corrupt scores BELOW absent has the policy inverted.
    """
    absent = mastering.banding_score(_FULL_RANGE)
    corrupt = mastering.banding_score(_FULL_RANGE | {"YLOW": 200.0, "YHIGH": 50.0})
    equal = mastering.banding_score(_FULL_RANGE | {"YLOW": 128.0, "YHIGH": 128.0})

    assert corrupt == absent
    assert equal == absent, "a zero-width 10-90 band is degenerate, i.e. unmeasured too"


def test_upper_clamp_survives_and_still_flags() -> None:
    """The control: the UPPER clamp is a legitimate rounding artifact, keep it.

    ``p90 - p10 > yrange`` can come out of averaged per-frame stats. It still
    means "the band covers the whole range", so it clamps to 1.0 and flags —
    it must NOT be swept up by the non-positive fallback.
    """
    over = _FULL_RANGE | {"YLOW": 0.0, "YHIGH": 300.0}
    assert mastering.banding_score(over) == mastering.banding_score(_FULL_RANGE)


def test_a_genuinely_narrow_band_is_still_suppressed() -> None:
    """The other control: the fix must not blanket-disarm the spatial term.

    A POSITIVE, narrow 10-90 band is real evidence of detail-with-outliers and
    must still push the score under the ceiling. Without this, deleting the
    spatial term outright would pass every test above.
    """
    detailed = _FULL_RANGE | {"YLOW": 110.0, "YHIGH": 145.0}
    assert mastering.banding_score(detailed) < mastering.BANDING_GATE_CEILING


# --------------------------------------------------------------------------
# ACCEPTED MISSES — documented in the `banding_score` docstring, not fixed.
# --------------------------------------------------------------------------


def test_partial_ramp_is_a_known_miss() -> None:
    """A sky band over ~30% of the frame with detail below scores 0.0820.

    NOT fixable from whole-frame percentiles: mandelbrot — dense detail, no
    banding, correctly suppressed — scores 0.071 and this scores 0.082. They are
    the same number, so no ceiling separates them. Catching localized banding
    needs a SPATIALLY LOCALIZED (per-tile / per-region) measurement, which
    signalstats' whole-frame aggregates cannot provide.
    """
    partial_ramp = _FULL_RANGE | {"YLOW": 70.0, "YHIGH": 150.0}

    assert mastering.banding_score(partial_ramp) == pytest.approx(0.0820)
    assert mastering.banding_score(partial_ramp) < mastering.BANDING_GATE_CEILING


def test_limited_range_gradient_is_a_known_miss() -> None:
    """A limited-range (16-235-ish) 8-bit gradient scores 0.1206.

    This one is an EXPONENT/CEILING question, not a formula question:
    ``span * smoothness`` is 0.1709, linear uniformity gives 0.1436 and the
    shipped square gives 0.1206 — so only the exponent-0 (pre-spatial-term)
    variant clears 0.15. Squaring is what bought the mandelbrot fix, so it is
    paying for a real separation. Recording the trade, not resolving it.
    """
    limited = {"YMIN": 80.0, "YMAX": 180.0, "YDIF": 0.5, "YLOW": 88.0, "YHIGH": 172.0}

    assert mastering.banding_score(limited) == pytest.approx(0.1206)
    assert mastering.banding_score(limited) < mastering.BANDING_GATE_CEILING

    yrange = 180.0 - 80.0
    span = (yrange - mastering._BANDING_RANGE_MIN) / (255.0 - mastering._BANDING_RANGE_MIN)
    smoothness = (mastering._BANDING_YDIF_MAX - 0.5) / mastering._BANDING_YDIF_MAX
    uniformity = (172.0 - 88.0) / yrange

    assert span * smoothness == pytest.approx(0.1709, abs=1e-4)
    assert span * smoothness * uniformity == pytest.approx(0.1436, abs=1e-4)
    assert span * smoothness >= mastering.BANDING_GATE_CEILING
    assert span * smoothness * uniformity < mastering.BANDING_GATE_CEILING
