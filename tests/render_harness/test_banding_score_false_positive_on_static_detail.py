"""KNOWN-DEFECT pin: ``banding_score`` has no spatial term, so a densely
detailed but near-static shot trips the 0.15 banding ceiling despite carrying
no banding at all.

⛔ THIS TEST DOES NOT ASSERT CORRECT BEHAVIOUR. It pins a MEASURED DEFECT so the
defect is visible in CI, not a guarantee the current score is right. Read
``mastering.banding_score``'s docstring first — it now spells out the mechanism:
the score is ``span(YMAX-YMIN) * smoothness(YDIF)``, i.e. "wide luma range" times
"low frame-to-frame delta". Both terms are real signals, but neither is spatial,
and banding is a spatial artifact (adjacent-pixel quantisation steps in a smooth
gradient). Nothing in the formula inspects a frame's own pixels for that
structure.

Measured today (2026-08-26) with the real ``measure.measure_shot_frames`` over
real ffmpeg-rendered clips, 60 frames each:

    material                                    max score   frac >= 0.15
    smooth 8-bit gradient (banded by construction)   0.810        51.7%
    same gradient + heavy dither (unbandable)        0.000         0.0%
    testsrc2 (textured, no banding)                  0.248        80.0%   <- FALSE POSITIVE
    mandelbrot (dense detail, near-static)           0.440        71.7%   <- FALSE POSITIVE
    life (noisy, high motion)                        0.000         0.0%
    flat grey                                        0.000         0.0%

And on a real delivered master: median ``max_banding_before`` was 0.9128 against
the 0.15 ceiling (98.7% of frames over it), SSIM before/after deband 0.9998 (the
filter barely touched the picture), and the score went UP on 13 of 13 segments —
consistent with a spatial filter being unable to move a score that has no
spatial term.

``mandelbrot`` is used here (not ``testsrc2``, also false-positive above) because
it is a lavfi SOURCE built into libavfilter core — no optional filter
(libass/libfreetype) dependency, so it needs no ``requires_filters`` capability
skip on top of the module-level ffmpeg/ffprobe guard.

⚠️ WHEN THE FIX LANDS (a real spatial term — e.g. a per-frame gradient/edge
probe, or comparing measured banding against a reference deband pass), THIS TEST
SHOULD START FAILING. That is the fix working, not a regression. Update or
delete this test then — do not silence it by loosening the assertion, and do not
treat a passing run of this test as evidence the metric is correct.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.services import mastering, measure, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_W, _H = 320, 240
_FPS = 30
_DURATION = 2.0  # 60 frames at 30fps


async def _make_mandelbrot_clip(path: Path) -> Path:
    """A densely detailed, near-static lavfi source: fine fractal texture with a
    fixed default zoom/pan speed slow enough that most of the frame barely moves
    frame-to-frame — exactly the "wide luma range + low temporal delta" shape
    ``banding_score`` rewards, with zero banding anywhere in the image."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"mandelbrot=size={_W}x{_H}:rate={_FPS}",
            "-t",
            str(_DURATION),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(path),
        ]
    )
    return path


@pytest.mark.asyncio
async def test_static_detailed_clip_with_no_banding_trips_the_ceiling(
    tmp_path: Path,
) -> None:
    """KNOWN DEFECT, pinned: mandelbrot has no smooth gradient anywhere — it is
    dense fractal edge detail, the opposite of a banding candidate — yet the
    span+smoothness formula convicts most of its frames.

    Measured 2026-08-26: max score 0.440, 71.7% of frames >= the 0.15 ceiling.
    Assert against the measured shape (majority of frames over ceiling, and the
    max comfortably clear of it) rather than the exact figures, since ffmpeg/libx264
    version drift could move them slightly — the point being pinned is "this
    clip has no banding and still fails the gate", not a specific decimal."""
    clip = await _make_mandelbrot_clip(tmp_path / "mandelbrot.mp4")

    frames = await measure.measure_shot_frames(clip)
    assert frames, "signalstats produced no frames — harness/ffmpeg setup broke"

    scores = [mastering.banding_score(f) for f in frames]
    assert scores, "no frame produced a usable signalstats reading"

    ceiling = mastering.BANDING_GATE_CEILING
    over_ceiling = [s for s in scores if s >= ceiling]
    frac_over = len(over_ceiling) / len(scores)

    # ✅ FIXED 2026-09-02 — this file was a KNOWN-DEFECT pin and is now a
    # REGRESSION test, exactly as its own instruction said to do:
    #
    #   before the spatial term: 41.7% of frames over the ceiling, max 0.533
    #   after:                   0% over the ceiling, max 0.071
    #
    # `mandelbrot` is dense detail with NO banding, so a metric that measures
    # banding must score it at or near zero. The old score fired because it read
    # "wide luma range x near-static shot" — a stillness detector. The added
    # spatial term (`YHIGH`-`YLOW` over the full range, squared) collapses it,
    # because detailed content concentrates its pixels in a narrow percentile
    # band while a true ramp spreads them evenly.
    #
    # ⚠️ If THIS starts failing, the spatial term has regressed — do not loosen
    # it; find what put the stillness confound back.
    assert frac_over == 0.0, (
        f"a clip with no banding must not trip the {ceiling} ceiling on ANY "
        f"frame now that banding_score has a spatial term; got {frac_over:.1%} "
        f"({len(over_ceiling)}/{len(scores)})"
    )
    assert max(scores) < 0.15, (
        f"measured 0.071 after the fix (was 0.533); got {max(scores)} — the "
        "stillness confound appears to be back"
    )
