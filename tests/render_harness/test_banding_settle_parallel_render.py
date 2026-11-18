"""Real-ffmpeg proof: a LONG clean film settles EXACTLY within budget, and the
parallel (worker-interpreter) settle returns exactly the serial result.

A 120s 30fps smooth full-range gradient (with light temporal noise, so it is
not a frozen frame — YDIF 0 scores 0): every one of its 3,600 frames clears
the prefilter and none is banded, so every frame must be measured before the
max is exact — the shape that pushed #254's serial settle (42.6s on a live 30s
film) past its 60s budget. Small resolution keeps CI fast; the production-size
timings live in the PR description.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from src.services import mastering, measure

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _long_clean_film(tmp_path: Path) -> Path:
    out = tmp_path / "long.mp4"
    subprocess.run(  # noqa: S603 — fixed argv, test-only
        [  # noqa: S607
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "nullsrc=s=192x108:r=30:d=120",
            "-vf", "geq=lum='16+220*X/W':cb=128:cr=128,noise=alls=4:allf=t",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


async def test_long_film_settles_exactly_within_budget_parallel_equals_serial(
    tmp_path: Path,
) -> None:
    path = _long_clean_film(tmp_path)
    timed = await measure.measure_shot_frames_timed(path)
    candidates = [
        pts
        for pts, m in timed
        if pts is not None and mastering.banding_score(m) >= mastering.BANDING_SPATIAL_PREFILTER
    ]
    assert len(candidates) >= 3000  # past #254's candidate cap: the long-film shape

    t0 = time.perf_counter()
    parallel = await mastering._settle_banding_spatial_factors(path, timed, None, workers=4)
    parallel_s = time.perf_counter() - t0
    serial = await mastering._settle_banding_spatial_factors(path, timed, None, workers=1)

    assert parallel_s < mastering._BANDING_SPATIAL_BUDGET_SECONDS
    assert set(candidates) <= set(parallel)  # every candidate measured: nothing left raw
    assert not mastering._banding_spatial_frontier(
        timed, None, parallel, set(parallel), per_group=len(timed)
    )  # EXACT
    assert parallel == serial  # value-identical, frame for frame
