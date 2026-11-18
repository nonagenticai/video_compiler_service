"""The banding settle pass scales with the pod's CPUs, not film length.

#254's serial settle took 42.6s of its 60s budget on a live 2560x1440 30s film,
so a longer clean film fell back to the RAW upper bound. The detector now runs
on worker interpreters sized to the pod's REAL CPU budget (the cgroup quota —
``os.cpu_count`` reports the node's 12 CPUs, the pod is limited to 10) and
frames that can no longer be any group's max are skipped as results land.

Pure pieces + a fake streamed measurer; the real-ffmpeg long-film proof is
``tests/render_harness/test_banding_settle_parallel_render.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.services import banding_spatial, mastering, measure
from src.services.mastering import _banding_spatial_frontier, _settle_banding_spatial_factors

TimedFrames = list[tuple[float | None, dict[str, float]]]


def _stats(ydif: float) -> dict[str, float]:
    return {"YMIN": 0.0, "YMAX": 255.0, "YDIF": ydif, "YLOW": 10.0, "YHIGH": 245.0}


# --- CPU budget -------------------------------------------------------------


def test_cgroup_v2_quota_is_read_in_cores(tmp_path: Path) -> None:
    v2 = tmp_path / "cpu.max"
    v2.write_text("1000000 100000\n")  # the video-compiler pod: limits.cpu "10"
    assert measure._cgroup_cpu_limit(v2, tmp_path / "q", tmp_path / "p") == 10.0
    v2.write_text("max 100000\n")
    assert measure._cgroup_cpu_limit(v2, tmp_path / "q", tmp_path / "p") is None


def test_cgroup_v1_quota_and_no_cgroup(tmp_path: Path) -> None:
    q, p = tmp_path / "quota", tmp_path / "period"
    q.write_text("250000\n")
    p.write_text("100000\n")
    assert measure._cgroup_cpu_limit(tmp_path / "missing", q, p) == 2.5
    q.write_text("-1\n")
    assert measure._cgroup_cpu_limit(tmp_path / "missing", q, p) is None
    assert measure._cgroup_cpu_limit(tmp_path / "a", tmp_path / "b", tmp_path / "c") is None


def test_available_cpus_is_capped_by_the_quota_rounded_up() -> None:
    assert measure.available_cpus(cgroup_limit=1.5) == 2
    assert measure.available_cpus(cgroup_limit=0.25) == 1
    assert 1 <= measure.available_cpus(cgroup_limit=10_000.0) <= (measure.os.cpu_count() or 1)


@pytest.mark.parametrize(("cpus", "workers"), [(1, 1), (2, 1), (3, 1), (4, 2), (10, 8), (12, 8)])
def test_worker_count_leaves_room_for_ffmpeg_and_the_loop(
    monkeypatch: pytest.MonkeyPatch, cpus: int, workers: int
) -> None:
    monkeypatch.setattr(measure, "available_cpus", lambda: cpus)
    assert mastering._banding_spatial_workers() == workers


# --- the worker pool really runs (else the "parallel" path is silently serial) --


def test_interpreter_pool_is_available_and_value_identical() -> None:
    pool = mastering._banding_executor(2)
    assert pool is not None, "InterpreterPoolExecutor unavailable: settle would be serial"
    try:
        frame = bytes((x * 7 + y * 3) % 256 for y in range(90) for x in range(160))
        rows, cols = banding_spatial.spatial_banding_lines(frame, 160, 90)
        got = pool.submit(banding_spatial.measure_lines_in_worker, rows, cols, 160, 90).result(30)
        assert got == banding_spatial.spatial_banding_factor(frame, 160, 90)
    finally:
        pool.shutdown(wait=True)


def test_one_worker_means_no_pool() -> None:
    assert mastering._banding_executor(1) is None


# --- exact pruning ----------------------------------------------------------


def _pruning_fake(monkeypatch: pytest.MonkeyPatch, factor: float) -> list[tuple[set[float], int]]:
    """Fake streamed measurer that honours ``needed`` / ``on_factor`` in time
    order like the real one. Records (wanted, frames actually measured)."""
    calls: list[tuple[set[float], int]] = []

    async def fake(
        path: object,
        all_pts: list[float],
        wanted: set[float],
        *,
        needed: Callable[[float], bool] | None = None,
        on_factor: Callable[[float, float], None] | None = None,
        **_: object,
    ) -> dict[float, float]:
        got: dict[float, float] = {}
        for pts in sorted(wanted):
            if needed is None or needed(pts):
                got[pts] = factor
                if on_factor is not None:
                    on_factor(pts, factor)
        calls.append((set(wanted), len(got)))
        return got

    monkeypatch.setattr(measure, "measure_spatial_factors_streamed", fake)
    return calls


async def test_banded_film_prunes_and_stays_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuinely banded (factor saturates): once a group's max is measured at
    its raw value nothing later can beat it, so later frames are skipped."""
    n = 900
    # Raw score RISES with time, so in time order every frame would beat the
    # last until pruning has the true max — worst case for in-order pruning.
    timed: TimedFrames = [(i / 30, _stats(0.6 - i * 0.0005)) for i in range(n)]
    calls = _pruning_fake(monkeypatch, 0.9)
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None, workers=1)
    assert not _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)
    assert len(calls) == 1  # pass 1's top 16 saturate the max; no pass 2
    assert max(mastering.banding_score(m, spatial_factor=factors.get(p)) for p, m in timed) == (
        mastering.banding_score(_stats(0.6 - (n - 1) * 0.0005))
    )


async def test_clean_film_measures_every_frame_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 3600  # 120s at 30 fps: over #254's 3000-frame cap
    timed: TimedFrames = [(i / 30, _stats(0.3 + (i % 50) * 0.001)) for i in range(n)]
    _pruning_fake(monkeypatch, 0.0)
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None, workers=1)
    assert len(factors) == n
    assert not _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)


async def test_slow_pool_start_does_not_eat_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI regression (PR #255): pool start-up ran INSIDE the budget clock, so a
    slow runner's start left pass 1 short. The clock now starts once the pool is
    ready, so pass 1 gets the WHOLE budget however slow the start."""
    import time

    real = mastering._banding_executor

    def slow(n: int):
        time.sleep(0.2)
        return real(n)

    budgets: list[float] = []

    async def fake(path: object, all_pts: list[float], wanted: set[float], **kw: object):
        budget = kw["budget_seconds"]
        assert isinstance(budget, float)
        budgets.append(budget)
        return {p: 0.0 for p in wanted}

    monkeypatch.setattr(mastering, "_banding_executor", slow)
    monkeypatch.setattr(measure, "measure_spatial_factors_streamed", fake)
    monkeypatch.setattr(mastering, "_BANDING_SPATIAL_BUDGET_SECONDS", 0.1)
    timed: TimedFrames = [(float(i), _stats(0.3 + i * 0.001)) for i in range(40)]
    await _settle_banding_spatial_factors("x.mp4", timed, None, workers=2)
    assert budgets and budgets[0] > 0.09  # the 0.2s start was not charged
