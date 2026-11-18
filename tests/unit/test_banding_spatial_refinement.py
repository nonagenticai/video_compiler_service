"""The reported banding max must be taken over CORRECTED frames.

``post_mux_verify`` reports ``max(banding_score(m, spatial_factor=f))`` over
every frame. Correcting only each group's raw-worst frame handed the max to
the next-worst, never-decoded frame at its RAW value (a showcase master
reported 0.2661 vs the 0.15 ceiling with no banding on inspection). The next
fix walked candidates one seek-decode at a time under a 96-decode budget, which
the 2026-10-20 generic masters (694-890 prefilter frames) exhausted every time.
``_settle_banding_spatial_factors`` now settles worst-first in streamed passes:
round 1 = each group's top raw frames, then the frontier (unmeasured frames that
could still be a group's max) in ever wider passes until it is empty (exact) or
a pathological bound is hit (upper bound).

Pure scoring + a fake streamed measurer; no ffmpeg.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping

import pytest

from src.services import mastering, measure
from src.services.mastering import (
    _BANDING_SPATIAL_FIRST_ROUND,
    BANDING_GATE_CEILING,
    BANDING_SPATIAL_PREFILTER,
    _banding_spatial_frontier,
    _settle_banding_spatial_factors,
    banding_score,
)

TimedFrames = list[tuple[float | None, dict[str, float]]]


def _stats(ydif: float) -> dict[str, float]:
    """Full-range, uniform-percentile frame; the raw score falls as ydif rises."""
    return {"YMIN": 0.0, "YMAX": 255.0, "YDIF": ydif, "YLOW": 10.0, "YHIGH": 245.0}


def _reported_max(timed: TimedFrames, factors: dict[float, float]) -> float:
    """Exactly what ``post_mux_verify`` publishes as the whole-file score."""
    return max((banding_score(m, spatial_factor=factors.get(pts)) for pts, m in timed), default=0.0)


def _fake_stream(monkeypatch: pytest.MonkeyPatch, factor_by_pts: Mapping[float, float | None]):
    """Fake ``measure.measure_spatial_factors_streamed``: one call per pass,
    returns the factor for each WANTED pts (``None`` = that frame unmeasured)."""
    calls: list[set[float]] = []

    async def fake(path: object, all_pts: list[float], wanted: set[float], **_: object):
        calls.append(set(wanted))
        return {p: f for p in wanted if (f := factor_by_pts.get(p)) is not None}

    monkeypatch.setattr(measure, "measure_spatial_factors_streamed", fake)
    return calls


async def test_next_worst_uncorrected_frame_no_longer_becomes_the_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported scenario: worst raw frame corrected low, next-worst never
    decoded. Before the fix the published score was the next-worst RAW value."""
    timed: TimedFrames = [(0.0, _stats(0.2)), (1.0, _stats(0.6)), (2.0, _stats(5.0))]
    raw_next_worst = banding_score(_stats(0.6))
    assert raw_next_worst >= BANDING_GATE_CEILING  # it WOULD have been a false positive
    calls = _fake_stream(monkeypatch, {0.0: 0.001, 1.0: 0.001})

    round1 = mastering._banding_spatial_candidates(timed, None)
    assert round1 == [0.0]
    assert _reported_max(timed, {0.0: 0.001}) == raw_next_worst  # the old bug

    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    # ONE pass: both prefilter frames fit round 1 (frame 2 scores 0 raw: not wanted).
    assert calls == [{0.0, 1.0}]
    assert _reported_max(timed, factors) < BANDING_GATE_CEILING


async def test_real_banding_still_reports_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real posterized ramp has spatial factor >= scale: the multiplier
    saturates and the whole-frame score passes through, unchanged."""
    timed: TimedFrames = [(0.0, _stats(0.2)), (1.0, _stats(0.6))]
    _fake_stream(monkeypatch, {0.0: 0.9, 1.0: 0.9})
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert _reported_max(timed, factors) == banding_score(_stats(0.2))
    assert _reported_max(timed, factors) >= BANDING_GATE_CEILING


async def test_long_dwell_of_similar_frames_is_exact_not_budget_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic-master shape: ~900 near-identical prefilter frames, none
    banded. Every frame can still be the max until measured, so the frontier
    walks all of them — worst first, each pass wider — and ends EXACT."""
    n = 900
    timed: TimedFrames = [(i / 30, _stats(0.3 + i * 0.0001)) for i in range(n)]
    calls = _fake_stream(monkeypatch, {i / 30: 0.002 for i in range(n)})
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert len(factors) == n
    assert len(calls) == 2  # round 1 (top 16), then the whole remaining frontier
    assert calls[0] == {i / 30 for i in range(_BANDING_SPATIAL_FIRST_ROUND)}  # worst first
    assert set().union(*calls) == set(factors)
    assert all(not (a & b) for a, b in itertools.combinations(calls, 2))  # no frame twice
    assert _reported_max(timed, factors) == _reported_max(timed, {i / 30: 0.002 for i in range(n)})
    assert _reported_max(timed, factors) < BANDING_GATE_CEILING
    assert not _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)


async def test_real_banding_on_the_worst_frame_settles_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worst-first payoff: when the raw-worst frame is GENUINELY banded its
    corrected score equals its raw one, nothing else can beat it, and the
    other ~900 prefilter frames are never decoded."""
    n = 900
    timed: TimedFrames = [(i / 30, _stats(0.3 + i * 0.0001)) for i in range(n)]
    calls = _fake_stream(monkeypatch, {i / 30: 0.9 for i in range(n)})
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert len(calls) == 1 and len(calls[0]) == _BANDING_SPATIAL_FIRST_ROUND
    assert _reported_max(timed, factors) == banding_score(_stats(0.3))
    assert not _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)


async def test_failed_later_pass_keeps_earlier_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round 2's decode fails ({}): round 1's measured factors stand, the
    rest stays raw (upper bound), and the loop stops instead of retrying."""
    n = 100
    timed: TimedFrames = [(float(i), _stats(0.3 + i * 0.001)) for i in range(n)]
    calls: list[set[float]] = []

    async def fake(path: object, all_pts: list[float], wanted: set[float], **_: object):
        calls.append(set(wanted))
        return {p: 0.0 for p in wanted} if len(calls) == 1 else {}

    monkeypatch.setattr(measure, "measure_spatial_factors_streamed", fake)
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert len(calls) == 2
    assert set(factors) == calls[0]
    assert _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)


async def test_budget_is_shared_across_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each pass gets only what is LEFT of the one wall budget."""
    n = 100
    timed: TimedFrames = [(float(i), _stats(0.3 + i * 0.001)) for i in range(n)]
    budgets: list[float] = []

    async def fake(path: object, all_pts: list[float], wanted: set[float], **kw: object):
        budget = kw["budget_seconds"]
        assert isinstance(budget, float)
        budgets.append(budget)
        if budget < 0.05:  # expires mid-pass: an in-order prefix, like the real one
            return {p: 0.0 for p in sorted(wanted)[: len(wanted) // 2]}
        await asyncio.sleep(0.05)
        return {p: 0.0 for p in wanted}

    monkeypatch.setattr(measure, "measure_spatial_factors_streamed", fake)
    monkeypatch.setattr(mastering, "_BANDING_SPATIAL_BUDGET_SECONDS", 0.08)
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert budgets[0] <= 0.08
    assert len(budgets) == 2 and budgets[1] < budgets[0]  # round 2 got only the remainder
    assert len(factors) < n
    assert _banding_spatial_frontier(timed, None, factors, set(factors), per_group=n)


async def test_candidate_cap_keeps_the_worst_and_leaves_an_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = 3000  # the real cap (18000) would need ydif past 3.0 -> raw 0, never over it
    monkeypatch.setattr(mastering, "_BANDING_SPATIAL_MAX_CANDIDATES", cap)
    n = cap + 10
    # Raw score FALLS with i, so the cap must keep i < MAX (the worst frames).
    timed: TimedFrames = [(float(i), _stats(0.3 + i * 0.0002)) for i in range(n)]
    calls = _fake_stream(monkeypatch, {float(i): 0.0 for i in range(n)})
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert set().union(*calls) == {float(i) for i in range(cap)}
    reported = _reported_max(timed, factors)
    assert reported >= _reported_max(timed, {float(i): 0.0 for i in range(n)})  # >= truth
    assert reported > 0.0  # unmeasured frames stay raw: conservative


async def test_failed_pass_leaves_every_frame_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    timed: TimedFrames = [(0.0, _stats(0.2)), (1.0, _stats(0.6))]
    _fake_stream(monkeypatch, {})  # the streamed measurer returned {} (decode error)
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert factors == {}
    assert _reported_max(timed, factors) == banding_score(_stats(0.2))


async def test_partially_measured_pass_stays_an_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget expiry: a measured prefix corrected, the rest raw."""
    timed: TimedFrames = [(0.0, _stats(0.6)), (1.0, _stats(0.2))]
    _fake_stream(monkeypatch, {0.0: 0.0})  # frame 1 (the raw worst) never measured
    factors = await _settle_banding_spatial_factors("x.mp4", timed, None)
    assert _reported_max(timed, factors) == banding_score(_stats(0.2))
    assert _banding_spatial_frontier(timed, None, factors, set(factors), per_group=1)


async def test_nothing_over_prefilter_decodes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    low = {"YMIN": 100.0, "YMAX": 110.0, "YDIF": 5.0}
    calls = _fake_stream(monkeypatch, {})
    assert await _settle_banding_spatial_factors("x.mp4", [(0.0, low)], None) == {}
    assert calls == []


async def test_per_shot_max_is_also_settled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same defect per shot: each shot's published max must be corrected."""
    timed: TimedFrames = [
        (0.0, _stats(0.2)),
        (0.5, _stats(0.6)),
        (1.0, _stats(0.3)),
        (1.5, _stats(0.7)),
    ]
    shots: list[dict[str, object]] = [
        {"line_id": "a", "start": 0.0, "end": 1.0},
        {"line_id": "b", "start": 1.0, "end": 2.0},
    ]
    _fake_stream(monkeypatch, {0.0: 0.0, 0.5: 0.0, 1.0: 0.0, 1.5: 0.0})
    factors = await _settle_banding_spatial_factors("x.mp4", timed, shots)
    per_shot = mastering.bucket_banding(timed, shots, spatial_factors=factors)
    assert [s["banding"] for s in per_shot] == [0.0, 0.0]
    assert not any(s["banding_residual_over_trigger"] for s in per_shot)


def test_frontier_skips_frames_under_prefilter_and_attempted() -> None:
    low = {"YMIN": 0.0, "YMAX": 255.0, "YDIF": 2.95, "YLOW": 10.0, "YHIGH": 245.0}
    assert banding_score(low) < BANDING_SPATIAL_PREFILTER
    timed: TimedFrames = [(0.0, _stats(0.2)), (1.0, low), (None, _stats(0.1))]
    assert _banding_spatial_frontier(timed, None, {}, set(), per_group=4) == [0.0]
    assert _banding_spatial_frontier(timed, None, {0.0: 0.0}, {0.0}, per_group=4) == []


def test_plan_intervals_attributes_frames_by_order() -> None:
    """The streamed pass attributes the i-th emitted frame to ``selected[i]``;
    the plan must list EXACTLY the frames its intervals admit, merged or not."""
    pts = [i / 10 for i in range(20)]
    wanted = {0.0, 0.1, 0.2, 0.5, 1.0, 1.1, 1.9}
    intervals, selected = measure.plan_spatial_intervals(pts, wanted)
    assert intervals == [(0.0, 0.2), (0.5, 0.5), (1.0, 1.1), (1.9, 1.9)]
    assert selected == [0.0, 0.1, 0.2, 0.5, 1.0, 1.1, 1.9]
    # Capped at 2 intervals: the closest runs merge and the frames between them
    # join `selected` in time order (measured, never mis-attributed).
    intervals, selected = measure.plan_spatial_intervals(pts, wanted, max_intervals=2)
    assert len(intervals) == 2
    assert set(wanted) <= set(selected)
    assert selected == sorted(selected)
    assert all(any(a <= p <= b for a, b in intervals) for p in selected)
    inside = [p for p in pts if any(a <= p <= b for a, b in intervals)]
    assert selected == inside
    assert measure.plan_spatial_intervals(pts, set()) == ([], [])


def test_chunked_plan_bounds_select_terms_and_preserves_order() -> None:
    """ffmpeg 7.1 cannot init a select over 100 ``between()`` terms (exit 244,
    'Cannot allocate memory'; live 2026-10-21 on a 101-term pass 2). The plan is
    split into invocations of <= ``_SPATIAL_MAX_TERMS_PER_PASS`` terms whose
    frames, concatenated, are exactly ``selected``."""
    pts = [i / 30 for i in range(1400)]
    wanted = {p for i, p in enumerate(pts) if i % 5 != 0 and i % 13 != 0}
    intervals, selected = measure.plan_spatial_intervals(pts, wanted)
    assert len(intervals) == measure._SPATIAL_MAX_INTERVALS > measure._SPATIAL_MAX_TERMS_PER_PASS
    chunks = measure.chunk_spatial_plan(intervals, selected)
    assert len(chunks) > 1
    assert all(1 <= len(c) <= measure._SPATIAL_MAX_TERMS_PER_PASS for c, _ in chunks)
    assert [iv for c, _ in chunks for iv in c] == intervals
    assert sum(n for _, n in chunks) == len(selected)
    emitted = [p for c, _ in chunks for p in pts if any(a <= p <= b for a, b in c)]
    assert emitted == selected
    for c, n in chunks:
        assert n == sum(1 for p in pts if any(a <= p <= b for a, b in c))
        cmd = measure._spatial_stream_cmd("x.mp4", c, 1920, decode_until=c[-1][1] + 1.0)
        assert cmd[cmd.index("-vf") + 1].count("between(") == len(c)
        assert cmd.index("-to") < cmd.index("-i")  # input-side early stop only


def test_stream_cmd_refuses_an_oversized_select() -> None:
    too_many = [(i * 1.0, i * 1.0 + 0.5) for i in range(measure._SPATIAL_MAX_TERMS_PER_PASS + 1)]
    with pytest.raises(ValueError, match="select terms"):
        measure._spatial_stream_cmd("x.mp4", too_many, 1920)
