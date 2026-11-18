"""The deband / source-match MEASURE passes must be concurrent, ordered and isolated.

Both passes run AFTER ``render_shots`` finishes, so the global ffmpeg semaphore
(``max_concurrent_ffmpeg_subprocesses``, 2 by default and 3 in production) sat
completely idle while these loops crawled one segment at a time — the deband pass
alone measured 56.4s of a 164.9s job. Making them ``asyncio.gather`` is only CORRECT if three properties hold,
and each has its own test below:

  1. the measures actually overlap (otherwise the change bought nothing);
  2. results map back to the RIGHT segment (gather preserves input order — a
     mismatch would deband the wrong shot, silently);
  3. one segment raising does NOT fail the job. Deband/source-match are documented
     best-effort polish that "never fails a compile"; a bare gather propagates the
     first exception and would turn picture polish into a job-killer. This is the
     property that matters most.

Plus the deband outcome counter, which exists because the SKIP path logged nothing
at all, making "does the deband re-encode ever actually fire?" unanswerable.

No ffmpeg: ``measure`` and the encode helpers are stubbed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from src.api.schemas import FinishProfile, GradeSpec
from src.config import Settings, get_settings
from src.services import measure, renderer

# A frame with the banding signature (wide luma span, low temporal diff) and its
# textured counterpart — same shapes the trigger/gate tests use.
_BANDED = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 0.5}
_CLEAN = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 10.0}


def _segs(n: int = 4) -> list[Path]:
    return [Path(f"/tmp/seg-{i:04d}.mp4") for i in range(n)]


def _deband_settings() -> Settings:
    return get_settings().model_copy(update={"render_deband": "on"})


def _match_settings() -> Settings:
    return get_settings().model_copy(update={"render_source_match": "on"})


def _finish() -> FinishProfile:
    return FinishProfile(grade=GradeSpec(enabled=True, match_sources=True))


def _deband_count(result: str) -> float:
    value = REGISTRY.get_sample_value("video_compile_deband_segments_total", {"result": result})
    return float(value or 0.0)


class _ConcurrencyProbe:
    """Records the peak number of measures in flight simultaneously."""

    def __init__(self, payload: object) -> None:
        self.inflight = 0
        self.peak = 0
        self._payload = payload

    async def __call__(self, seg: Path) -> object:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        # Yield twice so a SERIAL caller cannot possibly overlap by accident and a
        # concurrent one certainly does.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.inflight -= 1
        return self._payload


# ---------------------------------------------------------------------------
# 1. concurrency
# ---------------------------------------------------------------------------


async def test_deband_measure_pass_runs_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _ConcurrencyProbe([_CLEAN])
    monkeypatch.setattr(measure, "measure_shot_frames", probe)
    monkeypatch.setattr(renderer, "_deband_segment", _unreachable_deband)

    await renderer._apply_conditional_deband(_segs(4), _deband_settings())

    assert probe.peak == 4, f"measures must overlap; peak in flight was {probe.peak}"


async def test_source_match_measure_pass_runs_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _ConcurrencyProbe({"YAVG": 140.0})
    monkeypatch.setattr(measure, "measure_shot_stats", probe)
    monkeypatch.setattr(renderer, "_regrade_segment", _unreachable_regrade)

    await renderer._apply_source_match(_segs(4), _match_settings(), _finish())

    assert probe.peak == 4, f"measures must overlap; peak in flight was {probe.peak}"


async def _unreachable_deband(*args: object, **kwargs: object) -> None:
    raise AssertionError("clean segments must not be debanded")


async def _unreachable_regrade(*args: object, **kwargs: object) -> None:
    raise AssertionError("in-tolerance segments must not be regraded")


# ---------------------------------------------------------------------------
# 2. result order maps to the right segment
# ---------------------------------------------------------------------------


async def test_deband_results_map_back_to_their_own_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only seg-0002 is banded — the completion ORDER is deliberately reversed."""
    segs = _segs(4)

    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        # Later segments finish FIRST: a result-ordering bug shows up here.
        await asyncio.sleep(0.005 * (len(segs) - segs.index(seg)))
        return [_BANDED] if seg == segs[2] else [_CLEAN]

    debanded: list[Path] = []

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        debanded.append(seg)

    monkeypatch.setattr(measure, "measure_shot_frames", fake_measure)
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)

    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert debanded == [segs[2]]


async def test_source_match_stats_map_back_to_their_own_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """seg-0003 is the dark outlier; the other three set the median."""
    segs = _segs(4)
    luma = {segs[0]: 140.0, segs[1]: 141.0, segs[2]: 140.5, segs[3]: 60.0}

    async def fake_measure(seg: Path) -> dict[str, float]:
        await asyncio.sleep(0.005 * (len(segs) - segs.index(seg)))
        return {"YAVG": luma[seg]}

    regraded: list[Path] = []

    async def fake_regrade(
        seg: Path, filter_str: str, settings: Settings, deterministic: bool = False
    ) -> None:
        regraded.append(seg)

    monkeypatch.setattr(measure, "measure_shot_stats", fake_measure)
    monkeypatch.setattr(renderer, "_regrade_segment", fake_regrade)

    await renderer._apply_source_match(segs, _match_settings(), _finish())

    assert regraded == [segs[3]]


# ---------------------------------------------------------------------------
# 3. per-segment failure isolation — the one that must never regress
# ---------------------------------------------------------------------------


async def test_one_failing_measure_does_not_fail_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``gather`` would propagate seg-0001's error and kill the compile."""
    segs = _segs(4)

    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        if seg == segs[1]:
            raise RuntimeError("ffprobe blew up")
        return [_BANDED]

    debanded: list[Path] = []

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        debanded.append(seg)

    monkeypatch.setattr(measure, "measure_shot_frames", fake_measure)
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)

    # Must NOT raise.
    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert debanded == [segs[0], segs[2], segs[3]], "the surviving segments must still process"


async def test_one_failing_source_match_measure_does_not_fail_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = _segs(4)
    luma = {segs[0]: 140.0, segs[1]: 141.0, segs[2]: 140.5, segs[3]: 60.0}

    async def fake_measure(seg: Path) -> dict[str, float]:
        if seg == segs[1]:
            raise RuntimeError("ffprobe blew up")
        return {"YAVG": luma[seg]}

    regraded: list[Path] = []

    async def fake_regrade(
        seg: Path, filter_str: str, settings: Settings, deterministic: bool = False
    ) -> None:
        regraded.append(seg)

    monkeypatch.setattr(measure, "measure_shot_stats", fake_measure)
    monkeypatch.setattr(renderer, "_regrade_segment", fake_regrade)

    await renderer._apply_source_match(segs, _match_settings(), _finish())

    assert regraded == [segs[3]]


async def test_a_cancellation_is_not_swallowed_as_a_measure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort must not mean un-cancellable."""

    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        raise asyncio.CancelledError

    monkeypatch.setattr(measure, "measure_shot_frames", fake_measure)
    monkeypatch.setattr(renderer, "_deband_segment", _unreachable_deband)

    with pytest.raises(asyncio.CancelledError):
        await renderer._apply_conditional_deband(_segs(2), _deband_settings())


# ---------------------------------------------------------------------------
# 4. the deband outcome counter (fix 4)
# ---------------------------------------------------------------------------


async def test_counter_increments_applied_skipped_and_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = _segs(3)

    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        if seg == segs[0]:
            return [_BANDED]
        if seg == segs[1]:
            return [_CLEAN]
        raise RuntimeError("ffprobe blew up")

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        return None

    monkeypatch.setattr(measure, "measure_shot_frames", fake_measure)
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)

    before = {r: _deband_count(r) for r in ("applied", "skipped", "failed")}
    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert _deband_count("applied") == before["applied"] + 1
    assert _deband_count("skipped") == before["skipped"] + 1
    assert _deband_count("failed") == before["failed"] + 1


def test_the_deband_counter_carries_no_unbounded_label() -> None:
    from src.observability import metrics

    forbidden = {"tenant_id", "job_id", "run_id", "object_key", "url", "instance_id"}
    assert not forbidden & set(metrics.compile_deband_segments_total._labelnames)
