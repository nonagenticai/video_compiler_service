"""Unit tests for the compile-pipeline Prometheus metrics (pure, no DB, no ffmpeg).

These guard the two properties that make the metrics trustworthy rather than
decorative:

  1. a stage that RAISES is still timed — otherwise every quantile is biased
     toward the happy path and a slow-then-failing stage reads as fast;
  2. the ``stage`` label vocabulary stays CLOSED — an unbounded label is how a
     histogram turns into a TSDB incident.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from src.observability import metrics

# ---------------------------------------------------------------------------
# stage() timing
# ---------------------------------------------------------------------------


def _stage_count(stage: str) -> float:
    """Observation count for one stage, or 0.0 before the series exists."""
    value = REGISTRY.get_sample_value("video_compile_stage_seconds_count", {"stage": stage})
    return float(value or 0.0)


def test_stage_records_an_observation_on_success() -> None:
    before = _stage_count(metrics.Stage.PROBE)
    with metrics.stage(metrics.Stage.PROBE):
        pass
    assert _stage_count(metrics.Stage.PROBE) == before + 1


def test_stage_records_an_observation_when_the_body_raises() -> None:
    """A failing stage still consumed wall-clock and MUST be counted.

    This is the regression that matters: if the histogram only observed on the
    success path, a stage that reliably burns 60s and then throws would be
    invisible, and the p99 of every OTHER stage would look better than reality.
    """
    before = _stage_count(metrics.Stage.UPLOAD)
    with pytest.raises(RuntimeError, match="boom"), metrics.stage(metrics.Stage.UPLOAD):
        raise RuntimeError("boom")
    assert _stage_count(metrics.Stage.UPLOAD) == before + 1


def test_stage_does_not_swallow_the_exception() -> None:
    """The context manager is an observer; it must never change control flow."""
    with pytest.raises(ValueError, match="propagated"), metrics.stage(metrics.Stage.JOIN):
        raise ValueError("propagated")


def test_stage_observes_a_nonnegative_duration() -> None:
    stage_name = metrics.Stage.HASH
    before = REGISTRY.get_sample_value("video_compile_stage_seconds_sum", {"stage": stage_name})
    with metrics.stage(stage_name):
        pass
    after = REGISTRY.get_sample_value("video_compile_stage_seconds_sum", {"stage": stage_name})
    assert after is not None
    assert after >= float(before or 0.0)


# ---------------------------------------------------------------------------
# cardinality — the property that keeps this safe to ship
# ---------------------------------------------------------------------------


def test_stage_vocabulary_is_closed_and_stringly_typed() -> None:
    names = [
        getattr(metrics.Stage, attr) for attr in dir(metrics.Stage) if not attr.startswith("_")
    ]
    assert names, "Stage must declare at least one stage"
    assert all(isinstance(n, str) for n in names)
    assert len(set(names)) == len(names), "duplicate stage constants"
    # A closed vocabulary is the whole point; if this grows past a couple of
    # dozen, the label is no longer bounded in any useful sense.
    assert len(names) < 32


def test_no_metric_carries_an_unbounded_label() -> None:
    """tenant_id / job_id / run_id would each be unbounded. None may appear."""
    forbidden = {"tenant_id", "job_id", "run_id", "object_key", "url", "instance_id"}
    for metric in (
        metrics.compile_stage_seconds,
        metrics.compile_job_seconds,
        metrics.compile_output_seconds_total,
        metrics.compile_jobs_total,
        metrics.compile_deband_segments_total,
    ):
        assert not forbidden & set(metric._labelnames), f"{metric._name} carries an unbounded label"


def test_buckets_span_the_measured_job_range() -> None:
    """Real compiles ran 75s..2186s on 2026-08-26 — the ladder must cover that.

    A histogram whose top finite bucket sits below the real p99 reports every
    long job as +Inf, which makes quantiles useless exactly where they matter.
    """
    buckets = metrics.compile_job_seconds._upper_bounds
    finite = [b for b in buckets if b != float("inf")]
    assert min(finite) <= 1.0, "needs sub-second resolution for cheap stages"
    assert max(finite) >= 2186.0, "top bucket must exceed the longest observed job"


# ---------------------------------------------------------------------------
# counters/gauges exist with the labels the dashboards query
# ---------------------------------------------------------------------------


def test_job_counters_accept_the_documented_labels() -> None:
    metrics.compile_jobs_total.labels(kind="compile", status="completed").inc(0)
    metrics.compile_jobs_total.labels(kind="compile", status="failed").inc(0)
    metrics.compile_jobs_total.labels(kind="compile", status="interrupted").inc(0)
    metrics.compile_job_seconds.labels(kind="concat_audio").observe(0.0)
    metrics.compile_output_seconds_total.labels(kind="compile").inc(0)
    for status in ("completed", "failed", "interrupted"):
        labels = {"kind": "compile", "status": status}
        assert REGISTRY.get_sample_value("video_compile_jobs_total", labels) is not None
    assert (
        REGISTRY.get_sample_value("video_compile_job_seconds_count", {"kind": "concat_audio"})
        is not None
    )
    assert (
        REGISTRY.get_sample_value("video_compile_output_seconds_total", {"kind": "compile"})
        is not None
    )


def test_inflight_gauge_round_trips() -> None:
    before = float(REGISTRY.get_sample_value("video_compile_jobs_inflight") or 0.0)
    metrics.compile_jobs_inflight.inc()
    assert REGISTRY.get_sample_value("video_compile_jobs_inflight") == before + 1
    metrics.compile_jobs_inflight.dec()
    assert REGISTRY.get_sample_value("video_compile_jobs_inflight") == before
