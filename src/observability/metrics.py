"""Prometheus metrics for the compile pipeline.

WHY THIS EXISTS. Until 2026-08-26 this service exported process metrics only —
``src/asgi.py`` mounts ``make_asgi_app()`` over the default registry and NOT ONE
``Counter``/``Histogram``/``Gauge`` was defined anywhere in the codebase. The
practical consequence, hit while trying to size hardware for a coming increase
in video production:

  * how long a job takes           -> unanswerable (``compile_jobs`` is pruned by
                                      migration 009, so the table never holds
                                      more than a recent slice)
  * what FRACTION of a job is encode -> unanswerable, and it is the number that
                                      decides whether more CPU/GPU buys anything
  * throughput in video-seconds/hour -> unanswerable, nothing recorded output
                                      length
  * whether a backlog is forming    -> unanswerable

That gap made a real decision undecidable: a 2186s job cannot be 36 minutes of
encoding, because encode measures ~3x realtime — so encode is a MINORITY of job
time and the rest (asset fetch, mastering, upload) was invisible. Adding cores
would have optimised an unknown fraction.

CARDINALITY. Every label here is a bounded set: ``stage`` is the fixed vocabulary
in ``Stage`` below, ``kind`` is compile/concat_audio, ``status`` is the terminal
states, ``result`` is applied/skipped/failed. Deliberately NO tenant_id, job_id,
or run_id label — those are unbounded and would blow up the TSDB.

COST. A ``time.perf_counter()`` pair and one histogram observation per stage.
Nothing here awaits, allocates per-frame, or touches the render path's data.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Final

from prometheus_client import Counter, Gauge, Histogram

# Buckets span the MEASURED range rather than a generic ladder. Real compiles on
# 2026-08-26 ran 75s, 134s, 177s, 260s, 602s, 1575s and 2186s; the small
# per-stage probes (ffprobe, hash) land well under a second. The top bucket is
# 3600s so a pathological job still lands somewhere finite.
_STAGE_BUCKETS: Final = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1200.0,
    1800.0,
    3600.0,
)


class Stage:
    """The fixed ``stage`` label vocabulary.

    Keep this closed. A free-form stage name would make the label unbounded,
    which is the one way a histogram like this becomes a liability.
    """

    RENDER_SHOTS: Final = "render_shots"  # per-shot ffmpeg encode (the CPU/GPU stage)
    FETCH_BEDS: Final = "fetch_beds"  # music/ambience asset download
    DEBAND: Final = "deband"
    SOURCE_MATCH: Final = "source_match"
    JOIN: Final = "join"  # concat of the rendered segments
    FINALIZE_AUDIO: Final = "finalize_audio"
    COMPILE_SHOTS: Final = "compile_shots"  # the whole renderer call
    PROBE: Final = "probe"  # ffprobe dimensions + duration
    HASH: Final = "hash"  # output sha256
    LOUDNESS: Final = "loudness"
    MASTER_VERIFY: Final = "master_verify"
    UPLOAD: Final = "upload"  # MinIO PUT + presign


compile_stage_seconds = Histogram(
    "video_compile_stage_seconds",
    "Wall-clock seconds spent in one stage of a compile job.",
    ["stage"],
    buckets=_STAGE_BUCKETS,
)

compile_job_seconds = Histogram(
    "video_compile_job_seconds",
    "End-to-end wall-clock seconds for one job, from dispatch to terminal state.",
    ["kind"],
    buckets=_STAGE_BUCKETS,
)

compile_output_seconds_total = Counter(
    "video_compile_output_seconds_total",
    (
        "Cumulative seconds of finished video produced. "
        "Divide by wall-clock to get throughput as a multiple of realtime: "
        "rate(video_compile_output_seconds_total[1h])."
    ),
    ["kind"],
)

compile_jobs_total = Counter(
    "video_compile_jobs_total",
    "Compile jobs by terminal status.",
    ["kind", "status"],
)

compile_jobs_inflight = Gauge(
    "video_compile_jobs_inflight",
    "Jobs currently executing in THIS process. Sum across pods for pool concurrency.",
)

compile_shots_total = Counter(
    "video_compile_shots_total",
    "Shots rendered. Pair with render_shots stage seconds for per-shot cost.",
)

compile_deband_segments_total = Counter(
    "video_compile_deband_segments_total",
    (
        "Segments seen by the conditional-deband pass, by outcome. "
        "'applied' = the banding gate tripped and the segment was re-encoded; "
        "'skipped' = measured clean, left as a byte-identical single-pass encode; "
        "'failed' = the measure pass raised (best-effort, the compile continues). "
        "Answers 'does the deband re-encode ever actually fire?', which the logs "
        "could not: the skip path logged nothing at all."
    ),
    ["result"],
)


@contextmanager
def stage(name: str) -> Generator[None]:
    """Time a pipeline stage into ``video_compile_stage_seconds{stage=name}``.

    Synchronous on purpose: the body may contain ``await`` (a sync context
    manager does not interfere with the event loop), so this wraps async stages
    without needing an async CM or an extra task.

    Records on BOTH the success and the exception path — a stage that blew up
    still consumed wall-clock, and omitting failures would bias every quantile
    toward the happy path.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        compile_stage_seconds.labels(stage=name).observe(time.perf_counter() - started)


__all__ = [
    "Stage",
    "compile_deband_segments_total",
    "compile_job_seconds",
    "compile_jobs_inflight",
    "compile_jobs_total",
    "compile_output_seconds_total",
    "compile_shots_total",
    "compile_stage_seconds",
    "stage",
]
