"""Read-only per-segment picture measurement (observability + deband trigger).

MEASURE-ONLY: nothing here changes the pixels the compiler emits. It runs a
discarded (``-f null -``) ``signalstats`` pass over a rendered segment and
reports the averaged luma/chroma statistics the frame carried — the signal the
encode/delivery master uses to decide whether a shot needs deband and to
classify its content for the CRF ladder.

Two pieces, mirroring ``loudness.py``:

* ``parse_signalstats`` — a PURE, unit-testable parser. It turns an ffmpeg
  ``signalstats,metadata=print`` stderr dump into a small averaged-metrics dict
  and NEVER raises: garbage/empty in → ``{}`` out.
* ``parse_signalstats_per_frame`` — the PURE per-frame sibling: same dump →
  one metrics dict per frame, so a localized band in a few frames stays visible
  instead of being averaged away. NEVER raises: garbage/empty in → ``[]`` out.
* ``measure_shot_stats`` / ``measure_shot_frames`` — run the read-only measure
  pass through the renderer's ffmpeg concurrency gate (``renderer._run`` — never
  a bare ``create_subprocess_exec``, the async-hardening invariant). Best-effort:
  return ``{}`` / ``[]`` on ANY failure and never raise, so a measurement hiccup
  can never fail a compile job.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import math
import os
import re
from collections.abc import Callable
from concurrent.futures import Executor
from pathlib import Path

import structlog

from src.services import renderer

log = structlog.get_logger(__name__)

# ``metadata=print`` emits one ``lavfi.signalstats.<KEY>=<value>`` line per
# frame per metric. We average each KEY across all frames of the segment.
_SIGNALSTATS_RE = re.compile(r"lavfi\.signalstats\.([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)")

# ``metadata=print`` prefixes each frame's metrics with a ``frame:N pts:… …``
# header line — the boundary we split on for per-frame parsing. ffmpeg prefixes
# that header with a ``[Parsed_metadata_N @ 0x…] `` filter tag, so the marker is
# NOT at the line start — a ``^``-anchored match found zero frames and silently
# collapsed every per-frame pass to ONE whole-file aggregate (the per-frame
# banding gate was quietly averaging). Match ``frame:N`` anywhere on the line.
_FRAME_MARKER_RE = re.compile(r"\bframe:\d+", re.MULTILINE)

# The ``frame:N`` split consumes only ``frame:\d+``; the rest of that header line
# (``pts:… pts_time:…``) survives at the head of each per-frame block, so a timed
# parser can recover each frame's presentation timestamp for per-shot bucketing.
_PTS_TIME_RE = re.compile(r"pts_time:(-?[0-9.]+)")


def parse_signalstats(stderr: str) -> dict[str, float]:
    """Pure parser: a ``signalstats,metadata=print`` stderr dump → averaged
    metrics keyed by the signalstats field (``YMIN``, ``YMAX``, ``YAVG``,
    ``YDIF``, ``SATAVG`` …). NEVER raises — no matches → ``{}``."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, raw in _SIGNALSTATS_RE.findall(stderr or ""):
        try:
            val = float(raw)
        except ValueError:
            continue
        sums[key] = sums.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums if counts[key] > 0}


def _block_metrics(block: str) -> dict[str, float]:
    """Parse the ``lavfi.signalstats.<KEY>=<value>`` lines of ONE frame block →
    a metrics dict. Non-numeric values are skipped; NEVER raises."""
    metrics: dict[str, float] = {}
    for key, raw in _SIGNALSTATS_RE.findall(block):
        try:
            metrics[key] = float(raw)
        except ValueError:
            continue
    return metrics


def parse_signalstats_per_frame(stderr: str) -> list[dict[str, float]]:
    """Pure parser: split a ``signalstats,metadata=print`` dump into PER-FRAME
    metric dicts (one dict per ``frame:N`` block), keeping a localized band that
    ``parse_signalstats`` would average away visible. NEVER raises — no metrics →
    ``[]``. A dump with no ``frame:`` markers but present metrics (e.g. a
    single-frame segment) degrades to one aggregate frame so nothing is lost."""
    text = stderr or ""
    parts = _FRAME_MARKER_RE.split(text)
    # parts[0] is the preamble before the first frame marker; per-frame blocks
    # are parts[1:].
    frames = [m for block in parts[1:] if (m := _block_metrics(block))]
    if not frames:
        aggregate = _block_metrics(text)
        if aggregate:
            frames.append(aggregate)
    return frames


def _block_pts_time(block: str) -> float | None:
    """Parse the ``pts_time:`` of ONE frame block's surviving header → float, or
    ``None`` when the block carried no ``pts_time:`` marker / an unparseable value.
    NEVER raises. Sibling of ``_block_metrics`` — does not touch the metrics."""
    m = _PTS_TIME_RE.search(block)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_signalstats_per_frame_timed(
    stderr: str,
) -> list[tuple[float | None, dict[str, float]]]:
    """Timed sibling of ``parse_signalstats_per_frame``: the SAME per-frame split,
    but each element is ``(pts_time, metrics)`` so a per-shot bucketer can place
    each frame's banding on the film timeline. ``pts_time`` is ``None`` when a
    block carried no ``pts_time:`` marker. NEVER raises — no metrics → ``[]``. Same
    degrade as its sibling: a dump with no ``frame:`` markers but present metrics
    collapses to one aggregate frame with ``pts_time`` ``None`` so nothing is lost."""
    text = stderr or ""
    parts = _FRAME_MARKER_RE.split(text)
    frames = [(_block_pts_time(block), m) for block in parts[1:] if (m := _block_metrics(block))]
    if not frames:
        aggregate = _block_metrics(text)
        if aggregate:
            frames.append((None, aggregate))
    return frames


def _signalstats_cmd(path: Path | str) -> list[str]:
    """The read-only discarded ``signalstats,metadata=print`` ffmpeg command over
    ``path``. Shared by the averaged (``measure_shot_stats``) and per-frame
    (``measure_shot_frames``) measure passes so they stay byte-for-byte the same
    discarded pass."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-vf",
        "signalstats,metadata=print",
        "-an",
        "-f",
        "null",
        # /dev/null (not "-") so the discarded mux never collides with a stray
        # cwd file; the null muxer ignores the target either way.
        "/dev/null",
    ]


async def measure_shot_stats(path: Path | str) -> dict[str, float]:
    """Measure a rendered segment's averaged signalstats (read-only, best-effort).

    Runs a single discarded ``signalstats,metadata=print`` pass over ``path``
    through the ffmpeg concurrency gate and parses the stderr. The ``-f null -``
    output is thrown away, so this NEVER alters the segment. Returns ``{}`` on
    any failure — it must never fail a job (mirrors ``probe_dimensions``)."""
    try:
        stderr = await renderer._run(_signalstats_cmd(path), capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("measure_shot_stats.measure_failed", path=str(path), error=str(exc))
        return {}
    try:
        return parse_signalstats(stderr)
    except Exception as exc:  # noqa: BLE001 — parser is defensive, but never fail a job
        log.warning("measure_shot_stats.parse_failed", path=str(path), error=str(exc))
        return {}


async def measure_shot_frames(path: Path | str) -> list[dict[str, float]]:
    """Measure a rendered segment's PER-FRAME signalstats (read-only, best-effort).

    Same discarded ``signalstats,metadata=print`` pass as ``measure_shot_stats``,
    but returns EACH frame's metrics instead of the segment average — so a
    localized band that lives in a few frames is not diluted into a clean mean
    before the banding gate sees it. Returns ``[]`` on any failure; it must never
    fail a job (mirrors ``measure_shot_stats``)."""
    try:
        stderr = await renderer._run(_signalstats_cmd(path), capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("measure_shot_frames.measure_failed", path=str(path), error=str(exc))
        return []
    try:
        return parse_signalstats_per_frame(stderr)
    except Exception as exc:  # noqa: BLE001 — parser is defensive, but never fail a job
        log.warning("measure_shot_frames.parse_failed", path=str(path), error=str(exc))
        return []


async def measure_shot_frames_timed(
    path: Path | str,
) -> list[tuple[float | None, dict[str, float]]]:
    """Measure a rendered segment's PER-FRAME signalstats WITH each frame's
    ``pts_time`` (read-only, best-effort). The SAME discarded
    ``signalstats,metadata=print`` pass as ``measure_shot_frames`` — a second
    throwaway ``-f null`` read — but parsed via ``parse_signalstats_per_frame_timed``
    so a per-shot bucketer can place each frame's banding on the film timeline.
    Returns ``[]`` on any failure; it must never fail a job (mirrors
    ``measure_shot_frames``)."""
    try:
        stderr = await renderer._run(_signalstats_cmd(path), capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("measure_shot_frames_timed.measure_failed", path=str(path), error=str(exc))
        return []
    try:
        return parse_signalstats_per_frame_timed(stderr)
    except Exception as exc:  # noqa: BLE001 — parser is defensive, but never fail a job
        log.warning("measure_shot_frames_timed.parse_failed", path=str(path), error=str(exc))
        return []


# The banding SPATIAL term (``mastering.spatial_banding_factor``) is measured
# WITHOUT averaging: banding is a few-pixel-wide quantisation step, and the old
# 320px AREA downscale averaged exactly that step away (recalibrated
# 2026-10-21). Frames wider than 1920 (prod masters are 2560x1440) are capped
# with NEAREST-neighbour, which drops columns/rows but keeps every surviving
# pixel's value — so step HEIGHTS are untouched; plateaus narrow by the scale
# (the >= 6px plateau rule now means >= 8px at 2560; visible gradient bands
# are tens of px wide). Cost is linear in
# width + height (sampled lines), so 2560 -> 1920 is ~25% of the detector's
# time. Calibration regenerated at 2560x1440 through this exact filter:
# negatives <= 0.0066, positives >= 0.881 after libx264 crf18 (2026-10-21).
_SPATIAL_FRAME_MAX_WIDTH = 1920
# A select expression is one ``between(t,a,b)`` term per interval; past this
# many, the closest intervals are merged (the frames between them are decoded
# too and simply measured — never mis-attributed, see ``plan_spatial_intervals``).
_SPATIAL_MAX_INTERVALS = 128
# ⛔ HARD ffmpeg LIMIT, not a tuning knob: ffmpeg 7.1's expression parser
# rejects a ``select`` sum of more than 100 ``between()`` terms (expression-tree
# depth), and the filter graph then fails to init with a MISLEADING
# ``Cannot allocate memory`` (exit 244) — measured 2026-10-21 in the prod image
# (ffmpeg 7.1.5): 100 terms init, 101 fail, independent of resolution and
# memory (vm.overcommit=1; nothing was short of memory). Live it failed a
# generic master's pass 2 (101 intervals) and left the banding score at the RAW
# upper bound. So ``_SPATIAL_MAX_INTERVALS`` intervals are split into ffmpeg
# invocations of at most this many terms each (``chunk_spatial_plan``), well
# under the limit. Each invocation decodes only up to its last interval
# (input ``-to``), and a 2560x1440 30s decode is ~1.5s, so the extra passes are
# cheap; ``_spatial_stream_cmd`` refuses to build an expression over the bound.
_SPATIAL_MAX_TERMS_PER_PASS = 32
# Input ``-to`` margin past a chunk's last interval: ``-to`` only stops the
# decode early, ``select`` still picks the frames, so a generous margin costs a
# few decoded frames and can never drop a wanted one.
_SPATIAL_DECODE_END_MARGIN = 1.0
# Half-width (seconds) of each interval's guard band around the signalstats
# ``pts_time`` values, which ffmpeg prints rounded to 6 decimals. Far below any
# real frame spacing (1ms = 1000 fps), so it never admits a neighbouring frame.
_SPATIAL_PTS_EPS = 0.0005


def plan_spatial_intervals(
    all_pts: list[float], wanted: set[float], max_intervals: int = _SPATIAL_MAX_INTERVALS
) -> tuple[list[tuple[float, float]], list[float]]:
    """PURE. Cover ``wanted`` frames with at most ``max_intervals`` closed time
    intervals over the film's frame timeline ``all_pts`` (the signalstats
    pass's per-frame ``pts_time`` values). Returns ``(intervals, selected)``:
    ``selected`` is EVERY frame of ``all_pts`` inside an interval, in time order
    — exactly the frames a ``select`` over those intervals emits, so the i-th
    emitted frame IS ``selected[i]``. Merging two intervals pulls the frames
    between them into ``selected`` (they get measured too); it never shifts the
    attribution."""
    timeline = sorted(set(all_pts))
    if not timeline or not wanted:
        return [], []
    # Runs of consecutive timeline indices that are all wanted.
    runs: list[list[int]] = []
    for i, pts in enumerate(timeline):
        if pts not in wanted:
            continue
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    # Merge the runs separated by the fewest unwanted frames until within cap.
    while len(runs) > max(1, max_intervals):
        k = min(range(len(runs) - 1), key=lambda j: runs[j + 1][0] - runs[j][1])
        runs[k][1] = runs[k + 1][1]
        del runs[k + 1]
    intervals = [(timeline[a], timeline[b]) for a, b in runs]
    selected = [timeline[i] for a, b in runs for i in range(a, b + 1)]
    return intervals, selected


def chunk_spatial_plan(
    intervals: list[tuple[float, float]],
    selected: list[float],
    max_terms: int = _SPATIAL_MAX_TERMS_PER_PASS,
) -> list[tuple[list[tuple[float, float]], int]]:
    """PURE. Split a ``plan_spatial_intervals`` plan into consecutive chunks of
    at most ``max_terms`` intervals, one ffmpeg invocation each:
    ``[(chunk_intervals, n_frames)]``. ``n_frames`` is how many ``selected``
    frames fall in the chunk, so concatenating the chunks' emitted frames in
    order is EXACTLY ``selected`` (intervals are time-ordered and disjoint) and
    each invocation's frame count can be checked on its own."""
    step = max(1, max_terms)
    chunks: list[tuple[list[tuple[float, float]], int]] = []
    for i in range(0, len(intervals), step):
        part = intervals[i : i + step]
        lo, hi = part[0][0], part[-1][1]
        chunks.append((part, sum(1 for p in selected if lo <= p <= hi)))
    return chunks


def _spatial_stream_cmd(
    path: Path | str,
    intervals: list[tuple[float, float]],
    max_width: int,
    *,
    decode_until: float | None = None,
) -> list[str]:
    """The read-only ffmpeg command for ONE chunk of
    ``measure_spatial_factors_streamed``: decode (up to ``decode_until`` seconds
    when given — input ``-to``, an early stop only), keep ONLY frames whose
    timestamp falls in ``intervals``, emit each as a binary gray PGM on stdout
    (width-capped at ``max_width``, NEAREST-neighbour — never averaged). ``-fps_mode:v
    passthrough`` (never the removed ``-vsync``) so the image muxer never duplicates or drops a frame — the
    caller attributes frames by ORDER. PURE (builds the argv only). Raises
    ``ValueError`` past ``_SPATIAL_MAX_TERMS_PER_PASS`` intervals: ffmpeg would
    fail the whole pass (see the constant)."""
    if len(intervals) > _SPATIAL_MAX_TERMS_PER_PASS:
        raise ValueError(
            f"{len(intervals)} select terms > {_SPATIAL_MAX_TERMS_PER_PASS} per ffmpeg pass"
        )
    eps = _SPATIAL_PTS_EPS
    expr = "+".join(f"between(t,{a - eps:.6f},{b + eps:.6f})" for a, b in intervals)
    until = [] if decode_until is None else ["-to", f"{decode_until:.6f}"]
    return [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-nostdin",
        *until,
        "-i",
        str(path),
        "-an",
        "-vf",
        f"select='{expr}',scale=w='min(iw,{max_width})':h=-2:flags=neighbor,format=gray",
        "-fps_mode:v",
        "passthrough",
        "-c:v",
        "pgm",
        "-f",
        "image2pipe",
        "pipe:1",
    ]


_PGM_HEADER = re.compile(rb"\AP5\s+(\d+)\s+(\d+)\s+255\s")


def parse_pgm(raw: bytes) -> tuple[bytes, int, int] | None:
    """PURE. ``(pixels, width, height)`` from one binary 8-bit PGM (``P5``), or
    ``None`` when the header is missing/unsupported or the payload is short."""
    m = _PGM_HEADER.match(raw)
    if not m:
        return None
    width, height = int(m.group(1)), int(m.group(2))
    pixels = raw[m.end() : m.end() + width * height]
    if width <= 0 or height <= 0 or len(pixels) != width * height:
        return None
    return pixels, width, height


async def _read_pgm(reader: asyncio.StreamReader) -> tuple[bytes, int, int] | None:
    """One binary 8-bit PGM frame from a stream, or ``None`` at a clean EOF.
    Raises ``ValueError`` on a malformed header and ``IncompleteReadError`` on
    a truncated frame."""
    try:
        header = await reader.readuntil(b"\n")
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise
    while header.count(b"\n") < 3:  # "P5\n<w> <h>\n255\n" (ffmpeg's pgm encoder)
        header += await reader.readuntil(b"\n")
    m = _PGM_HEADER.match(header)
    if not m or m.end() != len(header):
        raise ValueError(f"unexpected PGM header {header[:40]!r}")
    width, height = int(m.group(1)), int(m.group(2))
    return await reader.readexactly(width * height), width, height


# cgroup v2 / v1 CPU quota files. A k8s CPU LIMIT is a CFS quota, invisible to
# ``os.cpu_count()`` and ``os.sched_getaffinity`` (both report the NODE's CPUs:
# 12 on worker4 vs the pod's 10-core limit).
_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cgroup_cpu_limit(
    v2: Path = _CGROUP_V2_CPU_MAX,
    v1_quota: Path = _CGROUP_V1_QUOTA,
    v1_period: Path = _CGROUP_V1_PERIOD,
) -> float | None:
    """The cgroup CPU quota in cores (``quota / period``), or ``None`` when
    unlimited / not in a cgroup / unreadable. Never raises."""
    try:
        quota, period = v2.read_text().split()[:2]
        if quota == "max":
            return None
        return int(quota) / int(period)
    except OSError, ValueError:
        pass
    try:
        q = int(v1_quota.read_text())
        per = int(v1_period.read_text())
    except OSError, ValueError:
        return None
    return q / per if q > 0 and per > 0 else None


def available_cpus(cgroup_limit: float | None = None) -> int:
    """CPUs this process may actually use: the affinity mask (``os.cpu_count``
    where ``sched_getaffinity`` does not exist, e.g. macOS) capped by the cgroup
    CPU quota (rounded UP — a 2.5-core limit still runs 3 threads at 5/6 each).
    ``cgroup_limit`` overrides the cgroup read (tests). Always >= 1."""
    n = os.cpu_count() or 1
    affinity = getattr(os, "sched_getaffinity", None)  # Linux only
    if affinity is not None:
        with contextlib.suppress(OSError):
            n = len(affinity(0))
    limit = _cgroup_cpu_limit() if cgroup_limit is None else cgroup_limit
    if limit is not None:
        n = min(n, max(1, math.ceil(limit)))
    return max(1, n)


class _SpatialFrameSink:
    """The ``consume`` side of ``measure_spatial_factors_streamed``: reads PGM
    frames off ffmpeg's stdout and measures each wanted one, serially in a
    thread or in parallel on ``executor`` with at most ``window`` in flight.
    Attribution is by the frame's position in ``selected`` (never by
    completion order), so both paths produce identical values."""

    def __init__(
        self,
        selected: list[float],
        executor: Executor | None,
        window: int,
        needed: Callable[[float], bool] | None,
        on_factor: Callable[[float, float], None] | None,
    ) -> None:
        self.selected = selected
        self.executor = executor
        self.window = max(1, window)
        self.needed = needed
        self.on_factor = on_factor
        self.factors: dict[float, float] = {}
        self.received = 0
        self.in_flight: collections.deque[tuple[float, asyncio.Future[float]]] = collections.deque()

    def _record(self, pts: float, factor: float) -> None:
        self.factors[pts] = factor
        if self.on_factor is not None:
            self.on_factor(pts, factor)

    async def _settle_head(self) -> None:
        pts, fut = self.in_flight.popleft()
        self._record(pts, await fut)

    async def _measure(self, pts: float, pixels: bytes, width: int, height: int) -> None:
        from src.services import banding_spatial

        if self.executor is None:
            factor = await asyncio.to_thread(
                banding_spatial.spatial_banding_factor, pixels, width, height
            )
            self._record(pts, factor)
            return
        rows, cols = banding_spatial.spatial_banding_lines(pixels, width, height)
        fut = asyncio.get_running_loop().run_in_executor(
            self.executor, banding_spatial.measure_lines_in_worker, rows, cols, width, height
        )
        self.in_flight.append((pts, fut))
        while len(self.in_flight) >= self.window:
            await self._settle_head()

    async def consume(self, reader: asyncio.StreamReader) -> None:
        while (frame := await _read_pgm(reader)) is not None:
            if self.received < len(self.selected):
                pts = self.selected[self.received]
                # Land finished heads first so ``needed`` prunes on fresh results.
                while self.in_flight and self.in_flight[0][1].done():
                    await self._settle_head()
                if self.needed is None or self.needed(pts):
                    await self._measure(pts, *frame)
            self.received += 1
        while self.in_flight:
            await self._settle_head()

    def cancel_in_flight(self) -> None:
        """Pass ended early (budget, error, cancellation): keep any result that
        already landed, drop the rest."""
        for pts, fut in self.in_flight:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self._record(pts, fut.result())
            else:
                fut.cancel()
        self.in_flight.clear()


async def measure_spatial_factors_streamed(
    path: Path | str,
    all_pts: list[float],
    wanted: set[float],
    *,
    budget_seconds: float,
    max_width: int = _SPATIAL_FRAME_MAX_WIDTH,
    executor: Executor | None = None,
    window: int = 1,
    needed: Callable[[float], bool] | None = None,
    on_factor: Callable[[float, float], None] | None = None,
) -> dict[float, float]:
    """``mastering.spatial_banding_factor`` for every ``wanted`` frame of
    ``path`` (keyed by its signalstats ``pts_time``), streamed frame by frame
    (one native gray frame in memory at a time) from ffmpeg invocations of at
    most ``_SPATIAL_MAX_TERMS_PER_PASS`` select terms each (``chunk_spatial_plan``
    — ffmpeg 7.1 cannot init a select over 100 terms), run in time order, each
    with its own frame-count check.

    Parallel when ``executor`` is given (``mastering`` passes an
    ``InterpreterPoolExecutor``): only each frame's sampled LINES
    (``banding_spatial.spatial_banding_lines``, ~100 KB) are shipped to a worker,
    with at most ``window`` frames in flight so ffmpeg is back-pressured instead
    of buffered. Without one, the pure-Python measure runs in a worker thread
    (serial, GIL-bound — the pre-2026-10-21 path). Results are attributed by
    the frame's position in the plan, never by completion order, so both paths
    return identical values.

    ``needed(pts)`` is consulted just before a frame is measured; ``False``
    skips it (the settle pass's exact pruning: that frame can no longer be any
    group's max). ``on_factor(pts, factor)`` reports each result as it lands so
    ``needed`` sees it.

    Best-effort, read-only, never raises (except cancellation):
      * decode/ffmpeg error, malformed stream, a failed measure, or a frame
        COUNT that does not match the plan -> ``{}``: the order-based
        attribution can no longer be trusted, so every frame stays RAW (a
        measurement gap, the fail-open direction ``banding_score`` already
        takes for ``spatial_factor=None``).
      * ``budget_seconds`` expiry -> the frames measured so far (attributed
        exactly; with an executor not necessarily a contiguous prefix); the
        rest stay raw, an upper bound.
    May also return factors for un-wanted frames that interval merging pulled
    in — measured, correctly attributed, harmless."""
    from src.services import renderer

    intervals, selected = plan_spatial_intervals(all_pts, wanted)
    if not selected:
        return {}
    sink = _SpatialFrameSink(selected, executor, window, needed, on_factor)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_seconds
    completed = True
    try:
        # One ffmpeg invocation per chunk of <= _SPATIAL_MAX_TERMS_PER_PASS
        # select terms, in time order; the sink attributes across all of them by
        # the frame's global position in ``selected``.
        for chunk, expected in chunk_spatial_plan(intervals, selected):
            remaining = deadline - loop.time()
            if remaining <= 0:
                completed = False
                break
            before = sink.received
            completed = await renderer._stream_raw_stdout(
                _spatial_stream_cmd(
                    path,
                    chunk,
                    max_width,
                    decode_until=chunk[-1][1] + _SPATIAL_DECODE_END_MARGIN,
                ),
                sink.consume,
                budget_seconds=remaining,
            )
            if not completed:
                break
            if sink.received - before != expected:
                log.warning(
                    "measure_spatial_factors_streamed.frame_count_mismatch",
                    path=str(path),
                    expected=expected,
                    received=sink.received - before,
                )
                return {}
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("measure_spatial_factors_streamed.failed", path=str(path), error=str(exc))
        return {}
    finally:
        sink.cancel_in_flight()
    factors = sink.factors
    if not completed:
        log.info(
            "measure_spatial_factors_streamed.budget_exhausted",
            path=str(path),
            measured=len(factors),
            planned=len(selected),
            budget_seconds=budget_seconds,
        )
    return factors
