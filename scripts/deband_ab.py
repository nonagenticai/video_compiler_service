"""A/B measurement harness for the conditional deband stage — DOES DEBAND HELP?

WHY THIS EXISTS
----------------
``_apply_conditional_deband`` (``src/services/renderer.py``) re-encodes 99.4%
of shots measured in prod (157 applied / 1 skipped) through the ffmpeg
``deband`` filter — ~23s of a ~77s job. Nobody has ever measured whether that
re-encode IMPROVES the picture or SMEARS it: ``grep -rln
"ssim\\|psnr\\|vmaf" src/ tests/ scripts/`` returns nothing before this file.
Every gate in this repo is predicate-level (does ``banding_score`` cross
0.15?) — none compares pixels before vs. after the filter runs.

This is a standalone CLI, not a pytest module: it needs REAL rendered
material (not the lavfi synthetic gradients ``tests/render_harness/harness.py``
generates — those are exactly the case that would beg the banding question)
and minutes of wall clock, neither of which a per-PR CI job can spend. It
changes NO production code — ``renderer.py``, ``mastering.py`` and
``config.py`` are read-only imports here.

USAGE
-----
Against a directory of already-split real segments (``seg-*.mp4``, e.g.
pulled by hand off a job's working dir before cleanup)::

    uv run python scripts/deband_ab.py --segments /path/to/seg/dir

Against a delivered master pulled from MinIO (stream-copy split on keyframes,
which approximates the real per-shot segments closely enough for this
question — the deband filter runs on this same GOP structure in prod)::

    uv run python scripts/deband_ab.py --minio-key videos/<job_id>/master.mp4 \\
        [--bucket w2v-compiled-videos] [--segment-seconds 6]

Either mode writes one JSON line per segment to stdout (or ``--out FILE``)
plus a final ``{"aggregate": ..., "verdict": ..., "reason": ...}`` line. Needs
``ffmpeg``/``ffprobe`` on PATH and (for ``--minio-key``) real
``MINIO_ENDPOINT``/``MINIO_ACCESS_KEY``/``MINIO_SECRET_KEY`` env — the same
ones the pod runs with.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# ``Settings`` carries a required ``DatabaseSettings.url`` even though this
# script never touches the durable-job store — set a harmless placeholder
# before importing so construction doesn't need a live DB (mirrors
# tests/conftest.py's ``os.environ.setdefault`` for the same reason).
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test@127.0.0.1:5478/video_compiler_service_ci",  # pragma: allowlist secret
)

from src.config import Settings
from src.services import mastering, measure, renderer
from tests.render_harness.harness import available_filters

# Mirrors mastering.BANDING_GATE_CEILING / the spec's "13% of the 0.15
# ceiling" no-op floor. Named here (not imported) because it is a DECISION
# threshold for this harness, independent of the production gate constant —
# an accidental prod-constant edit must not silently change what this script
# concludes.
REDUCTION_FLOOR = 0.02
NOOP_SSIM_FLOOR = 0.995
DESTRUCTION_SSIM_CEILING = 0.98
BROAD_HELP_FRACTION = 0.60


# ---------------------------------------------------------------------------
# Per-segment measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRecord:
    segment: str
    max_banding_before: float
    max_banding_after: float
    p95_banding_before: float
    p95_banding_after: float
    mean_banding_before: float
    mean_banding_after: float
    frac_over_ceiling_before: float
    frac_over_ceiling_after: float
    ssim: float | None
    vmaf: float | None
    measure_before_seconds: float
    deband_seconds: float
    measure_after_seconds: float
    compare_seconds: float

    @property
    def max_banding_reduction(self) -> float:
        return self.max_banding_before - self.max_banding_after


def _banding_stats(frames: list[dict[str, float]]) -> tuple[float, float, float, float]:
    """(max, p95, mean, fraction >= ceiling) over a segment's per-frame banding
    scores. PURE. Empty ``frames`` -> all zeros (mirrors ``banding_score``'s
    "0.0 when unmeasured" contract)."""
    scores = [mastering.banding_score(f) for f in frames]
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    ordered = sorted(scores)
    p95_idx = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    frac_over = sum(1 for s in scores if s >= mastering.BANDING_GATE_CEILING) / len(scores)
    return max(scores), ordered[p95_idx], statistics.fmean(scores), frac_over


def _parse_ssim(stderr: str) -> float | None:
    """Pull the ``All:`` aggregate out of ffmpeg's ``-lavfi ssim`` stderr log.
    PURE. Returns ``None`` on any parse miss — never raises."""
    last = None
    for line in stderr.splitlines():
        if "All:" in line:
            last = line
    if last is None:
        return None
    for token in last.split():
        if token.startswith("All:"):
            try:
                return float(token.removeprefix("All:"))
            except ValueError:
                return None
    return None


def _parse_vmaf(stderr: str) -> float | None:
    """Pull the aggregate VMAF score out of ffmpeg's ``-lavfi libvmaf`` stderr
    log (``VMAF score: NN.NNNNNN``). PURE. ``None`` on any parse miss."""
    for line in stderr.splitlines():
        if "VMAF score:" in line:
            try:
                return float(line.rsplit("VMAF score:", 1)[1].strip())
            except ValueError:
                return None
    return None


async def _compare(
    orig: Path, debanded: Path, *, use_vmaf: bool
) -> tuple[float | None, float | None, float]:
    """SSIM (+ VMAF when available) between ``orig`` and ``debanded``, timed."""
    # ssim and libvmaf both need two inputs mapped into a filter_complex; run
    # them as two separate discarded passes rather than one filter_complex so
    # a libvmaf model-load failure on a runner without the model file can
    # never take the (cheap, always-available) SSIM measurement down with it.
    t0 = time.perf_counter()
    ssim_stderr = await renderer._run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(orig),
            "-i",
            str(debanded),
            "-lavfi",
            "ssim",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    ssim = _parse_ssim(ssim_stderr)
    vmaf = None
    if use_vmaf:
        try:
            vmaf_stderr = await renderer._run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostats",
                    "-i",
                    str(orig),
                    "-i",
                    str(debanded),
                    "-lavfi",
                    "libvmaf",
                    "-f",
                    "null",
                    "-",
                ],
                capture_stderr=True,
            )
            vmaf = _parse_vmaf(vmaf_stderr)
        except Exception as exc:  # noqa: BLE001 — vmaf is best-effort, ssim is the load-bearing metric
            print(f"  vmaf measurement failed (non-fatal): {exc}", file=sys.stderr)
    return ssim, vmaf, time.perf_counter() - t0


async def measure_segment(
    seg_path: Path, settings: Settings, work_dir: Path, *, use_vmaf: bool
) -> SegmentRecord:
    t0 = time.perf_counter()
    frames_before = await measure.measure_shot_frames(seg_path)
    t1 = time.perf_counter()
    mb, pb, meanb, fb = _banding_stats(frames_before)

    copy_path = work_dir / f"{seg_path.stem}-deband.mp4"
    shutil.copyfile(seg_path, copy_path)
    t2 = time.perf_counter()
    await renderer._deband_segment(copy_path, settings)
    t3 = time.perf_counter()

    frames_after = await measure.measure_shot_frames(copy_path)
    t4 = time.perf_counter()
    ma, pa, meana, fa = _banding_stats(frames_after)

    ssim, vmaf, compare_s = await _compare(seg_path, copy_path, use_vmaf=use_vmaf)

    return SegmentRecord(
        segment=seg_path.name,
        max_banding_before=mb,
        max_banding_after=ma,
        p95_banding_before=pb,
        p95_banding_after=pa,
        mean_banding_before=meanb,
        mean_banding_after=meana,
        frac_over_ceiling_before=fb,
        frac_over_ceiling_after=fa,
        ssim=ssim,
        vmaf=vmaf,
        measure_before_seconds=t1 - t0,
        deband_seconds=t3 - t2,
        measure_after_seconds=t4 - t3,
        compare_seconds=compare_s,
    )


# ---------------------------------------------------------------------------
# Pure decision function — the thing the unit tests exercise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str
    median_max_banding_reduction: float | None
    median_ssim: float | None
    helped_fraction: float | None
    n_segments: int
    n_with_ssim: int


def decide(records: Sequence[SegmentRecord]) -> Decision:
    """Pure classification over a batch of :class:`SegmentRecord`. Never touches
    the filesystem or the network — the unit tests drive this directly with
    synthetic records.

    Branch order (destruction check runs FIRST and outranks everything else,
    per spec):

    1. **destroys** — SSIM < 0.98 on ANY measured segment. Deband is smearing
       detail; report this regardless of how the banding numbers look.
    2. **no-op** — median ``max_banding_reduction`` < 0.02 (13% of the 0.15
       banding ceiling) AND median SSIM > 0.995. The re-encode is not doing
       anything visible, whichever direction.
    3. **helps_broadly** — reduction >= 0.02 on >= 60% of segments.
    4. **helps_selectively** — reduction >= 0.02 on < 60% of segments.
    """
    if not records:
        return Decision(
            verdict="no_data",
            reason="no segments measured",
            median_max_banding_reduction=None,
            median_ssim=None,
            helped_fraction=None,
            n_segments=0,
            n_with_ssim=0,
        )

    ssim_values = [r.ssim for r in records if r.ssim is not None]
    reductions = [r.max_banding_reduction for r in records]
    median_reduction = statistics.median(reductions)
    median_ssim = statistics.median(ssim_values) if ssim_values else None
    n = len(records)
    helped = sum(1 for r in reductions if r >= REDUCTION_FLOOR)
    helped_fraction = helped / n

    if any(v < DESTRUCTION_SSIM_CEILING for v in ssim_values):
        worst = min(ssim_values)
        return Decision(
            verdict="destroys_detail",
            reason=(
                f"SSIM {worst:.4f} < {DESTRUCTION_SSIM_CEILING} on at least one segment — "
                "deband is DESTROYING detail, not polishing it. This outranks the "
                "banding-reduction numbers below."
            ),
            median_max_banding_reduction=median_reduction,
            median_ssim=median_ssim,
            helped_fraction=helped_fraction,
            n_segments=n,
            n_with_ssim=len(ssim_values),
        )

    if (
        median_reduction < REDUCTION_FLOOR
        and median_ssim is not None
        and median_ssim > NOOP_SSIM_FLOOR
    ):
        return Decision(
            verdict="no_op",
            reason=(
                f"median max-banding reduction {median_reduction:.4f} < {REDUCTION_FLOOR} "
                f"and median SSIM {median_ssim:.4f} > {NOOP_SSIM_FLOOR} — the deband "
                "pass is not visibly changing the picture either way."
            ),
            median_max_banding_reduction=median_reduction,
            median_ssim=median_ssim,
            helped_fraction=helped_fraction,
            n_segments=n,
            n_with_ssim=len(ssim_values),
        )

    if helped_fraction >= BROAD_HELP_FRACTION:
        return Decision(
            verdict="helps_broadly",
            reason=(
                f"{helped}/{n} segments ({helped_fraction:.0%}) reduced max banding by "
                f">= {REDUCTION_FLOOR} — deband is earning its cost on most shots."
            ),
            median_max_banding_reduction=median_reduction,
            median_ssim=median_ssim,
            helped_fraction=helped_fraction,
            n_segments=n,
            n_with_ssim=len(ssim_values),
        )

    return Decision(
        verdict="helps_selectively",
        reason=(
            f"only {helped}/{n} segments ({helped_fraction:.0%}) reduced max banding by "
            f">= {REDUCTION_FLOOR} — deband helps a minority of shots; the trigger "
            "(banding_risk_frames) is over-broad relative to where it actually helps."
        ),
        median_max_banding_reduction=median_reduction,
        median_ssim=median_ssim,
        helped_fraction=helped_fraction,
        n_segments=n,
        n_with_ssim=len(ssim_values),
    )


# ---------------------------------------------------------------------------
# Material acquisition
# ---------------------------------------------------------------------------


async def _split_master(master_path: Path, out_dir: Path, segment_seconds: float) -> list[Path]:
    """Stream-copy split on keyframes — the same GOP structure the compiler's
    own per-shot segments carry, without re-encoding (so the split itself
    introduces no picture change to confound the A/B)."""
    pattern = out_dir / "seg-%04d.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(master_path),
            "-c",
            "copy",
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    return sorted(out_dir.glob("seg-*.mp4"))  # noqa: ASYNC240 — one-shot CLI script, not a hot path


async def _fetch_minio_segments(
    settings: Settings, *, bucket: str, key: str, out_dir: Path, segment_seconds: float
) -> list[Path]:
    from src.storage.minio_client import get_storage

    storage = await get_storage(settings, bucket=bucket)
    master_path = out_dir / "master-source.mp4"
    stream = await storage.get_object_stream(key, bucket=bucket)
    async with stream:
        if stream.status == 404:
            raise SystemExit(f"deband_ab: key not found: {bucket}/{key}")
        with master_path.open("wb") as fh:
            async for chunk in stream.iter_chunks():
                fh.write(chunk)
    return await _split_master(master_path, out_dir, segment_seconds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A/B measure the conditional deband stage against real segments.",
    )
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--segments", type=Path, help="directory of real seg-*.mp4 files")
    src_group.add_argument(
        "--minio-key", type=str, help="object key of a delivered master to pull + split"
    )
    p.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="override minio bucket (default: settings.minio_bucket)",
    )
    p.add_argument(
        "--segment-seconds",
        type=float,
        default=6.0,
        help="split interval for --minio-key (default 6s)",
    )
    p.add_argument("--out", type=Path, default=None, help="write JSON lines here instead of stdout")
    p.add_argument("--limit", type=int, default=None, help="only measure the first N segments")
    p.add_argument("--no-vmaf", action="store_true", help="skip libvmaf even if the runner has it")
    return p


async def _amain(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings()
    use_vmaf = (not args.no_vmaf) and "libvmaf" in available_filters()

    out_fh = args.out.open("w") if args.out else sys.stdout

    with tempfile.TemporaryDirectory(prefix="deband-ab-") as tmp:
        work_dir = Path(tmp)
        if args.segments:
            segments = sorted(args.segments.glob("seg-*.mp4"))
            if not segments:
                print(f"deband_ab: no seg-*.mp4 files under {args.segments}", file=sys.stderr)
                return 2
        else:
            bucket = args.bucket or settings.minio_bucket
            segments = await _fetch_minio_segments(
                settings,
                bucket=bucket,
                key=args.minio_key,
                out_dir=work_dir,
                segment_seconds=args.segment_seconds,
            )
            if not segments:
                print(
                    f"deband_ab: split produced no segments for {args.minio_key}", file=sys.stderr
                )
                return 2

        if args.limit is not None:
            segments = segments[: args.limit]

        print(
            f"deband_ab: measuring {len(segments)} segment(s), vmaf={'on' if use_vmaf else 'off'}",
            file=sys.stderr,
        )

        records: list[SegmentRecord] = []
        for i, seg in enumerate(segments, 1):
            print(f"  [{i}/{len(segments)}] {seg.name}", file=sys.stderr)
            record = await measure_segment(seg, settings, work_dir, use_vmaf=use_vmaf)
            records.append(record)
            out_fh.write(json.dumps(asdict(record)) + "\n")
            out_fh.flush()

        decision = decide(records)
        out_fh.write(json.dumps({"aggregate": True, **asdict(decision)}) + "\n")
        out_fh.flush()
        if args.out:
            out_fh.close()

        print(f"deband_ab: verdict={decision.verdict} — {decision.reason}", file=sys.stderr)
        if decision.verdict == "destroys_detail":
            print("deband_ab: ⚠️  DEBAND IS DESTROYING DETAIL — see reason above", file=sys.stderr)
        return 0


def main() -> int:
    return asyncio.run(_amain(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
