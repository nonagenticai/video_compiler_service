"""Thumbnail / poster-parity frame extraction (spec 07-color-finish §E, AC:147).

Extract N DETERMINISTIC candidate frames from a video and return them as
presigned PNG URLs plus title-safe / action-safe overlay geometry.

Poster parity: when a ``GradeSpec`` is supplied the frames run through the SAME
``renderer._grade_vf`` filter chain the whole-film grade uses, so a thumbnail cut
from an UNGRADED hero source reproduces the delivered film's static grade. A
source-matched grade (``match_sources=True``) biases the eq midpoints toward the
whole film's *measured* signalstats — which a single extracted frame cannot
reproduce — so that mode is rejected here (422): for a source-matched deliverable,
cut the poster from the already-graded film with ``grade=None`` instead. When the
input is already the graded delivery master, ``grade`` is None and the frames are
extracted as-is. Everything reuses the compiler's existing
machinery — the ``_run`` ffmpeg-exec gate, ``_ffprobe_duration`` /
``probe_dimensions`` probes, the compile route's format→dims map, and the MinIO
uploader — so the thumbnail path can never drift from the film path.
"""

from __future__ import annotations

from pathlib import Path

import structlog

# Single source of the deliverable format→canvas-dims map (mirrors the
# orchestrator's motion_slides.format_dims); reused so a poster's aspect can
# never diverge from the compiled film's.
from src.api.routes.compile import _FORMAT_DIMS
from src.api.schemas import GradeSpec, SafeRect, ThumbnailCandidate
from src.config import Settings
from src.services.renderer import (
    _ffprobe_duration,
    _grade_vf,
    _run,
    _sws_flags,
    probe_dimensions,
)
from src.storage.minio_client import upload_and_presign

log = structlog.get_logger(__name__)

_MIN_CANDIDATES = 1
_MAX_CANDIDATES = 12
# Trim the first/last 5% of the clip so evenly-spaced samples avoid the lead-in
# black / title card and the tail card — the middle 90% is the body of the film.
_EDGE_TRIM = 0.05
# Broadcast-style safe areas as a fraction of the frame (centered).
_ACTION_SAFE_FRAC = 0.90
_TITLE_SAFE_FRAC = 0.80


def _clamp_count(count: int) -> int:
    """Clamp the candidate count to a sane range IN THE SERVICE (not only the
    schema), so a direct caller can't request 0 or 10_000 frames."""
    return max(_MIN_CANDIDATES, min(_MAX_CANDIDATES, count))


def _timestamps(duration: float, count: int) -> list[float]:
    """DETERMINISTIC evenly-spaced sample points across the middle 90% of the clip.

    No RNG and no wall-clock: the same ``(duration, count)`` always yields the
    same, strictly-increasing timestamps, so a thumbnail set is reproducible for a
    given input. Each point is the centre of its equal slice of the trimmed span.
    """
    start = duration * _EDGE_TRIM
    span = duration * (1.0 - 2.0 * _EDGE_TRIM)
    return [start + span * (i + 0.5) / count for i in range(count)]


def _safe_rect(width: int, height: int, frac: float) -> SafeRect:
    """Centered safe-area rectangle at ``frac`` of the frame, in pixels."""
    w = round(width * frac)
    h = round(height * frac)
    return SafeRect(x=(width - w) // 2, y=(height - h) // 2, width=w, height=h)


async def extract_candidates(
    video: Path,
    count: int,
    *,
    fmt: str | None,
    grade: GradeSpec | None,
    settings: Settings,
    work_dir: Path,
) -> list[ThumbnailCandidate]:
    """Extract ``count`` grade-parity poster frames from ``video``.

    ``fmt`` (a deliverable aspect like ``16x9``) retargets each frame to the film's
    delivery canvas via the compile route's format→dims map; ``None`` keeps the
    source's encoded dimensions. ``grade`` — when supplied — is applied through the
    identical whole-film ``_grade_vf`` chain so an ungraded source matches the film;
    a ``None`` / disabled / identity grade extracts the frame unchanged (same empty
    filter the compile ``-c copy`` fast path relies on). A source-matched grade
    (``match_sources=True``) is rejected with ``ValueError`` — see module docstring.
    Uploads every PNG under ``work_dir.name`` and returns the presigned candidates.
    """
    if grade is not None and grade.match_sources:
        # match_sources biases the grade toward the WHOLE FILM's measured signalstats
        # (renderer._grade_vf's ``stats`` arg); a single extracted frame can't
        # reproduce those, so a source-matched poster would silently DIVERGE from the
        # film's look — the exact opposite of the parity this route promises. The only
        # faithful source for a source-matched deliverable is the already-graded film
        # (pass grade=None). Reject rather than ship a look that merely approximates.
        raise ValueError(
            "match_sources grades cannot be reproduced from a single frame; "
            "extract the poster from the already-graded film with grade=None"
        )
    duration = await _ffprobe_duration(video)
    if duration <= 0:
        raise ValueError(f"cannot probe a positive duration from {video}")
    count = _clamp_count(count)

    # Output frame dims: an explicit deliverable format retargets the poster to the
    # film's canvas (reusing compile's map); otherwise keep the source's dims.
    dims = _FORMAT_DIMS.get((fmt or "").strip())
    if dims is not None:
        out_w, out_h = dims
    else:
        out_w, out_h = await probe_dimensions(video)
        # ``probe_dimensions`` returns the (0, 0) SENTINEL meaning "unknown" on any
        # probe failure. That is fine for its reporting-only caller, but here the
        # dims are GEOMETRY: they size the action/title safe rects below, and
        # ``SafeRect`` uses ``Field(ge=0)`` so zeros validate — the route would
        # answer 200 OK with width=0/height=0 and zero-area boxes, and a caller
        # would place a title in a degenerate rect at the origin. Unknown is not a
        # measurement: refuse it the same way the duration probe above does (422).
        if out_w <= 0 or out_h <= 0:
            raise ValueError(f"cannot probe positive dimensions from {video}")

    # Grade the frame through the IDENTICAL whole-film chain ONLY when a grade is
    # supplied (ungraded hero source → matches the film). An already-graded input
    # passes grade=None; a disabled/identity grade yields "" — no -vf, same as the
    # compile fast path.
    grade_vf = _grade_vf(grade) if grade is not None else ""
    # Same swscale flags + derived in/out_color_matrix as every other scale=
    # node in the compiler (see renderer._sws_flags): unflagged, this scale
    # runs swscale's BILINEAR default and its BT.601 matrix default, silently
    # mismatching the source's bt709-tagged encode on every retargeted poster.
    scale = f"scale={out_w}:{out_h}:{_sws_flags(settings)}" if dims is not None else ""
    vf = ",".join(part for part in (grade_vf, scale) if part)

    run_id = work_dir.name
    action_safe = _safe_rect(out_w, out_h, _ACTION_SAFE_FRAC)
    title_safe = _safe_rect(out_w, out_h, _TITLE_SAFE_FRAC)

    candidates: list[ThumbnailCandidate] = []
    for i, t in enumerate(_timestamps(duration, count)):
        out_png = work_dir / f"thumb-{i}.png"
        # -ss BEFORE -i is the fast keyframe-accurate seek; -frames:v 1 grabs one
        # frame. -vf is added ONLY when there is a grade and/or a format scale.
        cmd = ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1"]
        if vf:
            cmd += ["-vf", vf]
        cmd.append(str(out_png))
        await _run(cmd)
        url = await upload_and_presign(
            settings,
            bucket=settings.minio_bucket,
            object_key=f"{run_id}/thumb-{i}.png",
            file_path=out_png,
            expiry_seconds=settings.minio_url_expiry_seconds,
            content_type="image/png",
        )
        candidates.append(
            ThumbnailCandidate(
                url=url,
                timestamp_seconds=round(t, 3),
                width=out_w,
                height=out_h,
                action_safe=action_safe,
                title_safe=title_safe,
            )
        )
    log.info(
        "thumbnail.extracted",
        n=len(candidates),
        run_id=run_id,
        graded=bool(grade_vf),
        width=out_w,
        height=out_h,
    )
    return candidates
