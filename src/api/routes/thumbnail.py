"""Synchronous thumbnail / poster-parity extraction (spec 07-color-finish §E).

Unlike the async ``/api/compile`` job, frame extraction is sub-second, so
``POST /api/thumbnails`` renders inline and returns the candidates directly —
this service has no DB / job store to hang a poll on. The input video is fetched
to a per-request scratch dir (reusing the renderer's SSRF-safe ``_fetch``), run
through ``thumbnail.extract_candidates`` (which reuses the same ffmpeg-exec gate,
whole-film grade chain, format→dims map, and MinIO uploader as the compiler), and
the scratch dir is removed in a ``finally``.

Route is UNVERSIONED (``/api/thumbnails`` — house rule, never ``/api/v1/``) and
reads the caller identity from the inherited request context exactly like the
compile route, so the log line is tenant-attributed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid7

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request
from w2v_common.auth import LICENSE_GATE
from w2v_common.auth.identity_context import (
    get_license_id_from_context,
    get_subject_id_from_context,
)

from src.api.routes.compile import _safe_run_id
from src.api.schemas import ThumbnailRequest, ThumbnailResponse
from src.config import get_settings
from src.services.renderer import (
    _BAD_ASSET_ERRORS,
    _FETCH_TIMEOUT_SECONDS,
    LocalFileInputError,
    _fetch,
)
from src.services.thumbnail import extract_candidates

log = structlog.get_logger(__name__)

# ENTITLEMENT GATE — same rationale as src/api/routes/compile.py. This router's
# sole route (POST /api/thumbnails) is tenant-scoped synchronous render work,
# so it gets the gate too; dormant until W2V_LICENSE_GATE_ENABLED is armed.
router = APIRouter(tags=["thumbnail"], dependencies=list(LICENSE_GATE))


@router.post(
    "/api/thumbnails",
    response_model=ThumbnailResponse,
    operation_id="extract_thumbnails",
    summary="Extract N grade-parity poster/thumbnail candidates from a video (synchronous).",
)
async def extract_thumbnails(request: Request, body: ThumbnailRequest) -> ThumbnailResponse:
    settings = get_settings()
    # LFI guard: a file:// / bare-path video_url is read straight off the pod
    # filesystem. The compile route guards its submit (PR #136), but the thumbnail
    # route shares the same SSRF-safe _fetch and needs the same gate — otherwise a
    # `file:///etc/passwd` video_url is a thumbnail-route LFI. Gated by
    # allow_local_file_inputs (default False/secure; tests+local dev opt in via
    # ALLOW_LOCAL_FILE_INPUTS=true).
    if not settings.allow_local_file_inputs and not body.video_url.startswith(
        ("http://", "https://")
    ):
        raise HTTPException(
            status_code=400, detail=f"local-file input rejected: {body.video_url[:80]}"
        )
    # License-tenant bound by AuthMiddleware. Log-only here (thumbnails persist no
    # tenant-scoped row), so it stays best-effort rather than fail-loud like the compile
    # routes — but it is NEVER defaulted to a sentinel.
    tenant_id = getattr(request.state, "tenant_id", None)
    run_id = _safe_run_id((body.tags or {}).get("run_id"))
    # A per-request scratch dir under work_dir; its name is the run-scoped,
    # collision-safe object-key namespace the service uploads the PNGs under.
    work_dir = Path(settings.work_dir) / f"{run_id}-{uuid7().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "thumbnail.accepted",
        run_id=run_id,
        count=body.count,
        format=body.format,
        tenant_id=tenant_id,
        license_id=get_license_id_from_context(),
        subject_id=get_subject_id_from_context(),
    )
    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as http:
            try:
                # video_url is CALLER-SUPPLIED. The prefix check above refuses local
                # refs outright when the flag is off; when it IS on, local_root
                # confines what the opt-in buys to work_dir, so a `/etc/passwd`
                # video_url (which sails past a prefix check) is still refused.
                video = await _fetch(
                    body.video_url,
                    work_dir / "source",
                    http,
                    local_root=Path(settings.work_dir),
                )
            except LocalFileInputError as exc:
                # The point-of-use confinement refused a path escaping work_dir (or
                # carrying a traversal token). Same 400 shape as the guard above —
                # a rejected input is a bad request, not a fetch failure.
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except _BAD_ASSET_ERRORS as exc:
                # An unfetchable / SSRF-blocked / oversized / MALFORMED video_url is a
                # bad request, not a 5xx — mirror the extractor's 422 below (previously
                # this propagated uncaught and 500'd the route).
                #
                # Reuses the renderer's tuple rather than restating it: this route calls
                # the SAME ``_fetch``, so the set of "the caller's URL is bad" errors is
                # by construction identical, and a local copy is exactly how the two
                # drifted. That tuple's ``ValueError`` / ``httpx.InvalidURL`` members are
                # load-bearing here too — ``safe_get`` validates the URL BEFORE any I/O
                # and raises a BARE ValueError for no hostname (``http://:8080/x.mp4``),
                # an out-of-range port, or an unsupported scheme, and the prefix guard
                # above admits all of those (it only checks the http(s) prefix). Before
                # this, they 500'd. ``LocalFileInputError`` is deliberately NOT in the
                # tuple and is handled as a 400 above; see its docstring.
                raise HTTPException(
                    status_code=422, detail=f"could not fetch video_url: {exc}"
                ) from exc
        try:
            candidates = await extract_candidates(
                video,
                body.count,
                fmt=body.format,
                grade=body.grade,
                settings=settings,
                work_dir=work_dir,
            )
        except ValueError as exc:
            # Bad request, not a 5xx: an unprobeable / zero-duration source, or a
            # source-matched grade that can't be reproduced from a single frame.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ThumbnailResponse(candidates=candidates)
    finally:
        # work_dir is a dir WE created directly under work_dir — safe to drop.
        shutil.rmtree(work_dir, ignore_errors=True)
