"""Health endpoints — liveness is static; readiness gates on the hard dependencies.

Liveness (`/health`, `/health/live`, `/healthz`) is a static OK so a transient
dependency blip never restarts the pod. Readiness (`/health/ready`, `/readyz`)
gates on the ONE hard dependency of the request path — Postgres: since the durable
job store, every submit/poll opens a session, so a pod that can't reach it would
202-accept and then 5xx every request at write/read time, and must leave the
Service's endpoints.

MinIO is still probed and reported in the readiness body, but is deliberately NOT
fatal. Accepting a compile never touches it (the renderer uploads later, in the
background); the one route that does need it in-request, the synchronous
`POST /api/thumbnails`, fails on a MinIO outage whether or not this pod is Ready,
so gating could not have saved it — it only added every OTHER request to the
casualties. A dependency shared by both replicas must not empty the whole Service.
See `health_ready` for the full reasoning.

Liveness additionally REPORTS (never gates on) whether the reaper is sweeping at all and
its consecutive sweep-failure count, so a reaper that is silently reclaiming nothing —
whether failing, stopped, or never started — is queryable instead of log-only.

Kept prefix-exempt from license-authz by path (`/health/`), not by an empty body.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from src.config import get_settings
from src.database.session import get_db_session
from src.storage.minio_client import get_storage

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])

# Bound each readiness dependency probe so a slow/hung MinIO or Postgres doesn't hang
# the kubelet probe (which has its own timeout, but fail fast and report cleanly here).
_READINESS_DEP_TIMEOUT_S = 3.0


def _reaper_status(request: Request) -> dict[str, object]:
    """Reaper sweep-health, reported on LIVENESS and deliberately never fatal.

    A reaper failing every sweep reclaims no orphaned renders, but the pod still serves
    submits/polls perfectly — so this must be *visible*, not fatal: failing readiness on it
    would pull a healthy pod from the Service (and a liveness failure would restart-loop
    it), turning a background-housekeeping fault into an outage. It rides liveness rather
    than readiness because the plausible cause (Postgres/function-signature drift) is
    exactly what makes readiness 503 and hide the detail.

    ``reaper_state`` exists because a MISSING reaper used to report as perfect health: the
    old ``getattr(reaper, "consecutive_failures", 0) or 0`` collapsed "swept fine" and
    "there is no reaper at all" onto the same 0, so a pod that never started its reaper (or
    whose loop task died/was stopped) looked identical to a healthy one — the exact mask
    that hides a whole defect class. The count is now ``None`` whenever no reaper is
    sweeping, and the state says which case it is:

    ``running``   sweeping — trust ``reaper_consecutive_failures``.
    ``stopped``   installed but its loop task is gone (drain, or a dead task) — DEGRADED.
    ``absent``    ``reaper_enabled`` is on but nothing is installed — DEGRADED, a bug.
    ``disabled``  ``reaper_enabled`` is off — intentional, not a fault.

    Degraded is reported, not fatal, for the same reason as sweep failures above: no
    reclaim is a housekeeping fault, not a reason to restart-loop or de-endpoint the pod.
    """
    reaper = getattr(request.app.state, "reaper", None)
    if reaper is None:
        state = "absent" if get_settings().reaper_enabled else "disabled"
    else:
        state = "running" if reaper.running else "stopped"
    sweeping = state == "running"
    failed_at = getattr(reaper, "last_failure_at", None)
    return {
        "reaper_state": state,
        # None (not 0) when nothing is sweeping: 0 must only ever mean "swept, no failures".
        "reaper_consecutive_failures": (
            reaper.consecutive_failures if sweeping and reaper is not None else None
        ),
        "reaper_last_failure": getattr(reaper, "last_failure", None),
        "reaper_last_failure_at": failed_at.isoformat() if failed_at is not None else None,
    }


@router.get("/health", operation_id="health_check")
async def health(request: Request) -> dict[str, object]:
    return {"status": "ok", **_reaper_status(request)}


@router.get("/health/live", operation_id="health_live")
async def health_live(request: Request) -> dict[str, object]:
    return {"status": "live", **_reaper_status(request)}


@router.get("/healthz", operation_id="health.live")
async def healthz(request: Request) -> dict[str, object]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await health_live(request)


async def _minio_reachable() -> bool:
    """True if MinIO answers a bucket_exists call (reachable). A missing bucket
    still counts as reachable — ensure_bucket creates it on first upload; only a
    connection/transport failure counts as unreachable. The result is REPORTED on
    readiness, not gated on (see `health_ready`). The whole probe (incl. the
    first-call client init, which does a bucket round-trip) is inside the timeout so
    an unreachable MinIO at boot can't hang past the bound on miniopy's own timeout."""
    settings = get_settings()

    async def _probe() -> bool:
        storage = await get_storage(settings, bucket=settings.minio_bucket)
        return bool(await storage.check_health())

    try:
        healthy = await asyncio.wait_for(_probe(), timeout=_READINESS_DEP_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — any failure (timeout, transport, config) = not ready
        # `error` MUST be repr(), not str(). `StorageService.check_health` already
        # swallows (OSError, S3Error, MinioException) and returns False, so in steady
        # state the ONLY exception reaching here is the asyncio.TimeoutError from the
        # wait_for above — and `str(TimeoutError()) == ""`. This line rendered as
        # `{"error": "", "event": "readiness.minio_unreachable"}` in production, which
        # is undiagnosable: it cannot distinguish a 3s timeout from a config error.
        log.warning(
            "readiness.minio_unreachable",
            error=repr(exc),
            error_type=type(exc).__name__,
            timeout_s=_READINESS_DEP_TIMEOUT_S,
            endpoint=settings.minio_endpoint,
        )
        return False
    if not healthy:
        # Distinct from the except branch above: check_health() returned False having
        # swallowed its own exception, so there is no `exc` to report. Without this the
        # path was entirely silent — a reachability failure with no log line at all.
        log.warning(
            "readiness.minio_health_false",
            endpoint=settings.minio_endpoint,
            bucket=settings.minio_bucket,
        )
    return healthy


async def _postgres_reachable() -> bool:
    """True if Postgres answers a trivial SELECT. Since the durable job store, every
    submit/poll opens a session, so an unreachable DB means the pod can't do any work
    and must be pulled from the Service's endpoints (else it 5xx's every request)."""

    async def _probe() -> bool:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return True

    try:
        return await asyncio.wait_for(_probe(), timeout=_READINESS_DEP_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — any failure (timeout, transport, auth) = not ready
        # repr(), not str() — same reason as _minio_reachable: a bare
        # asyncio.TimeoutError from the wait_for stringifies to "".
        log.warning(
            "readiness.postgres_unreachable",
            error=repr(exc),
            error_type=type(exc).__name__,
            timeout_s=_READINESS_DEP_TIMEOUT_S,
        )
        return False


@router.get("/health/ready", operation_id="health_ready")
async def health_ready(response: Response) -> dict[str, str]:
    """Readiness gates on POSTGRES ONLY. MinIO is probed and REPORTED, never fatal.

    MinIO used to be half of this gate, and that was wrong: it made an object store that
    the PRIMARY workload does not need in-request into a *serving* dependency of the whole
    Service.

    Nothing this pod does to ACCEPT a compile touches MinIO. `submit_compile`
    (`src/api/routes/compile.py`) runs `_guard_local_file_inputs` (pure string checks) →
    `_accept` (Postgres only) → `asyncio.create_task`, and returns 202. The poll endpoint
    reads Postgres. Startup touches neither bucket. MinIO is first contacted much later,
    by the background renderer, when it uploads the finished MP4/WAV.

    ⚠️ ONE ROUTE IS DIFFERENT, and an earlier revision of this docstring wrongly claimed
    the probe below was "the only MinIO call in the request path". It is not:
    `POST /api/thumbnails` is deliberately SYNCHRONOUS and calls `upload_and_presign`
    inside the request (`src/services/thumbnail.py:138`) to return presigned PNG URLs.
    With MinIO down that route fails.

    That does not change the decision — it sharpens it. If MinIO is unreachable, the
    thumbnail route fails whether or not this pod advertises readiness; taking the pod out
    of the endpoints cannot rescue it. What gating DID buy was failing everything else
    too. Both replicas share the one MinIO, so a sustained outage fails the probe on both
    pods and Kubernetes empties the entire Service's endpoint list — killing compile
    submits and job polls that never needed MinIO at all, to protect one route that was
    already broken. Gating trades one broken route for every broken route.

    On timing, precisely: the readiness probe is `periodSeconds: 30` with `failureThreshold`
    unset (Kubernetes default 3), so a single slow probe never de-routed anything — it took
    roughly 60-90s of SUSTAINED failure. The observed `readiness.minio_unreachable` warnings
    recur about hourly on both replicas; this change removes the class of outcome rather
    than tuning the threshold.

    A real MinIO outage still surfaces, in the right place and at the right granularity: the
    render fails at upload and the job is marked `failed` with its error, reported per job
    through `GET /api/jobs/{job_id}`. That is a JOB failure, which the caller can retry —
    it is not an inability to accept work, and readiness is the wrong instrument for it.

    So MinIO becomes visible-but-not-fatal, the same treatment (and for the same reason) as
    `_reaper_status` on liveness above: keep the truth queryable in the body, keep the
    `readiness.minio_unreachable` warning firing, and do not convert a background-path
    fault into a de-endpointed pod. Both dependency states are always in the body so
    `kubectl get --raw /readyz` / curl shows reality whatever the status code.

    Postgres genuinely does gate: `_accept` writes the job row, so with the DB unreachable
    a submit cannot be honoured at all and this pod must leave the endpoints.
    """
    minio_ok, pg_ok = await asyncio.gather(_minio_reachable(), _postgres_reachable())
    body = {
        "minio": "reachable" if minio_ok else "unreachable",
        "postgres": "reachable" if pg_ok else "unreachable",
    }
    if pg_ok:
        return {"status": "ready", **body}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "unavailable", **body}


@router.get("/readyz", operation_id="health.ready")
async def readyz(response: Response) -> dict[str, str]:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready(response)
