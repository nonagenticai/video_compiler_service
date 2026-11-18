"""Async compile / concat-audio jobs: submit → poll.

``POST /api/compile`` and ``POST /api/concat-audio`` accept a job and return a
``job_id`` immediately (202); the ffmpeg render + MinIO upload run in a
background task. ``GET /api/jobs/{job_id}`` returns the job status and, once
terminal, the result.

Why async: the render is unbounded (30s to many minutes for a big
website-review or course stitch). A synchronous endpoint held the connection
open for the
whole render, so a long/queued compile tripped the caller's read timeout — which
``w2v_common``'s BaseServiceClient hard-caps at 600s. Decoupling render from
request lifetime means every request is short (submit returns instantly; each
poll is a quick GET), so the wall-clock budget lives entirely in the caller's
poll loop, with no 600s ceiling. Matches the fleet submit_job/poll_job
convention.

Job state + the Idempotency-Key dedup index are DURABLE in Postgres (see
``src/services/compile_job_store.py``): a resend after a pod restart dedupes to
the persisted job instead of firing a SECOND paid render. The render itself stays
an in-process asyncio task (single-replica ``replicas: 1`` + ``Recreate``); a task
orphaned by a restart leaves its row ``processing``, which the next pod's startup
reclaim flips to ``interrupted`` so a resend re-runs it under the same job_id.

Production-safety guards on the async surface:
- backpressure: submits 429 once ``max_concurrent_compile_jobs`` renders are in
  flight (counted as LIVE tasks, RLS-agnostic), so a burst can't spawn unbounded
  renders and fill /tmp / OOM the pod;
- a per-job work timeout force-fails a wedged render instead of leaving it
  ``processing`` forever (and ``compile_job_store.prune`` flips such zombies to
  ``interrupted``);
- shutdown drain (see ``drain_background_tasks``) cancels in-flight renders on
  SIGTERM, best-effort stamping each ``interrupted`` (startup reclaim is the real
  guarantee) instead of orphaning it;
- jobs are tenant-stamped at submit and polls are tenant-scoped (cross-tenant
  polls 404);
- submits carrying an ``Idempotency-Key`` are deduped against the durable index,
  so a connection-reset replay of the POST (BaseServiceClient retries these for
  any method) — and a resume-after-timeout re-dispatch across a pod restart —
  return the original job_id instead of starting a second, non-idempotent render.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import secrets
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast
from uuid import uuid7

import structlog
from fastapi import APIRouter, HTTPException, Request
from w2v_common.auth import LICENSE_GATE, W2V_APP_KEY
from w2v_common.auth.identity_context import (
    clear_identity_context,
    get_attribution_subject_id_from_context,
    get_license_id_from_context,
    set_identity_context,
)
from w2v_common.metering import cost_event_key, schedule_cost_event

from src.api.schemas import (
    AudioConcatRequest,
    CompileRequest,
    FinishProfile,
    JobAccepted,
    JobCancelRequest,
    JobCancelResponse,
    JobStatus,
)
from src.config import Settings, get_settings
from src.database.session import get_db_session
from src.observability import metrics
from src.services import cadence, mastering
from src.services import compile_job_store as job_store
from src.services.captions_ass import caption_look_from_request
from src.services.compile_job_store import (
    CapacityError,
    ClaimedJob,
    MissingTenantError,
    require_tenant,
)
from src.services.durable_dispatch import (
    KIND_COMPILE,
    KIND_CONCAT,
    bind_job_attribution,
    build_job_options,
    inputs_reconstructable,
    restore_tenant_source,
    run_with_lease,
)
from src.services.loudness import final_mux_loudness
from src.services.metering import (
    COST_SOURCE,
    PRICE_PER_OUTPUT_MINUTE_USD,
    PROVIDER,
    SERVED_MODEL,
    SERVICE_KEY,
    funding_for,
)
from src.services.renderer import (
    RetryableRenderError,
    compile_shots,
    concat_audio,
    get_master_loudnorm_evidence,
    get_master_normalization_type,
    grade_report,
    probe_dimensions,
    probe_output_duration,
)
from src.storage.minio_client import upload_and_presign

log = structlog.get_logger(__name__)

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# (AuthMiddleware answers "who are you") and never asked "may you" — a tenant
# whose licence is expired or revoked still carried a syntactically valid
# token and was served the full compile/concat-audio/cancel/poll surface,
# including the billing-bearing render dispatch. LICENSE_GATE closes that:
# none/expired/revoked -> 402 license_<state>; an unestablishable verdict ->
# 503 license_unavailable + Retry-After; active/expiring_soon -> pass.
#
# Every route on THIS router is tenant-scoped compile-job work (submit
# compile/concat-audio, cancel-by-prefix, poll job status) — there is no
# internal/peer callback here, so nothing needs an exemption from this router.
#
# DORMANT ON MERGE: require_license is a no-op unless
# W2V_LICENSE_GATE_ENABLED is true, read per-call from os.environ. It
# defaults to False, so this change alters no behaviour until gitops arms
# the flag. Preserve that default.
router = APIRouter(tags=["compile"], dependencies=list(LICENSE_GATE))

# Accepted spellings for an opt-in boolean per-job tag (case-insensitive, trimmed).
_TRUTHY = frozenset({"1", "true", "on", "yes"})

# Bounded retry for the TERMINAL completion write of a SUCCESSFUL render. A single
# transient DB blip (pgbouncer hiccup / brief connection reset) on this one write used to
# leave the row ``processing`` — so the reaper later did a FULL paid re-render of an
# already-successful render. A few short retries absorb the blip in-process; only if ALL
# fail is the row left ``processing`` for the reaper (still the backstop, never the
# first line of defence). Kept small so a genuinely-wedged DB doesn't stall the task.
_COMPLETION_WRITE_RETRIES = 3
_COMPLETION_WRITE_BACKOFF = 0.5  # seconds; multiplied by (attempt + 1) for linear backoff
# SystemRandom only to keep bandit's B311 quiet — this is retry jitter, not a secret.
_JITTER = secrets.SystemRandom()


def _parse_truthy_tag(tags: Mapping[str, str], key: str) -> bool:
    """Whether a per-job ``tags`` entry reads as truthy.

    None-safe (a missing key reads falsey), case-insensitive, surrounding
    whitespace stripped. The single source of truth for the ``deterministic`` /
    ``freeze_pad_cap`` opt-in flags so their accepted spellings can't drift apart."""
    return (tags.get(key) or "").strip().lower() in _TRUTHY


# Keep a strong ref to in-flight tasks so they aren't GC'd mid-render.
_background_tasks: set[asyncio.Task[None]] = set()

# job_ids this pod is CURRENTLY rendering (added at spawn, discarded when the task
# ends). The reaper consults this before re-dispatching a reclaimed job so it can
# never start a second render of a job already running HERE: if a live render's lease
# lapses (e.g. a hung heartbeat renew under DB saturation) the same-pod reaper would
# otherwise reclaim its own in-flight job and double-render it. This in-process set is
# authoritative for the same-pod case regardless of what the (eventually-consistent)
# lease says; the lease still guards the genuine cross-pod (dead-pod) case.
_active_job_ids: set[str] = set()

# job_id → the render Task, for ``POST /api/jobs/cancel``. ``_background_tasks`` alone
# cannot serve a cancel: it is task-keyed, so there is no way to find the ONE task that
# belongs to a given job. Same lifetime as ``_active_job_ids`` (added at spawn, dropped in
# the done-callback), so a finished job leaves nothing behind.
_job_tasks: dict[str, asyncio.Task[None]] = {}

# True only while ``drain_background_tasks`` is cancelling in-flight renders on
# SIGTERM. During a drain, ANY render exception (not just the injected
# CancelledError) is a shutdown artifact, so the terminal write is 'interrupted'
# (a resend re-runs it), never 'failed' (which would dedup + never re-run → a lost
# render). Reset once the drain completes so it can't leak across the process.
_draining = False

# Wall-clock budget for ``drain_background_tasks``. Sits INSIDE the pod's
# terminationGracePeriodSeconds (k8s default 30s) with headroom for the rest of the
# ASGI shutdown (reaper stop, metering flush, engine dispose) — the point of the bound
# is that the drain finishes on ITS terms, not the kubelet's SIGKILL. A task's cancel
# handler stamps 'interrupted' via a SHIELDED write, and each such write is itself
# bounded at ~2.25s of retry sleep plus the write, so a healthy drain of the 8-job
# in-flight cap clears well inside this.
_DRAIN_GRACE_SECONDS: Final = 12.0

# Metering Redis client, set once from the ASGI lifespan (None = metering
# disabled / unreachable → emits no-op). Module-level because the render runs in
# a detached background task that has no `request`/`app` handle.
_metering_redis: Any = None


def set_metering_redis(client: Any) -> None:
    """Install the metering Redis client (called from the ASGI lifespan)."""
    global _metering_redis
    _metering_redis = client


# How the billed output duration was obtained. Stamped verbatim onto the cost
# event's ``usage_extra`` so an unmeasured (nominal-fallback) charge stays
# distinguishable from a probed one downstream.
_DURATION_PROBED: Final = "probed"
_DURATION_NOMINAL: Final = "nominal"
# A third state, distinct from both: the artifact WAS delivered but its length could
# be neither probed nor declared. Absent measurement is not a zero — conflating the
# two is what made an unprobeable audiobook bill nothing, silently (see
# ``_billed_duration`` and ``_emit_cost``).
_DURATION_UNMEASURED: Final = "unmeasured"


def _emit_cost(
    *,
    job_id: str,
    tenant_id: uuid.UUID,
    duration_seconds: float | None,
    duration_source: str,
    latency_ms: int,
    job_type: str,
) -> None:
    """Fire-and-forget a per-output-minute cost event for a completed render.

    Billed on OUTPUT duration (compiled video / stitched audio seconds). No-op
    when metering is disabled or the job produced NO OUTPUT AT ALL. An output that
    was delivered but could not be measured (``duration_source="unmeasured"``) is
    NOT a no-op: it emits a zero-quantity event flagged ``duration_unmeasured``
    and logs ``cost_duration_unprobed``, so unbillable delivered work is visible in
    both billing and the logs instead of vanishing.
    Identity (license/subject) is read from the request context the background
    task inherited when the render was spawned; ``tenant_id`` is the
    authoritative owner threaded through the job.

    G5 full attribution: ``app_id`` is the w2v app key, ``funding`` is stated
    (self-hosted -> ``local_gpu``), and ``event_key`` is derived from the
    durable ``job_id`` — only the attempt that WON the terminal write emits, so
    one job is one paid render and a reaper-resumed re-emit dedupes at billing.
    Project/actor come from w2v_common's ambient chain (signed cell claims).

    Raises ``MissingTenantError`` if the tenant is absent — a tripwire, not an
    expected path: ``compile_jobs.tenant_id`` is ``uuid NOT NULL``, so a job
    that reached a completed render always has one. Emitting an unattributed
    cost event would be silently unbilled, which is worse than a loud failure.
    Beyond that it never raises: the shared ``schedule_cost_event`` retains a
    strong reference to the emit task (so it can't be GC'd before the XADD
    lands) and swallows any Redis error.
    """
    if _metering_redis is None:
        return
    # The annotation says ``uuid.UUID``, but SQLAlchemy/attribute reads are
    # ``Any``-typed, so the gate cannot see a caller that threads ``None`` here.
    # Without this guard ``str(None)`` would emit the literal "None" as the
    # tenant -- not even a uuid, and worse than the nil sentinel this service
    # just removed, because billing's ``_normalize_tenant_id`` would bucket it
    # as unattributed and silently never charge it. Fail here instead.
    tenant = require_tenant(tenant_id)
    seconds = max(0, round(duration_seconds or 0.0))
    unmeasured = duration_source == _DURATION_UNMEASURED
    if seconds <= 0 and not unmeasured:
        # GENUINELY zero: a measurement or a declaration exists and it says there is
        # nothing to bill (no artifact was delivered). Distinct from ``unmeasured``,
        # which falls through to a flagged zero-quantity event below. This early
        # return used to swallow the unmeasured case too, jumping clean over the
        # ``cost_duration_unprobed`` warning: delivered work, billed nothing, and not
        # one log line to notice it by.
        return
    if duration_source != _DURATION_PROBED:
        # Either the final-mux ffprobe did not produce a measurement and this charge is
        # the DECLARED per-shot sum (which over-reports seeked/repeated clips), or there
        # was no declared sum either and the quantity is UNMEASURED (``seconds == 0``,
        # charged nothing). Loud here and marked on the event, so an over-report and an
        # unbillable delivery are both detectable rather than silent.
        log.warning(
            "compile.cost_duration_unprobed",
            job_type=job_type,
            duration_source=duration_source,
            billed_seconds=seconds,
        )
    cost_hint = seconds / 60.0 * PRICE_PER_OUTPUT_MINUTE_USD
    usage_extra: dict[str, str] = {
        "job_type": job_type,
        "unit": "video_seconds",
        "duration_source": duration_source,
    }
    if unmeasured:
        # Fleet shape (``modality_gateway`` ``tokens_unmeasured``): the event EXISTS so
        # the delivery is reconcilable, carries quantity 0 so nothing is fabricated and
        # nothing is over-charged (there is no refund path in this system — an
        # over-charge is permanent), and says so on its face so it can be re-priced by
        # hand rather than being invisible.
        usage_extra["duration_unmeasured"] = "true"
    schedule_cost_event(
        _metering_redis,
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id=str(tenant),
        provider=PROVIDER,
        model=SERVED_MODEL,
        input_tokens=seconds,
        output_tokens=0,
        latency_ms=latency_ms,
        cost_usd=cost_hint,
        license_id=get_license_id_from_context(),
        subject_id=get_attribution_subject_id_from_context(),
        app_id=W2V_APP_KEY,
        usage_extra=usage_extra,
        funding=funding_for(PROVIDER),
        event_key=cost_event_key(SERVICE_KEY, job_id, "render"),
    )


# run_id arrives untrusted via ``tags`` and is interpolated straight into the
# MinIO object key. Anything outside this set is replaced so a value like
# ``../../other-tenant`` can't traverse the bucket namespace.
_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_run_id(raw: str | None) -> str:
    """Sanitize a caller-supplied ``run_id`` before it becomes an object-key prefix."""
    cleaned = _RUN_ID_UNSAFE.sub("_", (raw or "").strip())[:64]
    return cleaned or "ad-hoc"


def _even_dim(raw: str | None, lo: int, hi: int) -> int | None:
    """Parse a caller-supplied pixel dimension tag into a safe even int.

    Resolution arrives untrusted via ``tags`` (the orchestrator's compile_tags)
    and feeds straight into ffmpeg scale/pad filters. Reject anything non-numeric
    or out of range, and force even (yuv420p requires even W/H — odd → ffmpeg
    fails the whole render). ``None`` means "tag absent/invalid → keep the env
    default", so a bad tag degrades to the global setting rather than 500-ing.
    """
    if raw is None:
        return None
    try:
        val = int(str(raw).strip())
    except ValueError:
        return None
    if val < lo or val > hi:
        return None
    return val - (val % 2)


def _settings_for_tags(settings: Settings, tags: dict[str, str] | None) -> Settings:
    """Per-job resolution/quality override from compile ``tags``.

    Lets a single run request 2K/4K (or a tighter CRF) without a service
    redeploy: the orchestrator emits ``render_width``/``render_height``/``crf``
    in compile_tags. The env-configured ``settings`` stays the default; only
    valid, bounded, even values override it.
    """
    tags = tags or {}
    update: dict[str, Any] = {}
    width = _even_dim(tags.get("render_width"), 256, 7680)
    height = _even_dim(tags.get("render_height"), 256, 4320)
    if width is not None:
        update["render_width"] = width
    if height is not None:
        update["render_height"] = height
    # CRF need not be even; just guard numeric + range.
    raw_crf = (tags.get("crf") or "").strip()
    if raw_crf.isdigit() and 0 <= int(raw_crf) <= 51:
        update["render_crf"] = int(raw_crf)
    # Loudness master (default-ON in settings): a job can disable it or retarget
    # the loudnorm integrated-loudness without a redeploy.
    raw_master = (tags.get("mastering") or "").strip().lower()
    if raw_master in {"off", "false", "0", "no"}:
        update["master_enabled"] = False
    elif raw_master in {"on", "true", "1", "yes"}:
        update["master_enabled"] = True
    raw_lufs = (tags.get("master_lufs") or "").strip()
    if raw_lufs:
        try:
            lufs = float(raw_lufs)
        except ValueError:
            lufs = None
        if lufs is not None and -30.0 <= lufs <= -5.0:
            update["master_lufs"] = lufs
    _apply_mastering_tags(tags, update)
    if not update:
        return settings
    return settings.model_copy(update=update)


def _clamped_float(raw: str, lo: float, hi: float) -> float | None:
    """Parse ``raw`` to a float within ``[lo, hi]``, else ``None`` (degrade)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if lo <= val <= hi else None


def _apply_mastering_tags(tags: dict[str, str], update: dict[str, Any]) -> None:
    """Fold the VC-2 encode/delivery-master tags (colorspace / deband / bitrate /
    platform / true-peak) into ``update`` in place, with the same clamp
    discipline as the rest of ``_settings_for_tags`` — an unknown value degrades
    to the env default (dropped), never a 5xx."""
    raw_cs = (tags.get("colorspace") or "").strip().lower()
    if raw_cs in {"bt709", "bt601", "bt2020", "srgb", "none", "off"}:
        update["render_colorspace"] = "" if raw_cs in {"none", "off"} else raw_cs
    raw_deband = (tags.get("deband") or "").strip().lower()
    if raw_deband in {"auto", "on", "true", "1", "yes"}:
        update["render_deband"] = "auto" if raw_deband == "auto" else "on"
    elif raw_deband in {"off", "false", "0", "no"}:
        update["render_deband"] = "off"
    raw_mode = (tags.get("bitrate_mode") or "").strip().lower()
    if raw_mode in {"auto", "vbv", "cbr", "capped"}:
        update["render_bitrate_mode"] = raw_mode
    raw_maxrate = (tags.get("maxrate") or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?[kKmMgG]?", raw_maxrate or ""):
        update["render_vbv_maxrate"] = raw_maxrate
    raw_platform = (tags.get("platform") or "").strip().lower()
    if raw_platform in {"youtube", "instagram", "tiktok", "x", "web", "vimeo"}:
        update["render_output_platform"] = raw_platform
    tp = _clamped_float(tags.get("master_tp") or "", -9.0, 0.0)
    if tp is not None:
        update["master_tp"] = tp


def _caller_tenant(request: Request) -> uuid.UUID:
    """License-tenant bound by w2v_common ``AuthMiddleware``, as a real UUID.

    FAIL LOUD (400) when absent or unparseable. There is no fallback: ``tenant_id`` may
    never hold a sentinel, so a request that forwards no registered tenant cannot write
    or read a tenant-scoped row at all. This used to return ``None`` and every consumer
    coalesced it to ``''``/the nil uuid — one shared bucket, readable by every
    unattributed caller under the strict RLS policy.
    """
    try:
        return require_tenant(getattr(request.state, "tenant_id", None))
    except MissingTenantError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _active_render_count() -> int:
    """Live in-flight render tasks on THIS pod — the backpressure gauge.

    Counts actual running renders, not DB rows: under ``TENANT_RLS_MODE=enforce``
    a DB ``processing`` count would only see the current tenant's rows,
    undercounting the global in-flight load this cap protects (the pod's /tmp +
    memory). The task set is also the accurate live-render gauge (no zombies).

    Note: this is a SOFT admission bound, not a hard limit. The count is read at
    accept-time and the task is added a few awaits later in ``_spawn``, so a burst
    of concurrent submits can over-admit by ~the number of simultaneous accepts.
    That is acceptable because peak MEMORY is separately + hard-bounded by the
    ffmpeg-subprocess semaphore (``max_concurrent_ffmpeg_subprocesses``), which
    caps actual ffmpeg parallelism regardless of how many jobs are ``processing``.
    """
    return len(_background_tasks)


def _instance_id(request: Request) -> str:
    """This pod's boot id (set once at startup), stamped on rows it is rendering."""
    return cast(str, request.app.state.instance_id)


def _billed_duration(result: Mapping[str, Any]) -> tuple[float | None, str]:
    """Output duration to METER on, PAIRED with how it was obtained.

    Returns ``(seconds, source)`` where ``source`` is ``"probed"``, ``"nominal"`` or
    ``"unmeasured"`` — three states, not two, because "we could not measure it" and
    "it was genuinely zero" must not share a branch. They did, and an unprobeable
    audiobook was the result: ``seconds <= 0`` returned from ``_emit_cost`` before the
    ``cost_duration_unprobed`` warning could fire, so a delivered master produced no
    cost event AND no log line.

    The nominal ``duration_seconds`` (sum of ``render_shot`` durations) OVER-reports a
    seeked no-narration footage clip — ``clip_anchor="end"`` / ``clip_lead_seconds`` /
    a repeated clip adds a ``-ss`` seek, so the encoded segment is shorter than the
    nominal clip length by the offset. ``actual_duration_seconds`` is the probed final
    output (``probe_output_duration``), the true deliverable, so billing it is both
    correct and immune to any nominal/actual drift.

    ``probe_output_duration`` returns the **0.0 sentinel meaning "unknown"** on any
    ffprobe failure/timeout, and 0.0 is falsy — so the fallback below is reached by
    "we could not measure it", not only by "no probe was attempted". Billing the
    nominal sum there is deliberate (we cannot bill nothing), but it is an
    OVER-report of unmeasured work, so the source is returned alongside the number
    and stamped onto the cost event: an unprobed charge must be auditable after the
    fact, never silently indistinguishable from a measured one.
    """
    probed = cast(float | None, result.get("actual_duration_seconds"))
    if probed:
        return probed, _DURATION_PROBED
    nominal = cast(float | None, result.get("duration_seconds"))
    if nominal:
        # The audiobook concat carries NO separate ``actual_duration_seconds``: its
        # ``duration_seconds`` IS an ffprobe of the delivered master, flagged
        # ``duration_probed`` by ``_concat_to_result``. Reporting that as "nominal"
        # slandered every audiobook charge as an unaudited over-report and fired the
        # ``cost_duration_unprobed`` warning on every single successful one, which is
        # exactly the noise that makes the real occurrences unfindable.
        if result.get("duration_probed"):
            return nominal, _DURATION_PROBED
        return nominal, _DURATION_NOMINAL
    if result.get("artifact_url"):
        # An artifact WAS delivered, but the probe carried no duration. Fall back to the
        # declared sum exactly as the compile path does — same reasoning, same marker.
        declared = cast(float | None, result.get("declared_duration_seconds"))
        if declared:
            return declared, _DURATION_NOMINAL
        # No probe and no complete declaration: the quantity is genuinely UNKNOWN.
        # Return the ABSENCE as an absence — never 0.0, which reads as "measured, and it
        # was nothing", which is what silently unbilled this path.
        return None, _DURATION_UNMEASURED
    # Nothing delivered and nothing declared: genuinely zero, and not billable.
    return None, _DURATION_NOMINAL


async def _stamp_completed(job_id: str, result: dict[str, Any], instance_id: str = "") -> bool:
    """Terminal 'completed' write; returns whether THIS write won the row.

    A superseded attempt (a timed-out original racing a reclaimed re-run of the same
    job_id, or — at replicas>1 — a render this pod no longer owns) hits 0 rows, so its
    caller must NOT then emit a duplicate paid cost event. ``instance_id`` fences the
    write to the current owner (see ``compile_job_store._owner_conds``).
    """
    async with get_db_session() as session:
        return await job_store.mark_completed(session, job_id, result, instance_id)


async def _stamp_failed(
    job_id: str,
    error: str,
    instance_id: str = "",
    *,
    result: dict[str, object] | None = None,
) -> None:
    async with get_db_session() as session:
        await job_store.mark_failed(session, job_id, error, instance_id, result=result)


def _failure_result(exc: BaseException) -> dict[str, object] | None:
    """The STRUCTURED verdict to persist alongside a failure's ``error`` string.

    VC-W14/V2. ``MasteringGateError`` is the one failure in this service that
    already knows exactly which check cleared the gate, with the measured value,
    the limit it breached and the tier — ``violations_payload()`` has projected
    that onto the published contract since wave 13 and had **zero callers**, so
    the structured ``violations`` contract was live only for masters that PASS.
    The failing master — the entire point of a gate — still delivered prose.

    ``get_job`` reads ``result["mastering"]`` onto ``JobStatus.mastering`` for any
    status, so returning it here is what puts the verdict on the wire.
    ``gate_pass`` is stamped ``False`` explicitly: a consumer must not have to
    infer the verdict from the presence of a non-empty list.

    Returns ``None`` for every other exception ⇒ ``mark_failed`` leaves ``result``
    alone and nothing about the existing failure path changes."""
    if isinstance(exc, mastering.MasteringGateError):
        return {"mastering": {"gate_pass": False, "violations": exc.violations_payload()}}
    return None


async def _stamp_interrupted(
    job_id: str, error: str, instance_id: str = "", *, reason: str
) -> None:
    """Flip to ``interrupted``, recording WHY (see ``job_store.InterruptReason``).

    ``reason`` is forwarded verbatim: only ``drain`` makes the sibling replica's reclaim
    free, so a genuine render fault must never be labelled as one.
    """
    async with get_db_session() as session:
        await job_store.mark_interrupted(session, job_id, error, instance_id, reason=reason)


async def _accept(
    request: Request,
    settings: Settings,
    *,
    kind: str,
    tenant_id: uuid.UUID,
    job_options: dict[str, Any],
) -> job_store.AcceptResult:
    """Durable Idempotency-Key dedup + backpressure for a submit.

    Persists the accept in Postgres (so a resend after a pod restart dedupes
    instead of double-rendering) with the reconstruction bundle + a fresh lease, so a
    crash mid-render lets another pod's reaper resume it. Raises 429 when the in-flight
    render cap is reached for a spawn/rerun.
    """
    try:
        async with get_db_session() as session:
            return await job_store.accept(
                session,
                tenant_id=tenant_id,
                idempotency_key=request.headers.get("Idempotency-Key"),
                kind=kind,
                instance_id=_instance_id(request),
                active_count=_active_render_count(),
                max_concurrent=settings.max_concurrent_compile_jobs,
                job_options=job_options,
                lease_seconds=settings.job_lease_seconds,
                max_attempts=settings.reclaim_max_attempts,
                max_lifetime_seconds=settings.reclaim_max_lifetime_seconds,
            )
    except CapacityError:
        raise HTTPException(
            status_code=429,
            detail="compiler at capacity; retry shortly",
            headers={"Retry-After": "5"},
        ) from None


async def _finalize_completed(
    *,
    job_id: str,
    result: dict[str, Any],
    instance_id: str,
    tenant_id: uuid.UUID,
    job_type: str,
    started: float,
) -> None:
    """Guarded terminal-completed write + metering for a SUCCESSFUL render.

    The completion write is in its own guard so a transient DB blip does NOT propagate to
    ``_run``'s failure handling and mis-stamp a good render ``failed`` (which would dedup a
    perfectly-good render terminal and orphan its MinIO object). On a stamp error the row is
    left ``processing`` for the reaper to resume (the deterministic per-job key overwrites).
    Cost is metered only when THIS attempt won the terminal write — a superseded attempt
    hits 0 rows and MUST NOT emit a second paid event — on the PROBED output duration.
    """
    won = False
    for attempt in range(_COMPLETION_WRITE_RETRIES):
        try:
            won = await _stamp_completed(job_id, result, instance_id)
            break
        except Exception:
            if attempt == _COMPLETION_WRITE_RETRIES - 1:
                # Exhausted retries: leave the row ``processing`` for the reaper to resume
                # (the deterministic per-job key overwrites) rather than mis-stamping a
                # good render ``failed``. The reaper is the backstop, not the first resort.
                log.error("compile.completion_write_failed", job_id=job_id, exc_info=True)
                return
            log.warning(
                "compile.completion_write_retry",
                job_id=job_id,
                attempt=attempt + 1,
                exc_info=True,
            )
            # Jittered so two replicas retrying the same blip (a pgbouncer restart hits
            # both at once) do not re-collide in lockstep on every attempt.
            await asyncio.sleep(
                _COMPLETION_WRITE_BACKOFF * (attempt + 1) * _JITTER.uniform(0.5, 1.5)
            )
    if won:
        billed_seconds, duration_source = _billed_duration(result)
        _emit_cost(
            job_id=job_id,
            tenant_id=tenant_id,
            duration_seconds=billed_seconds,
            duration_source=duration_source,
            latency_ms=int((time.monotonic() - started) * 1000),
            job_type=job_type,
        )


def _spawn(
    job_id: str,
    work: Callable[[], Awaitable[dict[str, Any]]],
    *,
    tenant_id: uuid.UUID,
    work_timeout: float,
    job_type: str,
    lease_seconds: int = 0,
    instance_id: str | None = None,
) -> None:
    """Run ``work`` in the background; record terminal status in the durable store.

    The render stays an in-process asyncio task (single-replica service); only the
    job STATE is durable now. The detached task inherits the request's
    tenant/identity ContextVars (asyncio copies the context at ``create_task``), so
    its own ``get_db_session`` binds the right tenant for the terminal write and
    ``_emit_cost`` reads the right identity. On SIGTERM cancel it best-effort stamps
    ``interrupted`` (a resend re-runs it); if that write is lost, the next pod's
    startup reclaim flips the still-``processing`` row to ``interrupted``.

    When ``lease_seconds`` > 0 the render runs under a lease heartbeat (see
    ``durable_dispatch.run_with_lease``) so the reaper on another pod never reclaims a
    job still rendering here; the lease lapses only if THIS pod dies.
    """

    async def _run() -> None:
        started = time.monotonic()
        # Terminal status for `video_compile_jobs_total`. Defaults to "failed" so a
        # path that somehow reaches `finally` without classifying itself is counted
        # as a failure rather than silently vanishing from the totals.
        outcome = "failed"
        metrics.compile_jobs_inflight.inc()
        try:
            result = await run_with_lease(
                job_id=job_id,
                instance_id=instance_id or "",
                lease_seconds=lease_seconds,
                work=asyncio.wait_for(work(), timeout=work_timeout),
            )
            # The render SUCCEEDED — finalize (guarded completion write + metering) in a
            # helper so a transient DB blip there can't fall through to the failure excepts
            # below and mis-stamp a good render ``failed``.
            # SHIELDED. By this point the render is done and the artifact is in
            # MinIO — the only thing left is the terminal row write. Its DB await
            # and its jittered backoff `sleep` are both cancellation points, and
            # `_finalize_completed` catches `Exception`, which CancelledError is
            # not. So a SIGTERM landing in that window fell through to the
            # `except asyncio.CancelledError` branch below, stamped
            # `interrupted/drain`, and the reaper re-ran a render that had ALREADY
            # succeeded — paying for the whole thing twice and emitting `_emit_cost`
            # a second time for one deliverable. The drain stamp beneath is shielded
            # for the same reason; this call was the half that was not.
            #
            # Bounded on purpose: 3 attempts with linear jittered backoff caps the
            # shielded window at ~2.25s of sleep plus the writes, well inside a
            # normal termination grace period.
            await asyncio.shield(
                _finalize_completed(
                    job_id=job_id,
                    result=result,
                    instance_id=instance_id or "",
                    tenant_id=tenant_id,
                    job_type=job_type,
                    started=started,
                )
            )
            outcome = "completed"
        except asyncio.CancelledError:
            # SIGTERM / shutdown drain (see drain_background_tasks): best-effort
            # flip to 'interrupted' so a resend re-runs it, then propagate. Shielded
            # so the cancellation doesn't abort the write; swallowed because the next
            # pod's startup reclaim is the real guarantee. work()'s finally has
            # already removed the scratch dir.
            log.warning("compile.job_cancelled", job_id=job_id)
            outcome = "interrupted"
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    _stamp_interrupted(
                        job_id,
                        "cancelled — service shutting down",
                        instance_id or "",
                        reason=job_store.INTERRUPT_REASON_DRAIN,
                    )
                )
            raise
        except RetryableRenderError as exc:
            # A TRANSIENT render failure (ffmpeg subprocess timeout / signal-kill / OOM) —
            # stamp 'interrupted' (re-runnable), NOT terminal 'failed'. Must precede
            # 'except TimeoutError': RetryableRenderError is not a TimeoutError, so the
            # outer whole-job work_timeout branch below is untouched. The poison-attempt +
            # lifetime caps bound a job that keeps failing this way — which is exactly why
            # the reason is 'render_failure' and NOT 'drain': a drain-labelled interrupt is
            # exempt from the attempt increment, so mislabelling this genuine ffmpeg crash
            # would let a crash-looping render be reclaimed and re-billed forever.
            log.warning("compile.job_interrupted_transient", job_id=job_id, error=str(exc))
            outcome = "interrupted"
            with contextlib.suppress(Exception):
                await _stamp_interrupted(
                    job_id,
                    str(exc),
                    instance_id or "",
                    reason=job_store.INTERRUPT_REASON_RENDER_FAILURE,
                )
        except TimeoutError:
            log.error("compile.job_timeout", job_id=job_id, work_timeout=work_timeout)
            outcome = "interrupted" if _draining else "failed"
            with contextlib.suppress(Exception):
                if _draining:
                    await _stamp_interrupted(
                        job_id,
                        "interrupted — service shutting down",
                        instance_id or "",
                        reason=job_store.INTERRUPT_REASON_DRAIN,
                    )
                else:
                    await _stamp_failed(
                        job_id,
                        f"render exceeded {work_timeout:.0f}s work timeout",
                        instance_id or "",
                    )
        except Exception as exc:  # noqa: BLE001 — any failure is reported via job status
            # A non-CancelledError raised as a SHUTDOWN artifact (a killed subprocess
            # surfacing as a plain exception) must be 'interrupted' (re-runnable), not
            # a terminal 'failed' that a resend would dedup + never re-run.
            outcome = "interrupted" if _draining else "failed"
            if _draining:
                log.warning("compile.job_interrupted_on_drain", job_id=job_id, error=str(exc))
                with contextlib.suppress(Exception):
                    await _stamp_interrupted(
                        job_id,
                        "interrupted — service shutting down",
                        instance_id or "",
                        reason=job_store.INTERRUPT_REASON_DRAIN,
                    )
            else:
                log.error("compile.job_failed", job_id=job_id, error=str(exc))
                with contextlib.suppress(Exception):
                    # VC-W14/V2: a gate failure carries its violations onto the row,
                    # so the poll returns the structured verdict and not only prose.
                    await _stamp_failed(
                        job_id, str(exc), instance_id or "", result=_failure_result(exc)
                    )
        finally:
            # Runs on EVERY path including the re-raised CancelledError, so a drained
            # pod's in-flight jobs still decrement the gauge and land in the totals.
            # Observability must never break a render: a metrics backend fault here
            # would otherwise mask the real terminal status recorded above.
            with contextlib.suppress(Exception):
                metrics.compile_jobs_inflight.dec()
                metrics.compile_job_seconds.labels(kind=job_type).observe(
                    time.monotonic() - started
                )
                metrics.compile_jobs_total.labels(kind=job_type, status=outcome).inc()

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    _active_job_ids.add(job_id)
    _job_tasks[job_id] = task

    def _done(t: asyncio.Task[None]) -> None:
        _background_tasks.discard(t)
        # Drop BOTH job-keyed entries only if it is still OUR task: a rerun of the same
        # job_id (an interrupted job re-accepted under the same key) installs a new task,
        # and a late callback from the old one must not evict the live entry. This guard
        # is load-bearing for ``_active_job_ids`` too, not just ``_job_tasks``:
        # ``_active_job_ids`` is the SOLE input to both same-pod double-render guards
        # (``reaper_service`` reclaim exclusion, and the ``/internal/claim`` check), so an
        # unguarded discard un-shields a LIVE render and can buy a second paid ffmpeg
        # pass. The two entries share one lifetime — added together at spawn above,
        # dropped together here — so gating them identically cannot leak either one.
        if _job_tasks.get(job_id) is t:
            _active_job_ids.discard(job_id)
            del _job_tasks[job_id]

    task.add_done_callback(_done)


def begin_drain() -> None:
    """Mark the pod as draining BEFORE any render is cancelled.

    ``asgi._shutdown`` calls this at the very start of shutdown — before
    ``reaper.stop()`` (which can block on an in-flight sweep) and before
    ``drain_background_tasks()``. Without it, a render killed in that pre-cancel
    window (an OOM'd ffmpeg, or the node SIGTERM killing the subprocess) would be
    stamped terminal ``failed`` (dedup'd, never re-run → lost paid render) instead of
    ``interrupted`` (re-runnable). ``drain_background_tasks`` clears the flag in its
    ``finally`` once the drain completes.
    """
    global _draining
    _draining = True


async def drain_background_tasks() -> None:
    """Cancel and await in-flight render tasks on shutdown.

    Called from the ASGI lifespan on SIGTERM so a rolling deploy doesn't orphan
    a render or leave a job wedged in ``processing``. Each cancelled task stamps
    its row ``interrupted`` (see ``_spawn._run``) so a resend re-runs it; the
    ``_draining`` flag makes even a non-CancelledError shutdown artifact resolve to
    ``interrupted`` rather than a terminal ``failed``.
    """
    global _draining
    tasks = list(_background_tasks)
    if not tasks:
        return
    log.info("compile.drain_background_tasks", n_tasks=len(tasks))
    _draining = True
    try:
        for task in tasks:
            task.cancel()
        # BOUNDED, not an unbounded gather. Every task's CancelledError handler awaits a
        # SHIELDED _stamp_interrupted, and a shield cannot be cancelled — so ONE stalled
        # pgbouncer write used to hold the whole drain until that write returned. Overrun
        # terminationGracePeriodSeconds and the kubelet SIGKILLs us: no
        # interrupt_reason='drain' is written for ANY task, the sibling's reclaim falls to
        # the SQL's ELSE 1, and a render that was merely rolled out is charged a poison
        # attempt. Bounding the wait keeps the drain graceful for the tasks that CAN
        # finish, which is precisely what migration 008's exemption needs.
        # asyncio.wait (not wait_for/gather): the timeout expiring must LEAVE the stragglers
        # alone rather than re-cancel them — they are already cancelled, and their shielded
        # write may still land before the process exits.
        done, pending = await asyncio.wait(tasks, timeout=_DRAIN_GRACE_SECONDS)
        for task in done:
            # asyncio.wait does not consume results; retrieve them so a render that died
            # with a real exception cannot surface as "exception was never retrieved" and
            # cannot propagate out of the shutdown path (the old gather(return_exceptions=
            # True) contract).
            with contextlib.suppress(BaseException):
                task.result()
        if pending:
            log.warning(
                "compile.drain_grace_exceeded",
                n_pending=len(pending),
                grace_seconds=_DRAIN_GRACE_SECONDS,
            )
    finally:
        _draining = False


# Deliverable ``format`` tag → output canvas (w, h). Mirrors the orchestrator's
# ``motion_slides.format_dims`` so the compiled canvas matches the format-keyed
# motion clips / title cards exactly: a 9x16 deliverable must render a 1080x1920
# PORTRAIT video, not letterbox portrait cards inside a 16:9 canvas. Unknown or
# absent format → the service default (16:9 via settings.render_{width,height}).
_FORMAT_DIMS: dict[str, tuple[int, int]] = {
    "16x9": (1920, 1080),
    "9x16": (1080, 1920),
    "1x1": (1080, 1080),
    "4x5": (1080, 1350),
}


def _settings_for_format(tags: dict[str, Any] | None, settings: Settings) -> Settings:
    """Retarget the render canvas to the deliverable ``format`` tag.

    The renderer reads every output dimension from ``settings.render_{width,height}``
    (scale/pad, scrolling, bg canvas, caption wrap), so a per-job ``model_copy`` of
    the cached settings retargets the whole compile to the requested aspect with no
    renderer-signature churn. Returns the settings unchanged when the format is
    absent/unknown or already matches the default.
    """
    fmt = str((tags or {}).get("format") or "").strip()
    dims = _FORMAT_DIMS.get(fmt)
    if not dims or (dims[0] == settings.render_width and dims[1] == settings.render_height):
        return settings
    return settings.model_copy(update={"render_width": dims[0], "render_height": dims[1]})


def _settings_for_finish(settings: Settings, finish: FinishProfile | None) -> Settings:
    """Fold a request-level ``FinishProfile`` (VC-5) onto the render settings.

    Presence-is-opt-in: an absent profile — or one whose fields are all at their
    defaults — returns the settings UNCHANGED, so every filter string stays
    byte-identical to a pre-VC-5 compile. Same ``model_copy`` seam as
    ``_settings_for_format``: the renderer reads the colour off ``settings``, so
    a per-job finish needs no renderer-signature churn.

    ``letterbox_color`` is pattern-validated (``#RRGGBB``) at the schema, which is
    also the injection guard — it is interpolated into the ffmpeg pad ``color=``.
    """
    if finish is None or not finish.letterbox_color:
        return settings
    return settings.model_copy(update={"render_letterbox_color": finish.letterbox_color})


def _cleanup_job_dir(out_path: Path | None, settings: Settings) -> None:
    """Remove the per-job scratch dir that produced ``out_path``.

    Guarded on purpose. The renderer always returns a file inside a
    ``work_dir/job-*`` (or ``work_dir/audio-*``) scratch dir, so the dir to drop
    is ``out_path.parent`` — but a bare ``rmtree(out_path.parent)`` deletes
    whatever that parent happens to be. If a compile ever hands back a path
    sitting directly in the temp root (``/tmp/x.mp4``), the parent is ``/tmp``
    and the cleanup wipes the whole of ``/tmp``. That is not hypothetical: a
    test stub returned ``Path("/tmp/fake.mp4")`` and this line deleted ``/tmp``
    wholesale — including pytest's basetemp — on every Linux run.

    So: only ever remove a strict *subdirectory* of ``work_dir``. Anything else
    is refused and logged rather than deleted.
    """
    if out_path is None:
        return
    job_dir = out_path.parent.resolve()
    work_root = Path(settings.work_dir).resolve()
    if job_dir == work_root or work_root not in job_dir.parents:
        log.warning("compile.cleanup_refused_unsafe_dir", job_dir=str(job_dir))
        return
    shutil.rmtree(job_dir, ignore_errors=True)


_SHA256_CHUNK = 1 << 20  # 1 MiB — hash the mp4 in chunks, never load it whole.


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of ``path``, read in chunks (never loads the whole mp4)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_SHA256_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _compile_to_result(
    body: CompileRequest, settings: Settings, job_id: str = ""
) -> dict[str, Any]:
    # A 9x16 / 1x1 / 4x5 deliverable retargets the output canvas before render.
    settings = _settings_for_format(body.tags, settings)
    out_path: Path | None = None
    # Per-run resolution/quality (2K/4K, tighter CRF) without a redeploy: the
    # orchestrator threads these in compile_tags; env stays the default. A
    # top-level ``master_lufs`` on the request is injected into the tags dict so
    # it reuses the EXISTING [-30, -5] clamp path in ``_settings_for_tags`` (no
    # second code path); an explicit ``tags['master_lufs']`` wins if both are set.
    tags = dict(body.tags or {})
    if body.master_lufs is not None and "master_lufs" not in tags:
        tags["master_lufs"] = str(body.master_lufs)
    # Fold a structured MasteringProfile into the SAME tag-resolution clamp path
    # (mirrors how master_lufs is folded) so the picture master reuses one code
    # path; an explicit tag wins over the profile field.
    if body.mastering is not None:
        _profile = body.mastering
        _profile_tags = {
            "colorspace": _profile.colorspace,
            "deband": _profile.deband,
            "bitrate_mode": _profile.bitrate_mode,
            "maxrate": _profile.maxrate,
            "crf": _profile.crf,
            "platform": _profile.platform,
            "master_lufs": _profile.loudness_target,
            "master_tp": _profile.true_peak,
        }
        for _key, _val in _profile_tags.items():
            if _val is not None and _key not in tags:
                tags[_key] = str(_val)
    settings = _settings_for_tags(settings, tags)
    # Opt-in bit-deterministic encode (substrate for a cross-run frame_md5 E2E
    # check). Parsed off the SAME per-job tags as crf/format/render_width; NOT a
    # Settings field — it is a per-request encode-mode flag threaded straight
    # into compile_shots. Absent/falsey → the render stays byte-identical to prod.
    deterministic = _parse_truthy_tag(tags, "deterministic")
    # Opt-in freeze-pad cap (Step B): bound a footage beat's freeze-pad DRIFT at its
    # per-shot ``max_seconds`` ceiling instead of ramping over the whole over-length
    # narration. Same per-job tag surface as ``deterministic``; a shot with no
    # ``max_seconds`` is unaffected. Absent/falsey → byte-identical to prod.
    freeze_pad_cap = _parse_truthy_tag(tags, "freeze_pad_cap")
    # Finishing (VC-5): brand letterbox/pillarbox colour.
    # Absent profile → settings untouched → byte-identical render.
    settings = _settings_for_finish(settings, body.finish)
    # An explicit request-level cadence (VC-3) pins the whole compile onto that
    # rate: snap to the nearest legal cadence and switch off the auto selector so
    # compile_shots renders at exactly this rate.
    if body.cadence is not None:
        settings = settings.model_copy(
            update={
                "render_fps": cadence.clamp_cadence(body.cadence),
                "render_cadence_mode": "fixed",
            }
        )
    try:
        # Filled in place by compile_shots (out-parameter, not a return value —
        # see compile_shots' docstring): each shot's local narration source clip,
        # index-aligned to body.shots/shot_timeline. Threaded into post_mux_verify
        # below so narration_presence_report reuses the render's OWN download
        # instead of re-fetching a presigned URL that may have expired by the time
        # post-mux verification runs.
        narration_audio_paths: list[Path | None] = []
        with metrics.stage(metrics.Stage.COMPILE_SHOTS):
            (
                out_path,
                total_seconds,
                music_bed_report,
                shot_timeline,
                master_fps,
            ) = await compile_shots(
                list(body.shots),
                settings,
                music=body.music,
                ambient=list(body.ambient),
                finish=body.finish,
                captions=body.captions,
                captions_render=body.captions_render,
                caption_mood=body.caption_mood,
                captions_karaoke=body.captions_karaoke,
                caption_look=caption_look_from_request(
                    body.caption_font_family,
                    body.caption_text_colour,
                    body.caption_box_colour,
                    body.caption_box_opacity,
                ),
                # Compute per-shot boundaries whenever a downstream check could need
                # them: the request asked for the timeline in the response, OR deband
                # is active (so post_mux_verify can bucket per-shot banding), OR any
                # shot carries narration (so post_mux_verify's narration_presence_report
                # has a per-shot window to correlate each clip against). The flag gates
                # ONLY read-only ffprobe boundary reads, never the rendered join.
                emit_shot_timeline=(
                    body.emit_shot_timeline
                    or mastering.deband_enabled(settings)
                    or any(s.audio_url for s in body.shots)
                ),
                deterministic=deterministic,
                freeze_pad_cap=freeze_pad_cap,
                narration_audio_sink=narration_audio_paths,
            )
        # Report the ACTUAL encoded dimensions (not the requested settings) so the
        # caller / e2e check can confirm the 2K/4K tier truly landed in the file.
        with metrics.stage(metrics.Stage.PROBE):
            out_width, out_height = await probe_dimensions(out_path)
            # Report the ACTUAL encoded duration so the orchestrator can flag a
            # truncated / short-rendered file (declared total_seconds vs this).
            actual_seconds = await probe_output_duration(out_path)
        # Content hash of the delivered mux (audit / provenance — the manifest's
        # output_sha256). Hashed off the event loop; over the finalized file,
        # before upload + cleanup, so it identifies exactly what ships.
        with metrics.stage(metrics.Stage.HASH):
            output_sha256 = await asyncio.to_thread(_hash_file, out_path)
        # Read-only loudness/silence measurement of the delivered mux. Best-effort
        # observability — returns None on any failure, never fails the render.
        with metrics.stage(metrics.Stage.LOUDNESS):
            loudness = await final_mux_loudness(str(out_path))
        # Encode/delivery master gate over the delivered mux (VC-2, C11). REUSES
        # the loudness measured just above (no re-measure) and probes the
        # colorspace tag + banding + framerate coherence. Under the DEFAULT
        # advisory mode it never fails the render: a verification error degrades
        # to a gate_pass=False report. Under `mastering_gate_mode=enforce` a
        # cleared gate raises MasteringGateError (which names the offending
        # metric) and a verification that CRASHED fails CLOSED.
        # VERIFY AGAINST WHAT WAS ENCODED, not what was requested. ``compile_shots``
        # resolves the master cadence (VC-3 auto selector) and rebinds ``render_fps``
        # on its OWN copy of the settings; ours still holds the pre-resolution rate.
        # ``post_mux_verify``'s ``framerate_coherence`` compares the probed frame rate
        # to ``settings.render_fps`` within ±0.5, so passing our stale copy reports
        # ``cfr_ok=False`` for every job whose shots declared a ``target_fps`` the
        # selector honoured (the default ``render_cadence_mode='auto'``). Rebinding
        # from the RETURNED ``master_fps`` — not re-deriving the selector here — makes
        # the expectation match the file. A no-op when the selector kept render_fps.
        verify_settings = (
            settings
            if master_fps == settings.render_fps
            else settings.model_copy(update={"render_fps": master_fps})
        )
        try:
            with metrics.stage(metrics.Stage.MASTER_VERIFY):
                mastering_report = await mastering.post_mux_verify(
                    out_path,
                    verify_settings,
                    loudness=loudness,
                    shot_timeline=shot_timeline,
                    # ORIGINAL request shots for audio_url, index-aligned to
                    # shot_timeline the same way _build_shot_timeline zipped them.
                    shots=list(body.shots),
                    # The render's OWN per-shot narration downloads, same index
                    # alignment — REUSED here, never re-fetched. work_root (and
                    # therefore these files) is not removed until _cleanup_job_dir
                    # below, well after this call returns.
                    narration_audio_paths=narration_audio_paths,
                )
        except mastering.MasteringGateError:
            # Propagates to `_spawn._run`, which stamps the job terminal `failed`
            # with this message — the metric, its measured value and its ceiling —
            # instead of uploading an off-spec master. NOTE the upload below is the
            # very next stage, so raising HERE is what actually prevents delivery;
            # anything that only annotated `mastering_report` would still ship.
            #
            # Raised under `mastering_gate_mode=enforce` for any cleared gate, AND
            # in BOTH modes for an always-enforced stage (`av_coverage`: the picture
            # ends before the narration). Do not narrow this to enforce mode.
            raise
        except Exception as exc:  # ADVISORY mode: a soft-gate never fails the render
            log.warning("compile.post_mux_verify_failed", error=str(exc))
            if mastering.gate_armed(settings, "verify"):
                # FAIL CLOSED. A verification that crashed proved NOTHING about the
                # master, and degraded_report() makes it indistinguishable from a
                # check that ran and passed. An unverified master is a failed
                # render.
                #
                # ARMED BY DEFAULT (`mastering_gate_verify_enabled`), not only
                # under `mastering_gate_mode='enforce'` as this branch used to be:
                # "the report is absent" is a STRUCTURAL finding — it reports
                # ABSENCE, not a degree against some ceiling — so it needs no fleet
                # calibration to be trustworthy, and the mode it was gated behind
                # was never set by any deployment. Back it out with
                # MASTERING_GATE_VERIFY_ENABLED=false, never by narrowing this
                # branch again.
                raise mastering.MasteringGateError(
                    [
                        mastering.MasteringGateViolation(
                            stage="verify",
                            metric="post_mux_verify",
                            tier=mastering.gate_tier("verify"),
                            detail=f"post_mux_verify could not run: {exc}",
                        )
                    ]
                ) from exc
            mastering_report = mastering.degraded_report()
        # Which normalization mode the master pass actually ran in, folded into the
        # mastering report. Recorded by the renderer off ffmpeg's own pass-2 stderr;
        # it CANNOT be derived post-hoc from the delivered file, which is why it
        # rides a ContextVar out of the render rather than being probed here. Only
        # set when a master pass ran, so a mastering-off job keeps the report shape
        # it has today.
        normalization_type = get_master_normalization_type()
        if normalization_type is not None:
            mastering_report = {
                **mastering_report,
                "master_normalization_type": normalization_type,
            }
        # ...and the EVIDENCE that makes that mode readable: whether `linear` was
        # requested at all, plus pass 1's own readings, which are the inputs to
        # loudnorm's refusal decision. Same ContextVar channel and the same reason —
        # pass 1's measurement is thrown away inside the render and cannot be
        # recovered from the delivered file. Keys are `master_`-prefixed at the
        # renderer boundary rather than inside it so the report stays flat and the
        # names cannot be confused with the gate's OWN loudness fields
        # (`integrated_lufs` / `true_peak_dbtp` are measured on the DELIVERED mux;
        # these are measured on the pre-master bus).
        #
        # Only observed keys are folded in, so a mastering-off job keeps the report
        # shape it has today — the `is not None` guard above, generalised.
        mastering_report = {
            **mastering_report,
            **{f"master_{key}": value for key, value in get_master_loudnorm_evidence().items()},
        }
        # VC-W14/V4 — GATE ON THE DEGRADED MASTER, don't just publish it.
        #
        # ``master_measure_failed`` is the render's own report that pass-1 loudness
        # measurement CRASHED, so ``_master_filter`` was built with ``measured=None``
        # and the file shipped with single-pass DYNAMIC gain riding instead of the
        # requested two-pass linear master. It was recorded and published and had NO
        # GATING CONSUMER: the master degraded audibly and still returned
        # ``gate_pass=True`` with no note, indistinguishable in the report from a
        # clean one.
        #
        # Raised HERE and not inside ``post_mux_verify`` for the same reason the
        # ``verify`` gate is: the flag rides a ContextVar out of the render and the
        # verifier never sees it. Raising before the upload stage below is what
        # actually prevents delivery — annotating the report would still ship.
        # Armed by default; back out with MASTERING_GATE_MEASURE_ENABLED=false.
        measure_violation = mastering.measure_violation(
            measure_failed=bool(mastering_report.get("master_measure_failed")),
            settings=settings,
        )
        # VC-W14/V6 — the same shape for the bed mix: a job that ASKED for
        # VO-relative bed levelling/ducking and could not measure the narration
        # bus shipped with the bed gain AND the ducker's open threshold both on
        # literal fallbacks, reported as a log line and a telemetry boolean that
        # nothing read. Not probeable from the delivered file (the measurement
        # happened during the bed mix), so it is folded here like ``measure``.
        bed_violation = mastering.bed_calibration_violation(music_bed_report, settings)
        # FOLD FIRST, RAISE SECOND, and the order is the point: an UNARMED gate
        # must still clear `gate_pass` and appear in `violations`, or it is
        # telemetry again. Arming decides only whether delivery is also refused.
        mastering_report = mastering.apply_route_violations(
            mastering_report, (measure_violation, bed_violation), settings
        )
        run_id = _safe_run_id((body.tags or {}).get("run_id"))
        # Deterministic per-job key: a reclaimed re-run of the SAME job_id (crash recovery,
        # or a rare replicas>1 double-render) OVERWRITES its prior object (MinIO PUT is
        # atomic, last-writer-wins) instead of orphaning it under a random uuid. Falls back
        # to a unique id when no job_id is threaded (defensive; prod always passes it).
        object_key = f"{run_id}/{job_id or uuid7().hex}.mp4"
        with metrics.stage(metrics.Stage.UPLOAD):
            url = await upload_and_presign(
                settings,
                bucket=settings.minio_bucket,
                object_key=object_key,
                file_path=out_path,
                expiry_seconds=settings.minio_url_expiry_seconds,
            )
        # Throughput denominator. rate() over this gives finished-video-seconds per
        # wall-second, i.e. the pipeline's x-realtime. Counted off the PROBED duration
        # under kind="compile" — but ``probe_output_duration`` returns the 0.0 sentinel
        # for "unknown", and we still have to count SOMETHING, so an unprobed render
        # falls back to the declared sum under a SEPARATE kind="compile_unprobed"
        # series. Only kind="compile" is truncation-proof; the split keeps the
        # inflatable fallback out of it instead of pretending it was probed.
        probed_output = actual_seconds is not None and actual_seconds > 0
        with contextlib.suppress(Exception):
            metrics.compile_output_seconds_total.labels(
                kind="compile" if probed_output else "compile_unprobed"
            ).inc(actual_seconds if probed_output else (total_seconds or 0.0))
            metrics.compile_shots_total.inc(len(body.shots))
        log.info(
            "compile.done",
            object_key=object_key,
            bucket=settings.minio_bucket,
            duration_seconds=total_seconds,
        )
        return {
            "artifact_url": url,
            "duration_seconds": total_seconds,
            "actual_duration_seconds": actual_seconds,
            "output_sha256": output_sha256,
            "object_key": object_key,
            "bucket": settings.minio_bucket,
            "width": out_width,
            "height": out_height,
            "loudness": loudness,
            "mastering": mastering_report,
            "music_bed": music_bed_report,
            # Resolved whole-film grade knobs + whether a grade re-encode engaged
            # (VC-5). None when the request carried no finish.grade.
            "grade": grade_report(body.finish.grade if body.finish else None),
            # Authoritative per-shot [start,end) on the final mux, keyed by line_id.
            # None unless the request set emit_shot_timeline — the boundaries may
            # also be computed to feed per-shot banding when deband is active, but
            # the RESPONSE field stays gated on the request flag ONLY (contract).
            "shot_timeline": shot_timeline if body.emit_shot_timeline else None,
        }
    finally:
        _cleanup_job_dir(out_path, settings)


# Content-Type for the presigned audiobook artifact, keyed by output_format.
_AUDIOBOOK_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4b": "audio/mp4",
}


def _declared_clip_seconds(body: AudioConcatRequest) -> float | None:
    """The request's DECLARED total audio length, or ``None`` when it is incomplete.

    The nominal billing fallback for the audiobook path, used only when the delivered
    master could not be probed (see ``_billed_duration``). ``AudioClip.seconds`` is
    advisory and optional, so an all-or-nothing sum is deliberate: a partial sum is not
    a smaller measurement of the same thing, it is a silent under-report of however many
    clips declared nothing, and under-reporting is the failure this whole path is being
    fixed for.

    Inter-clip ``gap_ms`` silence is NOT added. The delivered master is at least this
    long, so the fallback errs low — the safe direction, because this system has no
    refund path and an over-charge is permanent.
    """
    seconds = [c.seconds for c in body.clips]
    if not seconds or any(s is None for s in seconds):
        return None
    return sum(s for s in seconds if s is not None)


async def _concat_to_result(
    body: AudioConcatRequest, settings: Settings, job_id: str = ""
) -> dict[str, Any]:
    out_path: Path | None = None
    # Opt-in bit-deterministic stitch, parsed off the SAME per-job tags as the
    # compile route's `deterministic`. Absent/falsey → command byte-identical to prod.
    tags = body.tags or {}
    deterministic = _parse_truthy_tag(tags, "deterministic")
    try:
        acx = body.acx_master
        out_path, total_seconds, acx_report = await concat_audio(
            list(body.clips),
            settings,
            body.output_format,
            body.gap_ms,
            deterministic=deterministic,
            acx_master=bool(acx and acx.enabled),
            acx_lufs_target=acx.lufs_target if acx else None,
            acx_tp=acx.tp if acx else None,
            acx_lra=acx.lra if acx else None,
            chapters=body.chapters,
        )
        content_type = _AUDIOBOOK_CONTENT_TYPES.get(body.output_format, "audio/mpeg")
        run_id = _safe_run_id((body.tags or {}).get("run_id"))
        # Deterministic per-job key (see _compile_to_result): a re-run overwrites, not orphans.
        object_key = f"{run_id}/{job_id or uuid7().hex}.{body.output_format}"
        url = await upload_and_presign(
            settings,
            bucket=settings.minio_audio_bucket,
            object_key=object_key,
            file_path=out_path,
            expiry_seconds=settings.minio_url_expiry_seconds,
            content_type=content_type,
        )
        log.info(
            "concat_audio.done",
            object_key=object_key,
            bucket=settings.minio_audio_bucket,
            duration_seconds=total_seconds,
            output_format=body.output_format,
            acx_pass=(acx_report or {}).get("pass"),
        )
        return {
            "artifact_url": url,
            "duration_seconds": total_seconds,
            # Both internal-only: the poll response marshals named fields, so neither
            # reaches the wire.
            #
            # ``duration_probed`` -- ``total_seconds`` is ``_ffprobe_duration`` of the
            # delivered master, NOT a declared sum, so billing must read it as a
            # measurement. False means the probe carried no duration (ffprobe's 0.0 is
            # documented as "not carried", see ``renderer._duration_field``) -- an
            # UNMEASURED delivery, which must not be mistaken for zero work.
            #
            # ``declared_duration_seconds`` -- the nominal billing fallback the audio
            # path lacked entirely, mirroring the compile path's shot-duration sum.
            # Only set when EVERY clip declares its length: ``AudioClip.seconds`` is
            # advisory and optional, so a partial sum would under-report a delivery by
            # however many clips stayed silent about themselves, and quietly.
            "duration_probed": total_seconds > 0,
            "declared_duration_seconds": _declared_clip_seconds(body),
            "object_key": object_key,
            "bucket": settings.minio_audio_bucket,
            "acx_report": acx_report,
        }
    finally:
        _cleanup_job_dir(out_path, settings)


def _guard_local_file_inputs(settings: Settings, payload: Mapping[str, object]) -> None:
    """Reject a submit whose shot/clip inputs are file:// or bare local paths.

    A ``file:///etc/passwd`` image_url/audio_url is read straight off the pod's
    filesystem into the render (LFI). Gated by ``allow_local_file_inputs`` (default
    FALSE so prod is armed without a gitops env; tests/local dev that use ``file://``
    fixtures opt in via ``ALLOW_LOCAL_FILE_INPUTS=true``). Reuses
    ``inputs_reconstructable``, which already flags any non-http(s) URL, so the guard
    is exactly the "cross-pod reachable" predicate.
    """
    if settings.allow_local_file_inputs:
        return
    ok, reason = inputs_reconstructable(payload)
    if not ok:
        raise HTTPException(status_code=400, detail=f"local-file input rejected: {reason}")


@router.post(
    "/api/compile",
    status_code=202,
    response_model=JobAccepted,
    operation_id="compile_video",
    summary="Submit a shot-list compile job; poll /api/jobs/{job_id} for the MP4.",
)
async def submit_compile(request: Request, body: CompileRequest) -> JobAccepted:
    settings = get_settings()
    _guard_local_file_inputs(settings, body.model_dump())
    work_timeout = float(settings.job_work_timeout_seconds)
    tenant_id = _caller_tenant(request)
    job_options = build_job_options(
        kind=KIND_COMPILE,
        request=body,
        tenant_id=tenant_id,
        work_timeout=settings.job_work_timeout_seconds,
    )
    accepted = await _accept(
        request, settings, kind="compile", tenant_id=tenant_id, job_options=job_options
    )
    if accepted.action == "dedup":
        log.info("compile.deduped", job_id=accepted.job_id)
        return JobAccepted(job_id=accepted.job_id, poll_url=f"/api/jobs/{accepted.job_id}")
    log.info(
        "compile.accepted",
        job_id=accepted.job_id,
        action=accepted.action,
        n_shots=len(body.shots),
        total_seconds=sum(s.seconds for s in body.shots),
        tags=body.tags or {},
    )
    _spawn(
        accepted.job_id,
        lambda: _compile_to_result(body, settings, accepted.job_id),
        tenant_id=tenant_id,
        work_timeout=work_timeout,
        job_type="compile",
        lease_seconds=settings.job_lease_seconds,
        instance_id=_instance_id(request),
    )
    return JobAccepted(job_id=accepted.job_id, poll_url=f"/api/jobs/{accepted.job_id}")


@router.post(
    "/api/concat-audio",
    status_code=202,
    response_model=JobAccepted,
    operation_id="concat_audio",
    summary="Submit an audio-stitch job; poll /api/jobs/{job_id} for the WAV/MP3.",
)
async def submit_concat_audio(request: Request, body: AudioConcatRequest) -> JobAccepted:
    settings = get_settings()
    _guard_local_file_inputs(settings, body.model_dump())
    work_timeout = float(settings.job_work_timeout_seconds)
    tenant_id = _caller_tenant(request)
    job_options = build_job_options(
        kind=KIND_CONCAT,
        request=body,
        tenant_id=tenant_id,
        work_timeout=settings.job_work_timeout_seconds,
    )
    accepted = await _accept(
        request, settings, kind="concat", tenant_id=tenant_id, job_options=job_options
    )
    if accepted.action == "dedup":
        log.info("concat_audio.deduped", job_id=accepted.job_id)
        return JobAccepted(job_id=accepted.job_id, poll_url=f"/api/jobs/{accepted.job_id}")
    log.info(
        "concat_audio.accepted",
        job_id=accepted.job_id,
        action=accepted.action,
        n_clips=len(body.clips),
        output_format=body.output_format,
        gap_ms=body.gap_ms,
        tags=body.tags or {},
    )
    _spawn(
        accepted.job_id,
        lambda: _concat_to_result(body, settings, accepted.job_id),
        tenant_id=tenant_id,
        work_timeout=work_timeout,
        job_type="concat_audio",
        lease_seconds=settings.job_lease_seconds,
        instance_id=_instance_id(request),
    )
    return JobAccepted(job_id=accepted.job_id, poll_url=f"/api/jobs/{accepted.job_id}")


async def redispatch_reclaimed(
    claimed: ClaimedJob, *, lease_seconds: int, instance_id: str
) -> None:
    """Re-run a crash-orphaned job the reaper claimed, from its persisted job_options.

    The row is already ``processing`` (the reclaim flipped it + stamped this pod + a
    fresh lease). We rebind the ORIGINAL caller's identity (so the terminal write binds
    the right tenant under RLS and any cost event bills the right license/subject),
    reconstruct the request, and spawn the render under the lease — reusing the SAME
    ``_spawn`` terminal-write + cost-gating path as a live submit.

    Fail-fast, never loop: if the inputs can't be reconstructed on this pod (a
    ``file://``/bare-path input) or the payload no longer validates, the job is stamped
    ``failed`` with a clear reason. A ``failed`` row is terminal and not reclaimable, so
    a non-reconstructable job dies after ONE reclaim cycle instead of spinning.

    Self-reclaim guard: if THIS pod is already rendering ``claimed.job_id`` (its lease
    lapsed under a live render — e.g. a hung heartbeat renew — and the same-pod reaper
    reclaimed it), do NOT start a second render. The running task still owns the terminal
    write + its heartbeat will re-extend the lease; a second dispatch here would be a pure
    double-render of paid compute (the money guard dedups the BILL, not the work).
    """
    if claimed.job_id in _active_job_ids:
        log.warning("compile.reclaim_skip_live", job_id=claimed.job_id, tenant_id=claimed.tenant_id)
        return
    options = claimed.options
    kind = str(options.get("kind") or "")
    tenant_id = claimed.tenant_id
    request_payload = options.get("request")
    work_timeout = float(cast(int, options.get("work_timeout") or 0)) or float(
        get_settings().job_work_timeout_seconds
    )
    settings = get_settings()

    # Bind the original caller's identity for the whole re-dispatch (the terminal write's
    # tenant GUC + any cost event). Cleared in the finally so it can't bleed into the
    # reaper's next claim.
    #
    # ⚠️ tenant_source comes from the PERSISTED snapshot, never from the ContextVar.
    # This runs on the reaper's task, which claims rows across tenants, so the ambient
    # value belongs to whichever tenant was reclaimed last — reading it back here would
    # vouch one tenant's provenance for another tenant's job. That is the opposite of
    # the request-dispatched `_bg` case, where the inherited value IS this request's and
    # must be preserved. Restoring the enqueue-time value is what makes this a re-bind of
    # the ORIGINAL caller rather than a re-derive: without it the re-dispatch carried a
    # license and a subject but no provenance, and every metered hop failed closed as
    # 'unrecorded' even though the identity looked complete.
    set_identity_context(
        str(tenant_id),
        str(options.get("license_id") or "") or None,
        str(options.get("subject_id") or "") or None,
        tenant_source=restore_tenant_source(options.get("tenant_source")),
    )
    # The ORIGINAL caller's actor + project/job, so the re-run's cost event and outbound
    # calls carry them rather than this service's own SERVICE_ACTOR_ID.
    bind_job_attribution(options)
    try:
        if not isinstance(request_payload, dict):
            await _stamp_failed(
                claimed.job_id, "reclaim: job_options carried no request payload", instance_id
            )
            return
        try:
            ok, reason = inputs_reconstructable(cast("Mapping[str, object]", request_payload))
        except Exception as exc:  # noqa: BLE001 — a classifier error is terminal, not a silent attempt-burn
            # Without this, a raise (not a False return) propagates past the claim with no
            # terminal write, so the row waits a full lease then re-reclaims — burning
            # attempts toward the poison cap on a payload we can never re-render anyway.
            log.warning("compile.reclaim_reconstruct_error", job_id=claimed.job_id, error=str(exc))
            await _stamp_failed(claimed.job_id, f"reclaim: input check failed: {exc}", instance_id)
            return
        if not ok:
            log.warning("compile.reclaim_unreconstructable", job_id=claimed.job_id, reason=reason)
            await _stamp_failed(claimed.job_id, f"reclaim: {reason}", instance_id)
            return
        try:
            if kind == KIND_CONCAT:
                concat_body = AudioConcatRequest.model_validate(request_payload)
                work: Callable[[], Awaitable[dict[str, Any]]] = lambda: _concat_to_result(  # noqa: E731
                    concat_body, settings, claimed.job_id
                )
                job_type = "concat_audio"
            else:
                compile_body = CompileRequest.model_validate(request_payload)
                work = lambda: _compile_to_result(compile_body, settings, claimed.job_id)  # noqa: E731
                job_type = "compile"
        except Exception as exc:  # noqa: BLE001 — a payload that no longer validates is terminal
            log.warning("compile.reclaim_invalid_payload", job_id=claimed.job_id, error=str(exc))
            await _stamp_failed(
                claimed.job_id, f"reclaim: invalid job_options payload: {exc}", instance_id
            )
            return
        _spawn(
            claimed.job_id,
            work,
            tenant_id=tenant_id,
            work_timeout=work_timeout,
            job_type=job_type,
            lease_seconds=lease_seconds,
            instance_id=instance_id,
        )
    finally:
        clear_identity_context()


@router.post(
    "/api/jobs/cancel",
    response_model=JobCancelResponse,
    operation_id="cancel_compile_jobs",
    summary="Cancel this tenant's live compile jobs whose Idempotency-Key has a prefix.",
)
async def cancel_jobs(request: Request, body: JobCancelRequest) -> JobCancelResponse:
    """Stop paying for a render whose caller no longer wants it.

    Rendering is the most expensive stage in the product, and until this endpoint existed
    a cancelled run stopped the ORCHESTRATOR only: the compile kept running to completion
    and the money was spent anyway. There is no reverse channel from the orchestrator's
    cancel into the compiler other than this.

    Two halves, in this order and no other:

    1. the DURABLE flip (``cancel_by_key_prefix``) — terminal ``cancelled`` on every live
       row of THIS tenant under the prefix. This alone already stops the spend on a job
       that is merely QUEUED/``interrupted`` (nothing re-runs it: it is non-rerunnable, so
       a resend dedups, and the lease is NULLed so no reaper reclaims it);
    2. the LOCAL kill — cancel the in-flight render Task, whose ``_run`` propagates
       ``CancelledError`` down into ``renderer._run``'s ``proc.kill()``, so the ffmpeg
       child dies rather than being abandoned.

    Flip-then-kill is the DETERMINISTIC order, not merely the tidy one. The killed task's
    handler stamps ``interrupted`` — a re-runnable state that would undo the cancel — and
    flipping first makes that stamp a 0-row no-op (it is guarded on
    ``status='processing'``). Kill-then-flip happens to converge too, because ``interrupted``
    is itself cancellable, but only by racing two writes; there is no reason to depend on
    that. Both guards are what make the convergence total, so neither may be relaxed: see
    ``_CANCELLABLE`` and ``_owner_conds``.

    A job owned by a SIBLING replica has no local task here; step 1 still lands, and that
    pod's lease heartbeat sees the renew match 0 rows and aborts its own render within one
    beat (see ``durable_dispatch._heartbeat``). So the reported ``killed`` is a lower
    bound on renders stopped, never a measure of success.
    """
    tenant_id = _caller_tenant(request)
    async with get_db_session() as session:
        job_ids = await job_store.cancel_by_key_prefix(
            session, tenant_id=tenant_id, key_prefix=body.idempotency_key_prefix
        )
    killed = 0
    for job_id in job_ids:
        task = _job_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            killed += 1
    log.info(
        "compile.cancelled",
        prefix=body.idempotency_key_prefix,
        n_cancelled=len(job_ids),
        n_killed=killed,
    )
    return JobCancelResponse(cancelled=job_ids, killed=killed)


@router.get(
    "/api/jobs/{job_id}",
    response_model=JobStatus,
    operation_id="get_compile_job",
    summary="Poll a compile / concat-audio job until it reaches a terminal status.",
)
async def get_job(request: Request, job_id: str) -> JobStatus:
    async with get_db_session() as session:
        job = await job_store.get(session, job_id, _caller_tenant(request))
    # None = missing OR cross-tenant (the store scopes by tenant): 404 (not 403)
    # so a job's existence isn't leaked to a tenant that doesn't own it.
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    # 'interrupted' (a restart orphan awaiting re-run on the next resend) is
    # reported as 'processing' on the wire so the 3-value poll contract is
    # unchanged and the caller keeps polling until its stage timeout re-dispatches.
    # 'cancelled' passes through UNMAPPED: it is terminal and must stop the caller's poll
    # loop. Mapping it onto 'processing' (as 'interrupted' is) would hang the poller until
    # its stage timeout, and mapping it onto 'failed' would report a render fault where
    # there was none.
    wire_status = cast(
        Literal["processing", "completed", "failed", "cancelled"],
        "processing" if job.status == "interrupted" else job.status,
    )
    result: dict[str, Any] = job.result or {}
    return JobStatus(
        job_id=job_id,
        status=wire_status,
        artifact_url=result.get("artifact_url"),
        duration_seconds=result.get("duration_seconds"),
        actual_duration_seconds=result.get("actual_duration_seconds"),
        output_sha256=result.get("output_sha256"),
        object_key=result.get("object_key"),
        bucket=result.get("bucket"),
        width=result.get("width"),
        height=result.get("height"),
        loudness=result.get("loudness"),
        acx_report=result.get("acx_report"),
        mastering=result.get("mastering"),
        music_bed=result.get("music_bed"),
        grade=result.get("grade"),
        shot_timeline=result.get("shot_timeline"),
        # An 'interrupted' job is reported 'processing' (a resend re-runs it), so
        # don't surface its internal cancel note as if it were a terminal error.
        error=None if wire_status == "processing" else job.error,
    )
