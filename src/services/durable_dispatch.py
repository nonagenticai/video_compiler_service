"""Durable crash-resumption helpers (job-options bundle + lease heartbeat).

The render is a fire-and-forget ``asyncio`` task that dies with its pod, leaving
the ``compile_jobs`` row stuck ``processing``. This module is the durability layer
shared by the submit path and the reaper:

* :func:`build_job_options` — the JSONB bundle persisted on the row so a DIFFERENT
  pod (the reaper) can reconstruct + re-run the render after a crash.
* :func:`inputs_reconstructable` — a render is only reproducible on another pod when
  every input is an http(s) URL. A ``file://`` / bare-path input is pod-affine; the
  reaper fail-fasts such a job instead of rendering a broken compile.
* :func:`run_with_lease` — wraps the render coroutine with a lease heartbeat so the
  reaper never reclaims a job still actively rendering on a live pod.

The lease itself is stamped by ``compile_job_store.accept`` at submit and by the
SECURITY DEFINER reclaim function on reclaim; this module only RENEWS it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from w2v_common.auth import (
    ACTOR_ID_HEADER,
    PROJECT_ID_HEADER,
    W2V_JOB_ID_HEADER,
    W2V_PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    get_project_id_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
)
from w2v_common.auth.identity_context import (
    get_license_id_from_context,
    get_subject_id_from_context,
    get_tenant_source_from_context,
    mark_inbound_request,
    resolve_actor_id,
)
from w2v_common.tenant import TenantSource

from src.database.session import get_db_session
from src.services import compile_job_store as job_store

if TYPE_CHECKING:
    from src.api.schemas import AudioConcatRequest, CompileRequest

logger = logging.getLogger(__name__)

# Floor on the lease-heartbeat cadence: a tiny configured lease must not turn the beat
# into a DB hammer. It also bounds how fast a cancel propagates to a SIBLING replica's
# render (that pod learns its lease is gone on the next beat), so it is the cancel
# latency ceiling as much as a rate limit.
_MIN_BEAT_SECONDS = 5.0

# Dispatch kinds — the reaper routes a reclaimed job by this tag.
KIND_COMPILE = "compile"
KIND_CONCAT = "concat"

# Shot / bed / clip fields that carry a FETCHED input the renderer downloads.
# A crash re-dispatch can only resolve these on another pod when they are http(s).
# Only fields the renderer actually fetches belong here: a URL-shaped field that is
# never fetched (e.g. MasteringProfile.grade.lut_url — LUT application is not performed
# by this service) is deliberately excluded, so a pod-affine value there can't wrongly
# fail-fast an otherwise-reconstructable job.
_INPUT_URL_KEYS = (
    "video_url",
    "image_url",
    "audio_url",
    "url",
    "second_source_url",
)


def build_job_options(
    *,
    kind: str,
    request: CompileRequest | AudioConcatRequest,
    tenant_id: uuid.UUID,
    work_timeout: int,
) -> dict[str, object]:
    """Serialize everything a different pod needs to re-run this render after a crash.

    ``request`` is dumped in JSON mode (URLs/enums → strings) so it round-trips through
    JSONB and back into the request model. Identity (license/subject) is read from the
    live request context and persisted so the reaper can re-bind it — the cost event
    and the RLS tenant GUC on the re-run must carry the ORIGINAL caller's identity.

    ``tenant_source`` is part of that identity, not bookkeeping alongside it. The
    re-dispatch makes METERED outbound calls, and ``build_outbound_identity`` refuses
    to delegate a tenant whose provenance is ``None`` ("unrecorded"). Persisting
    license and subject but not provenance therefore restored an identity that cannot
    actually be used. Stored as the enum's ``.value`` so it round-trips through JSONB;
    ``""`` when the enqueue had no provenance, which reads back as ``None``.
    """
    context_source = get_tenant_source_from_context()
    return {
        "kind": kind,
        "request": request.model_dump(mode="json"),
        "tenant_id": str(tenant_id),  # real registered tenant; never a sentinel
        "license_id": get_license_id_from_context(),
        "subject_id": get_subject_id_from_context(),
        "tenant_source": context_source.value if context_source is not None else "",
        "work_timeout": work_timeout,
        # ATTRIBUTION (w2v-common 0.88). A reaper re-dispatch runs outside the request,
        # where w2v_common would treat it as SERVICE-originated and stamp this service's
        # own SERVICE_ACTOR_ID. It is caller-originated, so the caller's actor
        # (canonical UUID, "" when the caller sent none) and the project/job it worked
        # for are persisted here and re-bound by :func:`bind_job_attribution`.
        "actor_id": resolve_actor_id()[0] or "",
        "project_id": get_project_id_from_context(),
        "w2v_project_id": get_w2v_project_id_from_context(),
        "w2v_job_id": get_w2v_job_id_from_context(),
    }


def bind_job_attribution(options: Mapping[str, object]) -> None:
    """Re-bind the enqueue-time actor / project / job for a re-dispatched job.

    Call AFTER ``set_identity_context`` (which clears the actor and project).

    A persisted job is CALLER-originated work, so the context is marked inbound: a job
    carrying a stored actor bills that actor; a job WITHOUT one (a legacy row persisted
    before this snapshot existed, or a caller that sent no actor) stays actor-less and
    is counted by w2v_common as ``inbound_without_actor``. It must NOT fall back to this
    service's SERVICE_ACTOR_ID — that would hide the caller's gap from billing coverage.

    Values go through w2v_common's public ``bind_attribution_from_headers`` (trusted:
    they were captured from this service's own validated request context), so a
    malformed persisted value is ignored, never coerced.
    """
    mark_inbound_request()
    headers = {
        ACTOR_ID_HEADER: str(options.get("actor_id") or ""),
        PROJECT_ID_HEADER: str(options.get("project_id") or ""),
        W2V_PROJECT_ID_HEADER: str(options.get("w2v_project_id") or ""),
        W2V_JOB_ID_HEADER: str(options.get("w2v_job_id") or ""),
    }
    bind_attribution_from_headers({k: v for k, v in headers.items() if v}, trusted=True)
    if not headers[ACTOR_ID_HEADER]:
        logger.info(
            "Re-dispatched job has no stored caller actor; dispatching actor-less "
            "(counted as inbound_without_actor, no SERVICE_ACTOR_ID fallback)"
        )


def restore_tenant_source(value: object) -> TenantSource | None:
    """Re-hydrate the enqueue-time ``TenantSource`` from persisted job options.

    Returns ``None`` — never a guessed default — when the job predates this snapshot
    or carries an unrecognised value. ``None`` is the honest answer and it fails
    CLOSED: the delegation gate refuses an unvouched tenant, so a pre-existing job
    raises at its first metered call instead of silently spending an arbitrary
    tenant's credentials on provenance this process cannot vouch for. Substituting
    ``LICENSE_TOKEN`` here would be forging the vouch.

    That is also why this change needs no migration: in-flight rows have no
    ``tenant_source`` key, read back as ``None``, and behave exactly as they do today.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return TenantSource(value)
    except ValueError:
        # TenantSource is a plain StrEnum, so this is genuinely "unknown member",
        # not a swallowed tenant refusal. Degrade to unvouched rather than raise —
        # the delegation gate downstream is the thing that must refuse, and it will.
        logger.warning("Unrecognised persisted tenant_source %r; treating as unvouched", value)
        return None


def _collect_url_key_value(value: object, urls: list[str]) -> None:
    """Harvest the value sitting under a URL key: a string, or a list of them.

    Anything else is handed back to :func:`_walk_for_urls` so a nested object under a
    URL key keeps being explored exactly as before.
    """
    if isinstance(value, str):
        if value:
            urls.append(value)
    elif isinstance(value, list):
        for item in value:
            _collect_url_key_value(item, urls)
    else:
        _walk_for_urls(value, urls)


def _walk_for_urls(node: object, urls: list[str]) -> None:
    """Depth-first walk appending every input URL string reachable from ``node``."""
    if isinstance(node, Mapping):
        for key, val in node.items():
            if key in _INPUT_URL_KEYS:
                _collect_url_key_value(val, urls)
            else:
                _walk_for_urls(val, urls)
    elif isinstance(node, list):
        for item in node:
            _walk_for_urls(item, urls)


def _url_values(payload: Mapping[str, object]) -> list[str]:
    """Every input URL string reachable in a compile/concat request payload.

    A URL key may hold a single string OR a **list** of them (a multi-source field);
    both shapes are collected, and a list mixing strings with nested objects yields the
    strings AND the walk of the objects. The bias is deliberate: over-collecting a URL
    is harmless (it is only pattern-checked), while MISSING one would let a pod-affine
    ``file://`` input pass :func:`inputs_reconstructable` and be re-run on a pod where
    the file does not exist.
    """
    urls: list[str] = []
    _walk_for_urls(payload, urls)
    return urls


def inputs_reconstructable(request_payload: Mapping[str, object]) -> tuple[bool, str]:
    """Whether a crash re-dispatch on ANOTHER pod could resolve every input.

    Only http(s) inputs are cross-pod reachable. A ``file://`` or bare local path is
    pod-affine (the file lives in the dead pod's scratch), so re-running elsewhere would
    fetch nothing. Returns ``(False, reason)`` naming the first offending input so the
    reaper can fail the job with a clear error rather than render a broken compile.
    (A presigned http(s) URL that has since EXPIRED is not caught here — it surfaces as a
    normal fetch failure during the re-run and terminates the job as ``failed``; either
    way the job never loops.)
    """
    for url in _url_values(request_payload):
        if not url.startswith(("http://", "https://")):
            return False, f"input not cross-pod reconstructable: {url[:80]}"
    return True, ""


async def run_with_lease[T](
    *,
    job_id: str,
    instance_id: str,
    lease_seconds: int,
    work: Awaitable[T],
    heartbeat_enabled: bool = True,
) -> T:
    """Run ``work`` while heartbeating the job's lease; return ``work``'s result.

    Keeps the lease fresh for the whole (possibly multi-minute) render so a reaper on
    another pod does not reclaim a job still running here. If THIS pod dies the heartbeat
    stops, the lease lapses, and the reaper takes over.
    """
    if not heartbeat_enabled or lease_seconds <= 0:
        return await work
    stop = asyncio.Event()
    # ``work`` becomes a Task so the heartbeat has something to cancel: a lease renew that
    # matches 0 rows means this render no longer owns its job (cancelled by a caller, or
    # reclaimed), and continuing to burn ffmpeg on it is the money bug. Awaiting the task
    # keeps the OUTER cancellation path (SIGTERM drain) intact — asyncio propagates a
    # cancel through to the awaited future — and the explicit forward below makes that
    # guarantee independent of that implementation detail.
    work_task = asyncio.ensure_future(work)
    hb = asyncio.create_task(
        _heartbeat(job_id, instance_id, lease_seconds, stop, on_lease_lost=work_task.cancel)
    )
    try:
        return await work_task
    except asyncio.CancelledError:
        work_task.cancel()
        with contextlib.suppress(BaseException):
            await work_task
        raise
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await hb


async def _heartbeat(
    job_id: str,
    instance_id: str,
    lease_seconds: int,
    stop: asyncio.Event,
    *,
    interval: float | None = None,
    on_lease_lost: Callable[[], object] | None = None,
) -> None:
    """Renew ``job_id``'s lease every ``lease_seconds/3`` until ``stop`` is set.

    The renew itself is bounded by ``asyncio.wait_for``: a lease renew that HANGS (a
    connection stuck acquiring from a saturated pgbouncer, an UPDATE that never returns)
    would otherwise wedge this loop indefinitely, letting the lease lapse under a healthy
    render while no beat ever fires — and the same-pod reaper would then reclaim a live
    job. Capping each renew well inside the beat interval means a stuck renew is abandoned
    and the NEXT beat retries, so only a genuine multi-beat DB outage can lapse the lease.

    ``on_lease_lost`` is invoked (once, then the loop returns) when a renew comes back
    ``False`` — a DEFINITIVE "this row is no longer ours": a caller cancelled the job, it
    was terminalized, or another pod reclaimed it. That is the CROSS-REPLICA cancel path:
    a cancel request served by a replica that does not own the render can only flip the
    row, and this beat is what turns that flip into a killed ffmpeg on the owning pod,
    within one beat (``lease_seconds/3``). A renew that ERRORS is not this signal — the
    row's ownership is simply unknown, so the render keeps going and the next beat retries.

    ``interval`` overrides the beat cadence (seconds) for tests; production derives it from
    the lease.
    """
    beat = interval if interval is not None else max(_MIN_BEAT_SECONDS, lease_seconds / 3)
    renew_timeout = beat / 2
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=beat)
            return  # stop set → render finished; the terminal write owns the lease clear
        except TimeoutError:
            pass
        try:
            held = await asyncio.wait_for(
                _renew_once(job_id, instance_id, lease_seconds), renew_timeout
            )
        except Exception:  # best-effort; a missed/slow beat just shortens the lease
            logger.warning("Lease renew failed for job %s", job_id, exc_info=True)
            continue
        if not held and on_lease_lost is not None:
            logger.warning(
                "Lease lost for job %s (cancelled or reclaimed) — aborting the render", job_id
            )
            on_lease_lost()
            return


async def _renew_once(job_id: str, instance_id: str, lease_seconds: int) -> bool:
    async with get_db_session() as db:
        return await job_store.renew_lease(db, job_id, instance_id, lease_seconds)
