"""ASGI entrypoint for video_compiler_service.

Built on the shared ``w2v_common.asgi.create_app`` factory, which wires the
fleet-canonical middleware stack (RequestId → Auth → SecurityHeaders + CORS),
the license-token BearerAuth/TenantIdHeader OpenAPI customization, and the
canonical error envelope. This module only supplies the service-specific bits:
the two routers, the metering-bus lifespan, structlog config, the health-log
filter, and the manual Prometheus ``/metrics`` mount.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import structlog
from prometheus_client import REGISTRY, make_asgi_app
from prometheus_client.exposition import choose_encoder
from starlette.requests import Request
from starlette.responses import Response
from w2v_common.asgi import create_app
from w2v_common.observability.request_context import RequestIdLogFilter

from src.api.routes import compile as compile_routes
from src.api.routes import health as health_routes
from src.api.routes import thumbnail as thumbnail_routes
from src.config import get_settings
from src.services.renderer import configure_ffmpeg_concurrency

if TYPE_CHECKING:
    from fastapi import FastAPI


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress access log noise from health check endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


logging.getLogger("uvicorn.access").addFilter(HealthCheckLoggingFilter())


logger = logging.getLogger(__name__)


async def _startup(app: FastAPI) -> None:
    """Init the durable job store, reclaim restart orphans, connect metering.

    The compile job store + Idempotency-Key dedup index now live in Postgres, so a
    resend after a pod restart dedupes instead of firing a second PAID render.
    ``replicas: 1`` + ``Recreate`` guarantees the prior pod is gone, so every
    ``processing`` row not owned by THIS pod is a restart orphan whose in-process
    render task died — flip it to ``interrupted`` so the next same-key submit
    re-runs it under the same job_id. A missing/unreachable ``METERING_REDIS_URL``
    just disables metering (no-op emits), never fails boot.

    ⚠️ The ``replicas: 1`` + ``Recreate`` premise above is STALE as a statement
    about infra: the live deployment is ``replicas=2`` + ``RollingUpdate``
    (verified 2026-08-21). The reclaim is nevertheless safe, because it does not
    rest on that premise — migration ``005_lease_aware_startup_reclaim`` made it
    LEASE-gated, so a booting pod only flips rows whose lease has ALSO lapsed and
    cannot interrupt a live sibling's in-flight render. Read the reclaim SQL in
    ``src/database/rls_resolvers.py`` for the actual contract; do not re-derive a
    "prior pod is provably gone" argument from this docstring.
    """
    import uuid

    from w2v_common.database.rls import install_rls_tenant_rebind

    from src.database.connection import get_engine
    from src.database.session import get_db_session
    from src.services import compile_job_store
    from src.services.metering import init_metering_redis

    settings = get_settings()

    # Install the tenant-GUC rebind BEFORE any session opens; warm the engine; mint
    # this pod's boot id (stamped on the rows it renders so reclaim can tell its own
    # in-flight jobs from a dead pod's orphans).
    install_rls_tenant_rebind()
    app.state.db_engine = await get_engine()
    instance_id = uuid.uuid7().hex
    app.state.instance_id = instance_id

    async with get_db_session() as session:
        reclaimed = await compile_job_store.reclaim_orphans(session, instance_id)
        # Cross-tenant housekeeping once per boot (SECURITY DEFINER): flip any
        # live-pod zombie + evict terminal rows past the RETENTION TTL. Pods restart
        # on every deploy, so this is the store's periodic-cleanup cadence. The TTL is
        # compile_job_retention_seconds (#153) — the Idempotency-Key dedup horizon — NOT the
        # presigned artifact-URL expiry: evicting a completed row deletes its dedup entry,
        # so a same-key resend after that point re-renders and RE-BILLS.
        pruned = await compile_job_store.prune(
            session,
            ttl_seconds=settings.compile_job_retention_seconds,
            work_timeout_seconds=settings.job_work_timeout_seconds,
        )
    if reclaimed:
        logger.warning("compile startup reclaimed %d orphaned processing job(s)", reclaimed)
    if pruned:
        logger.info("compile startup pruned %d stale job row(s)", pruned)

    metering_redis = await init_metering_redis(settings.metering_redis_url)
    compile_routes.set_metering_redis(metering_redis)
    app.state.metering_redis = metering_redis

    # Start the crash-resumption reaper: it reclaims orphaned jobs (a render whose pod
    # died) and re-runs them from job_options, and drives periodic prune. The startup
    # reclaim above already flipped this predecessor's orphans → interrupted, so the
    # reaper's first sweep picks up the queue-managed ones immediately.
    app.state.reaper = None
    if settings.reaper_enabled:
        from src.services.reaper_service import ReaperService

        reaper = ReaperService(app)
        await reaper.start()
        app.state.reaper = reaper


async def _shutdown(app: FastAPI) -> None:
    """Drain in-flight renders, then close the metering bus + DB engine.

    On SIGTERM (rolling deploy) cancel and await the background compile/concat
    tasks so they don't get orphaned mid-render; each best-effort stamps
    ``interrupted`` (a resend re-runs it), and the next pod's startup reclaim is the
    real guarantee. Drain BEFORE closing the engine so those terminal writes land.
    """
    from src.database.connection import close_engine
    from src.services.metering import close_metering_redis

    # Mark draining BEFORE stopping the reaper. reaper.stop() can block on an in-flight
    # sweep, and during that window a render killed by the shutdown (OOM'd/SIGTERM'd ffmpeg)
    # must resolve to 'interrupted' (re-runnable), not terminal 'failed'.
    compile_routes.begin_drain()
    # Stop the reaper FIRST so it can't reclaim + re-dispatch a job while we are draining
    # this pod's renders (that would race a render this pod is about to interrupt).
    reaper = getattr(app.state, "reaper", None)
    if reaper is not None:
        await reaper.stop()
    await compile_routes.drain_background_tasks()
    metering_redis = getattr(app.state, "metering_redis", None)
    if metering_redis is not None:
        await close_metering_redis(metering_redis)
    await close_engine()


_settings = get_settings()
_configure_logging(_settings.log_level)
# Bound concurrent ffmpeg subprocesses per pod so concurrent compile jobs can't
# OOM-kill ffmpeg (exit=-9). MUST stay in lockstep with the pod memory limit.
configure_ffmpeg_concurrency(_settings.max_concurrent_ffmpeg_subprocesses)


app = create_app(
    _settings,
    routers=[health_routes.router, compile_routes.router, thumbnail_routes.router],
    title="video_compiler_service",
    description=(
        "Async video compiler. POST /api/compile {shots} → {job_id}; "
        "poll GET /api/jobs/{job_id} → {status, artifact_url, duration_seconds}."
    ),
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "compile",
            "description": (
                "Async compile/concat-audio jobs: submit shots → poll for the MP4/WAV."
            ),
        },
        {
            "name": "thumbnail",
            "description": (
                "Synchronous poster/thumbnail extraction: N grade-parity frames + safe-area geometry."
            ),
        },
    ],
    on_startup=[_startup],
    on_shutdown=[_shutdown],
)

# Bind the inbound request_id ContextVar into the root logger so JSON log lines
# carry the trace id (the factory registers RequestIdMiddleware which sets it).
logging.getLogger().addFilter(RequestIdLogFilter())


# Prometheus scrape endpoint. Auth-exempt: "/metrics" is in the shared
# PUBLIC_ENDPOINTS set.
#
# The bare `app.mount("/metrics", make_asgi_app())` below does NOT answer
# `GET /metrics` (no trailing slash) — only `/metrics/...`. Starlette's `Mount`
# compiles its match regex as `<path>/{path:path}`, which requires a literal
# "/" after the mount prefix, so the exact path "/metrics" never matches the
# Mount route at all. Normally Starlette's router falls back to a
# redirect-with-trailing-slash for that case, but this app is built with
# `redirect_slashes=False` (see below), which disables exactly that fallback —
# so an unmatched "/metrics" falls through to a plain 404, not a redirect.
# Prometheus's `prometheus.io/path` annotation convention scrapes the
# no-trailing-slash form, so a 404 there is a permanently-down scrape target.
#
# Fix: an explicit exact-path route for GET+HEAD "/metrics" that does REAL
# content negotiation via prometheus_client's own `choose_encoder` — the same
# function `make_asgi_app()`'s wrapped ASGI callable uses internally (see
# `prometheus_client.exposition._bake_output`) — rather than a fixed
# `generate_latest()` call. A fixed call would silently downgrade every real
# Prometheus scrape (which sends an OpenMetrics `Accept` by default) from
# OpenMetrics (with `# EOF` truncation detection + exemplar support) to the
# legacy 0.0.4 text format, on exactly the path Prometheus scrapes — an
# earlier version of this fix did precisely that via `w2v_common`'s
# fixed-format `metrics_router` and was caught in review. HEAD is answered
# with the negotiated Content-Type and an empty body (RFC 9110 §9.3.2: same
# headers as GET, no body). Reads the SAME default prometheus_client REGISTRY
# as the mount below (`make_asgi_app()`/`choose_encoder`'s callers both default
# to the same module-level singleton), so bodies are equivalent modulo the
# negotiated format. The mount is kept for "/metrics/" and any tooling that
# already depends on the sub-app's own routing.
@app.api_route("/metrics", methods=["GET", "HEAD"], include_in_schema=False)
async def _metrics(request: Request) -> Response:
    encoder, content_type = choose_encoder(request.headers.get("accept", ""))
    body = b"" if request.method == "HEAD" else encoder(REGISTRY)
    return Response(content=body, media_type=content_type)


app.mount("/metrics", make_asgi_app())


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.asgi:app",
        host="0.0.0.0",  # noqa: S104  # nosec B104 — in-container uvicorn, listen on all interfaces is correct
        port=settings.port,
        log_level=settings.log_level.lower(),
        # STDLIB asyncio, never uvloop (``uvicorn[standard]`` installs it and
        # ``loop="auto"`` picks it). uvloop's subprocess spawn forks and runs
        # CPython's after-fork-child hook in the child, which tears down every
        # other interpreter; with the banding settle's InterpreterPoolExecutor
        # alive that crashes the CHILD before exec — every ffmpeg the pod
        # spawned while a pool lived died with signal 11 (2026-10-21, #255).
        # The stdlib loop spawns via ``_posixsubprocess`` (vfork/exec, no Python
        # in the child). Reproduced + fixed in the prod image; see
        # ``mastering._banding_executor`` for the matching guard.
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
