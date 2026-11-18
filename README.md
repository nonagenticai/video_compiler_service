# Video Compiler Service

Async ffmpeg compiler that stitches storyboard shots, narration, and text cards into final MP4s (and concatenates audio clips into stitched audiobooks), uploads the result to MinIO, and returns a presigned URL. Port 8200.

## What Video Compiler Does

Video Compiler is the leaf of the w2v render pipeline. It runs the orchestrator's `video_compile` stage — the promo, lesson, app_review, website_review, audiobook, and course product types all end here — reached over HTTP at `POST /api/compile` from `w2v_orchestrator/src/clients/video_compile.py`. A caller submits a list of storyboard **shots**; each shot resolves its visual in precedence order — a pre-recorded `video_url`, a still `image_url`, or a solid text card drawn from `description` — and an optional `audio_url` muxes narration and sets the shot's duration. The shots are encoded to uniform H.264/AAC segments and joined with ffmpeg's fast concat demuxer into a single MP4. A parallel `POST /api/concat-audio` endpoint stitches an ordered list of narration clips into one WAV/MP3 for the orchestrator's `audiobook_stitch` step (per-chapter, then a combined book).

Because a render is unbounded — seconds to many minutes — the API is decoupled from request lifetime. Both submit endpoints return **202** with a `{job_id, poll_url}` and run the actual render in a background task; the caller polls `GET /api/jobs/{job_id}` until the status reaches `completed` or `failed`. The whole wall-clock budget lives in the caller's poll loop, which sidesteps the shared client's 600-second read cap that a synchronous endpoint would trip. When a render finishes, the compiled MP4 lands in the `w2v-compiled-videos` MinIO bucket (stitched audio in `w2v-audiobooks`), and the job record carries a presigned `artifact_url`, the probed output `width`/`height`, `duration_seconds`, `object_key`, and `bucket` back to the poller.

The service holds its job state **durably in Postgres** — a single `compile_jobs` table backs both the async job store (`job_id` → status/result) and the Idempotency-Key dedup index (`src/services/compile_job_store.py`, migrations `001`–`004`). Because the store survives a pod restart, submit-then-poll is cross-pod safe and the Idempotency-Key dedup no longer forgets keys on a bounce, and on boot a **crash-resumption reaper** reclaims any job a dead pod left `processing`/`interrupted` and re-renders it. The Deployment is nonetheless pinned to `replicas: 1` with `strategy: Recreate` — **but the reason is no longer state loss**: the render itself is an in-process asyncio task, so a second replica could accept a poll for a job whose render lives on the other pod, and a rolling surge would briefly run two pods. True multi-replica (lifting `replicas`) is **deferred** — it needs distributed task ownership (claim/lease of a job's render by a specific pod), not just the shared store it already has. Capacity is managed inside the single pod through backpressure, a concurrent-ffmpeg gate, and per-job timeouts.

Do not confuse this with `video_production_service` (port 6308, namespace `w2v`), which does browser-based recording only. The two are separate repos that are easy to confuse by name; a symptom in the `video_compile` stage belongs here, in namespace `video-compiler`. Tenancy is enforced at the row level: the runtime connects as the restricted `w2v_video_compiler_app` role with `TENANT_RLS_MODE=enforce`, so the durable `compile_jobs` store is tenant-isolated by RLS (a SECURITY DEFINER function handles the one cross-tenant path — reclaiming a dead pod's orphaned rows at startup), and the license tenant also drives per-minute output metering.

## Features

- **Async submit-then-poll contract** — every compile/concat returns 202 with a `job_id` immediately and renders in the background, so an unbounded render never blocks an HTTP request or trips the client's 600s read cap
- **Shot compositor** — resolves each shot's visual as pre-recorded video, still image, or generated text card, muxes optional narration, and joins uniform H.264/AAC segments with the fast concat demuxer for a single MP4
- **Audio stitching** — concatenates an ordered list of narration clips into one WAV/MP3 with configurable inter-clip gaps, serving the orchestrator's per-chapter and combined-book audiobook stitch
- **Idempotency-Key dedup** — a resubmit carrying the same `Idempotency-Key` (tenant-scoped) returns the original `job_id` instead of spawning a second render, so a connection-reset POST replay never double-renders or double-bills
- **Backpressure** — once `max_concurrent_compile_jobs` renders are in flight, further submits get 429 + `Retry-After`, so a burst can't fill `/tmp` and OOM the pod
- **ffmpeg concurrency gate** — a per-pod semaphore caps concurrent ffmpeg/ffprobe subprocesses, keeping memory in lockstep with the pod limit to avoid `ffmpeg exit=-9` OOM kills
- **Per-job work timeout** — each render runs under `asyncio.wait_for`, and a pruner evicts zombie `processing` jobs, so a wedged ffmpeg can never leave a job stuck forever or corrupt the backpressure count
- **SIGTERM drain** — the ASGI lifespan cancels in-flight renders on shutdown and stamps them terminally `failed`, so a rolling deploy never orphans a render mid-encode
- **Per-job render tags** — `format`, `render_width`/`render_height`, `crf`, and a sanitized `run_id` object-key prefix let a single request retarget resolution/quality without a redeploy; the response reports the actually probed dimensions
- **Presigned MinIO delivery** — compiled artifacts are streamed to MinIO and returned as short-lived presigned URLs, so downstream consumers fetch directly from object storage rather than through this pod
- **Best-effort metering** — a fire-and-forget cost event per completed render is emitted to a separate metering Redis stream, billed per output minute, and an empty URL disables metering without ever failing a job
- **Tenant-scoped job isolation** — jobs are stamped with the caller's tenant at submit; a cross-tenant poll returns 404 (not 403) so a job's existence isn't leaked
- **Health probes** — static liveness plus a MinIO-reachability readiness gate so a pod that can't write artifacts is pulled from the Service endpoints

## Authentication

Auth is always on: the `w2v_common.auth.middleware.AuthMiddleware` is added unconditionally at app construction (there is no in-service toggle). Inbound requests carry the license token as `X-Tenant-Id: Bearer <token>` — **not** the `Authorization` header, which is reserved for the SPA's Keycloak JWT. The middleware introspects the token against the licenses service and caches results for 5 minutes (`CACHE_LICENSE_TTL=300`).

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected via the licenses service, 5-minute TTL cache |
| 2 | Trusted network | Synthetic in-cluster identity for source IPs in `TRUSTED_CIDRS` (`127.0.0.1/32`, `::1/128`, `172.16.0.0/12`, `10.0.0.0/8`) — **fail-closed by default** (`AUTH_TRUST_INCLUSTER_IDENTITY=false`) |

Public (exempt) paths: `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics`, plus the `/health/`, `/docs`, and `/redoc` prefixes.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 8200
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Compile `/api/compile`, `/api/concat-audio`, `/api/jobs`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/compile` | Submit a shot-list compile job (returns 202 `{job_id, poll_url}`) |
| POST | `/api/concat-audio` | Submit an audio-stitch job (returns 202 `{job_id, poll_url}`) |
| GET | `/api/jobs/{job_id}` | Poll a compile/concat job until it reaches a terminal status |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Static liveness check |
| GET | `/health/live` | Liveness probe |
| GET | `/healthz` | Liveness probe (modern K8s alias) |
| GET | `/health/ready` | Readiness probe (gates on MinIO reachability) |
| GET | `/readyz` | Readiness probe (modern K8s alias) |
| GET | `/metrics` | Prometheus metrics |

## Configuration

All configuration via environment variables (flat `pydantic_settings` `Settings` — this service defines its own, no `API_` prefix or w2v_common base).

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8200` | Bind port (never 8000) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MINIO_ENDPOINT` | `""` | MinIO `host:port` (no scheme); supplied via gitops env |
| `MINIO_ACCESS_KEY` | `""` | MinIO access key |
| `MINIO_SECRET_KEY` | `""` | MinIO secret key |
| `MINIO_SECURE` | `false` | Use TLS for MinIO |
| `MINIO_BUCKET` | `w2v-compiled-videos` | Bucket for compiled MP4s |
| `MINIO_AUDIO_BUCKET` | `w2v-audiobooks` | Bucket for stitched audiobook WAV/MP3 |
| `MINIO_URL_EXPIRY_SECONDS` | `3600` | Presigned-URL TTL on the returned `artifact_url` |
| `RENDER_WIDTH` | `1920` | Default output canvas width |
| `RENDER_HEIGHT` | `1080` | Default output canvas height |
| `RENDER_FPS` | `30` | Output frame rate |
| `RENDER_CRF` | `20` | libx264 constant-rate-factor (per-job overridable via `crf` tag) |
| `RENDER_PRESET` | `medium` | libx264 `-preset` (libx264 path only) |
| `RENDER_ENCODER` | `libx264` | Video encoder; `h264_videotoolbox`/`h264_nvenc` are hardware paths |
| `RENDER_HW_BITRATE` | `6M` | Target bitrate for hardware encoder paths |
| `RENDER_BG_COLOR` | `0x111827` | Text-card background fill |
| `RENDER_TEXT_COLOR` | `white` | Text-card text color |
| `RENDER_FONT` | `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` | TrueType font path inside the container |
| `WORK_DIR` | `/tmp/video-compiler` | Scratch dir for per-request ffmpeg work; cleared after each compile |
| `MAX_CONCURRENT_FFMPEG_SUBPROCESSES` | `6` | Per-pod ffmpeg/ffprobe gate (OOM guard; keep in lockstep with pod memory) |
| `MAX_CONCURRENT_COMPILE_JOBS` | `8` | In-flight cap; further submits get 429 + `Retry-After` |
| `JOB_WORK_TIMEOUT_SECONDS` | `3600` | Hard ceiling before a render is force-failed |
| `METERING_REDIS_URL` | `""` | Cost-event stream (`metering:costs`); empty = metering disabled |

Per-job `tags` on a compile request override resolution/quality without a redeploy: `format` (`16x9`/`9x16`/`1x1`/`4x5`), `render_width`/`render_height` (bounded even ints, 2K/4K), `crf`, and a sanitized `run_id` object-key prefix.

## Architecture

```
src/
├── asgi.py                       # FastAPI app factory, AuthMiddleware, lifespan (metering + SIGTERM drain), health/compile routers, /metrics mount, custom OpenAPI
├── config.py                     # Flat pydantic_settings Settings (port 8200, MinIO, render, backpressure, metering)
├── api/
│   ├── schemas.py                # Request/response DTOs: Shot, AudioClip, sound cues, CompileRequest, JobAccepted, JobStatus
│   └── routes/
│       ├── compile.py            # /api/compile, /api/concat-audio, /api/jobs/{id}; durable Postgres job store, idempotency, backpressure, spawn/drain
│       └── health.py             # Static liveness + MinIO-reachability readiness
├── services/
│   ├── renderer.py               # Shot-list → MP4 and clip-list → WAV/MP3 via ffmpeg; concurrency semaphore; SSRF-guarded fetch
│   ├── compile_job_store.py      # Durable Postgres job store: accept/dedup, lease renew, terminal writes, reclaim
│   ├── durable_dispatch.py       # Background render dispatch + lease heartbeat over the durable store
│   ├── reaper_service.py         # Crash-resumption reaper: reclaims orphaned processing/interrupted jobs, re-renders
│   └── metering.py               # Best-effort per-output-minute cost events to metering Redis
├── database/
│   ├── models.py                 # SQLAlchemy compile_jobs table (job store + tenant idempotency index)
│   ├── rls_resolvers.py          # SECURITY DEFINER cross-tenant reclaim (startup orphan recovery under RLS enforce)
│   ├── connection.py / session.py # Async engine + session wiring
│   └── migrations/               # Alembic: 001 jobs, 002 RLS, 003 durable dispatch, 004 reclaim
├── storage/
│   └── minio_client.py           # Async MinIO client, ensure_bucket, upload + presign
└── middleware/
    └── request_id.py             # Request-ID middleware (outermost; binds request_id ContextVar)
```

## Testing

```bash
uv run pytest tests/ -v --cov=src          # unit tests; integration tests deselected by default
uv run pytest tests/ -m integration        # renderer/upload tests needing ffmpeg + MinIO on localhost
uv run ruff check src/ tests/              # Lint (target py314)
uv run ruff format --check src/ tests/     # Format check
uv run ty check                            # Type check
```

Unit coverage spans the renderer (compositor, audio concat, concurrency, SFX), the compile/concat routes and async-hardening guards, format-dims resolution, auth middleware, the error envelope, OpenAPI, health, metering, and the request-ID middleware. Integration tests (marker `integration`) are opt-in and require ffmpeg + ffprobe + a reachable MinIO on localhost. Coverage threshold: 70% (`fail_under` in pyproject.toml).

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`. No image is published from this repository.


Image: `registry.test:5000/video_compiler_service:sha-<commit>`, promoted to `:latest`. Deploy namespace is `video-compiler` (distinct from the `w2v` namespace), imagepolicy `video-compiler`. No postgres/redis/alembic flags — there is no DB.

## Docker

```bash
docker build -t video-compiler-service .
docker run -p 8200:8200 video-compiler-service
docker run -p 8200:8200 -e LOG_LEVEL=DEBUG -e MINIO_ENDPOINT=minio.test:9000 video-compiler-service
```

The runtime image is `python:3.14-slim` plus `ffmpeg` and `fonts-dejavu-core` (DejaVu is the default text-card font); the container listens on 8200 and `EXPOSE`s the same. These commands are for local testing only — production deploys via CI/CD.

## Design Principles

1. **Durable job store, single-replica render** — job state and the Idempotency-Key index are durable in Postgres (they survive a pod restart), but the Deployment stays pinned to `replicas: 1` because the render is an in-process asyncio task, not because of state loss. Scaling out is deferred: it needs distributed task ownership (claim/lease of a render by a specific pod), which the shared store alone does not provide.
2. **Recreate, never surge** — the Deployment uses `strategy: Recreate` (no rolling surge) because two pods briefly coexisting during a rollout would let the second pod field a poll for a job whose in-process render lives on the pod being torn down.
3. **Compiler leaf, no LLM** — no LLM, no external calls beyond MinIO/metering; it is the terminal, side-effect-light node of the render pipeline. It does own a small Postgres store for the durable job/idempotency state, tenant-isolated by RLS (`TENANT_RLS_MODE=enforce`, restricted `w2v_video_compiler_app` role).
4. **Async decoupled from request lifetime** — renders run as background tasks behind a 202 submit-then-poll contract, so an unbounded ffmpeg job never holds an HTTP connection open past the client's read cap.

## Dependencies

- **w2v_orchestrator** — drives the `video_compile` stage and `audiobook_stitch` step via `POST /api/compile` and `POST /api/concat-audio`, and owns the poll loop against `GET /api/jobs/{job_id}`
- **licenses service** — introspects the inbound `X-Tenant-Id: Bearer <token>` license token (via w2v_common AuthMiddleware)
- **MinIO** — object storage for compiled MP4s (`w2v-compiled-videos`) and stitched audio (`w2v-audiobooks`); artifacts returned as presigned URLs
- **Metering Redis** — separate cost-event stream (`metering:costs`) for per-output-minute billing; optional, best-effort
- **ffmpeg / ffprobe** — the render engine, bundled into the runtime image
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — request/response validation and env-driven settings
- **httpx** — async HTTP client for fetching shot/clip source assets
- **miniopy-async** — async MinIO client
- **redis** — metering cost-bus client
- **prometheus-client** — `/metrics` exposition
- **structlog** — structured logging
- **w2v-common** — auth middleware, error envelope, request-context/observability, SSRF-guarded fetch

## Known Issues

- **Single-replica by design (not scale-limited by state)** — the Deployment stays `replicas: 1` + `strategy: Recreate`. Job state is durable in Postgres and survives a restart — an interrupted render is stamped `interrupted` on SIGTERM (best-effort) and, more reliably, the startup **crash-resumption reaper** reclaims any job a dead pod left `processing`/`interrupted` and re-renders it from freshly-presigned inputs, so a bounce no longer loses in-flight work. The remaining blocker to horizontal scale is distributed task ownership: the render is an in-process asyncio task and the self-reclaim veto that stops a pod re-rendering its own live job is per-pod, so N replicas is deferred until a cross-pod claim/lease lands.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
