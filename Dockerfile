ARG BASE_REGISTRY=registry.test:5000
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime — ffmpeg + DejaVu (font referenced by renderer.py default)
# Burned-in captions may also ask for Inter: its TTFs ship in src/assets/fonts/caption
# (COPY src/ below) and reach libass via the ass filter's fontsdir, not fontconfig.
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ⛔ PIN THE FFMPEG MAJOR VERSION — as an ASSERTION, not an apt version pin.
#
# ffprobe's per-frame entry names are NOT stable across majors, and asking for a
# name it does not know is silent: ffprobe prints nothing and exits 0. ffmpeg 7.0
# renamed `frame=pkt_duration_time` to `frame=duration_time`, which turned the CFR
# probe in `mastering._frame_durations_are_uniform` into zero output on every
# master — `cfr_ok` published `null` fleet-wide while the code, the logs and the
# exit status all looked healthy. A silent vocabulary change must fail LOUDLY here
# instead of quietly nulling a metric months later.
#
# An apt `ffmpeg=<exact>` pin was deliberately NOT used: the digest-pinned base
# gets point releases and an exact pin breaks the build on every one of them, for
# a compatibility surface that only moves across MAJORS. This asserts the major
# and lets point releases through. Bump _FFMPEG_MAJOR_EXPECTED together with the
# probe vocabulary in `mastering.py`, never on its own.
ARG FFMPEG_MAJOR_EXPECTED=7
# No pipe here on purpose (hadolint DL4006): shell parameter expansion instead of
# `| head | sed`, so this RUN needs no `SHELL -o pipefail` to be safe.
RUN set -eu; \
    found="$(ffmpeg -version)"; \
    found="${found#ffmpeg version }"; \
    found="${found#n}"; \
    found="${found%%.*}"; \
    if [ "$found" != "${FFMPEG_MAJOR_EXPECTED}" ]; then \
        echo "FATAL: ffmpeg major ${found} != expected ${FFMPEG_MAJOR_EXPECTED}." >&2; \
        echo "ffprobe entry names change across majors - re-verify frame duration" >&2; \
        echo "keys in src/services/mastering.py (_CFR_DURATION_KEYS) before bumping." >&2; \
        exit 1; \
    fi; \
    echo "ffmpeg major ${found} OK"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
# alembic.ini must ship in the runtime image so the k8s migration initContainer can
# run `alembic upgrade head` from WORKDIR /app (script_location = src/database/migrations).
COPY alembic.ini ./alembic.ini

RUN useradd --system --uid 10001 --no-create-home appuser && \
    mkdir -p /tmp/video-compiler && \
    chown -R appuser:appuser /tmp/video-compiler
# DL3066 ("non-numeric user-id may not be resolvable") is a FALSE POSITIVE here:
# `appuser` is created with an explicit `--uid 10001` two lines above, so the id
# IS fixed and host-resolvable — hadolint just cannot follow the useradd. This
# finding pre-dates this change (it fires on origin/main's Dockerfile too); it
# only became blocking because touching this file brings it into the hook's
# changed-file set. Silenced with the reason rather than by loosening the
# repo-wide --ignore list, which would hide it on files where it is real.
# (The pragma must sit on the line DIRECTLY above the instruction.)
# hadolint ignore=DL3066
USER 10001:10001

EXPOSE 8200

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

CMD ["python", "-m", "src.asgi"]
