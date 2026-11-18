"""Cost-metering config — thin delegation to the shared w2v_common cost-bus.

This service emits one event to the cost-bus Redis stream per completed
compile/concat render. The emit machinery — field marshalling, stream
resolution, GC-safe fire-and-forget scheduling, and the Redis lifecycle — now
lives in :mod:`w2v_common.metering`, the single fleet-wide source of truth.

Historically every w2v service hardcoded its own ``DEFAULT_STREAM`` constant;
when the cost-bus stream was renamed ``metering:costs`` → ``improvement-loop:costs``
on 2026-06-24, services that hardcoded the old name silently routed events to a
dead stream. Centralising stream resolution in ``w2v_common`` makes the next
rename a one-line change there.

This module keeps only the video-compiler-specific identity/pricing constants
and re-exports the
shared Redis lifecycle helpers so existing importers keep working.

Billing unit: **MINUTE of OUTPUT media** (compiled video duration / stitched
audio duration). ``input_tokens`` = whole seconds of output, ``output_tokens``
= 0. The emitted ``cost_usd`` is only an advisory hint — the billing service
recomputes the authoritative cost from the centralised ``provider_pricing``
table (provider ``video-compiler``).

The headline rate is benchmarked against Shotstack's render API pay-as-you-go
price of **$0.30 / output-minute** (resolution-independent), the closest
commercial comparable for a programmatic ffmpeg compile/render service
(2026-06). The authoritative rate lives in ``provider_pricing``; this constant
only sizes the advisory hint.
"""

from __future__ import annotations

from typing import Final

from w2v_common.metering import Funding, close_metering_redis, init_metering_redis

__all__ = [
    "COST_SOURCE",
    "PRICE_PER_OUTPUT_MINUTE_USD",
    "PROVIDER",
    "SERVED_MODEL",
    "SERVICE_KEY",
    "UnknownProviderFundingError",
    "close_metering_redis",
    "funding_for",
    "init_metering_redis",
]

COST_SOURCE = "w2v-video-compiler"
SERVICE_KEY = "w2v-video-compiler"  # pragma: allowlist secret -- routing slug, not a credential
PROVIDER = "video-compiler"
SERVED_MODEL = "ffmpeg"

# Advisory hint rate only — the billing service recomputes from provider_pricing.
# $0.30 per output-minute (Shotstack render API pay-as-you-go benchmark, 2026-06).
PRICE_PER_OUTPUT_MINUTE_USD = 0.30

# ⛔ ``DEFAULT_TENANT_ID`` (the nil-UUID "unattributed" sentinel) was REMOVED.
# Product-owner ruling: ``tenant_id`` may never hold a sentinel — every event carries a
# real registered tenant forwarded from the requestor. Substituting the nil uuid here was
# the second spelling of the schema's empty-string default: it made an unattributed render
# look attributed-to-nobody instead of failing. An absent tenant now raises
# ``src.services.compile_job_store.MissingTenantError`` at the request boundary, so no
# render (and therefore no cost event) can ever reach this module without one.
# The ``UNATTRIBUTED_TENANT_ID`` re-export is gone too: it belongs to the billing
# flusher's own normalisation vocabulary and is NOT a value this service may ever emit,
# so it must not be reachable from here for someone to reach for "just this once".


class UnknownProviderFundingError(ValueError):
    """A provider this service does not know how to fund (G5: never guess funding)."""


#: WHO FUNDS each provider this service emits for. The compiler is self-hosted
#: ffmpeg on platform hardware -> ``local_gpu`` (billing ``LOCAL_PROVIDERS``
#: lists ``video-compiler``). A new provider must be added here explicitly.
_FUNDING_BY_PROVIDER: Final[dict[str, Funding]] = {PROVIDER: "local_gpu"}


def funding_for(provider: str) -> Funding:
    """``funding`` for ``provider``; an unknown provider raises (fail loud)."""
    try:
        return _FUNDING_BY_PROVIDER[provider]
    except KeyError:
        raise UnknownProviderFundingError(
            f"no funding declared for provider {provider!r}"
        ) from None
