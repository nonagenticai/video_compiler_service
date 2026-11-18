"""Unit tests for the video-compiler cost-event wiring.

The emit machinery (field marshalling, stream resolution, GC-safe scheduling,
Redis lifecycle) now lives in :mod:`w2v_common.metering` and is unit-tested
there. These tests verify the *service-local* wiring: the video-compiler
identity/pricing constants, that NO tenant sentinel is reachable from this module,
and that the compile route threads a REAL tenant through the shared
``schedule_cost_event`` into a correctly-shaped cost event.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services import metering as metering_module
from src.services.metering import (
    COST_SOURCE,
    PRICE_PER_OUTPUT_MINUTE_USD,
    PROVIDER,
    SERVED_MODEL,
    SERVICE_KEY,
    close_metering_redis,
    init_metering_redis,
)
from tests.conftest import TENANT_A


def test_video_compiler_identity_constants() -> None:
    # The video-compiler-specific identity constants this service contributes to
    # every emitted cost event. (PRICE only sizes the advisory cost hint.)
    assert COST_SOURCE == "w2v-video-compiler"
    # "w2v-video-compiler" is a routing slug, not a credential.
    assert SERVICE_KEY == "w2v-video-compiler"  # pragma: allowlist secret -- routing slug
    assert PROVIDER == "video-compiler"
    assert SERVED_MODEL == "ffmpeg"
    assert PRICE_PER_OUTPUT_MINUTE_USD == 0.30


@pytest.mark.parametrize("sentinel", ["DEFAULT_TENANT_ID", "UNATTRIBUTED_TENANT_ID"])
def test_no_tenant_sentinel_is_reachable_from_this_module(sentinel: str) -> None:
    """This module must expose NO substitutable tenant value — the inverse of what it
    used to assert.

    It previously asserted ``DEFAULT_TENANT_ID == UNATTRIBUTED_TENANT_ID ==`` the nil
    uuid, i.e. it PINNED the defect: a render that resolved no tenant was emitted as
    attributed-to-nobody rather than failing, so an unattributable cost event was the
    designed outcome. Product-owner ruling: every event carries a real registered
    tenant. An absent one now raises ``MissingTenantError`` at the request boundary, so
    nothing tenant-less can reach the emit path at all.

    ``UNATTRIBUTED_TENANT_ID`` is asserted gone too — it is the billing flusher's own
    normalisation vocabulary, never a value this service may emit, and leaving it
    importable here is exactly how someone reaches for it "just this once".
    """
    assert not hasattr(metering_module, sentinel), (
        f"{sentinel} is reachable from src.services.metering again — a tenant sentinel "
        "must not exist in this service"
    )
    assert sentinel not in metering_module.__all__


@pytest.mark.asyncio
async def test_init_metering_redis_empty_url_returns_none() -> None:
    assert await init_metering_redis("") is None


@pytest.mark.asyncio
async def test_close_metering_redis_none_is_noop() -> None:
    await close_metering_redis(None)  # must not raise


# --- compile-route wiring (_emit_cost + set_metering_redis) -------------------


@pytest.mark.asyncio
async def test_compile_emit_cost_fires_with_full_event_shape() -> None:
    import asyncio

    from src.api.routes import compile as compile_routes

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        compile_routes._emit_cost(
            job_id="job-1",
            tenant_id=TENANT_A,
            duration_seconds=90.4,  # rounds to 90 seconds of output
            duration_source="probed",
            latency_ms=1234,
            job_type="compile",
        )
        # schedule_cost_event spawns a fire-and-forget task; let it run.
        await asyncio.sleep(0)
        await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])
    finally:
        compile_routes.set_metering_redis(None)

    redis.xadd.assert_awaited_once()
    # Stream resolves to the canonical cost-bus stream (centralised in w2v_common).
    assert redis.xadd.await_args.args[0] == "billing:costs"
    fields: dict[str, str] = redis.xadd.await_args.args[1]
    # Every value is a string at the Redis layer.
    assert all(isinstance(v, str) for v in fields.values())
    assert fields["tenant_id"] == str(TENANT_A)
    assert fields["service_key"] == SERVICE_KEY
    assert fields["cost_source"] == COST_SOURCE
    assert fields["provider"] == PROVIDER == "video-compiler"
    assert fields["model"] == fields["served_model"] == SERVED_MODEL == "ffmpeg"
    assert fields["input_tokens"] == "90"  # round(90.4)
    assert fields["output_tokens"] == "0"
    # 90s of output at $0.30/min = $0.45.
    assert fields["cost_usd"] == "0.45000000"
    assert fields["latency_ms"] == "1234"
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["job_type"] == "compile"
    assert extra["unit"] == "video_seconds"
    assert redis.xadd.await_args.kwargs["maxlen"] == 100_000
    assert redis.xadd.await_args.kwargs["approximate"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [None, "", "   ", "legacy-string-tenant", "not-a-uuid"])
async def test_unattributed_submit_is_rejected_and_emits_no_cost_event(
    forwarded: object,
) -> None:
    """A submit forwarding no REAL tenant is refused at the boundary (400) and NOTHING
    is billed — the exact inverse of the deleted nil-sentinel test.

    That test asserted ``_emit_cost(tenant_id=None)`` emitted the nil uuid. The render
    still ran, still cost money, and produced a cost event nobody could attribute. The
    contract now is: ``_caller_tenant`` runs ``require_tenant`` and maps
    ``MissingTenantError`` to HTTP 400, so the request dies BEFORE any render or emit.

    Both halves are load-bearing: 400 alone would still pass if the route emitted a
    sentinel event on its way out, so the ``xadd`` absence is asserted too.
    """
    from fastapi import FastAPI, Request
    from httpx import ASGITransport, AsyncClient

    from src.api.routes import compile as compile_routes

    app = FastAPI()
    app.state.instance_id = "test-unattributed"

    @app.middleware("http")
    async def _stamp(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.tenant_id = forwarded
        return await call_next(request)

    app.include_router(compile_routes.router)

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            resp = await client.post(
                "/api/compile",
                json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]},
            )
    finally:
        compile_routes.set_metering_redis(None)

    assert resp.status_code == 400, resp.text
    assert "tenant_id" in resp.text
    redis.xadd.assert_not_awaited()  # no unattributed cost event was ever emitted


@pytest.mark.asyncio
async def test_emit_cost_carries_the_real_tenant_uuid_verbatim() -> None:
    """The emitted ``tenant_id`` is the caller's own uuid — never rewritten or bucketed."""
    import asyncio

    from src.api.routes import compile as compile_routes

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        compile_routes._emit_cost(
            job_id="job-1",
            tenant_id=TENANT_A,
            duration_seconds=60,
            duration_source="probed",
            latency_ms=1,
            job_type="compile",
        )
        await asyncio.sleep(0)
        await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])
    finally:
        compile_routes.set_metering_redis(None)

    fields = redis.xadd.await_args.args[1]
    assert fields["tenant_id"] == str(TENANT_A)
    assert fields["tenant_id"] != "00000000-0000-0000-0000-000000000000"  # never the nil sentinel


@pytest.mark.asyncio
async def test_compile_emit_cost_swallows_redis_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Redis hiccup must never surface from the fire-and-forget emit.
    import asyncio

    from src.api.routes import compile as compile_routes

    # w2v-common 0.67.9 added RuntimeError/OSError to the retryable set: a bare
    # asyncio "TCPTransport closed" RuntimeError used to bypass the retry and
    # drop the event. So this emit is now ATTEMPTED once per backoff plus a
    # final try. Pin the backoff list rather than inheriting the library
    # default, so this asserts our contract and not w2v-common's tuning.
    monkeypatch.setenv("METERING_EMIT_RETRY_BACKOFFS", "0,0")
    expected_attempts = 3  # two zero-delay retries + the final attempt

    redis = AsyncMock()
    redis.xadd.side_effect = RuntimeError("redis went away")
    compile_routes.set_metering_redis(redis)
    try:
        compile_routes._emit_cost(
            job_id="job-1",
            tenant_id=TENANT_A,
            duration_seconds=60,
            duration_source="probed",
            latency_ms=1,
            job_type="compile",
        )
        await asyncio.sleep(0)
        await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])
    finally:
        compile_routes.set_metering_redis(None)
    # The point of this test is that the error is SWALLOWED — that property is
    # unchanged. Only the attempt count moved, because the retry now covers it.
    assert redis.xadd.await_count == expected_attempts


@pytest.mark.asyncio
async def test_compile_emit_cost_noop_when_disabled() -> None:
    from src.api.routes import compile as compile_routes

    compile_routes.set_metering_redis(None)
    # No redis + no exception. (No task is scheduled.)
    compile_routes._emit_cost(
        job_id="job-1",
        tenant_id=TENANT_A,
        duration_seconds=60,
        duration_source="probed",
        latency_ms=1,
        job_type="compile",
    )


@pytest.mark.asyncio
async def test_compile_emit_cost_noop_on_zero_duration() -> None:
    import asyncio

    from src.api.routes import compile as compile_routes

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        compile_routes._emit_cost(
            job_id="job-1",
            tenant_id=TENANT_A,
            duration_seconds=0,
            duration_source="probed",
            latency_ms=1,
            job_type="compile",
        )
        await asyncio.sleep(0)
    finally:
        compile_routes.set_metering_redis(None)
    redis.xadd.assert_not_awaited()


def test_stream_resolution_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream resolution is centralised in ``w2v_common.metering.resolve_stream``:
    it honours ``$METERING_STREAM``, defaulting to the canonical cost-bus stream.
    History: ``metering:costs`` -> ``improvement-loop:costs`` (2026-06-24) ->
    ``billing:costs`` (billing-bus rename, w2v_common 0.44.0). The retired names
    have no consumer — events there never reach billing."""
    from w2v_common.metering import resolve_stream

    monkeypatch.delenv("METERING_STREAM", raising=False)
    assert resolve_stream() == "billing:costs"

    monkeypatch.setenv("METERING_STREAM", "custom:costs")
    assert resolve_stream() == "custom:costs"


@pytest.mark.asyncio
async def test_unprobed_billed_duration_is_marked_on_the_cost_event() -> None:
    """A render whose final-mux ffprobe FAILED is billed off the nominal declared sum
    (``probe_output_duration``'s 0.0 = unknown, and 0.0 is falsy), which OVER-reports
    seeked/repeated clips. That charge must be auditable: the cost event has to say the
    figure was NOT probed, so over-reporting is detectable after the fact instead of
    being indistinguishable from a measured charge.
    """
    import asyncio

    from src.api.routes import compile as compile_routes

    # The exact production shape: probe timed out -> the 0.0 sentinel; 72 nominal
    # seconds nobody measured.
    result = {"duration_seconds": 72.0, "actual_duration_seconds": 0.0}
    billed_seconds, duration_source = compile_routes._billed_duration(result)
    assert billed_seconds == 72.0  # we still bill; we do not bill nothing

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        compile_routes._emit_cost(
            job_id="job-1",
            tenant_id=TENANT_A,
            duration_seconds=billed_seconds,
            duration_source=duration_source,
            latency_ms=1,
            job_type="compile",
        )
        await asyncio.sleep(0)
        await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])
    finally:
        compile_routes.set_metering_redis(None)

    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["input_tokens"] == "72"
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["duration_source"] == "nominal"

    # ...and a genuinely probed render is NOT tarred with the same marker.
    probed_seconds, probed_source = compile_routes._billed_duration(
        {"duration_seconds": 72.0, "actual_duration_seconds": 65.0}
    )
    assert (probed_seconds, probed_source) == (65.0, "probed")


async def _drain_emit(fn: Any) -> None:
    """Run a fire-and-forget ``_emit_cost`` and await the task it schedules."""
    import asyncio

    fn()
    await asyncio.sleep(0)
    await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not asyncio.current_task()])


@pytest.mark.asyncio
async def test_unmeasured_audiobook_still_emits_a_flagged_cost_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A DELIVERED audiobook whose duration could not be probed must not vanish.

    The regression this pins: the concat-audio result carries no
    ``actual_duration_seconds`` and its ``duration_seconds`` is itself an ffprobe of the
    delivered master, so an unparseable/absent ffprobe duration made BOTH keys 0.0.
    ``_billed_duration`` returned ``(0.0, "nominal")`` and ``_emit_cost`` returned at
    ``seconds <= 0`` -- BEFORE the ``cost_duration_unprobed`` warning. Delivered work,
    no cost event, no log line, nothing to notice it by.

    Contract now: the absence is expressible (``"unmeasured"``), the warning fires, and
    a zero-quantity event is emitted so the delivery is reconcilable. Quantity stays 0
    on purpose -- there is no refund path in this system, so an invented duration would
    be a permanent over-charge.
    """
    from src.api.routes import compile as compile_routes

    # The exact production shape of an unprobeable audiobook master.
    result = {
        "artifact_url": "https://minio.test/audio/run/job.m4b",
        "duration_seconds": 0.0,
        "duration_probed": False,
    }
    billed_seconds, duration_source = compile_routes._billed_duration(result)
    assert (billed_seconds, duration_source) == (None, "unmeasured")

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        await _drain_emit(
            lambda: compile_routes._emit_cost(
                job_id="job-1",
                tenant_id=TENANT_A,
                duration_seconds=billed_seconds,
                duration_source=duration_source,
                latency_ms=7,
                job_type="concat_audio",
            )
        )
    finally:
        compile_routes.set_metering_redis(None)

    redis.xadd.assert_awaited_once()  # the delivery is no longer invisible to billing
    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["tenant_id"] == str(TENANT_A)
    assert fields["input_tokens"] == "0"  # nothing fabricated
    assert float(fields["cost_usd"]) == 0.0  # and nothing over-charged
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["duration_source"] == "unmeasured"
    assert extra["duration_unmeasured"] == "true"
    assert extra["job_type"] == "concat_audio"
    # ...and it is LOUD: the warning the early return used to jump over.
    assert "compile.cost_duration_unprobed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_genuinely_zero_job_emits_nothing_and_stays_quiet(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CONTROL: a job that delivered NO artifact and declared no duration is genuinely
    zero, not unmeasured, and must keep taking the silent non-billing path.

    Without this control the fix could 'work' by simply billing every zero, which would
    emit paid-looking events for jobs that produced nothing.
    """
    from src.api.routes import compile as compile_routes

    billed_seconds, duration_source = compile_routes._billed_duration(
        {"duration_seconds": 0.0, "actual_duration_seconds": 0.0}
    )
    assert duration_source != "unmeasured"

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        await _drain_emit(
            lambda: compile_routes._emit_cost(
                job_id="job-1",
                tenant_id=TENANT_A,
                duration_seconds=billed_seconds,
                duration_source=duration_source,
                latency_ms=7,
                job_type="compile",
            )
        )
    finally:
        compile_routes.set_metering_redis(None)

    redis.xadd.assert_not_awaited()
    assert "compile.cost_duration_unprobed" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_probeable_audiobook_bills_its_real_duration_as_probed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CONTROL: a normal audiobook bills the ffprobe'd seconds, marked ``probed``.

    ``duration_probed`` is what stops the concat path's ``duration_seconds`` -- which
    IS a measurement -- from being reported as an unaudited "nominal" charge and firing
    ``cost_duration_unprobed`` on every successful audiobook, which is exactly the noise
    that makes a real occurrence unfindable.
    """
    from src.api.routes import compile as compile_routes

    result = {
        "artifact_url": "https://minio.test/audio/run/job.m4b",
        "duration_seconds": 120.0,
        "duration_probed": True,
    }
    billed_seconds, duration_source = compile_routes._billed_duration(result)
    assert (billed_seconds, duration_source) == (120.0, "probed")

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        await _drain_emit(
            lambda: compile_routes._emit_cost(
                job_id="job-1",
                tenant_id=TENANT_A,
                duration_seconds=billed_seconds,
                duration_source=duration_source,
                latency_ms=7,
                job_type="concat_audio",
            )
        )
    finally:
        compile_routes.set_metering_redis(None)

    fields: dict[str, str] = redis.xadd.await_args.args[1]
    assert fields["input_tokens"] == "120"
    assert fields["cost_usd"] == "0.60000000"  # 120s at $0.30/min
    extra: dict[str, Any] = json.loads(fields["usage_extra"])
    assert extra["duration_source"] == "probed"
    assert "duration_unmeasured" not in extra
    assert "compile.cost_duration_unprobed" not in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probed_seconds", "clip_seconds", "expected"),
    [
        # A normal audiobook: the probe measured it, so it bills its real length.
        (94.5, None, (94.5, "probed")),
        # Unprobeable, but the request declared every clip -> nominal fallback, the
        # same shape the compile path has always had.
        (0.0, [30.0, 12.5], (42.5, "nominal")),
        # Unprobeable AND undeclared -> the gap is expressible, not a zero.
        (0.0, None, (None, "unmeasured")),
    ],
)
async def test_concat_to_result_marks_whether_its_duration_was_measured(
    monkeypatch: pytest.MonkeyPatch,
    probed_seconds: float,
    clip_seconds: list[float] | None,
    expected: tuple[float | None, str],
) -> None:
    """END-TO-END over the real audiobook path: what ``_concat_to_result`` returns must
    let ``_billed_duration`` tell a measured master from a declared one from an
    unmeasurable one.

    ``concat_audio`` returns ``_ffprobe_duration(out_path)``, whose 0.0 is documented as
    "duration not carried" -- i.e. UNMEASURED, not "zero seconds". The stub stands in
    for ffprobe exactly as the rest of this suite does; nothing here spawns a
    subprocess or opens a URL.
    """
    from pathlib import Path

    from src.api.routes import compile as compile_routes
    from src.api.schemas import AudioClip, AudioConcatRequest
    from src.config import get_settings

    out = Path("/tmp/vc-test-concat.wav")  # never created, opened or read

    async def _fake_concat(*_a: Any, **_k: Any) -> tuple[Path, float, None]:
        return out, probed_seconds, None

    async def _fake_upload(*_a: Any, **_k: Any) -> str:
        return "https://minio.test/audio/run/job.wav"

    monkeypatch.setattr(compile_routes, "concat_audio", _fake_concat)
    monkeypatch.setattr(compile_routes, "upload_and_presign", _fake_upload)
    monkeypatch.setattr(compile_routes, "_cleanup_job_dir", lambda *_a, **_k: None)

    declared = clip_seconds if clip_seconds is not None else [None, None]
    body = AudioConcatRequest(
        clips=[
            AudioClip(id=f"c{i}", audio_url="https://example.invalid/a.wav", seconds=sec)
            for i, sec in enumerate(declared)
        ]
    )
    result = await compile_routes._concat_to_result(body, get_settings(), "job-1")

    assert result["artifact_url"]  # the master WAS delivered in every case
    assert compile_routes._billed_duration(result) == expected


# --- G5: full cost attribution (app, funding, deterministic event_key) -------


async def _emit_and_capture(job_id: str = "job-1") -> dict[str, str]:
    from src.api.routes import compile as compile_routes

    redis = AsyncMock()
    compile_routes.set_metering_redis(redis)
    try:
        await _drain_emit(
            lambda: compile_routes._emit_cost(
                job_id=job_id,
                tenant_id=TENANT_A,
                duration_seconds=60.0,
                duration_source="probed",
                latency_ms=10,
                job_type="compile",
            )
        )
    finally:
        compile_routes.set_metering_redis(None)
    redis.xadd.assert_awaited_once()
    return redis.xadd.await_args.args[1]


@pytest.fixture
def _bound_licence() -> Any:
    import uuid

    from w2v_common.auth.identity_context import clear_identity_context, set_identity_context

    set_identity_context(TENANT_A, str(uuid.uuid7()), str(uuid.uuid7()))
    yield
    clear_identity_context()


@pytest.mark.asyncio
async def test_emit_carries_app_key_funding_and_deterministic_event_key(
    _bound_licence: None,
) -> None:
    first = await _emit_and_capture("job-1")
    again = await _emit_and_capture("job-1")
    other = await _emit_and_capture("job-2")
    assert first["app_id"] == "w2v"
    assert first["funding"] == "local_gpu"
    assert first["license_id"] and first["subject_id"]
    # Same job re-emitted (reaper resume) -> same key; another job -> another key.
    assert first["event_key"] == again["event_key"]
    assert first["event_key"] != other["event_key"]


@pytest.mark.asyncio
async def test_strict_attribution_emits_with_a_bound_licence(
    _bound_licence: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    fields = await _emit_and_capture()
    assert fields["app_id"] == "w2v"


@pytest.mark.asyncio
async def test_strict_attribution_refuses_an_event_without_a_licence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from w2v_common.metering import MissingAttributionError

    from src.api.routes import compile as compile_routes

    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    compile_routes.set_metering_redis(AsyncMock())
    try:
        with pytest.raises(MissingAttributionError, match="license_id"):
            compile_routes._emit_cost(
                job_id="job-1",
                tenant_id=TENANT_A,
                duration_seconds=60.0,
                duration_source="probed",
                latency_ms=10,
                job_type="compile",
            )
    finally:
        compile_routes.set_metering_redis(None)


def test_unknown_provider_funding_fails_loud() -> None:
    from src.services.metering import UnknownProviderFundingError, funding_for

    assert funding_for(PROVIDER) == "local_gpu"
    with pytest.raises(UnknownProviderFundingError):
        funding_for("sora")
