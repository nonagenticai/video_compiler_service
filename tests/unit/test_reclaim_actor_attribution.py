"""A reaper re-dispatch carries the ORIGINAL caller's actor, not this service's own.

w2v-common 0.88 uses ``SERVICE_ACTOR_ID`` only for SERVICE-originated work. The reaper
re-runs a crash-orphaned compile outside any request, so without a persisted actor
w2v_common sees it as service-originated and stamps this service's own sub on the
re-run's cost event and outbound calls. That is wrong: a caller asked for the render.

* enqueue with an actor -> the re-dispatch carries THAT actor (and the project/job),
  even with a different ``SERVICE_ACTOR_ID`` configured;
* a legacy job (no stored actor) stays actor-less and is COUNTED — it never falls back
  to ``SERVICE_ACTOR_ID``, because it was caller-originated.
"""

from __future__ import annotations

import json
import uuid

import pytest
from w2v_common.auth import (
    ACTOR_ID_HEADER,
    W2V_JOB_ID_HEADER,
    W2V_PROJECT_ID_HEADER,
    bind_attribution_from_headers,
    get_identity_headers,
    get_project_id_from_context,
    get_w2v_job_id_from_context,
    get_w2v_project_id_from_context,
)
from w2v_common.auth.identity_context import (
    ACTOR_MISSING_INBOUND,
    actor_missing_total,
    clear_identity_context,
    mark_inbound_request,
    resolve_actor_id,
    set_identity_context,
)
from w2v_common.metering.cost_bus import _apply_ambient_metadata
from w2v_common.tenant import TenantSource

from src.api.routes import compile as compile_routes
from src.api.schemas import CompileRequest, Shot
from src.services import compile_job_store as job_store
from src.services.durable_dispatch import KIND_COMPILE, build_job_options
from src.services.metering import SERVICE_KEY

TENANT = uuid.UUID("e148e8af-3751-49bf-a0fa-56ab46d31ddc")
CALLER_ACTOR = "a1a1a1a1-1111-4111-8111-111111111111"
OWN_SERVICE_ACTOR = "5e5e5e5e-5555-4555-8555-555555555555"
FR_PROJECT = "b2b2b2b2-2222-4222-8222-222222222222"
W2V_PROJECT = "c3c3c3c3-3333-4333-8333-333333333333"
W2V_JOB = "d4d4d4d4-4444-4444-8444-444444444444"


@pytest.fixture(autouse=True)
def _clean_identity(monkeypatch):
    monkeypatch.setenv("SERVICE_ACTOR_ID", OWN_SERVICE_ACTOR)
    clear_identity_context()
    yield
    clear_identity_context()


def _request() -> CompileRequest:
    return CompileRequest(shots=[Shot(id="s1", seconds=2.0, image_url="https://minio/x.png")])


def _enqueue_options() -> dict:
    """``build_job_options`` inside a request that named an actor + project + job."""
    set_identity_context(
        TENANT, "lic", "sub", tenant_source=TenantSource.LICENSE_TOKEN, actor_id=CALLER_ACTOR
    )
    mark_inbound_request()
    bind_attribution_from_headers(
        {
            "X-Project-Id": FR_PROJECT,
            W2V_PROJECT_ID_HEADER: W2V_PROJECT,
            W2V_JOB_ID_HEADER: W2V_JOB,
        },
        trusted=True,
    )
    opts = build_job_options(
        kind=KIND_COMPILE, request=_request(), tenant_id=TENANT, work_timeout=60
    )
    clear_identity_context()
    return json.loads(json.dumps(opts))  # JSONB round trip


async def _redispatch_and_observe(monkeypatch, options: dict) -> dict:
    """Run the real ``redispatch_reclaimed``; observe the context ``_spawn`` sees."""
    observed: dict = {}

    def _spawn(job_id, *a, **k) -> None:
        observed["actor"] = resolve_actor_id()
        observed["headers"] = get_identity_headers()
        observed["project"] = get_project_id_from_context()
        observed["w2v_project"] = get_w2v_project_id_from_context()
        observed["w2v_job"] = get_w2v_job_id_from_context()
        fields: dict[str, str] = {}
        _apply_ambient_metadata(fields, None, SERVICE_KEY)
        observed["cost_fields"] = fields

    monkeypatch.setattr(compile_routes, "_spawn", _spawn)
    claimed = job_store.ClaimedJob(job_id="job-1", tenant_id=TENANT, options=options)
    await compile_routes.redispatch_reclaimed(claimed, lease_seconds=0, instance_id="pod")
    assert observed, "redispatch never reached _spawn"
    return observed


def test_build_job_options_persists_actor_project_and_job() -> None:
    opts = _enqueue_options()
    assert opts["actor_id"] == CALLER_ACTOR
    assert opts["project_id"] == FR_PROJECT
    assert opts["w2v_project_id"] == W2V_PROJECT
    assert opts["w2v_job_id"] == W2V_JOB


async def test_enqueued_actor_is_carried_by_the_redispatch(monkeypatch) -> None:
    observed = await _redispatch_and_observe(monkeypatch, _enqueue_options())

    assert observed["actor"] == (CALLER_ACTOR, None)
    assert observed["headers"][ACTOR_ID_HEADER] == CALLER_ACTOR
    assert observed["cost_fields"]["actor_id"] == CALLER_ACTOR
    assert observed["cost_fields"]["job_id"] == W2V_JOB
    assert observed["project"] == FR_PROJECT
    assert observed["w2v_project"] == W2V_PROJECT
    assert observed["w2v_job"] == W2V_JOB
    assert OWN_SERVICE_ACTOR not in observed["headers"].values()


async def test_legacy_job_without_actor_stays_actorless_and_is_counted(monkeypatch) -> None:
    legacy = _enqueue_options()
    for key in ("actor_id", "project_id", "w2v_project_id", "w2v_job_id"):
        legacy.pop(key)
    counter = actor_missing_total.labels(service=SERVICE_KEY, reason=ACTOR_MISSING_INBOUND)
    before = counter._value.get()

    observed = await _redispatch_and_observe(monkeypatch, legacy)

    assert observed["actor"] == (None, ACTOR_MISSING_INBOUND)
    assert ACTOR_ID_HEADER not in observed["headers"]
    assert "actor_id" not in observed["cost_fields"]
    assert OWN_SERVICE_ACTOR not in observed["headers"].values()
    assert counter._value.get() == before + 1


async def test_redispatch_does_not_leak_actor_into_the_reaper(monkeypatch) -> None:
    await _redispatch_and_observe(monkeypatch, _enqueue_options())
    assert resolve_actor_id() == (OWN_SERVICE_ACTOR, None)
