"""Guardrail tests for the OpenAPI surface of video_compiler_service.

Locks in the org compliance standard:

- every route has an explicit ``operation_id="…"`` per
  ``OPERATION_ID_PLAN.md`` (no reliance on the
  ``generate_unique_id_function`` fallback — that is defence-in-depth only);
- the standard OpenAPI metadata (BearerAuth security scheme, openapi_tags
  grouping by domain) is present;
- protected paths advertise the security requirement; public health paths
  do not.

If a new route lands, update ``OPERATION_ID_PLAN.md`` *and* the
``EXPECTED_OPERATION_IDS`` set below — they should move together.
"""

from __future__ import annotations

import pytest

from src.asgi import app

# Source of truth — keep in sync with OPERATION_ID_PLAN.md.
EXPECTED_OPERATION_IDS: frozenset[str] = frozenset(
    {
        # health
        "health_check",
        "health_live",
        "health_ready",
        "health.live",
        "health.ready",
        # compile (async submit → poll)
        "compile_video",
        "concat_audio",
        "get_compile_job",
        "cancel_compile_jobs",
        # thumbnail (synchronous poster/thumbnail extraction)
        "extract_thumbnails",
    }
)


@pytest.fixture(scope="module")
def openapi_schema() -> dict:
    """Build the OpenAPI schema once per module."""
    app.openapi_schema = None
    return app.openapi()


def test_every_route_has_explicit_operation_id(openapi_schema: dict) -> None:
    """No route may rely on FastAPI's auto-generated operationId.

    FastAPI's auto-format is ``<function>_<path>_<method>`` and produces
    awful client names. The fallback ``generate_unique_id_function`` in
    asgi.py is defence-in-depth only — every production route must set
    ``operation_id="…"`` explicitly.
    """
    auto_suffixes = ("_post", "_get", "_put", "_delete", "_patch")
    offenders: list[tuple[str, str, str]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId", "")
            if op_id.endswith(auto_suffixes) and op_id not in EXPECTED_OPERATION_IDS:
                offenders.append((method.upper(), path, op_id))
    assert not offenders, f"Routes without explicit operation_id (auto-generated): {offenders}"


def test_operation_ids_match_plan(openapi_schema: dict) -> None:
    """The set of operationIds in the schema must equal the plan."""
    actual: set[str] = set()
    for item in openapi_schema.get("paths", {}).values():
        for op in item.values():
            if isinstance(op, dict) and op.get("operationId"):
                actual.add(op["operationId"])

    missing = EXPECTED_OPERATION_IDS - actual
    unexpected = actual - EXPECTED_OPERATION_IDS
    assert not missing and not unexpected, (
        f"operationId drift — missing: {sorted(missing)}, unexpected: {sorted(unexpected)}"
    )


def test_operation_ids_are_unique(openapi_schema: dict) -> None:
    """Every operationId must be unique across the whole schema."""
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, tuple[str, str], tuple[str, str]]] = []
    for path, item in openapi_schema.get("paths", {}).items():
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            if not op_id:
                continue
            if op_id in seen:
                duplicates.append((op_id, seen[op_id], (method.upper(), path)))
            else:
                seen[op_id] = (method.upper(), path)
    assert not duplicates, f"Duplicate operationIds: {duplicates}"


def test_bearer_auth_scheme_is_declared(openapi_schema: dict) -> None:
    """The BearerAuth security scheme must be present in components."""
    schemes = openapi_schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes, "BearerAuth scheme missing from OpenAPI"
    scheme = schemes["BearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"


def test_tenant_id_header_parameter_is_declared(openapi_schema: dict) -> None:
    """The reusable TenantIdHeader parameter must be in components.parameters."""
    parameters = openapi_schema.get("components", {}).get("parameters", {})
    assert "TenantIdHeader" in parameters, "TenantIdHeader parameter missing"
    param = parameters["TenantIdHeader"]
    assert param["in"] == "header"
    assert param["name"].lower() == "x-tenant-id"


def test_protected_routes_have_security_requirement(openapi_schema: dict) -> None:
    """Every protected operation must advertise BearerAuth.

    In this service the non-health, non-metrics routes are the async submit
    endpoints ``POST /api/compile`` / ``POST /api/concat-audio`` and the
    ``GET /api/jobs/{job_id}`` poll endpoint.
    """
    protected_paths = [
        p
        for p in openapi_schema.get("paths", {})
        if not (p.startswith(("/health", "/metrics")) or p in {"/", "/healthz", "/readyz"})
    ]
    assert protected_paths, "No protected paths found in schema"
    offenders: list[tuple[str, str]] = []
    for path in protected_paths:
        for method, op in openapi_schema["paths"][path].items():
            if not isinstance(op, dict):
                continue
            security = op.get("security", [])
            if not any("BearerAuth" in entry for entry in security):
                offenders.append((method.upper(), path))
    assert not offenders, f"Protected routes missing BearerAuth: {offenders[:5]}…"


def test_public_routes_have_no_security_requirement(openapi_schema: dict) -> None:
    """Health paths must not require auth in the schema."""
    public = {"/healthz", "/readyz", "/health", "/health/ready", "/health/live"}
    for path in public:
        item = openapi_schema.get("paths", {}).get(path)
        if item is None:
            continue
        for method, op in item.items():
            if not isinstance(op, dict):
                continue
            assert not op.get("security"), (
                f"Public route {method.upper()} {path} unexpectedly has "
                f"security={op.get('security')}"
            )


def test_vc3_schema_fields_present(openapi_schema: dict) -> None:
    """The VC-3 cadence fields are advertised on the OpenAPI component schemas:
    Shot.target_fps/retime, CompileRequest.cadence, and the measured-cadence
    metrics on MasteringReport."""
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    assert {"target_fps", "retime"} <= set(schemas["Shot"]["properties"])
    assert "cadence" in schemas["CompileRequest"]["properties"]
    master_props = set(schemas["MasteringReport"]["properties"])
    assert {"measured_fps", "duplicate_frame_ratio"} <= master_props
    # is_cfr was never introduced — cfr_ok remains the single coherence signal.
    assert "is_cfr" not in master_props
    assert "cfr_ok" in master_props


def test_shot_timeline_schema_fields_present(openapi_schema: dict) -> None:
    """The opt-in shot-timeline surface is advertised: the ``emit_shot_timeline``
    request flag, the ``shot_timeline`` field on the poll response (``JobStatus`` is
    the referenced response model — ``CompileResponse`` mirrors it but is not routed),
    and the ``ShotTimelineEntry`` component with its line_id/start/end keys."""
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    assert "emit_shot_timeline" in schemas["CompileRequest"]["properties"]
    assert "shot_timeline" in schemas["JobStatus"]["properties"]
    entry_props = set(schemas["ShotTimelineEntry"]["properties"])
    assert {"line_id", "start", "end"} <= entry_props


def test_tag_groups_are_defined(openapi_schema: dict) -> None:
    """``openapi_tags`` must define a group per domain.

    Tag names are lowercase to match the per-router ``tags=[…]`` declarations.
    """
    tag_names = {t["name"] for t in openapi_schema.get("tags", [])}
    required = {"health", "compile"}
    assert required <= tag_names, f"Missing required tag groups; have {sorted(tag_names)}"
