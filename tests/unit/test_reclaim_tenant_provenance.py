"""A reclaimed re-dispatch must restore the ORIGINAL caller's tenant provenance.

``build_job_options`` persisted ``license_id`` and ``subject_id`` so the reaper could
re-bind the enqueuer's identity, but not ``tenant_source``. That made the restored
identity look complete and be unusable: ``build_outbound_identity`` reports a ``None``
provenance as ``'unrecorded'`` and refuses every METERED destination, so a reclaimed
render died at its first outbound hop with an error that names the callee, not this
service.

⛔ The fix is NOT to read the ContextVar back at reclaim time. ``redispatch_reclaimed``
runs on the reaper's task, which claims rows across tenants, so the ambient value
belongs to whichever tenant was reclaimed last — using it would vouch one tenant's
provenance for another tenant's job. Provenance must travel IN the persisted snapshot.
That is the same distinction that makes the request-dispatched ``_bg`` case the
opposite: there the inherited value IS this request's, and must be preserved.

These tests prove the two halves TOGETHER. Each half is trivially "correct" alone: a
writer that stores a key nobody reads, or a reader that reads a key nobody writes, both
pass a single-sided test and ship a silently unvouched re-dispatch.
"""

from __future__ import annotations

import ast
import json
import pathlib
import uuid

import pytest
from w2v_common.auth.identity_context import (
    clear_identity_context,
    set_identity_context,
)
from w2v_common.tenant import TenantSource

from src.api.schemas import CompileRequest, Shot
from src.services.durable_dispatch import build_job_options, restore_tenant_source

TENANT = uuid.UUID("e148e8af-3751-49bf-a0fa-56ab46d31ddc")
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def request_model() -> CompileRequest:
    return CompileRequest(shots=[Shot(id="s1", seconds=1.0)])


def _options_through_jsonb(request_model: CompileRequest) -> dict:
    """Build options and round-trip them through JSON, as JSONB storage does."""
    opts = build_job_options(
        kind="compile", request=request_model, tenant_id=TENANT, work_timeout=60
    )
    return json.loads(json.dumps(opts))


def test_provenance_survives_the_full_enqueue_reclaim_round_trip(
    request_model: CompileRequest,
) -> None:
    """Writer → JSONB → reader, in one assertion. This is the property that matters."""
    clear_identity_context()
    set_identity_context(TENANT, "lic-1", "sub-1", tenant_source=TenantSource.LICENSE_TOKEN)
    try:
        stored = _options_through_jsonb(request_model)
    finally:
        clear_identity_context()

    # The reaper's task has no ambient provenance — exactly the condition under which
    # the old code produced None and the re-dispatch failed closed.
    assert restore_tenant_source(stored.get("tenant_source")) is TenantSource.LICENSE_TOKEN


def test_an_unvouched_enqueue_round_trips_as_unvouched_not_as_vouched(
    request_model: CompileRequest,
) -> None:
    """A bare header must not be laundered into a vouch by being persisted."""
    clear_identity_context()
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.UNVOUCHED_HEADER)
    try:
        stored = _options_through_jsonb(request_model)
    finally:
        clear_identity_context()

    assert restore_tenant_source(stored["tenant_source"]) is TenantSource.UNVOUCHED_HEADER


def test_a_job_enqueued_before_this_change_fails_closed(request_model: CompileRequest) -> None:
    """No migration is needed precisely because the absent key is the safe answer."""
    clear_identity_context()
    legacy = _options_through_jsonb(request_model)
    legacy.pop("tenant_source")  # a row written by the previous version
    assert restore_tenant_source(legacy.get("tenant_source")) is None
    assert restore_tenant_source("") is None
    assert restore_tenant_source(None) is None
    assert restore_tenant_source(12) is None


def test_an_unrecognised_persisted_value_degrades_rather_than_raising() -> None:
    """A future/renamed member must not crash the reaper — the gate downstream refuses."""
    assert restore_tenant_source("not_a_real_source") is None


def test_the_reader_reads_the_key_the_writer_writes(request_model: CompileRequest) -> None:
    """The seam itself: a key-name mismatch is invisible to either side alone.

    Read out of the source, not asserted as a literal in both places — a constant
    duplicated in the test is a third copy that can agree with neither.
    """
    clear_identity_context()
    written = set(_options_through_jsonb(request_model))

    compile_py = _SRC / "api" / "routes" / "compile.py"
    tree = ast.parse(compile_py.read_text())
    read_keys = [
        node.args[0].args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "restore_tenant_source"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and node.args[0].args
        and isinstance(node.args[0].args[0], ast.Constant)
    ]
    assert read_keys, "compile.py no longer calls restore_tenant_source(options.get(<key>))"
    for key in read_keys:
        assert key in written, (
            f"reclaim reads options[{key!r}] but build_job_options writes {sorted(written)} — "
            "the re-dispatch would restore no provenance and fail every metered hop"
        )


def test_reclaim_does_not_read_provenance_from_the_ambient_context() -> None:
    """⛔ The escalation guard: the reaper must never vouch with the last tenant's value.

    Without this, the natural "fix" — passing ``get_tenant_source_from_context()`` here,
    which is right in the request-dispatched ``_bg`` case — silently attributes one
    tenant's provenance to another tenant's reclaimed job.
    """
    compile_py = _SRC / "api" / "routes" / "compile.py"
    tree = ast.parse(compile_py.read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_identity_context"
        ):
            continue
        for kw in node.keywords:
            if kw.arg != "tenant_source":
                continue
            callee = (
                kw.value.func.id
                if isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name)
                else None
            )
            assert callee == "restore_tenant_source", (
                f"compile.py:{node.lineno} binds tenant_source from {callee!r}; the reclaim "
                "path must restore it from the persisted snapshot, not the ambient ContextVar"
            )
