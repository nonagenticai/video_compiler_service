"""The nil uuid is not a tenant — refused BY VALUE at the inbound boundary.

``uuid.UUID(nil)`` parses, so the pre-existing ``except ValueError`` arm in
``require_tenant`` never saw it and the sentinel reached ``compile_jobs.tenant_id``
and every metering event. These tests assert the REFUSAL (the raised type, and the
HTTP status the route boundary answers) — never a mock call count, which is how a
guard like this has shipped broken before.

Mutation proof: delete the ``_reject_nil`` call in ``require_tenant`` (or the
``tenant.int == 0`` test inside it) and every case below fails — the nil uuid is
returned as a tenant and ``_caller_tenant`` answers no error at all.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException, Request

from src.api.routes.compile import _caller_tenant
from src.services.compile_job_store import (
    MissingTenantError,
    NilTenantIdError,
    require_tenant,
)

NIL = "00000000-0000-0000-0000-000000000000"


@pytest.mark.parametrize(
    "spelling",
    [
        NIL,
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
        f"  {NIL}  ",
    ],
)
def test_require_tenant_refuses_every_nil_spelling(spelling: str) -> None:
    with pytest.raises(NilTenantIdError):
        require_tenant(spelling)


def test_require_tenant_refuses_an_already_parsed_nil_uuid() -> None:
    """The ``isinstance(..., uuid.UUID)`` fast path is a boundary too."""
    with pytest.raises(NilTenantIdError):
        require_tenant(uuid.UUID(int=0))


def test_nil_refusal_is_catchable_as_the_existing_missing_tenant_error() -> None:
    """Subclassing keeps the route's existing 400 mapping working unchanged."""
    assert issubclass(NilTenantIdError, MissingTenantError)


def test_a_real_tenant_still_passes() -> None:
    """The guard must reject the sentinel, not tenants in general."""
    real = uuid.UUID("0195f48b-d000-7000-8000-000000000001")
    assert require_tenant(str(real)) == real
    assert require_tenant(real) == real


def test_caller_tenant_answers_400_for_a_request_asserting_the_nil_tenant() -> None:
    """The inbound seam: a request whose bound tenant is nil is REFUSED, not served."""
    # A real Request, not a stub: ``_caller_tenant`` reads ``request.state``, and
    # Starlette builds that from ``scope["state"]`` — so this exercises the actual
    # binding the auth middleware writes, and stays type-checkable under ``ty``.
    request = Request(
        {"type": "http", "headers": [], "state": {"tenant_id": NIL}},
    )
    with pytest.raises(HTTPException) as excinfo:
        _caller_tenant(request)
    assert excinfo.value.status_code == 400
    assert "nil uuid" in str(excinfo.value.detail)
