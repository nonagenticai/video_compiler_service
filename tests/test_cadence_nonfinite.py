"""``cadence`` promises "never raises" — a non-finite source rate broke that.

``Shot.target_fps`` is ``float | None, gt=0`` with no upper bound, and JSON ``1e400``
decodes to ``inf``, which passes pydantic. ``_coerce_fps`` accepted it (``inf > 0``),
so ``math.floor(inf)`` / ``round(inf)`` raised ``OverflowError`` inside
``judder_risk`` / ``_fmt_rate`` / ``_is_integer_ratio``. With the default
``render_cadence_mode="auto"`` the master-cadence selector runs on any compile that
declares source rates, so the whole job died with an opaque ``OverflowError`` before
a single shot rendered.

A non-finite rate carries no cadence information, so it must be treated exactly like
an unparseable one: dropped, and the selector falls back to its default.
"""

from __future__ import annotations

import math

import pytest

from src.api.schemas import Shot
from src.services import cadence

_NON_FINITE = [float("inf"), float("-inf"), float("nan")]


def test_schema_admits_a_non_finite_target_fps() -> None:
    """Precondition: an infinite target_fps really does reach the cadence selector."""
    shot = Shot.model_validate_json('{"id":"a","seconds":3.0,"target_fps":1e400}')
    assert shot.target_fps is not None
    assert math.isinf(shot.target_fps), "precondition broken — 1e400 no longer decodes to inf"


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_coerce_fps_rejects_non_finite(bad: float) -> None:
    assert cadence._coerce_fps(bad) is None


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_judder_risk_never_raises(bad: float) -> None:
    assert cadence.judder_risk(bad, 30) == 0.0


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_retime_plan_never_raises(bad: float) -> None:
    assert cadence.retime_plan(bad, 30) == "fps=30"


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_clamp_cadence_degrades_to_the_default(bad: float) -> None:
    assert cadence.clamp_cadence(bad) == 30


def test_master_cadence_ignores_a_non_finite_source_rate() -> None:
    """A shot declaring inf must not poison the whole-film cadence selection."""
    profile = cadence.source_rate_profile([24.0, 24.0, float("inf")])
    # Precondition: the non-finite rate must not have been counted as a real rate.
    assert float("inf") not in profile["rates"]
    assert cadence.master_cadence([profile]) == 24
