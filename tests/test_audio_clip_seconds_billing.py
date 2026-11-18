"""``AudioClip.seconds`` is a BILLING input, and its ``gt=0`` was a money validator.

THE FIELD IS NOT A RENDERING ONE. ``renderer.concat_audio`` reads only
``clip.audio_url``; there is not one read of ``.seconds`` in ``renderer.py`` or
``mastering.py``. Its single reader is ``compile._declared_clip_seconds``, whose
all-or-nothing sum becomes ``declared_duration_seconds`` →
``_billed_duration`` → the emitted cost event — the nominal charge used when the
delivered master could not be probed.

TWO defects met on that field:

1. An ADVISORY field that HARD-REJECTED. ``gt=0`` turned a ``0.0`` into a 422 that
   failed the terminal stitch AFTER every clip's audio had been generated and paid
   for. (Found by a peer session.)

2. The 422 was ACCIDENTALLY HOLDING A MONEY INVARIANT. Removing it naively — a
   reader-side skip — trades a loud failure for a silent UNDER-BILL: four 100s
   clips with one zeroed sum to 300.0 and bill a nominal 300 for 400 seconds
   delivered, with nothing saying a number was dropped. An over-bill is permanent;
   an under-bill is invisible.

Normalising to ``None`` at parse time satisfies both: the request is accepted (no
422 on already-paid audio) AND the all-or-nothing sum makes the WHOLE request
unmeasured, so it is not billed at all rather than billed short.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.routes.compile import (
    _DURATION_NOMINAL,
    _DURATION_UNMEASURED,
    _billed_duration,
    _declared_clip_seconds,
)
from src.api.schemas import AudioClip, AudioConcatRequest


def _request(*seconds: float | None) -> AudioConcatRequest:
    return AudioConcatRequest.model_validate(
        {
            "clips": [
                {"id": f"c{i}", "audio_url": f"file:///tmp/c{i}.wav", "seconds": s}
                for i, s in enumerate(seconds)
            ]
        }
    )


def _billed_for(*seconds: float | None) -> tuple[float | None, str]:
    """The FULL chain a request's declared seconds actually travel, so the
    assertion is about the money and not about an intermediate."""
    return _billed_duration(
        {
            # An artifact WAS delivered, and its probe carried no duration — the one
            # branch in which the declared sum becomes the charge.
            "artifact_url": "https://minio.test/audiobook.m4b",
            "declared_duration_seconds": _declared_clip_seconds(_request(*seconds)),
        }
    )


# ---------------------------------------------------------------------------
# 1. The 422 on already-paid audio is gone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.001, float("nan"), float("inf"), float("-inf")])
def test_a_non_positive_advisory_duration_is_accepted_and_normalised(bad: float) -> None:
    """The parse SUCCEEDS (no 422 after the audio was paid for) and the value that
    means nothing is stored as nothing.

    ``-1.0`` is the load-bearing case for the normaliser's shape: it is TRUTHY, so
    a naive ``if not value`` would let it through and BILL A NEGATIVE duration.
    ``inf`` is strictly positive, so a naive ``<= 0`` would let it through and bill
    an unbounded one.
    """
    clip = AudioClip(id="c0", audio_url="file:///tmp/c0.wav", seconds=bad)

    assert clip.seconds is None


@pytest.mark.parametrize("good", [0.001, 1.0, 100.0, 3600.5])
def test_a_positive_advisory_duration_is_preserved_exactly(good: float) -> None:
    """NOT VACUOUS: the normaliser must not swallow real measurements, or the
    whole billing fallback would silently stop existing."""
    assert AudioClip(id="c0", audio_url="file:///tmp/c0.wav", seconds=good).seconds == good


def test_the_field_is_still_a_float_and_still_rejects_nonsense() -> None:
    """Normalising non-positive numbers is not the same as accepting anything —
    a non-numeric value is still a client error, caught before any audio is made."""
    with pytest.raises(ValidationError):
        AudioClip(id="c0", audio_url="file:///tmp/c0.wav", seconds="ninety")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. THE UNDER-BILL. The measured 400 -> 300 regression, as a test.
# ---------------------------------------------------------------------------


def test_four_full_clips_bill_the_full_declared_sum() -> None:
    """The control. Without it the assertion below cannot distinguish "the fix
    works" from "this path never bills anything"."""
    assert _billed_for(100.0, 100.0, 100.0, 100.0) == (400.0, _DURATION_NOMINAL)


@pytest.mark.parametrize("zeroed", [0.0, -1.0, float("nan"), float("inf")])
def test_one_unmeasured_clip_makes_the_whole_request_unbilled_never_short(
    zeroed: float,
) -> None:
    """THE MONEY INVARIANT, and the reason a reader-side skip was rejected.

    The regression this pins: with the value KEPT, these four clips declare 300.0
    and bill a nominal 300 for 400 seconds of delivered audio — short, and silent.
    The invariant asserted is not a call count and not the sum: it is that the
    request is UNMEASURED and therefore UNCHARGED. Partial credit is the failure.
    """
    billed, source = _billed_for(100.0, zeroed, 100.0, 100.0)

    assert source == _DURATION_UNMEASURED
    assert billed is None
    # Said the other way, because this is the specific number the bug produced:
    # never a smaller measurement of the same delivery.
    assert billed != 300.0


@pytest.mark.parametrize("zeroed", [0.0, -1.0])
def test_the_declared_sum_itself_is_absent_not_short(zeroed: float) -> None:
    """The same invariant one level down, at the single reader — so a future change
    to ``_billed_duration`` cannot quietly restore the partial sum upstream of it."""
    assert _declared_clip_seconds(_request(100.0, zeroed, 100.0, 100.0)) is None


def test_an_explicit_null_is_still_unmeasured() -> None:
    """The pre-existing contract the normalisation is deliberately routed INTO:
    ``None`` already meant unmeasured-and-unbilled, so a non-positive number now
    reaches an established behaviour rather than a new one."""
    assert _billed_for(100.0, None, 100.0, 100.0) == (None, _DURATION_UNMEASURED)


# ---------------------------------------------------------------------------
# 3. CROSS-SERVICE CONTRACT.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("advisory", [0.0, -1.0, float("nan"), float("inf"), None])
def test_the_compiler_guarantees_what_the_orchestrator_merely_practises(
    advisory: float | None,
) -> None:
    """The audiobook stitch caller collapses non-finite/<=0/absent to ``None``
    before it ever POSTs (``w2v_orchestrator``'s ``audiobook/stitch.py``
    ``_advisory_seconds``), which is why nothing is bleeding today.

    THAT IS CALLER DISCIPLINE, NOT AN INVARIANT. Nothing tested that the two sides
    agreed, and the compiler is one careless caller — a retry path, a second
    consumer, a hand-rolled curl — away from a silent mis-bill. This asserts the
    COMPILER-SIDE guarantee, which is the half this repo can actually keep: the
    same inputs the orchestrator normalises away are normalised here too, so the
    contract holds even when the caller stops honouring it.

    Deliberately NOT importing from ``w2v_orchestrator`` (a peer owns it and this
    service must not depend on its caller) — the contract is restated as the
    compiler's own postcondition.
    """
    clip = AudioClip(id="c0", audio_url="file:///tmp/c0.wav", seconds=advisory)

    assert clip.seconds is None
    # And it propagates: an unnormalised value here would re-enter the billing sum.
    assert _declared_clip_seconds(_request(100.0, advisory)) is None


def test_a_request_the_orchestrator_would_send_bills_normally() -> None:
    """Non-vacuity for the contract test: the ordinary, well-formed request the
    orchestrator actually sends is unaffected and still bills its full sum."""
    assert _billed_for(12.5, 30.25, 7.0) == (49.75, _DURATION_NOMINAL)
