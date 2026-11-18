"""A loudness measurement must be JSON-safe, or it poisons the completion write.

ffmpeg's ``loudnorm`` prints ``"input_i": "-inf"`` for a digitally-silent mux —
which is a real reading, so ``_as_float`` deliberately preserves it. But that
value does not stop at the parser: ``compile.py`` puts the whole dict on
``result["loudness"]``, ``job_store.mark_completed`` writes ``result`` into the
``CompileJob.result`` JSONB column, and no custom ``json_serializer`` is
configured on the engine — so ``json.dumps`` emits the bare token ``-Infinity``
and postgres rejects the entire write::

    psycopg.errors.InvalidTextRepresentation:
      invalid input syntax for type json
      DETAIL: Token "-Infinity" is invalid.

``_finalize_completed`` then retries three times with the IDENTICAL payload — a
retry that varies nothing can never succeed — the row stays ``processing``, the
reaper reclaims the lease and RE-RENDERS the whole job, and it eventually
reports ``failed`` for an artifact that was actually delivered and never billed.

``acx_measure`` already scrubs exactly this through ``_finite_or_none``;
``parse_loudness`` did not. These tests pin the parity.
"""

from __future__ import annotations

import json
import math

import pytest

from src.services.loudness import parse_loudness

# What ffmpeg really prints for a silent mux.
SILENT_BLOCK = '{ "input_i" : "-inf", "input_tp" : "-inf", "input_lra" : "0.00" }'


def test_a_silent_mux_measurement_carries_no_non_finite_value() -> None:
    """RED before the fix: ``integrated_lufs``/``true_peak_dbtp`` were ``-inf``.

    An unmeasurable value must be ABSENT (``None``), never ``±inf``/``nan`` —
    the same contract ``acx_measure`` already honours."""
    result = parse_loudness(SILENT_BLOCK)
    assert result["integrated_lufs"] is None
    assert result["true_peak_dbtp"] is None
    # A finite sibling on the same block is untouched — the fix scrubs, not drops.
    assert result["lra"] == pytest.approx(0.0)


def test_no_key_on_the_measurement_is_non_finite() -> None:
    """Every key, not just the two loudnorm ones — the JSONB write serializes the
    WHOLE dict, so one surviving ``-inf`` anywhere fails the same way."""
    result = parse_loudness(SILENT_BLOCK, "  Duration: 00:00:03.02, start: 0.0\n")
    for key, value in result.items():
        assert value is None or math.isfinite(value), f"{key}={value!r} is not finite"


def test_the_measurement_survives_the_serializer_postgres_actually_uses() -> None:
    """The defect proved at the boundary that broke: ``json.dumps`` with the
    default (non-``allow_nan``) settings postgres' JSONB codec requires.

    RED before the fix this raised ``ValueError: Out of range float values are
    not JSON compliant``; in production the permissive default emitted
    ``-Infinity`` and postgres rejected the row instead."""
    payload = {"loudness": parse_loudness(SILENT_BLOCK)}
    encoded = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in encoded
    assert "NaN" not in encoded


def test_a_normal_measurement_is_unchanged() -> None:
    """GUARD: the fix must scrub non-finite readings, not blunt real ones."""
    block = '{ "input_i" : "-14.20", "input_tp" : "-2.10", "input_lra" : "5.50" }'
    result = parse_loudness(block)
    assert result["integrated_lufs"] == pytest.approx(-14.20)
    assert result["true_peak_dbtp"] == pytest.approx(-2.10)
    assert result["lra"] == pytest.approx(5.50)
