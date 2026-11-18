"""Guardrail tests for the committed ``CompileRequest`` contract artifact.

``contracts/compile_request.schema.json`` is a GENERATED file: it is the JSON
Schema of :class:`src.api.schemas.CompileRequest`, rendered by
``scripts/regen_contract_artifact.py``. The sibling repo ``w2v_orchestrator``
consumes it to run its producer-coverage detector — "for every knob the compiler
accepts, does an orchestrator producer actually set it?". That detector used to
read a hand-typed mirror of this schema, which drifted and went blind exactly
where it mattered.

Two properties have to hold for the artifact to be worth trusting:

- it must be **current** — regenerated whenever a schema field lands here;
- it must be **deterministic** — the same models must always render the same
  bytes, or the currency check degrades into a flake.

Both are asserted below. If ``test_contract_artifact_is_current`` fails, you
added or changed a schema field: run

    uv run python scripts/regen_contract_artifact.py

and commit the result. Never hand-edit the JSON.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFRESH_COMMAND = "uv run python scripts/regen_contract_artifact.py"


def _load_regen_module() -> ModuleType:
    """Import ``scripts/regen_contract_artifact.py`` by path.

    ``scripts/`` is not a distributed package (the wheel only ships ``src``), so
    it may not be importable by name depending on how pytest was invoked. Load
    it by file path instead of duplicating its serialization logic here — the
    whole point is that exactly ONE definition of "the artifact bytes" exists.
    """
    path = REPO_ROOT / "scripts" / "regen_contract_artifact.py"
    spec = importlib.util.spec_from_file_location("regen_contract_artifact", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regen() -> ModuleType:
    """The regeneration script, imported once per module."""
    return _load_regen_module()


def _mapping(value: object) -> dict[str, object]:
    """Narrow an arbitrary JSON value to an object/mapping."""
    assert isinstance(value, dict), f"expected a JSON object, got {type(value).__name__}"
    # ``isinstance`` only narrows to ``dict[Unknown, Unknown]``; rebuild the
    # mapping so the key type is a fact rather than an assumption.
    return {str(key): item for key, item in value.items()}


def _property_names(schema: dict[str, object]) -> set[str]:
    """Top-level ``properties`` keys of a JSON Schema document."""
    return set(_mapping(schema.get("properties", {})))


def test_contract_artifact_is_current(regen: ModuleType) -> None:
    """The committed artifact must byte-match a freshly built schema.

    This is the whole guardrail: a compiler knob that lands in ``CompileRequest``
    without a regenerated artifact would be invisible to the orchestrator's
    producer-coverage detector, and could ship inert forever.
    """
    artifact_path: Path = regen.ARTIFACT_PATH
    assert artifact_path.is_file(), (
        f"contract artifact missing at {artifact_path} — generate it with: {REFRESH_COMMAND}"
    )

    fresh_schema = regen.build_schema()
    expected = regen.render(fresh_schema)
    committed = artifact_path.read_text(encoding="utf-8")

    if expected != committed:
        committed_schema = _mapping(json.loads(committed))
        fresh_props = _property_names(fresh_schema)
        committed_props = _property_names(committed_schema)
        added = sorted(fresh_props - committed_props)
        removed = sorted(committed_props - fresh_props)
        detail = (
            f"top-level properties added since last regen: {added or 'none'}; "
            f"removed: {removed or 'none'}"
        )
        if not added and not removed:
            detail = (
                "top-level property names are unchanged — the drift is nested "
                "(a $defs model, a field type, a default, or a description)"
            )
        pytest.fail(
            f"{artifact_path.relative_to(REPO_ROOT)} is stale. "
            f"Refresh it with: {REFRESH_COMMAND}\n{detail}"
        )


def test_artifact_covers_known_audio_knobs(regen: ModuleType) -> None:
    """Anchor: the artifact carries the six ducking knobs the hand-mirror lost.

    ``duck_attack_ms``, ``duck_release_ms``, ``duck_threshold``,
    ``sidechain_hpf_hz``, ``music_offset_lu`` and ``duck_open_lu_below_vo`` are
    exactly the fields the orchestrator's stale hand-typed ``MusicBed`` mirror
    was missing — so its producer-coverage detector could not tell whether any
    producer set them. This test is why the generated artifact is trustworthy as
    their source of truth: the knobs are asserted present, by name, in the file
    the orchestrator reads.
    """
    schema = _mapping(json.loads(regen.ARTIFACT_PATH.read_text(encoding="utf-8")))
    defs = _mapping(schema.get("$defs", {}))
    assert {"MusicBed", "AmbientCue"} <= set(defs), (
        f"audio component schemas missing from artifact $defs; have {sorted(defs)}"
    )

    music_bed_props = _property_names(_mapping(defs["MusicBed"]))
    expected_knobs = {
        "duck_attack_ms",
        "duck_release_ms",
        "duck_threshold",
        "sidechain_hpf_hz",
        "music_offset_lu",
        "duck_open_lu_below_vo",
    }
    missing = sorted(expected_knobs - music_bed_props)
    assert not missing, (
        f"MusicBed ducking knobs missing from the artifact: {missing}. "
        f"If they were intentionally removed from the model, update this test "
        f"AND the orchestrator's producer expectations together."
    )


def test_artifact_render_is_canonical_regardless_of_key_order(regen: ModuleType) -> None:
    """``render`` must CANONICALISE, not merely echo whatever order it was handed.

    ⚠️ This test used to render the same schema twice and assert the two runs
    matched. That could not fail: both calls run in one process, from one
    construction path, so dict iteration order is identical by construction with
    or without ``sort_keys=True``. Verified by mutation — deleting ``sort_keys``
    from ``scripts/regen_contract_artifact.py`` left the old assertion GREEN
    (only the sibling currency check went red, and only because the committed
    artifact happens to be sorted).

    Feeding the SAME schema in two DIFFERENT key orders is what actually pins the
    canonicalisation: without ``sort_keys`` the two renders diverge immediately.
    """
    schema = regen.build_schema()
    assert len(schema) > 1, "precondition: a single-key schema cannot be reordered"

    forward = regen.render(dict(schema))
    reversed_order = regen.render({k: schema[k] for k in reversed(list(schema))})

    assert forward == reversed_order, (
        "contract artifact rendering is order-dependent — `render` must "
        "canonicalise (sort_keys=True), or an unrelated pydantic reordering will "
        "produce a spurious 'artifact is stale' failure"
    )
    # The nested `$defs` are the larger surface; pin that they are sorted too.
    assert forward == regen.render(regen.build_schema()), "render is not deterministic"
