"""Regenerate the committed ``CompileRequest`` JSON-Schema contract artifact.

WHY THIS EXISTS
---------------
``w2v_orchestrator`` needs to know the *full* shape of this service's compile
request in order to run its producer-coverage detector: for every knob the
compiler accepts, is there an orchestrator producer that actually sets it? That
detector is the only guard against this project's signature bug — "the compiler
grows a knob, no producer ever sets it, and the field is silently inert in
production".

Historically the orchestrator carried a HAND-TYPED mirror of ``CompileRequest``
in its contract test. It drifted badly (``MusicBed`` had 5 of 11 real fields,
``Shot`` had 11 of 25), so the detector was blind to precisely the fields it
existed to catch. A hand mirror can only ever be as fresh as the last human who
remembered to update it.

So this repo — the *owner* of the schema — commits a generated artifact at
``contracts/compile_request.schema.json`` and the orchestrator consumes it. This
service is the single source of truth; the orchestrator does not retype
anything.

WHY ``sort_keys=True``
----------------------
The artifact is guarded by a byte-for-byte comparison in
``tests/test_contract_artifact.py``. Pydantic builds the schema out of dicts and
sets whose iteration order is not guaranteed to be stable across runs, versions,
or interpreters. ``sort_keys=True`` (plus a fixed ``indent=2`` and a trailing
newline) canonicalises the rendering, so the only thing a diff can ever mean is
"the schema really changed" — never "the dict came out in a different order".
That stability is what makes the byte-compare a usable CI gate instead of a
flake.

USAGE
-----
From the repo root::

    uv run python scripts/regen_contract_artifact.py

Then commit the resulting ``contracts/compile_request.schema.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``python scripts/regen_contract_artifact.py`` from the repo root: the
# script's own directory (not the root) is what lands on sys.path by default.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.schemas import CompileRequest  # noqa: E402

#: Committed artifact consumed by the orchestrator's producer-coverage detector.
ARTIFACT_PATH: Path = REPO_ROOT / "contracts" / "compile_request.schema.json"

#: A parsed JSON Schema document — nested values are arbitrary JSON.
type JsonSchema = dict[str, object]


def build_schema() -> JsonSchema:
    """Return the JSON Schema for ``CompileRequest``, ``$defs`` and all."""
    # pydantic types this as ``dict[str, Any]``; bind it to the narrower alias
    # here so ``Any`` stops at this boundary instead of leaking to callers.
    schema: JsonSchema = dict(CompileRequest.model_json_schema())
    return schema


def render(schema: JsonSchema) -> str:
    """Canonicalise ``schema`` to the exact bytes stored on disk.

    ``sort_keys=True`` is load-bearing — see the module docstring. Any change to
    this function's output format is a change to the committed artifact and must
    be followed by a regeneration.
    """
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write the artifact and report the path. Returns a process exit code."""
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(render(build_schema()), encoding="utf-8")
    print(f"wrote {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
