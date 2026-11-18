"""A marker that CI never re-selects makes its tests run on NO machine.

``pyproject.toml``'s ``addopts`` carries ``-m 'not integration'`` and
``.github/workflows/ci-cd.yaml`` re-selects that marker NOWHERE — no
``integration_test_select`` input, no second pytest invocation. So until
2026-09-07 every test wearing ``integration`` was deselected in PR CI, on main,
and nightly: SIX tests across ``tests/test_audiobook_acx.py`` and
``tests/test_loudness.py``, including the regression proof that the combined ACX
concat must RE-ENCODE its AAC/m4b inputs rather than ``-c copy`` them (the bug
silently dropped ~30% of the book). Reverting that fix to its original defect
would have been a green diff.

Those six needed nothing CI lacks — only real ffmpeg/ffprobe, which the
archetype installs via ``extra_apt: ffmpeg``. They now wear ``render``, the
marker this repo already uses for real-encode tests and which nothing
deselects.

This gate makes the hole un-reopenable: a marker listed in ``addopts``'s
deselection expression may only be worn by a file that appears in
``ALLOWED_UNRUN`` below, with a written reason that is PRINTED on every run.

Three deliberate design choices, each learned the hard way in this fleet:

* **AST, not regex.** A fleet gate once used a bare regex for its escape hatch
  and matched its own marker string inside a DOCSTRING, silencing itself; a
  regex that matches nothing also passes vacuously. ``ast`` sees only real
  attribute references, so the three modules here that merely *talk about*
  ``pytest.mark.integration`` in prose (this one, ``test_audiobook_acx.py``,
  ``test_loudness.py``, ``tests/unit/test_deband_ab_decision.py``) are invisible
  to it.
* **The allowlist is a Python dict whose reason is ECHOED.** The same incident
  shipped a gate that parsed its waiver reason then DISCARDED it, so every
  waiver rendered identically and a deliberate exemption was indistinguishable
  from an accident.
* **It has VACUITY FLOORS.** "Nothing is marked wrong" is trivially satisfied by
  deleting the tests. ``REQUIRED_CI_SELECTED`` names the specific real-ffmpeg
  proofs that must exist AND be CI-selected, and ``MIN_CI_SELECTED_TESTS`` is a
  count floor over the whole suite — so deleting or re-hiding fails just as
  loudly as marking.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"

#: Files permitted to wear a never-selected marker, mapped to WHY they cannot
#: run in CI. Adding an entry is a deliberate, reviewable statement that the test
#: needs something CI genuinely does not have — a live peer service or real
#: outbound network. Needing Postgres is NOT a qualifying reason (the archetype
#: provisions an ephemeral Postgres per job via ``has_postgres: true``), and
#: neither is needing ffmpeg (``extra_apt: ffmpeg``).
#:
#: EMPTY BY DESIGN as of 2026-09-07: nothing in this repo needs a live peer.
ALLOWED_UNRUN: dict[str, str] = {}

#: VACUITY FLOOR 1 — the real-ffmpeg proofs un-hidden on 2026-09-07. They must
#: exist AND be CI-selected. Do not shorten this list to make the file pass.
REQUIRED_CI_SELECTED: dict[str, tuple[str, ...]] = {
    "tests/test_audiobook_acx.py": (
        "test_acx_master_lands_in_band_and_neg_control",
        "test_acx_master_silent_stitch_degrades_not_crashes",
        "test_m4b_has_chapter_markers",
        "test_acx_master_m4b_inputs_not_truncated",
        "test_m4b_without_master_still_chaptered",
    ),
    "tests/test_loudness.py": ("test_final_mux_loudness_measures_real_file",),
}

#: The files whose tests were hidden, and the floor beneath their combined test
#: count. Measured 2026-09-07: 14 + 11 = 25 test functions.
UNHIDDEN_FILES: tuple[str, ...] = (
    "tests/test_audiobook_acx.py",
    "tests/test_loudness.py",
)
MIN_UNHIDDEN_TESTS = 22

#: VACUITY FLOOR 2 — the minimum number of test functions under ``tests/`` that
#: do NOT wear a deselected marker. Measured 2026-09-07: 1011. The floor sits
#: below that so ordinary churn does not trip it, but wholesale deletion — or
#: mass re-marking, which moves tests out of this count — does.
MIN_CI_SELECTED_TESTS = 940


def _deselected_markers() -> set[str]:
    """The markers ``addopts`` excludes, read from pyproject — not hardcoded.

    Reading it back means adding a SECOND marker to the deselection expression
    (``render``, say) without allowlisting or wiring it also trips this gate.
    """
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    addopts = cfg["tool"]["pytest"]["ini_options"]["addopts"]
    if isinstance(addopts, list):
        addopts = " ".join(addopts)
    tokens = addopts.replace("'", " ").replace('"', " ").split()
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok == "not" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in {"not", "and", "or"}:
                out.add(nxt)
    return out


def _marker_names_used(path: Path) -> set[str]:
    """Every ``<something>.mark.<name>`` referenced anywhere in the module.

    Covers decorators (function AND class), module-level ``pytestmark`` (scalar
    or list), marks bound to a module-level name and reused as a decorator, and
    marks inside ``pytest.param``. Comments and docstrings are not AST attribute
    nodes, so prose about a marker cannot trip this.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = node.value
        if isinstance(parent, ast.Attribute) and parent.attr == "mark":
            names.add(node.attr)  # pytest.mark.<name>
        elif isinstance(parent, ast.Name) and parent.id == "mark":
            names.add(node.attr)  # from pytest import mark; mark.<name>
    return names


def _is_mark_ref(node: ast.AST) -> bool:
    """True for an AST node that is a real ``*.mark.<name>`` attribute access."""
    if not isinstance(node, ast.Attribute):
        return False
    parent = node.value
    return (isinstance(parent, ast.Attribute) and parent.attr == "mark") or (
        isinstance(parent, ast.Name) and parent.id == "mark"
    )


def _marks_in(nodes: list[ast.expr], markers: set[str]) -> set[str]:
    """The deselected markers referenced anywhere inside the given expressions."""
    return {
        sub.attr
        for node in nodes
        for sub in ast.walk(node)
        if _is_mark_ref(sub) and isinstance(sub, ast.Attribute)
    } & markers


def _module_level_marks(tree: ast.Module, markers: set[str]) -> set[str]:
    """Deselected markers applied to the whole module via ``pytestmark``."""
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets)
    ]
    return _marks_in(values, markers)


def _marked_tests(path: Path, markers: set[str]) -> dict[str, set[str]]:
    """Map each ``test_*`` function to the deselected markers IT wears.

    ``_marker_names_used`` answers "does this FILE contain a hidden marker",
    which is enough to fail the build but not enough to fix it: a 1000-line
    module named alone sends the reader hunting. This attributes the marker to
    the decorated function, so the failure message names the test that will not
    run. Module-level ``pytestmark`` and class decorators hide every test
    beneath them, so they are attributed to each such test rather than to none.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    inherited = _module_level_marks(tree, markers)

    # Marks a test inherits from an enclosing decorated class.
    from_class: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        cls_marks = _marks_in(node.decorator_list, markers)
        if not cls_marks:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef) and sub.name.startswith(
                "test_"
            ):
                from_class.setdefault(sub.name, set()).update(cls_marks)

    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        worn = (
            _marks_in(node.decorator_list, markers) | inherited | from_class.get(node.name, set())
        )
        if worn:
            out[node.name] = worn
    return out


def _collected_test_names(path: Path) -> set[str]:
    """``test_*`` functions in a module, via AST (no import side effects).

    Walks the whole tree, not just module top level, so class-nested tests count.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
    }


def test_addopts_deselection_is_covered_by_this_gate() -> None:
    """The expression must be parseable and non-degenerate, or the gate is inert."""
    markers = _deselected_markers()
    assert markers, (
        "Could not parse any deselected marker out of pyproject.toml's "
        "[tool.pytest.ini_options] addopts. This gate reads that expression to "
        "know what to police; if the parse yields nothing the gate silently "
        "polices nothing. Fix the parser in _deselected_markers(), do not "
        "delete this assertion."
    )
    declared = set(
        tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
            "markers"
        ]
    )
    declared_names = {m.split(":", 1)[0].strip() for m in declared}
    unknown = markers - declared_names
    assert not unknown, (
        f"addopts deselects marker(s) {sorted(unknown)} that are not declared in "
        "[tool.pytest.ini_options].markers. A typo'd marker name in the "
        "deselection expression deselects nothing and hides the mistake."
    )


def test_no_test_wears_a_marker_ci_never_selects(capsys: pytest.CaptureFixture[str]) -> None:
    markers = _deselected_markers()
    offenders: dict[str, dict[str, set[str]]] = {}
    waived: list[str] = []

    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        used = _marker_names_used(path) & markers
        if not used:
            continue
        if rel in ALLOWED_UNRUN:
            reason = ALLOWED_UNRUN[rel].strip()
            assert reason, (
                f"{rel} is in ALLOWED_UNRUN with an EMPTY reason. A waiver whose "
                "reason is blank is indistinguishable from an accident — state "
                "what CI genuinely lacks (a live peer service, real network)."
            )
            # PRINTED, not merely parsed.
            waived.append(f"  {rel} [{','.join(sorted(used))}] — {reason}")
            continue
        # Attribute the marker to the individual tests, so the failure names the
        # tests that will not run rather than only the file they live in.
        per_test = _marked_tests(path, markers)
        offenders[rel] = per_test or {"<file-level reference>": used}

    if waived:
        with capsys.disabled():
            print(
                "\nDELIBERATELY UNRUN IN CI (each waiver and its stated reason):\n"
                + "\n".join(sorted(waived))
            )

    if offenders:
        detail = "\n".join(
            f"  {f}::{name} wears {sorted(m)}"
            for f, per_test in sorted(offenders.items())
            for name, m in sorted(per_test.items())
        )
        pytest.fail(
            "These tests wear a pytest marker that pyproject's `addopts` "
            "DESELECTS and that CI re-selects NOWHERE — so they execute on no "
            f"machine at all:\n{detail}\n\n"
            "Six ACX/loudness proofs sat here until 2026-09-07; reverting the "
            "AAC-in-wav truncation fix to its original bug was a green diff.\n\n"
            "Fix it by using `render` instead (nothing deselects it, and the "
            "archetype installs ffmpeg via `extra_apt`; an ephemeral Postgres is "
            "provisioned per job, so 'it needs a DB' is not a reason to hide a "
            "test), or — if the test truly needs a live peer service or real "
            "outbound network — add it to ALLOWED_UNRUN in "
            f"{Path(__file__).name} with a written reason, which is a reviewable "
            "act rather than a silent one."
        )


def test_the_real_ffmpeg_proofs_are_present_and_ci_selected() -> None:
    """VACUITY FLOOR 1: deleting the tests must fail as loudly as hiding them."""
    markers = _deselected_markers()
    problems: list[str] = []

    for rel, required in REQUIRED_CI_SELECTED.items():
        path = _REPO_ROOT / rel
        if not path.exists():
            problems.append(f"{rel} is GONE — it held {len(required)} required tests.")
            continue
        present = _collected_test_names(path)
        missing = [n for n in required if n not in present]
        if missing:
            problems.append(f"{rel}: missing required test(s) {missing}")
        per_test = _marked_tests(path, markers)
        for name in required:
            if worn := per_test.get(name):
                problems.append(
                    f"{rel}::{name} wears {sorted(worn)}, which CI deselects — "
                    "this required proof would run on no machine."
                )

    assert not problems, (
        "The real-ffmpeg proofs this gate exists to keep RUNNING are not all "
        "present and CI-selected:\n  " + "\n  ".join(problems) + "\n\n"
        "These are the DELIVERED-BYTES proofs: that the ACX master lands the book "
        "inside the ACX loudness window (with a negative control), that a silent "
        "stitch degrades instead of crashing, that the m4b carries real chapter "
        "markers, and that the combined concat does not truncate AAC inputs. The "
        "command-shape tests above them prove only the argv. Restore them — do "
        "not shorten REQUIRED_CI_SELECTED to make this pass."
    )


def test_the_unhidden_files_still_carry_their_tests() -> None:
    """VACUITY FLOOR 1b: the 25 tests in the un-hidden files must not evaporate."""
    total = 0
    missing_files: list[str] = []
    for rel in UNHIDDEN_FILES:
        path = _REPO_ROOT / rel
        if not path.exists():
            missing_files.append(rel)
            continue
        total += len(_collected_test_names(path))

    assert not missing_files, (
        f"File(s) un-hidden on 2026-09-07 no longer exist: {missing_files}. If a "
        "file was legitimately split or renamed, update UNHIDDEN_FILES in the same "
        "commit so this floor keeps measuring something."
    )
    assert total >= MIN_UNHIDDEN_TESTS, (
        f"Only {total} tests remain across the files this gate un-hid, below the "
        f"floor of {MIN_UNHIDDEN_TESTS} (25 at the time of measurement)."
    )


def test_the_suite_has_not_been_emptied_to_satisfy_this_gate() -> None:
    """VACUITY FLOOR 2: a count floor over everything CI actually selects."""
    markers = _deselected_markers()
    total = 0
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if _marker_names_used(path) & markers:
            continue  # not CI-selected; deliberately excluded from the count
        total += len(_collected_test_names(path))

    assert total >= MIN_CI_SELECTED_TESTS, (
        f"Only {total} CI-selected test functions remain under tests/, below the "
        f"floor of {MIN_CI_SELECTED_TESTS} (1011 at the 2026-09-07 measurement). "
        "A gate that only asserts 'nothing is marked wrong' is satisfiable by "
        "deleting the tests, or by mass-marking them `integration`; this is the "
        "assertion that catches that."
    )


def test_stale_allowlist_entries_are_removed() -> None:
    """A waiver for a file that no longer wears the marker is dead weight."""
    markers = _deselected_markers()
    stale = []
    for rel, reason in ALLOWED_UNRUN.items():
        path = _REPO_ROOT / rel
        if not path.exists() or not (_marker_names_used(path) & markers):
            stale.append(f"{rel} (waived because: {reason})")
    assert not stale, (
        "ALLOWED_UNRUN entries no longer needed (file gone, or marker removed):\n  "
        + "\n  ".join(stale)
        + "\nDelete them so the allowlist keeps meaning what it says."
    )
