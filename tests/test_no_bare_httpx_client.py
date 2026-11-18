"""CI-enforced guard: no UNDOCUMENTED bare ``httpx`` client under ``src/``.

Why this exists
---------------
Outbound service-to-service calls must forward the identity triple
``X-Tenant-Id`` / ``X-License-Id`` / ``X-Subject-Id``. Where the convention is
used it works and needs no discipline:
``w2v_common.auth.identity_context.get_identity_headers`` is auto-installed as an
httpx ``event_hooks={"request": [...]}`` handler by
``w2v_common.clients.base.BaseServiceClient``, so **every** subclass forwards
correctly without a single line of per-call-site code.

The droppers are therefore never the clients — they are the raw
``httpx.AsyncClient(...)`` / ``httpx.Client(...)`` constructions that bypass the
base class entirely. That is a *class* of defect, not an instance: each one is
individually plausible at review time ("it's just a fetch"), and nothing tells
you the headers silently vanished. Hence a static gate rather than a round of
one-off fixes.

How it decides
--------------
An AST walk over every ``src/**/*.py`` finds each construction of ``httpx.Client``
/ ``httpx.AsyncClient`` (attribute form, and the ``from httpx import AsyncClient``
form). Each hit is keyed by ``(path, enclosing-function-qualname)`` — stable
across line-number churn, unlike a ``file:line`` allowlist. Any key not in
:data:`ALLOWED` fails the test.

An exemption is NOT just a line in a list. Every entry carries a **category** and
a **written reason**, and the categories that make a claim about code are
*structurally substantiated* — the guard re-checks the claim against the AST of
the exempted function, so an exemption cannot quietly outlive the thing that
justified it:

* ``SSRF_SINK`` — the function must still CALL one of the guarded fetch helpers
  (:data:`_SSRF_MARKERS`), checked against AST identifiers rather than source
  text. If someone drops ``_fetch`` for a raw ``http.get``, the exemption stops
  applying and this test reds — and it stays red even though the function's
  docstring still talks about SSRF-safe fetching at length. (The orchestrator's
  first cut of this guard grepped source text and passed a mutation, because a
  *docstring* mentioned ``safe_get``. Prose is not code.)
* ``GUARDED_POOL`` — a long-lived pooled client that mostly does not fetch
  anything itself; it is handed down to consumers that each fetch through the
  guard. Those consumers are named in the entry and re-verified one level down:
  each must exist AND reference a guard marker. Rename or unguard a consumer and
  this reds.
* ``IDENTITY_FORWARDED`` — the function must still reference
  ``get_identity_headers`` / ``forward_headers``. Currently unused in this repo:
  no bare client here forwards the triple by hand. Kept because the next one
  might, and an unclaimed category is cheaper than a hand-rolled exception.
* ``NON_W2V_PEER`` — reason only. Something outside the w2v S2S convention where
  the triple is not the auth channel. Also currently unused here: this service
  has no ``src/clients/`` at all — it is a terminal renderer, not a caller.

Two directions of non-vacuity, because an allowlist rots both ways:

* a **stale entry that matches nothing fails** — when a call site is deleted or
  routed through ``BaseServiceClient``, its exemption must go with it;
* a **scanner that finds nothing fails** — a broken finder passes every other
  test in this file, so a floor and the three known keys are pinned.

And one guard on the guard: :func:`test_ssrf_markers_actually_reach_safe_get`
recomputes, from the AST, that every declared marker transitively reaches
``w2v_common.net.safe_get``. Adding a plausible-looking name that fetches raw is
the obvious way to defeat the substantiation check; it is now a red test.

Why every entry and zero "fixes"
--------------------------------
An audit named three sites here as identity-dropping S2S calls. All three are
**false positives**, and the same finding was already disproved in
``w2v_orchestrator`` (13 sites, zero droppers) and ``analysis_service`` (20
sites, zero droppers). Every construction in this repo is a transport handed to
``_fetch``, which calls ``w2v_common.net.safe_get`` (per-hop redirect
re-validation + IP pinning + a streaming byte cap) to pull a CALLER-SUPPLIED
media URL: a shot clip, an audiobook clip, a poster source. Those URLs are
normally our own presigned MinIO artifacts, but they arrive in the request body,
so attaching ``X-Tenant-Id: Bearer <license>`` would post a live credential to
whatever host the caller put in the brief. Forwarding identity there would be the
vulnerability, not the fix. Nothing in ``src/`` constructs a bare client aimed at
a w2v peer — there is no peer client in this service to construct one for. That
is the honest finding. The value of this file is the floor it sets for the NEXT
call site.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

_SRC = Path(__file__).resolve().parent.parent / "src"

# The estate SSRF chokepoint, and the in-repo chain that terminates in it.
# Every name here is re-proved to reach ``safe_get`` by
# ``test_ssrf_markers_actually_reach_safe_get`` — do NOT add a name on the
# strength of it sounding like a fetch helper.
#   safe_get      w2v_common.net — resolve once, pin the IP, re-validate every
#                 redirect hop, abort the transfer at ``max_fetch_bytes``
#   safe_get_to_path  w2v_common.net — the SAME redirect/pinning/cap loop
#                 (``_safe_request_to_sink``) with a file sink instead of a
#                 ``list[bytes]`` terminal, so shot assets up to
#                 ``max_fetch_bytes`` (now 4 GiB) stream to disk in constant
#                 memory. Equally a chokepoint, NOT a bypass of safe_get: it is
#                 the same guard code, and the reachability seed below names both
#                 for exactly that reason.
#   _fetch        renderer.py — the only caller of safe_get_to_path in this repo
#   _try_fetch    renderer.py — best-effort _fetch (degrade one shot, not the job)
#   _fetch_asset  renderer.py — _try_fetch behind the per-job source dedupe cache
_SSRF_MARKERS = (
    "safe_get",
    "safe_get_to_path",
    "_fetch",
    "_try_fetch",
    "_fetch_asset",
)

# Names whose presence proves the function still stamps the identity triple by
# hand instead of inheriting the BaseServiceClient event hook.
_IDENTITY_MARKERS = ("get_identity_headers", "forward_headers")

SSRF_SINK = "SSRF_SINK"
GUARDED_POOL = "GUARDED_POOL"
IDENTITY_FORWARDED = "IDENTITY_FORWARDED"
NON_W2V_PEER = "NON_W2V_PEER"

_MARKERS: dict[str, tuple[str, ...]] = {
    SSRF_SINK: _SSRF_MARKERS,
    GUARDED_POOL: (),
    IDENTITY_FORWARDED: _IDENTITY_MARKERS,
    NON_W2V_PEER: (),
}

# A reason short enough to be a shrug is not a reason.
_MIN_REASON_CHARS = 60


class Exemption(NamedTuple):
    category: str
    reason: str
    # GUARDED_POOL only: (path, qualname) of each consumer that actually fetches
    # with this client. Each must exist and reference an _SSRF_MARKERS name.
    consumers: tuple[tuple[str, str], ...] = ()


class Construction(NamedTuple):
    path: str  # posix, relative to the repo root
    qualname: str  # enclosing def/class chain, or "<module>"
    lineno: int
    # Identifiers actually REFERENCED in the enclosing function — names and
    # attribute accesses, collected from the AST. Deliberately not the source
    # text: a guard that reads prose proves nothing about code (same defect
    # class as an Any-ratchet that counts comments).
    identifiers: frozenset[str]


# ── The allowlist ────────────────────────────────────────────────────────────
# Key: (path relative to repo root, enclosing function qualname).
# Adding an entry here is a DECISION, not a formality — read the module docstring
# before you add one. The default answer for a new S2S call is
# "subclass BaseServiceClient", which needs no entry at all.
ALLOWED: dict[tuple[str, str], Exemption] = {
    ("src/api/routes/thumbnail.py", "extract_thumbnails"): Exemption(
        SSRF_SINK,
        "POST /api/thumbnails fetches body.video_url — CALLER-SUPPLIED — through _fetch, i.e. "
        "safe_get with the MinIO allow_hosts and the streaming byte cap, plus a route-level "
        "local-file guard (file:// / bare paths refused unless allow_local_file_inputs, and "
        "then confined to work_dir). The destination is whatever URL arrived in the request "
        "body, so attaching X-Tenant-Id: Bearer <license> would hand a live credential to a "
        "third-party host. Named by the audit as an identity-dropping S2S call; it is not one.",
    ),
    ("src/services/renderer.py", "concat_audio"): Exemption(
        SSRF_SINK,
        "Audiobook stitch: pulls every clip.audio_url in the request through _fetch/safe_get "
        "into the job scratch before ffmpeg concat. Same contract as the thumbnail route — the "
        "URLs are supplied by the caller (normally presigned MinIO, but never trusted as such), "
        "so no identity rides along. Second audit finding; also a false positive.",
    ),
    ("src/services/renderer.py", "compile_shots"): Exemption(
        GUARDED_POOL,
        "The job-scoped transport pool for a whole compile: one client, opened per compile and "
        "closed with the job, handed down to the per-shot render fan-out (_render_shot_cached → "
        "render_shot → _fetch_shot_sfx / _fetch_overlay_clips / _resolve_shot_assets) and used "
        "directly for the music bed, every one of which fetches via _fetch_asset → _try_fetch → "
        "_fetch → safe_get, deduped per source URL by SourceFetchCache. Shot asset URLs arrive "
        "in the compile request, so identity headers must never be baked into this pool. Third "
        "audit finding; also a false positive.",
        (
            ("src/services/renderer.py", "_fetch_asset"),
            ("src/services/renderer.py", "_fetch_shot_sfx"),
            ("src/services/renderer.py", "_fetch_overlay_clips"),
            ("src/services/renderer.py", "_resolve_shot_assets"),
        ),
    ),
}


def _walk_scoped(
    tree: ast.Module,
) -> tuple[list[tuple[ast.AST, str]], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return (node, enclosing-qualname) pairs plus a qualname -> function-node map.

    A nested ``def`` extends the qualname but keeps REPORTING its outermost
    enclosing function — which is the unit a reviewer reasons about — and the
    outermost function is what lands in the map, so the substantiation check sees
    the whole body (nested helpers included) rather than one closure. That matters
    here: ``compile_shots`` builds its client and then uses it from an inner
    ``_render`` closure.
    """
    pairs: list[tuple[ast.AST, str]] = []
    scopes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                reported = prefix if in_function else name
                if not in_function:
                    scopes[name] = child
                pairs.append((child, reported))
                visit(child, reported, True)
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                pairs.append((child, name))
                visit(child, name, in_function)
            else:
                pairs.append((child, prefix))
                visit(child, prefix, in_function)

    visit(tree, "", False)
    return pairs, scopes


def _is_httpx_client_call(node: ast.Call, imported_names: set[str]) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in ("Client", "AsyncClient"):
        return isinstance(func.value, ast.Name) and func.value.id == "httpx"
    return isinstance(func, ast.Name) and func.id in imported_names


def _referenced_identifiers(node: ast.AST) -> frozenset[str]:
    """Every name / attribute actually referenced inside ``node``.

    AST-based on purpose — see :class:`Construction`. Docstrings and comments are
    not Name/Attribute nodes, so a marker can only match code.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return frozenset(names)


def _httpx_imported_names(tree: ast.Module) -> set[str]:
    """Names bound by ``from httpx import Client/AsyncClient [as x]``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "httpx":
            names.update(
                a.asname or a.name for a in node.names if a.name in ("Client", "AsyncClient")
            )
    return names


def _function_identifiers(root: Path = _SRC) -> dict[tuple[str, str], frozenset[str]]:
    """(path, qualname) -> identifiers referenced, for EVERY top-level function/method."""
    repo = root.parent
    table: dict[tuple[str, str], frozenset[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(repo).as_posix()
        _, scopes = _walk_scoped(tree)
        for qualname, node in scopes.items():
            table[(rel, qualname)] = _referenced_identifiers(node)
    return table


def find_constructions(root: Path = _SRC) -> list[Construction]:
    """Every bare ``httpx`` client construction under ``root``, keyed for the allowlist."""
    repo = root.parent
    found: list[Construction] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "httpx" not in text:
            continue
        tree = ast.parse(text, filename=str(path))
        imported = _httpx_imported_names(tree)
        rel = path.relative_to(repo).as_posix()
        pairs, scopes = _walk_scoped(tree)
        for node, qualname in pairs:
            if not isinstance(node, ast.Call) or not _is_httpx_client_call(node, imported):
                continue
            owner = scopes.get(qualname)
            names = _referenced_identifiers(owner) if owner else frozenset()
            found.append(Construction(rel, qualname or "<module>", node.lineno, names))
    return found


# ── The gate ─────────────────────────────────────────────────────────────────


def test_no_undocumented_bare_httpx_client() -> None:
    """Every bare httpx client under src/ must carry a written exemption."""
    offenders = [c for c in find_constructions() if (c.path, c.qualname) not in ALLOWED]
    assert not offenders, (
        "bare httpx client construction(s) with no exemption:\n"
        + "\n".join(f"  {c.path}:{c.lineno} in {c.qualname}()" for c in offenders)
        + "\n\nA raw httpx client bypasses BaseServiceClient's identity event hook, so the "
        "X-Tenant-Id / X-License-Id / X-Subject-Id triple is silently DROPPED on that call. "
        "Subclass w2v_common.clients.base.BaseServiceClient instead. If this is an "
        "SSRF-guarded fetch of a caller-supplied URL (where forwarding a license token would "
        "leak a credential to a third party), add an entry to ALLOWED in "
        "tests/test_no_bare_httpx_client.py with a category and a written reason."
    )


def test_allowlist_has_no_stale_entries() -> None:
    """Non-vacuity: an exemption may not outlive the call site that excused it."""
    live = {(c.path, c.qualname) for c in find_constructions()}
    stale = sorted(key for key in ALLOWED if key not in live)
    assert not stale, (
        "ALLOWED entries match nothing in src/ — the call site was moved, renamed or "
        f"fixed, so the exemption must be deleted: {stale}"
    )


def test_every_exemption_states_a_category_and_a_reason() -> None:
    thin = {
        key: exemption
        for key, exemption in ALLOWED.items()
        if exemption.category not in _MARKERS or len(exemption.reason) < _MIN_REASON_CHARS
    }
    assert not thin, (
        "every exemption needs a known category and a reason of at least "
        f"{_MIN_REASON_CHARS} chars explaining why the identity triple must not be "
        f"forwarded here: {sorted(thin)}"
    )


def test_exemption_claims_are_still_true_in_the_source() -> None:
    """A structurally substantiated exemption must keep being substantiated.

    ``SSRF_SINK`` and ``IDENTITY_FORWARDED`` are claims about the code, so they
    are re-verified rather than trusted. Drop ``_fetch`` for a raw ``http.get``,
    or delete manual ``forward_headers``, and the exemption stops applying here —
    exactly the moment a reviewer needs to be told.
    """
    by_key = {(c.path, c.qualname): c for c in find_constructions()}
    unsubstantiated: list[str] = []
    for key, exemption in ALLOWED.items():
        markers = _MARKERS[exemption.category]
        if not markers:
            continue
        construction = by_key.get(key)
        if construction is None:  # covered by the staleness test
            continue
        if not any(marker in construction.identifiers for marker in markers):
            unsubstantiated.append(
                f"{key[0]}:{key[1]} claims {exemption.category} but references none of {markers}"
            )
    assert not unsubstantiated, "\n".join(unsubstantiated)


def test_guarded_pool_consumers_are_still_guarded() -> None:
    """GUARDED_POOL is a claim about someone ELSE's function — check it there.

    A pooled client is safe only because every consumer fetches through the SSRF
    guard. So each declared consumer must (a) exist and (b) still reference a
    guard marker; a renamed or unguarded consumer reds this.
    """
    identifiers = _function_identifiers()
    broken: list[str] = []
    for key, exemption in ALLOWED.items():
        if exemption.category != GUARDED_POOL:
            continue
        if not exemption.consumers:
            broken.append(f"{key} claims GUARDED_POOL but names no consumers")
            continue
        for consumer in exemption.consumers:
            names = identifiers.get(consumer)
            if names is None:
                broken.append(f"{key} names consumer {consumer}, which does not exist")
            elif not any(marker in names for marker in _SSRF_MARKERS):
                broken.append(
                    f"{key} pool is consumed by {consumer}, which references none of "
                    f"{_SSRF_MARKERS} — the pool is no longer demonstrably guarded"
                )
    assert not broken, "\n".join(broken)


def test_ssrf_markers_actually_reach_safe_get() -> None:
    """Guard the guard: a marker that does not reach ``safe_get`` is a backdoor.

    The substantiation check is only as good as :data:`_SSRF_MARKERS`. The cheap
    way to defeat it is to add a plausible-sounding helper that fetches raw, and
    nothing in the sibling repos' version of this file would notice. So recompute
    the reachability from the AST: seed with the ``w2v_common.net`` chokepoints
    (``safe_get`` and its constant-memory twin ``safe_get_to_path`` — same redirect
    loop, same IP pinning, same mid-stream byte cap, only the terminal differs) and
    close over "function F references a name already known to reach it". Every
    declared marker must land in that set.

    The seed is a TWO-name set on purpose and is the one place that may name a
    chokepoint without proof, so it stays minimal: a name goes in here only if it
    is a ``w2v_common.net`` entry point running the shared guard loop. Adding an
    in-repo helper here instead of to ``_SSRF_MARKERS`` would let it substantiate
    an exemption while fetching raw, which is precisely the backdoor this test
    exists to close.
    """
    identifiers = _function_identifiers()
    by_name: dict[str, list[frozenset[str]]] = {}
    for (_path, qualname), names in identifiers.items():
        by_name.setdefault(qualname.rsplit(".", 1)[-1], []).append(names)

    reaching = {"safe_get", "safe_get_to_path"}
    changed = True
    while changed:
        changed = False
        for name, bodies in by_name.items():
            if name in reaching:
                continue
            if any(ref in reaching for body in bodies for ref in body):
                reaching.add(name)
                changed = True

    unreachable = [m for m in _SSRF_MARKERS if m not in reaching]
    assert not unreachable, (
        f"declared SSRF marker(s) {unreachable} do not transitively reach "
        "w2v_common.net.safe_get / safe_get_to_path anywhere in src/ — an exemption "
        "substantiated by one of these proves nothing. Remove the marker, or route the "
        "helper through one of those chokepoints."
    )


def test_guard_actually_finds_the_known_constructions() -> None:
    """The scanner must not be silently finding nothing (self-check).

    A guard whose finder is broken passes every other test in this file. Pin a
    floor and all three known sites so a scanner regression reds instead of
    relaxing.
    """
    found = find_constructions()
    keys = {(c.path, c.qualname) for c in found}
    assert len(found) >= 3, f"scanner found only {len(found)} constructions; it is broken"
    assert ("src/api/routes/thumbnail.py", "extract_thumbnails") in keys
    assert ("src/services/renderer.py", "compile_shots") in keys
    assert ("src/services/renderer.py", "concat_audio") in keys
