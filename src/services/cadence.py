"""Frame-rate cadence SELECTION (VC-3) — pure, never-raise, no ffmpeg.

The compiler snaps every shot onto ONE master cadence (a single output frame
rate) so the concat demuxer's ``-c copy`` fast path stays valid — mixed source
rates would otherwise force a re-encode at join time. This module only *chooses*
that rate and describes *how* a source should be retimed onto it; it never
probes, never execs ffmpeg, and never raises. Verification that the delivered
file actually landed on the chosen cadence lives in
``mastering.framerate_coherence`` / ``mastering.post_mux_verify`` (contract
C11) — this module deliberately owns NONE of that.

Pieces:

* ``source_rate_profile`` — summarize a set of source clip frame rates.
* ``clamp_cadence`` — snap an arbitrary rate to the nearest legal delivery rate.
* ``judder_risk`` — a 0..1 heuristic for the visible judder of a non-integer
  source→target frame-rate ratio (0 for a clean integer ratio).
* ``master_cadence`` — pick the minimum-retime-cost target rate for a set of
  source profiles.
* ``retime_plan`` — the ffmpeg filter FRAGMENT that retimes a source onto the
  target: a clean ``fps=N`` for an exact/integer ratio (byte-identical to the
  pre-VC-3 ``fps={render_fps}``), motion-compensated interpolation for a
  high-motion non-integer retime, or a cheap blend variant on a budget.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

# The legal delivery cadences. 24 (film), 25 (PAL), 30 (web default), 50/60
# (high-motion). Everything is clamped/selected into this set.
_LEGAL_RATES: tuple[int, ...] = (24, 25, 30, 50, 60)

# Below this deviation from a whole number a frame-rate ratio counts as an exact
# integer ratio (a clean frame drop/dup with no judder) — e.g. 60/30=2, 50/25=2.
_INTEGER_RATIO_EPSILON = 0.01


def _usable(f: float) -> bool:
    """A rate is usable only when it is positive AND FINITE.

    ``inf``/``nan`` must be dropped here, not downstream: ``Shot.target_fps`` is
    ``gt=0`` with no upper bound and JSON ``1e400`` decodes to ``inf``, which passes
    pydantic. An ``inf`` that survived this coercion reached ``math.floor(inf)`` /
    ``round(inf)`` in ``judder_risk`` / ``_fmt_rate`` / ``_is_integer_ratio`` and
    raised ``OverflowError`` — killing the whole compile in the default
    ``render_cadence_mode="auto"`` selector before a single shot rendered, from a
    module whose contract is that it never raises. A non-finite rate carries no
    cadence information, so it degrades exactly like an unparseable one.
    """
    return f > 0 and math.isfinite(f)


def _coerce_fps(value: Any) -> float | None:
    """Best-effort parse of a frame-rate to a positive FINITE float, else ``None``.

    Accepts an int/float, a bare numeric string, or an ffprobe rational
    (``"30000/1001"``, ``"25/1"``). NEVER raises — bad/zero/negative/non-finite →
    ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if _usable(f) else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            if "/" in s:
                num, den = s.split("/", 1)
                d = float(den)
                if not d:
                    return None
                f = float(num) / d
            else:
                f = float(s)
        except ValueError, ZeroDivisionError, OverflowError:
            return None
        return f if _usable(f) else None
    return None


def _fmt_rate(value: Any) -> str:
    """Render a rate for an ffmpeg ``fps=`` fragment: an integral rate as a bare
    int (``30`` → ``"30"``, keeping the byte-identical fragment), else ``%g``."""
    f = _coerce_fps(value)
    if f is None:
        return "30"
    if abs(f - round(f)) < 1e-6:
        return str(round(f))
    return f"{f:g}"


def clamp_cadence(x: Any) -> int:
    """Snap an arbitrary rate to the nearest legal delivery cadence.

    Ties break toward the lower rate. Unparseable input degrades to ``30`` (the
    fleet default) rather than raising."""
    f = _coerce_fps(x)
    if f is None:
        return 30
    return min(_LEGAL_RATES, key=lambda r: (abs(r - f), r))


def source_rate_profile(fps_values: Iterable[Any]) -> dict[str, Any]:
    """Summarize a set of source clip frame rates into ``{rates, count, dominant}``.

    ``rates`` maps each observed rate (rounded to 3 dp) to how many clips carried
    it; ``dominant`` is the most common rate (``None`` when nothing parsed).
    Unparseable entries are silently dropped — never raises."""
    rates: dict[float, int] = {}
    total = 0
    for value in fps_values or ():
        f = _coerce_fps(value)
        if f is None:
            continue
        key = round(f, 3)
        rates[key] = rates.get(key, 0) + 1
        total += 1
    dominant = max(rates, key=lambda r: rates[r]) if rates else None
    return {"rates": rates, "count": total, "dominant": dominant}


def judder_risk(source_fps: Any, target_fps: Any) -> float:
    """A 0..1 judder heuristic for retiming ``source_fps`` onto ``target_fps``.

    An exact integer ratio (24→24, 30→60, 60→30) reads clean → ``0.0``. A
    non-integer ratio (25→30, 24→30) needs a fractional frame cadence that
    stutters; risk peaks at a half-frame ratio. ``0.0`` when either rate is
    unknown. PURE."""
    a = _coerce_fps(source_fps)
    b = _coerce_fps(target_fps)
    if a is None or b is None:
        return 0.0
    hi, lo = (a, b) if a >= b else (b, a)
    ratio = hi / lo
    frac = ratio - math.floor(ratio)
    return round(min(frac, 1.0 - frac) * 2.0, 4)


def master_cadence(
    source_profiles: Iterable[Mapping[str, Any]],
    candidates: Iterable[int] = _LEGAL_RATES,
) -> int:
    """Pick the minimum-retime-cost master cadence for a set of source profiles.

    Cost is the count-weighted total ``judder_risk`` of retiming every source
    rate onto a candidate; ties break toward the candidate nearest the dominant
    source rate, then toward the lower rate. With NO source rates it returns
    ``30`` (the fleet default) so the caller's default path stays byte-identical.
    PURE — never probes, never raises."""
    cands = sorted({clamp_cadence(c) for c in candidates}) or list(_LEGAL_RATES)
    weights: dict[float, float] = {}
    for profile in source_profiles or ():
        for rate, count in (profile.get("rates") or {}).items():
            r = _coerce_fps(rate)
            if r is None:
                continue
            weights[r] = weights.get(r, 0.0) + float(count)
    if not weights:
        return 30 if 30 in cands else cands[len(cands) // 2]
    dominant = max(weights, key=lambda r: weights[r])

    def cost(candidate: int) -> tuple[float, float, int]:
        judder = sum(judder_risk(rate, candidate) * w for rate, w in weights.items())
        return (round(judder, 6), abs(candidate - dominant), candidate)

    return min(cands, key=cost)


def _is_integer_ratio(source_fps: float | None, target_fps: float | None) -> bool:
    """Whether source→target is an exact integer ratio (a clean drop/dup).

    An unknown source rate (``None``) counts as clean so the retime degrades to a
    plain ``fps=`` resample rather than an unwanted interpolation."""
    if not source_fps or not target_fps:
        return True
    hi, lo = (source_fps, target_fps) if source_fps >= target_fps else (target_fps, source_fps)
    ratio = hi / lo
    return abs(ratio - round(ratio)) <= _INTEGER_RATIO_EPSILON


def retime_plan(
    source_fps: Any,
    target_fps: Any,
    *,
    high_motion: bool = False,
    budget: bool = False,
) -> str:
    """The ffmpeg filter FRAGMENT retiming ``source_fps`` onto ``target_fps``.

    * exact / integer ratio → ``fps={target}`` — a clean frame drop/dup, and
      byte-identical to the pre-VC-3 ``fps={render_fps}`` fragment.
    * non-integer + ``high_motion`` → ``minterpolate=fps={target}:mi_mode=mci``
      (motion-compensated interpolation — smooth, expensive).
    * non-integer + ``budget`` → a cheap ``tblend`` blend variant.
    * non-integer, neither flag → the clean ``fps={target}`` (accept the judder).

    PURE — returns a string, never execs ffmpeg, never raises."""
    tgt = _fmt_rate(target_fps)
    if _is_integer_ratio(_coerce_fps(source_fps), _coerce_fps(target_fps)):
        return f"fps={tgt}"
    if high_motion:
        return f"minterpolate=fps={tgt}:mi_mode=mci"
    if budget:
        return f"tblend=all_mode=average,framestep=1,fps={tgt}"
    return f"fps={tgt}"
