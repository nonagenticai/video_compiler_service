"""The deband trigger and the mastering gate must read the SAME measurement.

They did not, and the mismatch made the gate unsatisfiable:

* trigger — `measure_shot_stats`, signalstats AVERAGED over the whole segment,
  compared to 0.15;
* gate — `measure_shot_frames`, per-frame scores, **max** compared to 0.15.

A segment banded in a minority of frames averages BELOW the trigger, so deband
never runs, and maxes ABOVE the gate, so `gate_pass=False` with
`rerun_stages=["deband"]`. Re-running deband re-measures the same average and
again does nothing — the remediation the gate demands cannot satisfy it. Under
`mastering_gate_mode=enforce` (shipped in #171) that is a permanently-failing
render; under advisory, a permanent false red.

This is the same whole-file-vs-per-frame class already fixed once on the gate
side of this service. This file pins the trigger side.
"""

from __future__ import annotations

import ast
import inspect

from src.services import mastering


def _banded() -> dict[str, float]:
    """A frame with the banding signature: a wide, SPATIALLY UNIFORM luma ramp
    carrying a low temporal diff.

    ⚠️ ``YLOW``/``YHIGH`` (signalstats' 10th/90th luma percentiles) are part of
    every real frame's stats and the spatial term added 2026-09-02 reads them.
    A ramp spreads its pixels evenly, so the 10-90 band covers ~90% of the range
    — measured 0.914 on a real 8-bit gradient.
    """
    return {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 38.0, "YHIGH": 213.0, "YDIF": 0.5}


def _clean() -> dict[str, float]:
    """Textured content — same span, plenty of temporal diff, and pixels
    CONCENTRATED in a narrow band with outliers stretching YMIN/YMAX
    (measured 0.390 on `mandelbrot`)."""
    return {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 90.0, "YHIGH": 175.0, "YDIF": 10.0}


def test_a_segment_banded_in_a_minority_of_frames_triggers_deband() -> None:
    """The exact shape that used to slip through: 2 banded frames in 20."""
    frames = [_banded(), _banded()] + [_clean()] * 18

    assert mastering.banding_risk_frames(frames) is True, (
        "the worst frame is banded, so deband must run — averaging this segment "
        "hides it below the trigger while the gate still fails on it"
    )


def test_that_same_segment_is_exactly_what_the_gate_would_fail() -> None:
    """Trigger and gate must agree frame-for-frame, or the loop cannot converge."""
    frames = [_banded(), _banded()] + [_clean()] * 18
    gate_scores = [max((mastering.banding_score(f) for f in frames), default=0.0)]

    triggers = mastering.banding_risk_frames(frames)
    gate_passes = mastering._banding_gate_pass(gate_scores)

    assert triggers is not gate_passes, (
        "a segment the gate fails must be a segment the trigger debands, and "
        "vice versa — otherwise the gate asks for a remediation that cannot "
        "satisfy it"
    )


def test_clean_content_still_stays_a_single_pass_encode() -> None:
    """The trigger must not become trigger-happy: byte-identical encode matters."""
    assert mastering.banding_risk_frames([_clean()] * 20) is False


def test_no_measurement_does_not_trigger() -> None:
    """`measure_shot_frames` returns [] on any failure; that must not deband."""
    assert mastering.banding_risk_frames([]) is False


def test_trigger_and_gate_share_one_constant() -> None:
    """Named so they cannot drift apart again by an edit to one number.

    ⚠️ This used to be ``"BANDING_GATE_CEILING" in inspect.getsource(...)``, and
    that could not fail: ``getsource`` returns the DOCSTRING too, and
    ``banding_risk_frames``' docstring names the constant twice while explaining
    the very bug this test guards. Verified by mutation — rewriting the default
    to a literal ``0.30`` (a 2x drift) left the old assertion green. Same shape as
    the fleet-wide "the escape hatch matched its own marker in a docstring"
    defect, so read the AST, never the source text.
    """
    tree = ast.parse(inspect.getsource(mastering))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "banding_risk_frames"
    )
    defaults = [d for d in fn.args.kw_defaults if d is not None]
    assert len(defaults) == 1, (
        "banding_risk_frames no longer has exactly one keyword default — this "
        "guard is reading the wrong thing and must be updated with the signature"
    )
    default = defaults[0]
    assert isinstance(default, ast.Name) and default.id == "BANDING_GATE_CEILING", (
        "the trigger must default to the same NAMED ceiling the gate compares "
        f"against, not {ast.unparse(default)!r}; a literal here is how the two "
        "drifted in the first place"
    )

    # Belt and braces: the name must still resolve to the gate's own value, so a
    # rebind of BANDING_GATE_CEILING between the two call sites is caught too.
    assert (
        inspect.signature(mastering.banding_risk_frames).parameters["threshold"].default
        == mastering.BANDING_GATE_CEILING
    )
