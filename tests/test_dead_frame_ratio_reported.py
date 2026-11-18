"""``dead_frame_ratio`` must survive the response model — it was a dead signal.

``post_mux_verify`` computed it (``mastering.py:538``) and GATED on it
(``mastering.py:539-541``: ``gate_pass`` flips and ``"dead_frame"`` joins
``rerun_stages``), but ``MasteringReport`` never declared the field. Pydantic silently
drops undeclared keys, so every caller saw the verdict with none of the evidence behind
it: ``rerun_stages: ["dead_frame"]`` and no way to tell 11% of frames from 100%.

That distinction is the whole decision — re-render, or go look at the source. And
``luma_spread`` cannot substitute: it is the single WORST frame, which is precisely the
signal #152 stopped gating on after one intentional fade-to-black failed a whole film.

This is the same defect class as the readiness log that rendered its exception to the
empty string (#153): the code did the work, and the answer never reached anyone.
"""

from __future__ import annotations

from src.api.schemas import MasteringReport
from src.services import mastering


def test_dead_frame_ratio_survives_the_response_model() -> None:
    """The load-bearing assertion: validate a real report payload and read it back.

    ``model_validate`` is what the route does. Before the field was declared this dropped
    the key entirely and the attribute did not exist.
    """
    report = MasteringReport.model_validate(
        {
            "colorspace_ok": True,
            "integrated_lufs": -16.0,
            "banding_scores": [],
            "cfr_ok": True,
            "luma_spread": 2.0,
            "dead_frame_ratio": 0.42,
            "gate_pass": False,
            "rerun_stages": ["dead_frame"],
        }
    )

    assert report.dead_frame_ratio == 0.42
    # And it must survive serialisation back out to the caller, not just inbound parsing.
    assert report.model_dump()["dead_frame_ratio"] == 0.42


def test_degraded_report_declares_the_ratio_as_unknown() -> None:
    """A report from a verification that could not RUN must say "unknown", not "clean".

    ``degraded_report()`` is returned when ``post_mux_verify`` itself raised. Omitting the
    key would make it default to None anyway — but omitting it also means a future reader
    of that function cannot tell the field was considered. Asserting it explicitly pins
    that ``None`` (unknown) is deliberate and that it is never 0.0 (measured-and-clean).
    """
    degraded = mastering.degraded_report()

    assert "dead_frame_ratio" in degraded
    assert degraded["dead_frame_ratio"] is None
    assert MasteringReport.model_validate(degraded).dead_frame_ratio is None


def test_every_key_post_mux_verify_returns_is_declared_on_the_model() -> None:
    """The GENERAL guard — this defect can recur for any future field.

    ``dead_frame_ratio`` was lost because a producer key had no consumer field and nothing
    connected the two. Comparing the two key sets catches the next one automatically
    instead of waiting for someone to notice a missing number in a response.
    """
    produced = set(mastering.degraded_report())
    declared = set(MasteringReport.model_fields)

    assert produced <= declared, (
        f"post_mux_verify returns keys the response model drops: {sorted(produced - declared)}"
    )
