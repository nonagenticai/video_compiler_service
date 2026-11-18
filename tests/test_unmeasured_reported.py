"""``mastering.unmeasured`` must survive the response model — same defect as #153.

``post_mux_verify`` computes ``unmeasured`` (the gate inputs that could not be
measured) and returns it, with a comment stating precisely why it exists:
``gate_pass`` alone conflates "every check ran and none was violated" with "the
checks that RAN found nothing". ``MasteringReport`` never declared the field and
sets no ``extra=`` config, so pydantic's default ``extra="ignore"`` dropped it at
``api/routes/compile.py`` and no caller ever saw it.

The concrete loss: when the mpdecimate pass dies, ``duplicate_frame_ratio`` is
None, the duplicate gate deliberately does not fire on a measurement gap, and
``GET /api/jobs/{id}`` returns ``gate_pass=true, rerun_stages=[]`` with nothing
naming the check that never ran. The severity is honestly PARTIAL — a consumer
could infer a gap from the ``None`` metric values — so this is a degraded signal
rather than a blind one; the field makes it explicit and names only the Nones the
gate actually cared about.

``degraded_report()`` also omitted the key, so the two report shapes disagreed —
which is exactly why the general producer-vs-model guard in
``test_dead_frame_ratio_reported.py`` (built from ``degraded_report()``) could not
see this one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import MasteringReport
from src.config import Settings, get_settings
from src.services import mastering, renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _probe_stub(cmd: list[str], *, mpdecimate: str | None) -> str:
    """A clean probe round for every metric except mpdecimate, whose survivor count
    is ``mpdecimate`` (``None`` makes the pass die → duplicate_frame_ratio is None)."""
    joined = " ".join(cmd)
    if "codec_type,duration" in joined:
        # A/V coverage probe: a clean master whose picture covers its narration.
        # Returned for EVERY stub round so a "fully probed" master really is fully
        # probed — an unserved probe would name `av_gap_seconds` unmeasured and make
        # these tests assert the opposite of what they mean.
        return (
            '{"streams": ['
            '{"codec_type": "video", "duration": "30.000000"}, '
            '{"codec_type": "audio", "duration": "30.000000"}], '
            '"format": {"duration": "30.000000"}}'
        )
    if "stream=time_base:packet=pts" in joined:
        # The video pts-interval probe, the other half of ``cfr_ok``: three frames
        # on a regular 1/30 grid. Served for the same reason as the A/V probe above
        # — unserved, ``cfr_ok`` is None and every "fully probed" round names it.
        return (
            '{"streams": [{"time_base": "1/30"}], "packets": [{"pts": 0}, {"pts": 1}, {"pts": 2}]}'
        )
    if "color_primaries" in joined:
        return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
    if "r_frame_rate,avg_frame_rate" in joined:
        return "30/1\n30/1\n"
    if "nb_read_packets" in joined:
        return "300\n"
    if "mpdecimate" in joined:
        if mpdecimate is None:
            raise renderer.FfmpegError("mpdecimate exit=1")
        return mpdecimate
    if "avg_frame_rate" in joined:
        return "30/1\n"
    return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"


def test_unmeasured_survives_the_response_model() -> None:
    """The load-bearing assertion: ``model_validate`` is what the route does."""
    report = MasteringReport.model_validate(
        {
            "colorspace_ok": True,
            "gate_pass": True,
            "rerun_stages": [],
            "duplicate_frame_ratio": None,
            "unmeasured": ["duplicate_frame_ratio"],
        }
    )

    assert report.unmeasured == ["duplicate_frame_ratio"]
    # And outbound, not just inbound — the caller reads the dump.
    assert report.model_dump()["unmeasured"] == ["duplicate_frame_ratio"]


def test_degraded_report_names_every_input_as_unmeasured() -> None:
    """A verification that could not RUN measured nothing — and must say so.

    Both report shapes must carry the same key, or a consumer has to branch on
    which one it received before it can trust ``gate_pass``.
    """
    degraded = mastering.degraded_report()

    assert "unmeasured" in degraded
    assert degraded["unmeasured"] == sorted(
        [
            # ``degraded_report`` takes no settings, so it cannot claim the
            # colorspace check was deliberately off — it names the gap.
            "cfr_ok",
            "colorspace_ok",
            "dead_frame_ratio",
            "duplicate_frame_ratio",
            "integrated_lufs",
            "luma_spread",
            "measured_fps",
            "true_peak_dbtp",
            # Same reasoning as `colorspace_ok`: without probing the file this
            # function cannot know whether the master even HAS an audio stream, so
            # it cannot claim the A/V coverage check was deliberately off.
            "av_gap_seconds",
        ]
    )
    assert MasteringReport.model_validate(degraded).unmeasured == degraded["unmeasured"]


@pytest.mark.asyncio
async def test_a_dead_mpdecimate_pass_is_named_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported failure, end to end: the duplicate check never ran and the
    caller can now learn that from a ``gate_pass=True`` response.

    The invariant asserted is NOT "a probe was called" — it is that a report whose
    duplicate gate did not evaluate can never present as fully verified.
    """

    async def dead_mpdecimate(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub(cmd, mpdecimate=None)

    monkeypatch.setattr(renderer, "_run", dead_mpdecimate)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )

    # The premise of the bug: a gap does NOT gate, so this still reads as a pass.
    assert raw["duplicate_frame_ratio"] is None
    assert "duplicate_frames" not in raw["rerun_stages"]

    # The fix: the gap is named, and it reaches the caller through the response model.
    on_the_wire = MasteringReport.model_validate(raw)
    assert "duplicate_frame_ratio" in on_the_wire.unmeasured

    # NOT VACUOUS: a fully measured run reports an EMPTY list, so a non-empty
    # ``unmeasured`` genuinely discriminates.
    async def all_measured(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub(cmd, mpdecimate="frame=  300\n")

    monkeypatch.setattr(renderer, "_run", all_measured)
    clean = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )
    assert "duplicate_frame_ratio" not in clean["unmeasured"]


# ---------------------------------------------------------------------------
# The two BOOLEAN gate verdicts — the residual fail-open under the patch for it.
#
# ``colorspace_ok`` and ``cfr_ok`` were omitted from ``unmeasured``, so an
# unprobeable file yielded ``colorspace_ok=None``, the colorspace gate did not
# fire, and NOTHING named it. That is the exact fail-open this field exists to
# close, for the very gate the parent commit fixed.
#
# The distinction these tests pin: "the check was OFF" and "the check COULD NOT
# RUN" are different facts about the master and must not share a value.
# ---------------------------------------------------------------------------


async def _probe_stub_failing(cmd: list[str], *, dead: str) -> str:
    """A clean probe round except for the ``dead`` probe, which raises.

    ``dead`` is ``"colorspace"`` or ``"cadence"``; anything else keeps the round
    fully clean.
    """
    joined = " ".join(cmd)
    if "color_primaries" in joined:
        if dead == "colorspace":
            raise renderer.FfmpegError("ffprobe exit=1")
        return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
    if "r_frame_rate,avg_frame_rate" in joined:
        if dead == "cadence":
            raise renderer.FfmpegError("ffprobe exit=1")
        return "30/1\n30/1\n"
    return await _probe_stub(cmd, mpdecimate="frame=  300\n")


@pytest.mark.asyncio
async def test_an_unprobeable_colorspace_is_named_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ironic fail-open: the colorspace gate could not run and said nothing.

    Asserts the INVARIANT — a report whose colorspace gate did not evaluate can
    never present as fully verified — not that any probe was called.
    """

    async def dead_colorspace(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub_failing(cmd, dead="colorspace")

    monkeypatch.setattr(renderer, "_run", dead_colorspace)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="bt709"),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )

    # The premise: a gap does NOT gate, so this still reads as a clean pass.
    assert raw["colorspace_ok"] is None
    assert "colorspace" not in raw["rerun_stages"]
    assert raw["gate_pass"] is True

    # The fix: the gap is named, and it reaches the caller through the model.
    on_the_wire = MasteringReport.model_validate(raw)
    assert "colorspace_ok" in on_the_wire.unmeasured


@pytest.mark.asyncio
async def test_a_disabled_colorspace_gate_is_not_an_unmeasured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "The check was off" must NOT render as "the check could not run".

    This is the other half of the fix and the one that keeps it honest: a job that
    switched colour tagging off expected no verdict, so a fully-probed master must
    still report an EMPTY ``unmeasured``. Without this the fix would trade a
    fail-open for a permanent false "unverified" on the ordinary disabled path.
    """

    async def clean(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub_failing(cmd, dead="")

    monkeypatch.setattr(renderer, "_run", clean)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="off"),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )

    # Same None value as the unprobeable case above — a different FACT.
    assert raw["colorspace_ok"] is None
    assert "colorspace_ok" not in raw["unmeasured"]
    # Nothing else was missing either, so the master is genuinely fully verified.
    assert MasteringReport.model_validate(raw).unmeasured == []


@pytest.mark.asyncio
async def test_an_unprobeable_cadence_is_named_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cfr_ok`` is advisory, never gating — and still a measurement that failed.

    Precedent, not novelty: ``measured_fps``, ``luma_spread`` and
    ``true_peak_dbtp`` are advisory-only too and have always been listed here. The
    channel names verification inputs that could not be measured, and a caller told
    to "read ``cfr_ok is False`` directly" cannot distinguish coherent-unknown from
    coherent unless the gap is named. ``framerate_coherence`` has no disabled path,
    so ``None`` here can only mean the probe failed.
    """

    async def dead_cadence(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub_failing(cmd, dead="cadence")

    monkeypatch.setattr(renderer, "_run", dead_cadence)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="bt709"),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )

    assert raw["cfr_ok"] is None
    assert raw["gate_pass"] is True
    assert "cfr_ok" in MasteringReport.model_validate(raw).unmeasured


@pytest.mark.asyncio
async def test_a_fully_probed_master_names_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NOT VACUOUS: with every probe healthy the list is EMPTY, so the two names
    above genuinely discriminate rather than being always-present decoration."""

    async def clean(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return await _probe_stub_failing(cmd, dead="")

    monkeypatch.setattr(renderer, "_run", clean)
    raw = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(render_colorspace="bt709"),
        loudness={"integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    )

    assert raw["colorspace_ok"] is True
    assert raw["cfr_ok"] is True
    assert MasteringReport.model_validate(raw).unmeasured == []
