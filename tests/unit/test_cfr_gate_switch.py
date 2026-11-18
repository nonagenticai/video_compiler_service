"""``MASTERING_GATE_CFR_ENABLED`` — the constant-frame-rate gate's switch.

``cfr_ok`` was measured and published on every master but had no switch, so it
could never be enforced. The switch exists now and defaults OFF. These tests pin
the four states that matter:

* OFF + non-CFR master  -> delivered, report identical to the pre-switch one
* ON  + non-CFR master  -> the compile fails, naming ``cfr``
* ON  + CFR master      -> delivered
* ON  + UNMEASURED      -> delivered (a probe gap is not a violation — the same
  rule ``colorspace_ok is None`` follows)

``cfr_ok`` has TWO inputs since master 01a11032 shipped ``cfr_ok: true`` with a
missing frame slot: the header/declared-duration reading
(``framerate_coherence``) and the pts-interval reading
(``video_pts_gap_report``). The last tests pin how they combine:

* coherent headers + a pts gap        -> ``cfr_ok`` False, and ON fails the compile
* coherent headers + UNMEASURED pts   -> ``cfr_ok`` None (never True), delivered,
  named in ``unmeasured``

Both cadence PROBES are stubbed, not the gate: what is under test is how the gate
treats each ``cfr_ok`` value, not how that value is measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import mastering, renderer

_ALIVE_FRAME = (
    "lavfi.signalstats.YMIN=16\n"
    "lavfi.signalstats.YLOW=40\n"
    "lavfi.signalstats.YHIGH=200\n"
    "lavfi.signalstats.YMAX=235\n"
    "lavfi.signalstats.YDIF=9.0\n"
)
_LOUDNESS = {"integrated_lufs": -14.0}


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update={"master_lufs": -14.0, **overrides})


_NO_GAPS: tuple[int | None, list[dict[str, float]] | None] = (0, [])


def _stub_master(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfr_ok: bool | None,
    pts_gaps: tuple[int | None, list[dict[str, float]] | None] = _NO_GAPS,
) -> None:
    """An otherwise-CLEAN master (bt709 tags, live frames, A/V in step) whose
    header cadence reading is exactly ``cfr_ok`` and whose pts timeline reads
    ``pts_gaps`` (regular by default) — so cfr is the only signal that moves."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return (
                "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
                + _ALIVE_FRAME
                + "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
                + _ALIVE_FRAME
            )
        return ""

    async def fake_av_stream_seconds(path: object) -> tuple[float | None, float | None]:
        return (60.0, 60.1)

    async def fake_coherence(path: object, expected_fps: object = None) -> bool | None:
        return cfr_ok

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(mastering, "av_stream_seconds", fake_av_stream_seconds)

    async def fake_pts_gaps(
        path: object,
    ) -> tuple[int | None, list[dict[str, float]] | None]:
        return pts_gaps

    monkeypatch.setattr(mastering, "framerate_coherence", fake_coherence)
    monkeypatch.setattr(mastering, "video_pts_gap_report", fake_pts_gaps)


def test_the_switch_is_a_settings_field_defaulting_off_and_wired_to_the_stage() -> None:
    field = Settings.model_fields["mastering_gate_cfr_enabled"]
    assert field.annotation is bool
    assert field.default is False
    assert mastering._GATE_KILL_SWITCHES["cfr"] == "mastering_gate_cfr_enabled"
    assert mastering._GATE_ARMED_DEFAULTS["cfr"] is False
    assert mastering.gate_armed(_settings(), "cfr") is False
    assert mastering.gate_armed(_settings(mastering_gate_cfr_enabled=True), "cfr") is True


def test_the_env_var_name_arms_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the change: enabling it later is a CONFIG change."""
    monkeypatch.setenv("MASTERING_GATE_CFR_ENABLED", "true")
    assert Settings().mastering_gate_cfr_enabled is True
    monkeypatch.delenv("MASTERING_GATE_CFR_ENABLED")
    assert Settings().mastering_gate_cfr_enabled is False


@pytest.mark.asyncio
async def test_off_a_non_cfr_master_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default OFF: cfr stays advisory — published, named in ``rerun_stages``,
    and nothing else. No violation, ``gate_pass`` untouched."""
    _stub_master(monkeypatch, cfr_ok=False)

    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness=_LOUDNESS)

    assert report["cfr_ok"] is False
    assert report["rerun_stages"] == ["cfr"]
    assert report["violations"] == []
    assert report["gate_pass"] is True


@pytest.mark.asyncio
async def test_off_is_identical_whether_the_field_is_false_or_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Byte-identity with the pre-switch build, stated as a test: a ``Settings``
    that does not carry the field at all (the shape that existed before this
    change) and one carrying the default produce the SAME report."""
    _stub_master(monkeypatch, cfr_ok=False)
    explicit_off = _settings(mastering_gate_cfr_enabled=False)
    predates_field = _settings()
    del predates_field.__dict__["mastering_gate_cfr_enabled"]
    assert getattr(predates_field, "mastering_gate_cfr_enabled", "ABSENT") == "ABSENT"

    with_field = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), explicit_off, loudness=_LOUDNESS
    )
    without_field = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), predates_field, loudness=_LOUDNESS
    )

    assert with_field == without_field
    assert with_field["violations"] == []


@pytest.mark.asyncio
async def test_off_enforce_mode_alone_does_not_sweep_cadence_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mastering_gate_mode='enforce'`` never failed a compile on cadence before
    the switch existed (there was no cfr violation for it to arm), and adding a
    default-OFF switch must not change that."""
    _stub_master(monkeypatch, cfr_ok=False)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(mastering_gate_mode="enforce"), loudness=_LOUDNESS
    )

    assert report["cfr_ok"] is False
    assert report["violations"] == []
    assert report["gate_pass"] is True


@pytest.mark.asyncio
async def test_on_a_non_cfr_master_fails_with_the_cfr_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_master(monkeypatch, cfr_ok=False)

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(mastering_gate_cfr_enabled=True, render_fps=30),
            loudness=_LOUDNESS,
        )

    # Same published shape as the colorspace violation: non-numeric, so
    # value/limit are null and the stage/metric/tier carry the verdict.
    assert excinfo.value.violations_payload() == [
        {"stage": "cfr", "metric": "cfr_ok", "value": None, "limit": None, "tier": "tunable"}
    ]
    assert "cfr: delivered master is not constant-frame-rate" in str(excinfo.value)
    assert "render_fps 30" in str(excinfo.value)


@pytest.mark.asyncio
async def test_on_a_cfr_master_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_master(monkeypatch, cfr_ok=True)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(mastering_gate_cfr_enabled=True), loudness=_LOUDNESS
    )

    assert report["cfr_ok"] is True
    assert report["rerun_stages"] == []
    assert report["violations"] == []
    assert report["gate_pass"] is True


@pytest.mark.asyncio
async def test_on_an_unmeasured_cadence_passes_like_an_unprobed_colorspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ABSENT MEASUREMENT => PASS. ``cfr_ok is None`` means the probe produced
    nothing usable; the gate convicts only a MEASURED ``False`` — the rule
    ``colorspace_ok`` follows (``if colorspace_ok is False``). The gap is not
    silent: it is named in ``unmeasured``."""
    _stub_master(monkeypatch, cfr_ok=None)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(mastering_gate_cfr_enabled=True), loudness=_LOUDNESS
    )

    assert report["cfr_ok"] is None
    assert report["violations"] == []
    assert report["gate_pass"] is True
    assert "cfr" not in report["rerun_stages"]


@pytest.mark.asyncio
async def test_on_cfr_and_colorspace_both_report_in_rerun_stage_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the switch on, cfr joins ``violations`` at the position it already
    held in ``rerun_stages`` (first), and is not named there twice."""
    _stub_master(monkeypatch, cfr_ok=False)

    async def bt601_tags(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=smpte170m\ncolor_transfer=smpte170m\ncolor_primaries=smpte170m\n"
        if "signalstats" in joined:
            return _ALIVE_FRAME
        return "30/1\n" if "stream=avg_frame_rate" in joined else ""

    monkeypatch.setattr(renderer, "_run", bt601_tags)

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"), _settings(mastering_gate_cfr_enabled=True), loudness=_LOUDNESS
        )

    assert [v.stage for v in excinfo.value.violations] == ["cfr", "colorspace"]


# ---------------------------------------------------------------------------
# The pts-interval half of ``cfr_ok``
# ---------------------------------------------------------------------------

#: Master 01a11032's one hole, as ``video_pts_gaps`` reports it.
_ONE_GAP: tuple[int | None, list[dict[str, float]] | None] = (
    1,
    [{"at_seconds": 20.233333, "interval_ms": 66.667}],
)


@pytest.mark.parametrize(
    ("rate_coherent", "gap_count", "expected"),
    [
        (True, 0, True),
        (True, 1, False),  # master 01a11032: coherent headers over a missing frame
        (False, 0, False),
        (False, 1, False),
        (None, 1, False),  # one MEASURED bad reading is a verdict
        (False, None, False),
        (True, None, None),  # coherent headers alone are not "verified CFR"
        (None, 0, None),
        (None, None, None),
    ],
)
def test_cfr_verdict_needs_both_readings_good(
    rate_coherent: bool | None, gap_count: int | None, expected: bool | None
) -> None:
    assert mastering._cfr_verdict(rate_coherent, gap_count) is expected


@pytest.mark.asyncio
async def test_a_pts_gap_under_coherent_headers_is_not_cfr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delivered defect: every header reading says CFR, one frame slot is
    empty. ``cfr_ok`` is False and the report says where. Switch OFF: advisory."""
    _stub_master(monkeypatch, cfr_ok=True, pts_gaps=_ONE_GAP)

    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness=_LOUDNESS)

    assert report["cfr_ok"] is False
    assert report["video_pts_gap_count"] == 1
    assert report["video_pts_gaps"] == _ONE_GAP[1]
    assert report["rerun_stages"] == ["cfr"]
    assert report["violations"] == []
    assert report["gate_pass"] is True
    assert "cfr_ok" not in report["unmeasured"]


@pytest.mark.asyncio
async def test_on_a_pts_gap_fails_the_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_master(monkeypatch, cfr_ok=True, pts_gaps=_ONE_GAP)
    settings = _settings(mastering_gate_cfr_enabled=True, render_fps=30)

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(Path("/tmp/out.mp4"), settings, loudness=_LOUDNESS)

    assert [v["stage"] for v in excinfo.value.violations_payload()] == ["cfr"]
    # The reason names WHICH reading convicted it — the headers were fine.
    assert "1 irregular video pts interval(s)" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("armed", [False, True])
async def test_an_unmeasured_pts_timeline_is_never_a_pass(
    monkeypatch: pytest.MonkeyPatch, armed: bool
) -> None:
    """Coherent headers over a timeline nobody could read is the blind pass
    01a11032 got. It must read ``None`` — unmeasured, named, never gating — not
    ``True`` and not a fabricated zero."""
    _stub_master(monkeypatch, cfr_ok=True, pts_gaps=(None, None))

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(mastering_gate_cfr_enabled=armed), loudness=_LOUDNESS
    )

    assert report["cfr_ok"] is None
    assert report["video_pts_gap_count"] is None
    assert report["video_pts_gaps"] is None
    assert "cfr_ok" in report["unmeasured"]
    assert report["violations"] == []
    assert report["gate_pass"] is True
    assert "cfr" not in report["rerun_stages"]
