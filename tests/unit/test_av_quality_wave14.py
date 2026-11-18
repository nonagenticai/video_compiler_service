"""WAVE-14 — FINISH WHAT WAS MEASURED: signals that decided nothing now decide.

Five defects, one family, and the family is this project's signature failure
mode: a value is measured, published, and consumed by NOBODY.

* V1. ``colorspace_ok`` read FALSE on 23.3% of a n=120 master sample. The ENCODE
  tags correctly everywhere; the loss is at the terminal ``-c:v copy`` pass that
  rides an active audio re-encode. ``overlay_audio_bed`` re-stamped there (via
  ``_grade_video_args``) and the other three terminal passes did not — so a film
  WITH a music bed was tagged and a film WITHOUT one was not. An AUDIO branch,
  not an asset class.
* V2. ``MasteringGateError.violations_payload`` had ZERO callers. The structured
  ``violations`` contract was live only for masters that PASS; the failing
  master — the entire point of a gate — delivered prose.
* V3. ``true_peak_dbtp`` was measured and published on every report and compared
  to NOTHING at any value.
* V4. ``master_measure_failed`` had no gating consumer: the master silently
  degraded to dynamic loudnorm and shipped ``gate_pass=True``.
* V6. ``vo_measure_failed`` was a LOG LINE and ``music_was_measured`` a published
  boolean, while ``vo_lufs=None`` sent both the bed gain and the ducker's open
  threshold to literal fallbacks.

⭐ EVERY assertion here is MUTATION-TESTED: the line the assertion covers was
broken, the test observed to FAIL, and the break then reverted. The mutants are
real edits to production lines, not no-ops — four "surviving mutations" last wave
were the agent's own no-ops. This file asserts BEHAVIOUR (what the built ffmpeg
command contains, what the report carries, what the gate raises), never the
presence of a call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import Settings, get_settings
from src.services import mastering, renderer
from tests.conftest import is_delivered_master_measure

pytestmark = pytest.mark.anyio


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, has_audio: bool = True) -> list[list[str]]:
    """Capture the ffmpeg argv of every ``_run`` without executing anything."""
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        cmds.append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_has_audio(path: Path) -> bool:
        return has_audio

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    return cmds


_TAG_FLAGS = ("-color_primaries", "-color_trc", "-colorspace")


def _assert_tagged_copy(cmd: list[str], where: str) -> None:
    joined = " ".join(cmd)
    assert "-c:v copy" in joined, f"{where}: must stay a STREAM COPY, not a re-encode"
    for flag in _TAG_FLAGS:
        assert flag in cmd, (
            f"{where}: {flag} absent — the video map is a bare `-c:v copy` beside an "
            f"active audio re-encode, so the delivered master's tagging depends on "
            f"muxer codecpar propagation, which ffmpeg 7.1.5 does not guarantee"
        )
    assert "libx264" not in joined, f"{where}: re-tagging must not become a re-encode"


# ---------------------------------------------------------------------------
# V1 — the colorspace 23.3%: re-stamp on EVERY terminal copy pass, not just the
#      music-bed one.
# ---------------------------------------------------------------------------


async def test_master_audio_restamps_the_colour_triplet_on_its_copy_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE HEADLINE. ``master_audio`` is the NO-BED loudness master — the terminal
    pass for every film that carries no music or ambient bed. It mapped
    ``-map 0:v -c:v copy`` beside an active ``-filter_complex`` and emitted no
    colour tags, while the bed branch beside it re-stamped. That difference is an
    audio branch, and it is the 23.3%."""
    cmds = _install_fakes(monkeypatch)
    video = tmp_path / "concat.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", _settings())
    cmd = cmds[-1]
    assert "-filter_complex" in cmd, "guard: this must be the ACTIVE-audio-filter branch"
    _assert_tagged_copy(cmd, "master_audio")


async def test_apply_postclip_audio_restamps_the_colour_triplet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``apply_postclip_audio`` is the LAST pass to touch any compile that requests
    a per-clip gain or mute, so whatever it writes is what ships."""
    cmds = _install_fakes(monkeypatch)
    video = tmp_path / "mastered.mp4"
    video.write_bytes(b"\x00")
    await renderer.apply_postclip_audio(
        video, tmp_path / "postclip.mp4", "volume=-6.0dB", _settings()
    )
    cmd = cmds[-1]
    assert "-filter_complex" in cmd, "guard: this must be the ACTIVE-audio-filter branch"
    _assert_tagged_copy(cmd, "apply_postclip_audio")


async def test_encode_delivery_audio_restamps_the_colour_triplet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_encode_delivery_audio`` is the terminal pass on the two
    mastering-disabled configurations — the ALAC→AAC conversion that exists so a
    deliverable can never leave as lossless. Its video map is a copy beside an
    audio re-encode: the same exposure."""
    cmds = _install_fakes(monkeypatch)
    video = tmp_path / "bus.mp4"
    video.write_bytes(b"\x00")
    await renderer._encode_delivery_audio(video, tmp_path / "delivery.mp4", _settings())
    _assert_tagged_copy(cmds[-1], "_encode_delivery_audio")


async def test_the_audioless_guards_stay_plain_remuxes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ THE INVERSE, and it is load-bearing. The three no-audio-stream guards are
    ``-c copy`` with NO filter at all — measured to preserve tags — and adding a
    video-map rewrite there would turn a whole-file remux into a per-stream map
    and silently DROP the audio-less file's other streams. Only the branches that
    ride an active audio filter get the re-stamp."""
    src = tmp_path / "silent.mp4"
    src.write_bytes(b"\x00")

    def _assert_plain_remux(cmd: list[str], where: str) -> None:
        joined = " ".join(cmd)
        assert "-c copy" in joined, where
        assert "-filter_complex" not in joined, where
        for flag in _TAG_FLAGS:
            assert flag not in cmd, (
                f"{where}: the no-audio guard is a WHOLE-FILE remux; a video-map "
                f"rewrite here would drop every stream the map does not name"
            )

    cmds = _install_fakes(monkeypatch, has_audio=False)
    await renderer.master_audio(src, tmp_path / "out1.mp4", _settings())
    _assert_plain_remux(cmds[-1], "master_audio")

    cmds = _install_fakes(monkeypatch, has_audio=False)
    await renderer.apply_postclip_audio(src, tmp_path / "out2.mp4", "volume=0", _settings())
    _assert_plain_remux(cmds[-1], "apply_postclip_audio")

    cmds = _install_fakes(monkeypatch, has_audio=False)
    await renderer._encode_delivery_audio(src, tmp_path / "out3.mp4", _settings())
    _assert_plain_remux(cmds[-1], "_encode_delivery_audio")


# ---------------------------------------------------------------------------
# V2 — the FAILING master delivers the structured verdict, not prose.
# ---------------------------------------------------------------------------


def test_a_gate_failure_persists_its_violations_not_only_a_detail_string() -> None:
    """``MasteringGateError.violations_payload`` had ZERO callers, so the whole
    structured contract was live only for masters that PASS. ``_failure_result``
    is the consumer: what it returns is written to the job row's ``result``, which
    ``get_job`` already projects onto ``JobStatus.mastering`` for any status."""
    from src.api.routes import compile as compile_route

    exc = mastering.MasteringGateError(
        [
            mastering.MasteringGateViolation(
                stage="true_peak",
                metric="true_peak_dbtp",
                value=0.8,
                limit=0.0,
                tier="structural",
                detail="true_peak: delivered 0.80 dBTP exceeds full scale",
            )
        ]
    )
    result = compile_route._failure_result(exc)
    assert result is not None
    report = result["mastering"]
    assert isinstance(report, dict)
    assert report["gate_pass"] is False, "a failing master must not read as passing"
    (entry,) = report["violations"]
    # The NUMBERS, which the joined `detail` prose can only be string-matched for.
    assert entry == {
        "stage": "true_peak",
        "metric": "true_peak_dbtp",
        "value": 0.8,
        "limit": 0.0,
        "tier": "structural",
    }


def test_every_other_failure_leaves_the_result_column_alone() -> None:
    """``None`` means ``mark_failed`` does not touch ``result`` — so an ffmpeg
    crash, a timeout or an upload fault keeps exactly the behaviour it has."""
    from src.api.routes import compile as compile_route

    assert compile_route._failure_result(RuntimeError("ffmpeg exit=-9")) is None
    assert compile_route._failure_result(TimeoutError()) is None


# ---------------------------------------------------------------------------
# V3 — true_peak_dbtp is COMPARED to something.
# ---------------------------------------------------------------------------


def test_a_master_above_full_scale_produces_a_structural_violation() -> None:
    """The comparison that never existed. Above 0 dBTP the master limiter did not
    run or did not hold; a downstream lossy transcode reconstructs a waveform that
    cannot be represented and distorts."""
    violation = mastering._true_peak_violation(0.7, _settings(master_enabled=True))
    assert violation is not None
    assert violation.stage == "true_peak"
    assert violation.metric == "true_peak_dbtp"
    assert violation.value == 0.7
    assert violation.limit == mastering.TRUE_PEAK_CEILING_DBTP
    assert violation.tier == "structural", (
        "a clipped master needs no fleet calibration — full scale is a property "
        "of the format, not a percentile"
    )


@pytest.mark.parametrize("peak", [-1.0, -0.3, 0.0])
def test_a_master_at_or_below_full_scale_is_clean(peak: float) -> None:
    """⛔ NOT GATED AT ``master_tp``. The limiter AIMS at -1.0 dBTP and
    ``_LIMITER_TP_HEADROOM_DB`` deliberately aims 0.3 dB below that to absorb a
    measured 0.07-0.13 dB of intersample overshoot — so a ceiling at ``master_tp``
    would convict masters the limiter handled exactly as designed. -0.3 and 0.0
    are the two values that fail if anyone "tightens" this to the target."""
    assert mastering._true_peak_violation(peak, _settings(master_enabled=True)) is None


def test_true_peak_does_not_gate_a_job_that_switched_mastering_off() -> None:
    """UNSATISFIABLE-GATE GUARD, the same one ``_loudness_violation`` carries. With
    ``master_enabled`` False the loudnorm+alimiter chain is SKIPPED ENTIRELY, so a
    raw mix peaks wherever it peaks and re-running the named stage would change
    nothing — a gate that can never be satisfied fails a correct render forever."""
    assert mastering._true_peak_violation(3.0, _settings(master_enabled=False)) is None


def test_an_unmeasured_true_peak_is_not_a_violation() -> None:
    """NO-GATE-ON-A-GAP, deliberately the opposite of loudness's fail-closed: an
    unmeasured peak says nothing about clipping, and is already reported through
    ``unmeasured``. Failing closed here would convict every silent reel."""
    assert mastering._true_peak_violation(None, _settings(master_enabled=True)) is None


# ---------------------------------------------------------------------------
# V4 — master_measure_failed GATES instead of merely being published.
# ---------------------------------------------------------------------------


def test_a_failed_pass_one_measurement_is_a_structural_violation() -> None:
    """When pass 1 crashes, ``_master_filter`` is built with ``measured=None`` and
    the master runs single-pass DYNAMIC loudnorm — audibly different gain riding
    than the two-pass linear master that was requested. It shipped
    ``gate_pass=True`` with no note."""
    violation = mastering.measure_violation(measure_failed=True, settings=_settings())
    assert violation is not None
    assert violation.stage == "measure"
    assert violation.metric == "master_measure_failed"
    assert violation.tier == "structural"
    assert not mastering.gate_armed(_settings(), "measure"), (
        "NOT armed: a dynamically-normalised master is a SHIPPABLE film — worse "
        "than requested, not undeliverable — and the pass-1 failure rate is "
        "n=0. The fix is that the degradation is now VISIBLE, not that delivery "
        "is refused; see the fold tests below"
    )


def test_a_clean_measurement_and_a_mastering_off_job_produce_nothing() -> None:
    assert mastering.measure_violation(measure_failed=False, settings=_settings()) is None
    assert (
        mastering.measure_violation(measure_failed=True, settings=_settings(master_enabled=False))
        is None
    ), "with mastering off there is no measurement pass that could have failed"


# ---------------------------------------------------------------------------
# V6 — the bed's calibration failure DEGRADES the master.
# ---------------------------------------------------------------------------


def test_an_uncalibrated_bed_is_a_violation() -> None:
    """``vo_lufs=None`` sends BOTH ``_offset_gain_db`` and
    ``_vo_relative_threshold`` to literal fallbacks: the bed sits at an arbitrary
    level under the narration and the ducker's open threshold becomes an absolute
    constant the bus may never cross. That was a log line."""
    violation = mastering.bed_calibration_violation(
        {"vo_measure_failed": True, "vo_lufs": None}, _settings()
    )
    assert violation is not None
    assert violation.stage == "bed_calibration"
    assert violation.metric == "vo_measure_failed"
    assert violation.tier == "structural"
    assert not mastering.gate_armed(_settings(), "bed_calibration"), (
        "NOT armed: the trigger is a loudness-probe failure whose fleet rate is "
        "n=0, so arming would convict the PROBE and destroy paid deliverables"
    )


def test_a_bed_that_never_asked_for_calibration_is_not_a_violation() -> None:
    """⛔ THE DISCRIMINATOR. ``vo_lufs`` is legitimately ``None`` on EVERY job that
    requested no VO-relative knobs — the overwhelming majority — so a gate keyed on
    ``vo_lufs is None`` would fire on the whole default population. Only the
    renderer's conjunction (calibration wanted AND unmeasurable) says anything."""
    assert (
        mastering.bed_calibration_violation(
            {"vo_measure_failed": False, "vo_lufs": None}, _settings()
        )
        is None
    )
    assert mastering.bed_calibration_violation(None, _settings()) is None


async def test_the_renderer_publishes_the_conjunction_not_the_ambiguous_half(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The producer side of the discriminator above: ``overlay_audio_bed`` is the
    only place that knows BOTH whether calibration was wanted and whether it
    succeeded, so it must publish the conjunction rather than leaving a consumer
    to guess from ``vo_lufs``."""
    from src.api.schemas import MusicBed

    _install_fakes(monkeypatch)

    async def fake_measure(path: str) -> dict[str, float] | None:
        return None  # the probe failed

    async def fake_probe(path: Path) -> float:
        return 30.0

    monkeypatch.setattr("src.services.loudness.final_mux_loudness", fake_measure)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    asset = tmp_path / "bed.mp3"
    asset.write_bytes(b"\x00")
    video = tmp_path / "concat.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        30.0,
        (asset, MusicBed(url="http://x/bed.mp3", music_offset_lu=14.0)),
        [],
        _settings(),
    )
    assert report is not None
    assert report["vo_measure_failed"] is True
    assert report["vo_lufs"] is None
    assert report["music_was_measured"] is False
    # …and that report, handed to the gate, is a real degradation.
    assert mastering.bed_calibration_violation(report, _settings()) is not None


async def test_a_bed_with_no_relative_knobs_never_reports_a_measure_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ THE OTHER HALF OF THE DISCRIMINATOR, and the assertion that was MISSING:
    a mutation replacing ``wants_measurement and vo_lufs is None`` with the bare
    ``vo_lufs is None`` SURVIVED every other test in this file.

    A plain music bed with no ``music_offset_lu`` / ``duck_open_lu_below_vo``
    never triggers a narration measurement at all, so ``vo_lufs`` is ``None`` by
    DESIGN. Without the ``wants_measurement`` conjunction this default —
    overwhelmingly the common case — would report ``vo_measure_failed=True`` and
    the gate would fire on the entire fleet."""
    from src.api.schemas import MusicBed

    _install_fakes(monkeypatch)

    async def unreachable(path: str) -> dict[str, float] | None:
        raise AssertionError("no narration measurement may run without relative knobs")

    async def fake_probe(path: Path) -> float:
        return 30.0

    monkeypatch.setattr("src.services.loudness.final_mux_loudness", unreachable)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    asset = tmp_path / "bed.mp3"
    asset.write_bytes(b"\x00")
    video = tmp_path / "concat.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        30.0,
        (asset, MusicBed(url="http://x/bed.mp3")),
        [],
        _settings(),
    )
    assert report is not None
    assert report["vo_lufs"] is None, "guard: no measurement was requested"
    assert report["vo_measure_failed"] is False, (
        "an unmeasured-BY-DESIGN bed is not a failure — gating on `vo_lufs is "
        "None` alone would convict every default compile"
    )
    assert mastering.bed_calibration_violation(report, _settings()) is None


# ---------------------------------------------------------------------------
# The FOLD: an UNARMED gate must still degrade the report.
# ---------------------------------------------------------------------------


def test_an_unarmed_route_gate_still_clears_gate_pass() -> None:
    """⛔ THE WHOLE POINT OF THE WAVE. If a disarmed violation simply vanished, it
    would be telemetry again — recorded, published, consumed by nothing, with the
    report still saying ``gate_pass=True``. Folding happens regardless of arming;
    arming decides only whether delivery is ALSO refused."""
    report: dict[str, Any] = {"gate_pass": True, "rerun_stages": [], "violations": []}
    violation = mastering.bed_calibration_violation({"vo_measure_failed": True}, _settings())
    # Disarmed by default — proven above — so this must NOT raise.
    folded = mastering.apply_route_violations(report, (violation,), _settings())
    assert folded["gate_pass"] is False
    assert folded["rerun_stages"] == ["bed_calibration"]
    entries = folded["violations"]
    assert isinstance(entries, list)
    (entry,) = entries
    assert entry["stage"] == "bed_calibration"
    assert entry["tier"] == "structural"


def test_an_armed_route_gate_refuses_delivery() -> None:
    """With its switch ON, the same fold path must RAISE — the raise is what
    actually prevents the upload stage that follows it. This is also the
    back-out contract: the kill switch is the ONLY difference between degrade
    and refuse, so arming ``measure`` later needs no code change."""
    report: dict[str, Any] = {"gate_pass": True, "rerun_stages": [], "violations": []}
    armed_settings = _settings(mastering_gate_measure_enabled=True)
    violation = mastering.measure_violation(measure_failed=True, settings=armed_settings)
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        mastering.apply_route_violations(report, (violation,), armed_settings)
    assert [v["stage"] for v in excinfo.value.violations_payload()] == ["measure"]


def test_a_clean_compile_passes_its_report_through_untouched() -> None:
    """Identity on the clean path: nothing moves for a compile with no finding."""
    report: dict[str, Any] = {"gate_pass": True, "rerun_stages": [], "violations": []}
    assert mastering.apply_route_violations(report, (None, None), _settings()) == report
