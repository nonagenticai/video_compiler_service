"""Unit tests for the encode/delivery master helpers (VC-2, contract C11)."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable, Coroutine
from pathlib import Path

import pytest

from src.api.schemas import JobStatus, MasteringReport
from src.config import Settings, get_settings
from src.services import mastering, measure, renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


def _frame_durations(values: list[str]) -> str:
    """Render an ffprobe per-frame duration dump the way the REAL probe does.

    ``_frame_durations_are_uniform`` asks for two entry names at once
    (``frame=duration_time,pkt_duration_time``), so it cannot use ``nokey=1`` and
    the output is ``key=value`` lines. Fakes that returned bare values were
    encoding the pre-ffmpeg-7 single-key shape and silently stopped matching the
    probe at all."""
    return "".join(f"duration_time={v}\n" for v in values)


# ---------------------------------------------------------------------------
# colorspace_tag_args (pure)
# ---------------------------------------------------------------------------


def test_colorspace_tag_args_default_is_bt709() -> None:
    args = mastering.colorspace_tag_args(_settings())
    assert args == [
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
    ]


def test_colorspace_tag_args_disabled_is_empty() -> None:
    for val in ("", "none", "off"):
        assert mastering.colorspace_tag_args(_settings(render_colorspace=val)) == []


def test_colorspace_tag_args_unknown_degrades_to_empty() -> None:
    assert mastering.colorspace_tag_args(_settings(render_colorspace="rec2100")) == []


def test_colorspace_tag_args_bt2020() -> None:
    args = mastering.colorspace_tag_args(_settings(render_colorspace="bt2020"))
    assert args[args.index("-colorspace") + 1] == "bt2020nc"
    assert args[args.index("-color_trc") + 1] == "bt2020-10"


# ---------------------------------------------------------------------------
# banding (pure)
# ---------------------------------------------------------------------------


def test_banding_score_zero_for_clean_or_unmeasured() -> None:
    assert mastering.banding_score({}) == 0.0
    # High temporal diff (textured / dithered) → no banding risk.
    assert mastering.banding_score({"YMIN": 16, "YMAX": 235, "YDIF": 10.0}) == 0.0
    # Narrow luma range → no smooth ramp to band.
    assert mastering.banding_score({"YMIN": 100, "YMAX": 120, "YDIF": 0.5}) == 0.0


def test_banding_score_trips_on_smooth_wide_gradient() -> None:
    score = mastering.banding_score(
        {"YMIN": 16, "YMAX": 235, "YLOW": 38, "YHIGH": 213, "YDIF": 0.5}
    )
    assert score > 0.15
    assert (
        mastering.banding_risk({"YMIN": 16, "YMAX": 235, "YLOW": 38, "YHIGH": 213, "YDIF": 0.5})
        is True
    )


def test_banding_risk_false_for_textured_content() -> None:
    assert mastering.banding_risk({"YMIN": 16, "YMAX": 235, "YDIF": 10.0}) is False


# ---------------------------------------------------------------------------
# deband_filter + encode_ladder (pure)
# ---------------------------------------------------------------------------


def test_deband_filter_is_a_deband_fragment() -> None:
    frag = mastering.deband_filter()
    assert frag.startswith("deband")


def test_deband_filter_takes_no_settings_argument() -> None:
    """C3 honesty fix: the function took a ``Settings`` and ignored it, which
    read as "these thresholds are configurable". They are not. Pin the signature
    so the dead parameter cannot be reintroduced by a caller that "just passes
    settings along"."""
    import inspect

    assert list(inspect.signature(mastering.deband_filter).parameters) == []


def test_deband_enabled_policy() -> None:
    assert mastering.deband_enabled(_settings(render_deband="auto")) is True
    assert mastering.deband_enabled(_settings(render_deband="on")) is True
    assert mastering.deband_enabled(_settings(render_deband="off")) is False
    assert mastering.deband_enabled(_settings(render_deband="false")) is False


def test_encode_ladder_default_cell_is_render_crf_no_vbv() -> None:
    """The default cell must equal render_crf with NO VBV args → byte-identical
    encode versus the pre-master path."""
    crf, vbv = mastering.encode_ladder(_settings(render_crf=20))
    assert crf == 20
    assert vbv == []


def test_encode_ladder_motion_nudges_crf_but_default_content_does_not() -> None:
    assert mastering.encode_ladder(_settings(render_crf=20), "unknown")[0] == 20
    assert mastering.encode_ladder(_settings(render_crf=20), "flat")[0] == 20
    assert mastering.encode_ladder(_settings(render_crf=20), "motion")[0] == 21


def test_encode_ladder_vbv_only_when_configured() -> None:
    crf, vbv = mastering.encode_ladder(
        _settings(render_crf=18, render_bitrate_mode="vbv", render_vbv_maxrate="12M")
    )
    assert crf == 18
    assert vbv == ["-maxrate", "12M", "-bufsize", "24M"]  # bufsize derived ~2x


def test_encode_ladder_vbv_respects_explicit_bufsize() -> None:
    _, vbv = mastering.encode_ladder(
        _settings(
            render_bitrate_mode="cbr",
            render_vbv_maxrate="8000k",
            render_vbv_bufsize="8000k",
        )
    )
    assert vbv == ["-maxrate", "8000k", "-bufsize", "8000k"]


def test_encode_ladder_maxrate_ignored_when_mode_auto() -> None:
    """A stray maxrate must NOT emit VBV under the default auto (CRF) mode."""
    _, vbv = mastering.encode_ladder(
        _settings(render_bitrate_mode="auto", render_vbv_maxrate="12M")
    )
    assert vbv == []


# ---------------------------------------------------------------------------
# _video_codec_args regression — default = pre-master args + additive colr only
# ---------------------------------------------------------------------------


def test_video_codec_args_default_is_pre_master_plus_colr() -> None:
    """With default knobs the libx264 args are the pre-master args with ONLY the
    additive BT.709 colr tags inserted — no CRF/preset/VBV change."""
    args = renderer._video_codec_args(_settings())
    assert args[:4] == ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    assert args[args.index("-crf") + 1] == "18"
    assert args[args.index("-preset") + 1] == "medium"
    # additive colr present
    assert args[args.index("-colorspace") + 1] == "bt709"
    assert "-maxrate" not in args  # no VBV by default


def test_video_codec_args_untagged_when_colorspace_disabled() -> None:
    args = renderer._video_codec_args(_settings(render_colorspace="none"))
    assert "-colorspace" not in args
    assert args[args.index("-crf") + 1] == "18"


# ---------------------------------------------------------------------------
# framerate_coherence + post_mux_verify (ffmpeg-touching, mocked _run)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_framerate_coherence_true_when_rates_match(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return "30/1\n30/1\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering.framerate_coherence(Path("/tmp/x.mp4"), 30) is True


@pytest.mark.asyncio
async def test_framerate_coherence_false_when_vfr(monkeypatch: pytest.MonkeyPatch) -> None:
    # Headers disagree AND the frames really are irregular → VFR.
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if "frame=duration_time" in " ".join(cmd):
            return _frame_durations(["0.033333", "0.050000", "0.016667", "0.041667"])
        return "30/1\n2997/100\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering.framerate_coherence(Path("/tmp/x.mp4"), 30) is False


@pytest.mark.asyncio
async def test_disagreeing_headers_over_uniform_frames_is_cfr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ THE FALSE NEGATIVE THIS GUARDS — measured on three delivered masters.

    Each probed ``r_frame_rate=30/1`` vs ``avg_frame_rate=8678400/289403`` (29.9873)
    — 0.0127 apart, over the old 0.01 ceiling, so each was reported VFR. Each was
    in fact CFR frame-for-frame: 565 of 567 frames at exactly 0.033333. The header
    average is frames/duration and carries the duration's rounding; it cannot
    answer CFR. The frames can, so they are asked.
    """

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if "frame=duration_time" in " ".join(cmd):
            # The real shape: one dominant duration plus ffprobe's N/A tail.
            return _frame_durations(["0.033333"] * 565 + ["N/A", "N/A"])
        return "30/1\n8678400/289403\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering.framerate_coherence(Path("/tmp/x.mp4"), 30) is True


@pytest.mark.asyncio
async def test_framerate_coherence_none_when_frame_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unmeasurable confirmation must read UNMEASURED, never as a pass — this
    # codebase's stated #1 defect class.
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if "frame=duration_time" in " ".join(cmd):
            raise renderer.FfmpegError("ffprobe exit=1")
        return "30/1\n2997/100\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering.framerate_coherence(Path("/tmp/x.mp4"), 30) is None


@pytest.mark.asyncio
async def test_framerate_coherence_none_on_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("ffprobe exit=1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await mastering.framerate_coherence(Path("/tmp/x.mp4"), 30) is None


@pytest.mark.asyncio
async def test_post_mux_verify_happy_path_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed BT.709, CFR, low-banding, on-target file passes the gate."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=time_base:packet=pts" in joined:
            return '{"streams": [{"time_base": "1/30"}], "packets": [{"pts": 0}, {"pts": 1}, {"pts": 2}]}'
        # signalstats measure — textured, no banding
        return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(),
        loudness={"integrated_lufs": -12.4, "true_peak_dbtp": -1.1},
    )
    assert report["colorspace_ok"] is True
    assert report["cfr_ok"] is True
    assert report["integrated_lufs"] == -12.4  # reused, not re-measured
    assert report["true_peak_dbtp"] == -1.1
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_flags_unmeasured_loudness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loudness that could not be measured (integrated_lufs=None) must NOT silently
    pass the gate — it is flagged inconclusive, not treated as on-target."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        # textured, no banding, non-dead luma
        return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    # loudness={} → final_mux_loudness measure/parse gap → integrated_lufs is None.
    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})
    assert report["integrated_lufs"] is None
    assert "loudness" in report["rerun_stages"]
    assert report["gate_pass"] is False  # previously (bug) silently True


@pytest.mark.asyncio
async def test_post_mux_verify_flags_colorspace_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=smpte170m\ncolor_transfer=smpte170m\ncolor_primaries=smpte170m\n"  # the bt601 triplet, not the requested bt709
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        return "lavfi.signalstats.YDIF=9.0\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})
    assert report["colorspace_ok"] is False
    assert report["gate_pass"] is False
    assert "colorspace" in report["rerun_stages"]


@pytest.mark.asyncio
async def test_post_mux_verify_never_raises_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("boom")

    monkeypatch.setattr(renderer, "_run", boom)
    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})
    # colorspace probe failed → None (unknown, not a hard fail); no crash.
    assert report["colorspace_ok"] is None
    assert report["cfr_ok"] is None
    assert isinstance(report["gate_pass"], bool)
    # ⚠️ WAVE-13, AND THIS IS WHY ``banding_measurement`` WAS LEFT DISARMED: a
    # total probe failure means the picture was never MEASURED, which is a
    # structural FINDING (gate_pass=False + a 'structural' violations entry) but
    # not an armed one. Arming it would convict the PROBE rather than the master,
    # so a transient ffmpeg failure over a clean film would fail a paid compile.
    # The report still says so, in the contracted shape.
    assert report["gate_pass"] is False
    assert [v["stage"] for v in report["violations"]] == ["banding_measurement", "loudness"]
    assert report["violations"][0]["tier"] == "structural"
    # ...and the switch is WIRED: an operator who measures the probe-failure rate
    # can arm it from config alone.
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(mastering_gate_banding_measurement_enabled=True),
            loudness={},
        )
    assert "banding_measurement" in {v["stage"] for v in excinfo.value.violations_payload()}


def test_degraded_report_shape() -> None:
    report = mastering.degraded_report()
    assert report["gate_pass"] is False
    assert report["rerun_stages"] == ["verify"]
    assert report["banding_scores"] == []
    assert report["per_shot"] == []  # S20 per-shot banding degrades to empty
    # VC-3 measured-cadence fields degrade to None.
    assert report["measured_fps"] is None
    assert report["duplicate_frame_ratio"] is None


# ---------------------------------------------------------------------------
# VC-3 measured-cadence metrics (ffmpeg-touching, mocked _run)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_measured_fps_parses_avg_frame_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return "30000/1001\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    fps = await mastering.measured_fps(Path("/tmp/x.mp4"))
    assert fps is not None
    assert abs(fps - 29.97) < 0.01


@pytest.mark.asyncio
async def test_measured_fps_none_on_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("ffprobe exit=1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await mastering.measured_fps(Path("/tmp/x.mp4")) is None


@pytest.mark.asyncio
async def test_duplicate_frame_ratio_from_frame_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        # mpdecimate pass keeps fewer frames than the raw count.
        return "frame=   90 fps=..." if "mpdecimate" in cmd else "frame=  100 fps=..."

    monkeypatch.setattr(renderer, "_run", fake_run)
    ratio = await mastering.duplicate_frame_ratio(Path("/tmp/x.mp4"))
    assert ratio == 0.1  # (100 - 90) / 100


@pytest.mark.asyncio
async def test_duplicate_frame_ratio_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("boom")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await mastering.duplicate_frame_ratio(Path("/tmp/x.mp4")) is None


@pytest.mark.asyncio
async def test_duplicate_frame_ratio_real_ffmpeg(tmp_path: Path) -> None:
    """End-to-end mpdecimate probe over a real tiny clip: a solid-colour clip is
    almost all duplicate frames, so the ratio is high — and a bad path → None."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    clip = tmp_path / "solid.mp4"
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=64x64:d=1:r=10",
        "-pix_fmt",
        "yuv420p",
        str(clip),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    ratio = await mastering.duplicate_frame_ratio(clip)
    assert ratio is not None
    assert 0.0 < ratio <= 1.0  # solid clip → mostly duplicate frames
    # never-raises on a missing file → None
    assert await mastering.duplicate_frame_ratio(tmp_path / "nope.mp4") is None


@pytest.mark.asyncio
async def test_post_mux_verify_populates_measured_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return "lavfi.signalstats.YDIF=9.0\n"
        if "mpdecimate" in cmd:
            return "frame=   95 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    # On-target loudness so the loudness gate is a no-op and this test isolates the
    # measured-cadence property (an unmeasured `loudness={}` now flags the gate).
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["measured_fps"] == 30.0
    assert report["duplicate_frame_ratio"] == 0.05  # (100 - 95) / 100
    # Advisory only — does not flip the gate.
    assert report["gate_pass"] is True


# ---------------------------------------------------------------------------
# DEAD/BLANK-master gate: a uniform (near-zero luma spread) delivered file must
# fail the soft-gate, while a legit static screen (wide spread, ~all duplicate
# frames) must NOT be false-flagged. duplicate_frame_ratio can't discriminate;
# intra-frame luma spread (YMAX-YMIN) can.
# ---------------------------------------------------------------------------


def test_dead_frame_predicate() -> None:
    # Now a RATIO gate: fraction of measured frames below the luma floor.
    # Half the film blank → dead.
    assert mastering._dead_frame(0.5, 0.10) is True
    # Exactly at the threshold → dead (>=, so the boundary is not a silent gap).
    assert mastering._dead_frame(0.10, 0.10) is True
    # A 2s fade in a 72s master (~2.8% of frames) → NOT dead. This is the whole
    # point: the old `min(spreads) < floor` rule failed the entire film on it.
    assert mastering._dead_frame(0.028, 0.10) is False
    # None (no luma data) → never gate on a measurement gap.
    assert mastering._dead_frame(None, 0.10) is False
    # min_ratio 0.0 restores the old any-single-frame behaviour, so an operator can
    # get the strict rule back without a code change.
    assert mastering._dead_frame(0.001, 0.0) is True
    assert mastering._dead_frame(0.0, 0.0) is False


def test_dead_frame_ratio_counts_only_measured_frames() -> None:
    """A partial measurement gap must not dilute the ratio toward 'alive'."""
    blank = {"YMIN": 16.0, "YMAX": 17.0}  # spread 1 < floor
    alive = {"YMIN": 16.0, "YMAX": 235.0}  # spread 219
    gap = {"YAVG": 100.0}  # no luma keys at all

    # 1 blank of 2 MEASURED frames = 0.5, even though 4 frames were sampled.
    assert mastering._dead_frame_ratio([blank, alive, gap, gap], 8.0) == 0.5
    # No measured frame → None, never gated.
    assert mastering._dead_frame_ratio([gap, gap], 8.0) is None
    assert mastering._dead_frame_ratio([], 8.0) is None
    # All blank → 1.0.
    assert mastering._dead_frame_ratio([blank, blank], 8.0) == 1.0


def test_a_fade_to_black_does_not_fail_the_film_but_a_blank_half_does() -> None:
    """The contested case, stated end-to-end.

    `signalstats` measures EVERY frame (no select/fps filter in `_signalstats_cmd`),
    so at 30fps a 1s fade is ~1.4% of a 72s master. Under the previous
    `min(spreads) < floor` rule that single dark frame failed the whole render.
    """
    blank = {"YMIN": 16.0, "YMAX": 17.0}
    alive = {"YMIN": 16.0, "YMAX": 235.0}
    floor, min_ratio = 8.0, 0.10

    # 72s @ 30fps = 2160 frames, 1s fade to black = 30 of them.
    fade = [blank] * 30 + [alive] * 2130
    assert mastering._dead_frame(mastering._dead_frame_ratio(fade, floor), min_ratio) is False

    # The case the per-frame check exists for: blank for half its length.
    half_dead = [blank] * 1080 + [alive] * 1080
    assert mastering._dead_frame(mastering._dead_frame_ratio(half_dead, floor), min_ratio) is True

    # And a fully dead master, obviously.
    assert mastering._dead_frame(mastering._dead_frame_ratio([blank] * 100, floor), min_ratio)


def test_worst_luma_spread_picks_the_worst_frame() -> None:
    # A file blank for only PART of its length: the worst frame collapses to ~0
    # even though the others are rich, so a mean-based check would hide it.
    frames = [
        {"YMIN": 16.0, "YMAX": 235.0},  # rich
        {"YMIN": 100.0, "YMAX": 100.0},  # blank
        {"YMIN": 16.0, "YMAX": 235.0},  # rich
    ]
    assert mastering._worst_luma_spread(frames) == 0.0
    # Single rich frame → its own spread.
    assert mastering._worst_luma_spread([{"YMIN": 16.0, "YMAX": 235.0}]) == 219.0
    # No frames / no luma keys → None (never treated as dead).
    assert mastering._worst_luma_spread([]) is None
    assert mastering._worst_luma_spread([{"YDIF": 1.0}]) is None


@pytest.mark.asyncio
async def test_post_mux_verify_gates_blank_dead_master(monkeypatch: pytest.MonkeyPatch) -> None:
    """A uniform/blank delivered master (luma spread ~0) fails the soft-gate with
    a ``dead_frame`` rerun even though colorspace/cfr/loudness all look fine."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:  # blank frame: YMAX == YMIN
            return (
                "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=16\nlavfi.signalstats.YDIF=0.0\n"
            )
        if "mpdecimate" in cmd:
            return "frame=  100 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    # WAVE-13: `dead_frame` is ARMED (structural), so a blank master no longer
    # returns an annotated report — it fails the compile and never uploads.
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})
    assert "dead_frame" in {v["stage"] for v in excinfo.value.violations_payload()}
    # Disarmed via its own kill switch, the historical advisory report shape is
    # exactly what comes back — which is what makes the switch a back-out path
    # rather than decoration.
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(mastering_gate_dead_frame_enabled=False),
        loudness={},
    )
    assert report["luma_spread"] == 0.0
    assert report["gate_pass"] is False
    assert "dead_frame" in report["rerun_stages"]


@pytest.mark.asyncio
async def test_post_mux_verify_static_still_is_not_dead_and_duplicates_never_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bright, wide-spread, ~all-duplicate master PASSES its gate.

    ⚠️ This assertion has flipped twice; read the reason before flipping it again.
    Originally it passed on the strength of a ``duplicate_frame_advisory`` flag;
    W1-3 armed the ratio and made it FAIL; the fleet measurement (120 of 120
    masters over the ceiling, minimum ratio 0.643) retired the gate for good,
    because duplicate frames are intentional dwell — 6500ms holds captured at a
    hardcoded 25fps — and not a defect at any ratio.

    What this still proves is the discrimination that mattered all along:
    ``dead_frame`` does NOT fire here. Blankness is intra-frame luma spread and
    this frame has spread 219, so the two signals stay independent."""

    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.99))
    # On-target loudness so the gate reflects only the properties under test.
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["luma_spread"] == 219.0  # 235 - 16 → emphatically not blank
    assert "dead_frame" not in report["rerun_stages"]
    assert report["duplicate_frame_ratio"] == 0.99
    assert report["rerun_stages"] == []
    assert report["gate_pass"] is True


# ---------------------------------------------------------------------------
# DUPLICATE FRAMES ARE TELEMETRY, NOT A GATE (2026-09-11). ``duplicate_frame_ratio``
# gated between W1-3 and this change. Measured over n=120 completed masters — the
# whole fleet, 2026-08-27→09-11 — it ran min .643 / p50 .896 / max .998 against
# the 0.10 ceiling and therefore convicted 120 of 120 deliveries, which is what
# made ``gate_pass`` carry no information. The frames are intentional dwell
# (6500ms hold x Playwright's hardcoded 25fps), so no ceiling can separate a
# correct master from a frozen one. These tests pin that the ratio is still
# MEASURED and REPORTED, that it never reaches ``gate_pass``, and that neither the
# predicate nor its setting exists to be re-armed.
# ---------------------------------------------------------------------------


def _fake_dup(value: float | None) -> Callable[..., Coroutine[None, None, float | None]]:
    """A fake ``mastering.duplicate_frame_ratio`` returning a fixed ratio."""

    async def fake(path: Path | str) -> float | None:
        return value

    return fake


def test_there_is_no_duplicate_frame_predicate_or_setting() -> None:
    """The gate was DELETED, not neutralised with ``ceiling=1.0``.

    A disabled-but-present predicate plus its knob is an invitation to re-arm
    something the fleet measurement disqualified, so this pins their absence: the
    only way back is a deliberate code change with a new measurement behind it."""
    assert not hasattr(mastering, "_duplicate_frames")
    assert "render_duplicate_frame_ratio_ceiling" not in Settings.model_fields
    # And no CODE in the verification module still reads a ceiling for it. Comment
    # lines are stripped first: this module deliberately DOCUMENTS the removed
    # setting by name, and matching that prose would make the assertion vacuous in
    # the other direction (it would fail on the explanation, not on a gate).
    code = "\n".join(
        line
        for line in Path(mastering.__file__).read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "render_duplicate_frame_ratio_ceiling" not in code


@pytest.mark.asyncio
async def test_post_mux_verify_does_not_gate_a_92_percent_duplicate_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE REGRESSION TEST FOR THE FALSE POSITIVE. The 0.9197 master W1-3 built its
    gate around — every other signal clean, 92% duplicate frames — now PASSES, and
    queues no rerun. At p50 .896 that master is not an outlier, it is the typical
    delivery; failing it failed everything. The ratio is still reported."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9197))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["duplicate_frame_ratio"] == 0.9197
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_does_not_gate_the_worst_observed_duplicate_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet MAXIMUM (0.998) passes too — there is no residual ceiling hiding
    above the numbers the p50 test exercises."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.998))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["duplicate_frame_ratio"] == 0.998
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_duplicate_ratio_under_ceiling_does_not_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low end of the measured range keeps passing: a briskly-cut master (5%
    duplicates) reports its ratio and gates on nothing. Paired with the two tests
    above, this pins that the verdict is INDEPENDENT of the ratio across its whole
    domain, not merely tolerant at one end."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.05))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["duplicate_frame_ratio"] == 0.05
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_unmeasured_duplicate_ratio_does_not_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An mpdecimate PROBE GAP (ratio None) must not gate — a naive
    ``if dup_ratio > ceiling`` written without the None branch would raise, and a
    naive ``if not dup_ratio`` would red every unprobed master."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(None))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["duplicate_frame_ratio"] is None
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


def test_mastering_report_dropped_the_dead_advisory_flag() -> None:
    """``duplicate_frame_advisory`` is gone from both the payload and the schema.
    It was the gate's own comparison published a second time with zero consumers;
    leaving it would let the advisory and the verdict drift apart."""
    assert "duplicate_frame_advisory" not in MasteringReport.model_fields
    assert "duplicate_frame_advisory" not in mastering.degraded_report()


@pytest.mark.asyncio
async def test_post_mux_verify_gates_partially_dead_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master blank for only PART of its length: the whole-file AVERAGE spread is
    wide (~110, escapes a mean check) but the WORST frame collapses to ~0 → the
    per-frame gate still fails it with ``dead_frame``. This is the case a mean-based
    check silently passed."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:  # frame 0 rich, frame 1 blank (real ffmpeg prefix)
            return (
                "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
                "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"
                "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
                "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=16\nlavfi.signalstats.YDIF=0.0\n"
            )
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    # WAVE-13: armed → raises. Disarmed → the historical report shape.
    with pytest.raises(mastering.MasteringGateError):
        await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(mastering_gate_dead_frame_enabled=False),
        loudness={},
    )
    assert report["luma_spread"] == 0.0  # the WORST frame, not the ~110 average
    assert report["gate_pass"] is False
    assert "dead_frame" in report["rerun_stages"]


# ---------------------------------------------------------------------------
# FIX #5 — localized banding must surface as the WORST frame, not the segment
# mean (a band in a few frames must not be averaged away before the gate).
# ---------------------------------------------------------------------------

# Three clean frames (narrow luma range, high temporal diff → score 0) around a
# single heavily-banded frame (wide smooth ramp). The segment MEAN of these
# dilutes the band below the trigger; the per-frame MAX preserves it.
_BANDED_FRAMES_DUMP = """\
frame:0    pts:0       pts_time:0
lavfi.signalstats.YMIN=100
lavfi.signalstats.YLOW=104
lavfi.signalstats.YHIGH=116
lavfi.signalstats.YMAX=120
lavfi.signalstats.YDIF=8.0
frame:1    pts:512     pts_time:0.017
lavfi.signalstats.YMIN=16
lavfi.signalstats.YLOW=38
lavfi.signalstats.YHIGH=213
lavfi.signalstats.YMAX=235
lavfi.signalstats.YDIF=0.5
frame:2    pts:1024    pts_time:0.034
lavfi.signalstats.YMIN=100
lavfi.signalstats.YLOW=104
lavfi.signalstats.YHIGH=116
lavfi.signalstats.YMAX=120
lavfi.signalstats.YDIF=8.0
"""


@pytest.mark.asyncio
async def test_localized_band_not_averaged_away(monkeypatch: pytest.MonkeyPatch) -> None:
    """A band living in ONE frame of an otherwise-clean segment must reach the
    gate as the worst-frame score, not be diluted into a clean mean."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return _BANDED_FRAMES_DUMP
        if "mpdecimate" in cmd:
            return "frame=  100 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(Path("/tmp/out.mp4"), _settings(), loudness={})

    worst = mastering.banding_score(
        {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 38.0, "YHIGH": 213.0, "YDIF": 0.5}
    )
    assert report["banding_scores"] == [worst]
    assert report["banding_scores"][0] > 0.15  # the band survived to the REPORT
    # The old segment-MEAN path would have averaged the band away to ~0:
    assert mastering.banding_score(measure.parse_signalstats(_BANDED_FRAMES_DUMP)) == 0.0
    # ...so the worst-frame value is what a reader sees. It is NOT a verdict:
    # banding went advisory-only on 2026-09-11 (no usable spatial term; 89.2% of
    # 120 measured masters over the ceiling), so nothing is queued for rerun and
    # the gate holds.
    assert "deband" not in report["rerun_stages"]
    assert report["rerun_stages"] == ["loudness"]  # only because loudness={} here


# ---------------------------------------------------------------------------
# S20 — PER-SHOT banding: a discarded timed measure pass buckets each frame's
# banding onto its shot's [start, end) window, ADDITIVELY — the whole-file gate
# (banding_scores / gate_pass) stays byte-identical whether or not boundaries are
# supplied.
# ---------------------------------------------------------------------------

# A ramp spreads pixels EVENLY: the 10-90 band covers ~90% of the range (measured
# 0.914 on a real 8-bit gradient). YLOW/YHIGH are in every real frame's stats and
# the spatial term added 2026-09-02 reads them.
_BANDED = {"YMIN": 16.0, "YMAX": 235.0, "YLOW": 38.0, "YHIGH": 213.0, "YDIF": 0.5}
_CLEAN = {"YMIN": 100.0, "YMAX": 120.0, "YDIF": 8.0}  # narrow + textured → score 0


def test_bucket_banding_localizes_band_to_its_shot() -> None:
    """A band living in shot 2's time window surfaces ONLY on shot 2; the clean
    shots score ~0 and do not need deband."""
    timed: list[tuple[float | None, dict[str, float]]] = [
        (0.5, _CLEAN),
        (1.5, _BANDED),  # inside shot b's [1, 2) window
        (2.5, _CLEAN),
    ]
    timeline: list[dict[str, object]] = [
        {"line_id": "a", "start": 0.0, "end": 1.0},
        {"line_id": "b", "start": 1.0, "end": 2.0},
        {"line_id": "c", "start": 2.0, "end": 3.0},
    ]
    per_shot = mastering.bucket_banding(timed, timeline)
    assert [p["line_id"] for p in per_shot] == ["a", "b", "c"]
    assert per_shot[0]["banding"] == 0.0
    assert per_shot[0]["needs_deband"] is False
    banding_b = per_shot[1]["banding"]
    assert isinstance(banding_b, float) and banding_b > 0.15
    assert per_shot[1]["needs_deband"] is True
    assert per_shot[2]["banding"] == 0.0
    assert per_shot[2]["needs_deband"] is False


def test_bucket_banding_ignores_untimed_frames_and_empty_shots() -> None:
    """A frame with no pts_time cannot be placed; a shot with no in-window frame
    scores 0.0 and never raises."""
    timed: list[tuple[float | None, dict[str, float]]] = [(None, _BANDED)]
    per_shot = mastering.bucket_banding(timed, [{"line_id": "a", "start": 0.0, "end": 1.0}])
    assert per_shot == [
        {
            "line_id": "a",
            "banding": 0.0,
            "banding_residual_over_trigger": False,
            "needs_deband": False,
        }
    ]


@pytest.mark.asyncio
async def test_per_shot_does_not_alter_whole_file_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole-file banding_scores + gate are byte-identical whether or not a
    shot_timeline is passed; per_shot is populated only when it is."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return _BANDED_FRAMES_DUMP
        if "mpdecimate" in cmd:
            return "frame=  100 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    without = await mastering.post_mux_verify(Path("/tmp/o.mp4"), _settings(), loudness={})
    with_tl = await mastering.post_mux_verify(
        Path("/tmp/o.mp4"),
        _settings(),
        loudness={},
        shot_timeline=[{"line_id": "x", "start": 0.0, "end": 1.0}],
    )
    assert without["banding_scores"] == with_tl["banding_scores"]
    assert without["gate_pass"] == with_tl["gate_pass"]
    assert without["rerun_stages"] == with_tl["rerun_stages"]
    assert without["per_shot"] == []  # no boundaries → no per-shot data
    # The band (frame:1 @ pts_time 0.017) lands in shot x's [0, 1) window.
    assert len(with_tl["per_shot"]) == 1
    assert with_tl["per_shot"][0]["line_id"] == "x"
    assert with_tl["per_shot"][0]["banding"] > 0.15
    assert with_tl["per_shot"][0]["needs_deband"] is True


@pytest.mark.asyncio
async def test_post_mux_verify_per_shot_empty_without_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shot_timeline None or [] → per_shot == [], never raises, no timed pass."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "signalstats" in joined:
            return "lavfi.signalstats.YDIF=9.0\n"
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    none_report = await mastering.post_mux_verify(
        Path("/tmp/o.mp4"), _settings(), loudness={}, shot_timeline=None
    )
    empty_report = await mastering.post_mux_verify(
        Path("/tmp/o.mp4"), _settings(), loudness={}, shot_timeline=[]
    )
    assert none_report["per_shot"] == []
    assert empty_report["per_shot"] == []


@pytest.mark.asyncio
async def test_per_shot_survives_response_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per_shot data must SURVIVE pydantic validation into the real response
    path (MasteringReport nested in JobStatus). Without a ShotBanding field on
    MasteringReport, pydantic's default extra='ignore' would silently drop it and
    the orchestrator would never see it — this locks that boundary."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return _BANDED_FRAMES_DUMP
        if "mpdecimate" in cmd:
            return "frame=  100 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(
        Path("/tmp/o.mp4"),
        _settings(),
        loudness={},
        shot_timeline=[{"line_id": "x", "start": 0.0, "end": 1.0}],
    )
    assert len(report["per_shot"]) == 1  # populated in the raw dict

    # Nested exactly as the GET /api/jobs/{id} route validates it (compile.py):
    # the raw report dict is coerced into MasteringReport during JobStatus validation.
    status = JobStatus.model_validate({"job_id": "j1", "status": "completed", "mastering": report})
    assert status.mastering is not None
    assert len(status.mastering.per_shot) == 1  # survived validation, not dropped
    entry = status.mastering.per_shot[0]
    assert entry.line_id == "x"
    assert entry.banding > 0.15
    assert entry.needs_deband is True
    # ...and survives round-trip back out to the wire.
    assert status.model_dump()["mastering"]["per_shot"][0]["line_id"] == "x"
    # Standalone validation of a raw dict into MasteringReport keeps per_shot too.
    assert len(MasteringReport.model_validate(report).per_shot) == 1


# ---------------------------------------------------------------------------
# FIX #1 — characterization: the per-job LUFS override is the effective loudness
# target the gate compares against (NOT the static default). Locks the invariant
# that post_mux_verify receives the per-job settings copy.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_job_lufs_override_gated_against_effective_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.api.routes.compile import _settings_for_tags

    # Build the per-job Settings exactly as the compile handler does (tag fold →
    # model_copy), then verify the gate uses THAT target, not the -12.5 default.
    settings = _settings_for_tags(get_settings(), {"master_lufs": "-9.0"})
    assert settings.master_lufs == -9.0

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "signalstats" in joined:
            return "lavfi.signalstats.YDIF=9.0\n"
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), settings, loudness={"integrated_lufs": -9.2}
    )
    # -9.2 is within 3 LU of the per-job -9.0 target → NO loudness rerun. It WOULD
    # be flagged (|-9.2 - -12.5| = 3.3 > 3) if the gate used the static default.
    assert "loudness" not in report["rerun_stages"]


# ---------------------------------------------------------------------------
# FIX #2 — the hardcoded default LUFS is a single named constant.
# ---------------------------------------------------------------------------


def test_default_master_lufs_constant() -> None:
    assert mastering.DEFAULT_MASTER_LUFS == -14.0


# ---------------------------------------------------------------------------
# cfr_ok is ADVISORY, not gating — and the docstring must say so.
#
# ``cfr_ok`` is measured next to colorspace / deband / loudness / dead_frame but,
# unlike all four, it deliberately never clears ``gate_pass`` and never appends to
# ``rerun_stages``: no stage of this pipeline can re-cut cadence (VC-3 owns it at
# encode time; framerate_coherence is probe-only), and an incoherent-but-playable
# master is suboptimal, not unshippable. These tests pin that decision so a future
# "obvious" fix has to argue with them instead of silently wedging runs.
# ---------------------------------------------------------------------------


def _clean_master_run(r_avg: str) -> Callable[..., Coroutine[None, None, str]]:
    """A fake ``renderer._run`` for an otherwise-CLEAN master whose
    ``r_frame_rate``/``avg_frame_rate`` probe returns ``r_avg``."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return r_avg
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "stream=time_base:packet=pts" in joined:
            # The pts timeline itself is regular: what these callers vary is the
            # HEADER pair, and ``cfr_ok`` must follow that alone.
            return '{"streams": [{"time_base": "1/30"}], "packets": [{"pts": 0}, {"pts": 1}, {"pts": 2}]}'
        if "frame=duration_time" in joined:
            # The frame-level CONFIRMATION `framerate_coherence` runs when the two
            # header rates disagree. Callers that pass a mismatched pair (e.g.
            # "30/1\n24/1\n") mean a genuinely irregular stream, so the frames must
            # be irregular too — a uniform answer here would describe a DIFFERENT
            # file than the header line the caller declared.
            return _frame_durations(["0.033333", "0.041667", "0.033333", "0.050000"])
        if "signalstats" in joined:  # wide spread, textured → not dead, not banded
            return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"
        return ""

    return fake_run


def _blank_master_run() -> Callable[..., Coroutine[None, None, str]]:
    """A fake ``renderer._run`` for a DEAD (uniform/blank) master: every frame has
    ``YMAX == YMIN``, so ``dead_frame_ratio`` is 1.0 and the dead-frame gate — one
    of the four that still gate — fires."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return (
                "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=16\nlavfi.signalstats.YDIF=0.0\n"
            )
        return ""

    return fake_run


@pytest.mark.asyncio
async def test_post_mux_verify_incoherent_cfr_is_advisory_not_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r_frame_rate 30 vs avg_frame_rate 24 → ``cfr_ok is False``, but the master is
    otherwise clean, so ``gate_pass`` stays True — cadence is emission-only (3d):
    it names itself in ``rerun_stages`` but never clears the gate."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n24/1\n"))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["cfr_ok"] is False
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == ["cfr"]


@pytest.mark.asyncio
async def test_post_mux_verify_coherent_cfr_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverse: a frame-rate-coherent master reports ``cfr_ok is True`` and is
    likewise clean — proving the advisory branch changes nothing on good input."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["cfr_ok"] is True
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_unprobed_cfr_does_not_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cadence PROBE GAP (ffprobe returned one rate, so ``cfr_ok is None``) must
    not gate either — a naive ``if not cfr_ok`` fix would red every such master."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n"))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["cfr_ok"] is None
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_still_gates_its_gating_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the other direction: making cfr advisory must not have declawed the
    real gates. Same incoherent cadence, but an untagged colorspace → gate fails,
    and ``rerun_stages`` carries BOTH the real gating stage and the emission-only
    ``cfr`` entry (cadence is measured first, see the ordering above)."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=smpte170m\ncolor_transfer=smpte170m\ncolor_primaries=smpte170m\n"  # the bt601 triplet, not bt709
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n24/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "frame=duration_time" in joined:
            # The header pair above declares an outright cadence mismatch, so the
            # frames must be irregular to match it — `framerate_coherence` now
            # confirms a header disagreement against the frames themselves.
            return _frame_durations(["0.033333", "0.041667", "0.033333", "0.050000"])
        if "signalstats" in joined:
            return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["cfr_ok"] is False
    assert report["colorspace_ok"] is False
    assert report["gate_pass"] is False
    assert report["rerun_stages"] == ["cfr", "colorspace"]


def test_post_mux_verify_docstring_matches_the_code() -> None:
    """The docstring used to list frame-rate coherence among the things this
    function "verifies", which read as a gate it never was. Pin the corrected
    contract: every name in ``rerun_stages`` is documented as GATING, and every
    non-gating measurement (cfr_ok first among them) is documented as ADVISORY."""
    doc = mastering.post_mux_verify.__doc__
    assert doc is not None
    gating, _, rest = doc.partition("ADVISORY-ONLY")
    assert rest, "docstring must separate gating from advisory signals"
    # The advisory section runs up to the WAVE-13 arming note; the trailing prose
    # after it documents the tiering + kill switches and names every gate
    # (including the disarmed ones), so it is not an advisory claim.
    #
    # ⚠️ The partition marker used to be "NEVER raises" — a phrase the docstring
    # no longer contains, because the function now RAISES BY DEFAULT for the
    # structural tier. A `partition` on an absent marker returns the WHOLE string
    # as the head and an EMPTY tail, so this test would have silently gone
    # vacuous (advisory == "") rather than failing, had the assertions below not
    # been positive ones.
    advisory, _, _tail = rest.partition("THIS FUNCTION RAISES BY DEFAULT")
    assert _tail, "docstring must document that the structural tier now raises"
    # The five signals that really do clear gate_pass + name a rerun stage.
    for stage in ("colorspace", "deband", "loudness", "dead_frame", "duplicate_frames"):
        assert stage in gating, f"{stage} gates but is not documented as gating"
        assert stage not in advisory
    # cfr_ok is measured but never gates — it must be on the advisory side only.
    assert "cfr_ok" in advisory
    assert "cfr_ok" not in gating
    for signal in ("measured_fps", "true_peak_dbtp"):
        assert signal in advisory
    # WAVE-13: the arming note must name the tier, the per-gate switch mechanism,
    # and the two gates held DISARMED — a docstring that says "raises" without
    # saying which gates and how to back one out is the dangerous half of honest.
    arming = "THIS FUNCTION RAISES BY DEFAULT" + _tail
    for claim in ("STRUCTURAL", "TUNABLE", "gate_armed", "kill switch", "colorspace"):
        assert claim in arming, f"the arming note does not mention {claim}"


# ---------------------------------------------------------------------------
# mastering_gate_mode — the soft gate can be made a HARD one (T0-1). The gate
# has never been able to fail a render: it computes gate_pass, returns a dict,
# and the route uploads the artifact regardless. `advisory` (DEFAULT) keeps that
# byte-for-byte; `enforce` raises with the offending metric named.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_mode_advisory_is_the_default_and_still_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master with a REAL defect under the shipped default: reported, NOT raised.
    This is the behaviour every deployment runs today and it must not move.

    The defect is loudness 17.5 LU off target — a still-gating check. This test
    used to drive the same assertion from ``duplicate_frame_ratio=0.9197``, which
    no longer gates anything; the 0.9197 is kept in the fixture to prove exactly
    that, since the only rerun stage named is the loudness one."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9197))
    settings = _settings()
    assert settings.mastering_gate_mode == "advisory"
    assert mastering.gate_enforced(settings) is False

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), settings, loudness={"integrated_lufs": -30.0}
    )
    assert report["gate_pass"] is False
    assert report["rerun_stages"] == ["loudness"]
    assert report["duplicate_frame_ratio"] == 0.9197  # measured, reported, ignored


@pytest.mark.asyncio
async def test_gate_mode_enforce_raises_naming_the_offending_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same master, mode flipped: the render FAILS, and the message names WHICH
    metric cleared the gate with its measured value against its ceiling — not a
    generic "mastering gate failed". Driven by loudness since the duplicate-frame
    gate is gone; the 92%-duplicate ratio stays in the fixture and contributes
    NOTHING to the violation list, which is the point."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9197))
    settings = _settings(mastering_gate_mode="enforce")
    assert mastering.gate_enforced(settings) is True

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"), settings, loudness={"integrated_lufs": -30.0}
        )

    message = str(excinfo.value)
    assert "loudness" in message
    assert "-30.00 LUFS" in message  # the measured value
    assert "-14.00" in message  # the target it missed
    assert "duplicate" not in message
    violations = excinfo.value.violations
    assert [v.stage for v in violations] == ["loudness"]
    assert violations[0].metric == "integrated_lufs"
    assert violations[0].value == -30.0
    assert violations[0].limit == -14.0


@pytest.mark.asyncio
async def test_gate_mode_enforce_names_every_cleared_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master that clears TWO checks reports both — an operator must not have to
    re-run once per defect to discover the next one. The pair used to be
    loudness + duplicate_frames; it is now loudness + dead_frame, i.e. two checks
    that both still gate."""
    monkeypatch.setattr(renderer, "_run", _blank_master_run())
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9197))

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(mastering_gate_mode="enforce"),
            loudness={"integrated_lufs": -30.0},  # 16.0 LU off the -14.0 target
        )

    assert [v.stage for v in excinfo.value.violations] == ["loudness", "dead_frame"]
    assert "-30.00 LUFS" in str(excinfo.value)
    assert "16.00 LU off target" in str(excinfo.value)


@pytest.mark.asyncio
async def test_gate_mode_enforce_passes_a_clean_master(monkeypatch: pytest.MonkeyPatch) -> None:
    """The false-positive guard: enforce must fail BAD masters, not every master."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.05))

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(mastering_gate_mode="enforce"),
        loudness={"integrated_lufs": -12.5},
    )
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []


@pytest.mark.asyncio
async def test_gate_mode_enforce_does_not_gate_an_unprobed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PROBE GAP is not a violation: mpdecimate returning None must not fail a
    render under enforce either, or every unprobed master reds."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(None))

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(mastering_gate_mode="enforce"),
        loudness={"integrated_lufs": -12.5},
    )
    assert report["duplicate_frame_ratio"] is None
    assert report["gate_pass"] is True


# ---------------------------------------------------------------------------
# DECODE BUDGET (perf, behaviour-preserving). ``post_mux_verify`` used to decode
# the delivered master FOUR times; two of those were pure waste:
#   1. ``measure_shot_frames`` AND ``measure_shot_frames_timed`` issued the
#      BYTE-IDENTICAL signalstats command and differed only in their parser.
#   2. ``duplicate_frame_ratio`` full-decoded the master a second time purely to
#      COUNT frames, which a demux-only ffprobe packet walk answers.
# These tests pin the decode budget itself, not just the numbers — a regression
# that silently reinstates a full pass keeps every value correct and would go
# unnoticed by every other test in this file.
# ---------------------------------------------------------------------------

# A realistic two-frame ``signalstats,metadata=print`` dump WITH pts_time, as the
# real ffmpeg filter tag prefixes it.
_TIMED_SIGNALSTATS_DUMP = (
    "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMIN=16\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMAX=235\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YDIF=9.0\n"
    "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMIN=20\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMAX=225\n"
    "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YDIF=7.0\n"
)


def _verify_runner(calls: list[list[str]]) -> Callable[..., Coroutine[None, None, str]]:
    """A mocked ``renderer._run`` covering every probe ``post_mux_verify`` issues,
    recording each argv so a test can assert on the DECODE BUDGET."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        calls.append(list(cmd))
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "nb_read_packets" in joined:
            return "100\n"
        if "signalstats,metadata=print" in joined:
            return _TIMED_SIGNALSTATS_DUMP
        if "mpdecimate" in cmd:
            return "frame=   95 fps=..."
        if "null" in cmd:
            return "frame=  100 fps=..."
        return ""

    return fake_run


def _signalstats_calls(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if any("signalstats" in a for a in c)]


def _plain_full_decode_calls(calls: list[list[str]]) -> list[list[str]]:
    """A plain full DECODE of the master with no filter — the shape
    ``duplicate_frame_ratio`` used to issue only to count frames."""
    return [
        c for c in calls if c and c[0] == "ffmpeg" and "-f" in c and "null" in c and "-vf" not in c
    ]


@pytest.mark.asyncio
async def test_post_mux_verify_issues_exactly_one_signalstats_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on origin/main (2 calls): the untimed and timed measure passes built the
    byte-identical command and were both executed."""
    calls: list[list[str]] = []
    monkeypatch.setattr(renderer, "_run", _verify_runner(calls))
    await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(),
        loudness={"integrated_lufs": -12.5},
        shot_timeline=[{"line_id": "l1", "start": 0.0, "end": 1.0}],
    )
    assert len(_signalstats_calls(calls)) == 1


@pytest.mark.asyncio
async def test_post_mux_verify_still_one_signalstats_decode_without_shot_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single pass is unconditional — no boundaries must not mean a second one."""
    calls: list[list[str]] = []
    monkeypatch.setattr(renderer, "_run", _verify_runner(calls))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert len(_signalstats_calls(calls)) == 1
    assert report["per_shot"] == []


@pytest.mark.asyncio
async def test_post_mux_verify_derived_untimed_frames_match_the_untimed_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frames handed to the whole-file gate must be EXACTLY what the untimed
    parser would have returned from the same stderr — the derivation
    ``[m for _, m in timed]`` is only legitimate if it is byte-identical."""
    calls: list[list[str]] = []
    monkeypatch.setattr(renderer, "_run", _verify_runner(calls))
    seen: dict[str, object] = {}
    real_spread = mastering._worst_luma_spread

    def spy(frames: list[dict[str, float]]) -> float | None:
        seen["frames"] = frames
        return real_spread(frames)

    monkeypatch.setattr(mastering, "_worst_luma_spread", spy)
    await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    expected = measure.parse_signalstats_per_frame(_TIMED_SIGNALSTATS_DUMP)
    assert expected  # fixture sanity: the untimed parser really finds frames
    assert seen["frames"] == expected


@pytest.mark.asyncio
async def test_duplicate_frame_ratio_counts_total_by_demux_not_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED on origin/main: the total came from a plain full decode. It must now
    come from an ffprobe packet walk, and the plain decode must not be issued."""
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        calls.append(list(cmd))
        if "nb_read_packets" in " ".join(cmd):
            return "100\n"
        if "mpdecimate" in cmd:
            return "frame=   90 fps=..."
        return "frame=  100 fps=..."

    monkeypatch.setattr(renderer, "_run", fake_run)
    ratio = await mastering.duplicate_frame_ratio(Path("/tmp/x.mp4"))
    assert ratio == 0.1  # (100 - 90) / 100
    assert _plain_full_decode_calls(calls) == []
    probes = [c for c in calls if "-count_packets" in c]
    assert len(probes) == 1
    assert "nb_read_packets" in " ".join(probes[0])


@pytest.mark.asyncio
async def test_duplicate_frame_ratio_falls_back_to_decode_on_garbage_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong denominator is a false "frozen video" verdict on a delivered
    master, so an empty/non-numeric ffprobe must fall back to the real decode."""

    def _runner(
        probe_out: str, calls: list[list[str]]
    ) -> Callable[..., Coroutine[None, None, str]]:
        async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
            calls.append(list(cmd))
            if "nb_read_packets" in " ".join(cmd):
                return probe_out
            if "mpdecimate" in cmd:
                return "frame=   90 fps=..."
            return "frame=  100 fps=..."

        return fake_run

    for probe_out in ("", "N/A\n", "   \n"):
        calls: list[list[str]] = []
        monkeypatch.setattr(renderer, "_run", _runner(probe_out, calls))
        ratio = await mastering.duplicate_frame_ratio(Path("/tmp/x.mp4"))
        assert ratio == 0.1, f"probe_out={probe_out!r}"
        assert len(_plain_full_decode_calls(calls)) == 1, f"probe_out={probe_out!r}"


def test_the_report_names_the_checks_that_could_not_run() -> None:
    """`gate_pass=True` alone conflates "verified" with "nothing to check against".

    The no-gate-on-a-measurement-gap policy is deliberate — failing closed on a
    probe timeout would reject good masters — but a gap invisible to the caller is
    the fail-open half of it. `unmeasured` names which inputs were absent so a
    consumer can tell "passed" from "not measured".
    """
    from src.services.mastering import _unmeasured_metrics

    assert _unmeasured_metrics(duplicate_frame_ratio=0.02, integrated_lufs=-18.0) == []
    assert _unmeasured_metrics(duplicate_frame_ratio=None, integrated_lufs=-18.0) == [
        "duplicate_frame_ratio"
    ]
    # Sorted and complete, so the caller can compare two reports.
    assert _unmeasured_metrics(luma_spread=None, dead_frame_ratio=None, measured_fps=30.0) == [
        "dead_frame_ratio",
        "luma_spread",
    ]


# ---------------------------------------------------------------------------
# PHASE 2 — WHAT ``gate_pass`` MEANS NOW (2026-09-11).
#
# Measured over n=120 completed masters, the whole fleet, 2026-08-27→09-11:
#   duplicate_frame_ratio   fired 120/120 (100%)  min .643 p50 .896 max .998
#   banding worst-frame     fired 89.2%           min 0   p50 .642 max 1.00
#   colorspace_ok False     23.3%   (an upstream TAGGING bug — left gating)
#   dead_frame_ratio        4.2%    at the 0.10 default
#   loudness |delta| > 3 LU 0%
#   av_coverage             0%      (already always-enforced)
# The first two were removed from the verdict; the rest are untouched. These
# tests pin the three shapes the removal must produce and the one it must not.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_90_percent_duplicate_master_with_nothing_else_wrong_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE HEADLINE CASE, and 120 of 120 masters were exactly this shape: a clean,
    correctly-tagged, on-target master whose only "defect" is dwell. It passed
    NONE of them before and it must pass all of them now — and the ratio must
    still be in the report, because it is the telemetry the dwell/fps
    relationship is read from."""
    monkeypatch.setattr(renderer, "_run", _clean_master_run("30/1\n30/1\n"))
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9))
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
    )
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []
    # TELEMETRY PRESERVED — not merely absent from the verdict.
    assert report["duplicate_frame_ratio"] == 0.9
    assert "duplicate_frame_ratio" in MasteringReport.model_fields


@pytest.mark.asyncio
async def test_a_heavily_banded_master_with_nothing_else_wrong_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Banding alone, with the deband stage ENABLED (the fleet default, so this is
    not the disabled-stage carve-out): worst-frame score far over the 0.15 trigger
    threshold, everything else clean → PASSES, and the score is still reported."""
    monkeypatch.setattr(renderer, "_run", _fully_banded_master_run())
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.05))
    settings = _settings()
    assert mastering.deband_enabled(settings) is True
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), settings, loudness={"integrated_lufs": -12.5}
    )
    # The fixture really is over the threshold — this test is not vacuous.
    assert max(report["banding_scores"]) >= mastering.BANDING_GATE_CEILING
    assert report["gate_pass"] is True
    assert report["rerun_stages"] == []
    # TELEMETRY PRESERVED, whole-file and per-shot.
    assert report["banding_scores"]
    assert "banding_scores" in MasteringReport.model_fields
    assert "per_shot" in MasteringReport.model_fields


@pytest.mark.asyncio
async def test_a_half_dead_master_is_still_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE GUARD AGAINST OVERSHOOT. Phase 2 removes false positives; it must not
    disarm the 4.2% of masters that are genuinely bad. A master blank for half its
    length (dead_frame_ratio 0.5, 5x the 0.10 default) still clears ``gate_pass``
    and still names ``dead_frame`` — even while its duplicate ratio and banding
    score are both at values that used to fail it for the wrong reason."""
    blank = "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=16\nlavfi.signalstats.YDIF=0.0\n"
    alive = "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return _frames_dump_timed([blank, alive])
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(mastering, "duplicate_frame_ratio", _fake_dup(0.9))
    # WAVE-13: the guard against overshoot now bites HARDER than a flag — a
    # half-blank master fails the compile outright and is never delivered.
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"), _settings(), loudness={"integrated_lufs": -12.5}
        )
    dead = next(v for v in excinfo.value.violations_payload() if v["stage"] == "dead_frame")
    assert dead["value"] == pytest.approx(0.5)
    assert dead["limit"] == pytest.approx(0.10)
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(mastering_gate_dead_frame_enabled=False),
        loudness={"integrated_lufs": -12.5},
    )
    assert report["dead_frame_ratio"] == pytest.approx(0.5)
    assert report["gate_pass"] is False
    assert report["rerun_stages"] == ["dead_frame"]


def _frames_dump_timed(frames: list[str]) -> str:
    """Wrap per-frame signalstats blocks in the ``metadata=print`` frame markers the
    real filter emits, so ``measure_shot_frames_timed`` parses them as N frames
    instead of one aggregate."""
    out = []
    for i, block in enumerate(frames):
        out.append(f"[Parsed_metadata_1 @ 0x1] frame:{i}    pts:{i * 512}    pts_time:{i * 0.033}")
        out.extend(f"[Parsed_metadata_1 @ 0x1] {line}" for line in block.splitlines())
    return "\n".join(out) + "\n"


def _fully_banded_master_run() -> Callable[..., Coroutine[None, None, str]]:
    """A fake ``renderer._run`` for a master that is a smooth wide gradient in every
    frame — the genuine banding signature (wide luma span, sub-threshold temporal
    delta, near-uniform 10-90 percentile band) — and clean in every other respect."""
    banded = (
        "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\n"
        "lavfi.signalstats.YLOW=38\nlavfi.signalstats.YHIGH=213\n"
        "lavfi.signalstats.YDIF=0.5\n"
    )

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return _frames_dump_timed([banded, banded])
        return ""

    return fake_run


# ---------------------------------------------------------------------------
# WAVE 11 — intermediate encode cell, CFR probe vocabulary, per-shot naming
# ---------------------------------------------------------------------------


def test_encode_ladder_intermediate_cell_is_below_the_delivery_cell() -> None:
    """⭐ THE HEADLINE OF C1(b). The delivered picture ran THREE x264 generations
    at the SAME delivery CRF: the segment encode, the deband/source-match
    re-encode, and the whole-film grade/bed pass. Generations 2+ re-quantise
    material already quantised at that exact cell, which is pure generational
    loss. The intermediate cell must be strictly lower than the delivery cell."""
    settings = _settings(render_crf=18, render_intermediate_crf_delta=6)
    delivery_crf, _ = mastering.encode_ladder(settings)
    intermediate_crf, _ = mastering.encode_ladder(settings, intermediate=True)
    assert delivery_crf == 18
    assert intermediate_crf == 12
    assert intermediate_crf < delivery_crf


def test_encode_ladder_intermediate_floors_at_zero() -> None:
    """CRF 0 is the floor libx264 accepts; a delta larger than the cell must not
    produce a negative -crf token (ffmpeg would reject the command outright and
    the polish pass would silently degrade to "keep the clean segment")."""
    # The DELIVERY cell is untouched by the delta - only the intermediate floors.
    crf, _ = mastering.encode_ladder(_settings(render_crf=4, render_intermediate_crf_delta=6))
    assert crf == 4
    crf, _ = mastering.encode_ladder(
        _settings(render_crf=4, render_intermediate_crf_delta=6), intermediate=True
    )
    assert crf == 0


def test_encode_ladder_intermediate_delta_zero_restores_previous_behaviour() -> None:
    """The reversibility contract: ``render_intermediate_crf_delta=0`` must make
    an intermediate encode use the delivery cell again, exactly as before this
    change — so the knob is a true rollback, not an approximation of one."""
    settings = _settings(render_crf=18, render_intermediate_crf_delta=0)
    assert mastering.encode_ladder(settings, intermediate=True)[0] == 18


def test_encode_ladder_intermediate_drops_vbv() -> None:
    """-maxrate/-bufsize model a DELIVERY constraint (a player's decode buffer).
    Leaving them on the intermediate would cap exactly the extra bits the lower
    CRF exists to keep, silently undoing the fix while the CRF token still looks
    correct. The delivered cell keeps its full VBV."""
    settings = _settings(
        render_crf=18,
        render_intermediate_crf_delta=6,
        render_bitrate_mode="vbv",
        render_vbv_maxrate="10M",
    )
    _, delivery_vbv = mastering.encode_ladder(settings)
    _, intermediate_vbv = mastering.encode_ladder(settings, intermediate=True)
    assert delivery_vbv == ["-maxrate", "10M", "-bufsize", "20M"]
    assert intermediate_vbv == []


def test_encode_ladder_crf_fallback_tracks_the_field_default() -> None:
    """C3 honesty fix. The fallback used to restate ``20`` while ``config.py``
    had moved ``render_crf`` to 18, so a Settings-shaped stub without the field
    laddered a different cell than the real service. Read the field, don't
    restate it."""

    class _NoCrf:
        pass

    from src.config import Settings as _Settings

    field_default = _Settings.model_fields["render_crf"].default
    assert mastering.encode_ladder(_NoCrf())[0] == field_default  # ty: ignore
    assert field_default == 18


def test_parse_frame_durations_reads_the_ffmpeg7_key() -> None:
    """⛔ C2 ROOT CAUSE. ``frame=pkt_duration_time`` was REMOVED in ffmpeg 7.0
    (renamed ``duration_time``). Asking for an unknown entry is SILENT — ffprobe
    prints nothing and exits 0 — so on the prod build (7.1.5) the probe returned
    zero lines and ``cfr_ok`` published null on every master."""
    out = "duration_time=0.033333\nduration_time=0.033333\n"
    assert mastering._parse_frame_durations(out) == ["0.033333", "0.033333"]


def test_parse_frame_durations_still_reads_the_pre_ffmpeg7_key() -> None:
    """A build older than 7.0 emits only ``pkt_duration_time``. Both names are
    requested and whichever PARSES is used, so no version detection is needed."""
    out = "pkt_duration_time=0.041667\npkt_duration_time=0.041667\n"
    assert mastering._parse_frame_durations(out) == ["0.041667", "0.041667"]


def test_parse_frame_durations_prefers_one_key_when_a_build_emits_both() -> None:
    """Merging both would contribute two lines per frame — harmless for the
    uniformity RATIO but it doubles the apparent sampled-frame count."""
    out = "duration_time=0.033333\npkt_duration_time=0.033333\n" * 3
    assert mastering._parse_frame_durations(out) == ["0.033333"] * 3


def test_parse_frame_durations_drops_the_na_tail() -> None:
    """ffprobe prints ``N/A`` for the tail frames of a stream. ``N/A`` is not a
    duration; counting it as its own 'dominant value' candidate would drag a
    clean master under the uniformity floor."""
    out = _frame_durations(["0.033333", "0.033333", "N/A"])
    assert mastering._parse_frame_durations(out) == ["0.033333", "0.033333"]


@pytest.mark.asyncio
async def test_frame_durations_probe_asks_for_the_ffmpeg7_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression that produced fleet-wide null ``cfr_ok``: pin the entry
    names ON THE COMMAND LINE, not just the parser. Also pin that ``nokey=1`` is
    NOT used — with two keys requested the values would be indistinguishable."""
    seen: list[str] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen.append(" ".join(cmd))
        return _frame_durations(["0.033333"] * 30)

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering._frame_durations_are_uniform(Path("/tmp/x.mp4")) is True
    joined = seen[0]
    assert "frame=duration_time,pkt_duration_time" in joined
    assert "nokey=1" not in joined


def test_bucket_banding_publishes_the_residual_name_and_the_alias() -> None:
    """C3 naming fix. The flag is measured on the DELIVERED file, i.e. AFTER the
    conditional deband already ran, so it reports a RESIDUAL — ``needs_deband``
    read as an unmet ACTION and is how ``rerun_stages=['deband']`` -> 'no stage
    was re-run; degrading' kept being chased. The alias is NOT optional:
    ``w2v_orchestrator``'s demo_video encode-critic names ``needs_deband`` in its
    prompt and pins it in a structured-output JSON schema."""
    timed: list[tuple[float | None, dict[str, float]]] = [(0.5, _BANDED)]
    per_shot = mastering.bucket_banding(timed, [{"line_id": "a", "start": 0.0, "end": 1.0}])
    assert per_shot[0]["banding_residual_over_trigger"] is True
    assert per_shot[0]["needs_deband"] is True
    assert per_shot[0]["needs_deband"] == per_shot[0]["banding_residual_over_trigger"]


# ---------------------------------------------------------------------------
# NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB — real-production calibration pin
# ---------------------------------------------------------------------------
#
# These numbers are NOT synthetic. They are the actual ``correlation_db``
# values ``mastering._narration_clip_correlation_db`` measured (ffmpeg 4.4,
# 2026-09-17) against a real production master's real narration audio, for the
# UNPADDED (``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS = 0.0``) statistic —
# see ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB`` for the full provenance
# table and the two production runs (clean + the real missing-clip defect)
# they came from. A pure classification test: no ffmpeg, no audio, just "does
# the threshold comparison put every one of these on the truth side".
_REAL_MEASURED_PRESENT_DB = [
    -7.75,  # good shot-000 (2.29s)
    -6.63,  # good clip2 (2.53s)
    -7.92,  # good clip3 (4.96s) — the longer clip the OLD statistic false-negatived
    -6.51,  # good clip4 (4.73s) — same
    -13.66,  # broken master line1 — worst real present reading
    -12.48,  # broken master line3
]
_REAL_MEASURED_ABSENT_DB = -90.0  # broken master line2 — the real missing clip (clamped -inf)


def test_narration_presence_threshold_classifies_all_real_measured_present_clips() -> None:
    for db in _REAL_MEASURED_PRESENT_DB:
        assert db >= mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB, (
            f"real present clip at {db} dB misclassified as absent"
        )


def test_narration_presence_threshold_classifies_the_real_measured_absent_clip() -> None:
    assert _REAL_MEASURED_ABSENT_DB < mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB


def test_narration_presence_threshold_is_below_the_worst_real_present_reading() -> None:
    """Pins the margin itself, not just the classification: if a future edit
    moves the threshold above the worst real present clip, this catches it even
    though the classification tests above would too — this one names the
    number that would have to change and why."""
    worst_present = min(_REAL_MEASURED_PRESENT_DB)
    assert worst_present == -13.66
    margin = worst_present - mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB
    assert margin > 0


# ---------------------------------------------------------------------------
# (a) LENGTH DEFECT — mutation-proof, calibrated on real measured values
# ---------------------------------------------------------------------------


def _write_silent_wav(dest: Path, *, duration: float, sample_rate: int = 8000) -> None:
    """A trivially real, ffmpeg-decodable mono 16-bit PCM WAV — pure ``wave``
    module, no ffmpeg needed to GENERATE it (only the production code's own
    decode step touches ffmpeg). Content is irrelevant here: the correlation
    step itself is monkeypatched below to isolate exactly one variable — the
    WINDOW DURATION ``_narration_clip_correlation_db`` builds — from the
    acoustic content, which real narration audio supplies in production."""
    import wave as wave_mod

    n_frames = int(duration * sample_rate)
    with wave_mod.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)


@pytest.mark.asyncio
async def test_long_clip_length_defect_mutation_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) — reproduces the LENGTH defect this PR fixes, calibrated on the REAL
    production readings for good_clip3 (4.96s): -7.92 dB against its own exact
    [start, end) window (the FIX), -21.03 dB against the OLD padded window
    (start-0.5, end+0.5) — both pinned in
    ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``'s provenance table.
    ``_axcorrelate_rms_db`` (the ffmpeg axcorrelate+astats call) is
    monkeypatched to return exactly those two real numbers keyed on the
    WINDOW file's actual duration — so this test exercises the real window-
    construction code in ``_narration_clip_correlation_db``
    (``_slice_pcm16_wav`` really slices a real WAV) while isolating it from
    needing real narration audio content.

    MUTATION-PROOF: setting ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS`` back
    to its old value (0.5) makes ``corr_db`` read -21.03 dB — BELOW the old
    -20.0 dB threshold this PR replaced, i.e. exactly the false-negative this
    PR exists to fix. Restoring the module's actual (fixed) value makes it
    read -7.92 dB, clear of both the old and the new threshold. Both phases
    are asserted in this one test (rather than only performed by hand) so a
    future regression back to a padded window is caught automatically."""
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, duration=4.96)
    master_pcm = tmp_path / "master.wav"
    _write_silent_wav(master_pcm, duration=20.0)

    start, end = 5.0, 5.0 + 4.96

    async def fake_axcorrelate(_clip: Path, window: Path) -> float:
        import wave as wave_mod

        with wave_mod.open(str(window), "rb") as w:
            window_duration = w.getnframes() / w.getframerate()
        if window_duration <= 4.96 + 0.01:
            return -7.92  # REAL: exact [start, end) window, the FIX
        return -21.03  # REAL: OLD padded (±0.5s) window, the DEFECT

    monkeypatch.setattr(mastering, "_axcorrelate_rms_db", fake_axcorrelate)

    # Phase 1 — the FIX, as shipped: margin is 0.0.
    assert mastering.NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS == 0.0
    fixed_db = await mastering._narration_clip_correlation_db(
        clip, master_pcm, 20.0, start, end, tmp_path, index=0
    )
    assert fixed_db is not None
    assert fixed_db == -7.92
    assert fixed_db >= mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB  # present, correctly

    # Phase 2 — MUTATE: revert to the old padded-window statistic.
    monkeypatch.setattr(mastering, "NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS", 0.5)
    defect_db = await mastering._narration_clip_correlation_db(
        clip, master_pcm, 20.0, start, end, tmp_path, index=1
    )
    assert defect_db is not None
    assert defect_db == -21.03
    old_threshold = -20.0  # the threshold this PR replaced
    assert defect_db < old_threshold, (
        "expected the old padded-window statistic to false-negative this genuinely "
        "present, long (4.96s) clip against the old -20.0 dB threshold"
    )


@pytest.mark.asyncio
async def test_margin_mutation_proof_against_the_current_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GENUINE mutation proof against the CURRENT gate, not the retired -20.0 dB
    one: ``test_long_clip_length_defect_mutation_proof`` above only shows the old
    padded reading (-21.03 dB) beats the OLD threshold — it is still 3.97 dB
    clear of the CURRENT -25.0 dB threshold, so it does not actually turn this
    gate's classification RED. This test uses a different pair from the same
    real-audio measurement (``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``'s
    provenance table, "broken line3"): -32.94 dB padded vs -12.48 dB unpadded —
    a genuinely present clip whose OLD padded reading crosses below -25.0 dB.

    Confirmed by hand before this test existed: setting
    ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS`` to 0.5 (this test's Phase 2)
    does turn the classification red (``present`` computed the same way
    ``_narration_presence_entry`` does, via ``correlation_db >=
    NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``); restoring 0.0 (Phase 1) turns
    it green again. Unlike the render-harness test
    ``test_a_long_present_clip_is_not_flagged``, whose synthetic fixture does
    NOT reproduce enough degradation to cross -25.0 dB at any of 0.5/1.0/2.0 s
    margin (measured 2026-09-17) — this test is what actually pins the margin=0.0
    decision against a real regression back to padding."""
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, duration=4.73)
    master_pcm = tmp_path / "master.wav"
    _write_silent_wav(master_pcm, duration=20.0)

    start, end = 5.0, 5.0 + 4.73

    async def fake_axcorrelate(_clip: Path, window: Path) -> float:
        import wave as wave_mod

        with wave_mod.open(str(window), "rb") as w:
            window_duration = w.getnframes() / w.getframerate()
        if window_duration <= 4.73 + 0.01:
            return -12.48  # REAL: exact [start, end) window, the FIX
        return -32.94  # REAL: OLD padded (+-0.5s) window, the DEFECT

    monkeypatch.setattr(mastering, "_axcorrelate_rms_db", fake_axcorrelate)

    # Phase 1 — the FIX, as shipped: margin is 0.0. Classifies PRESENT.
    assert mastering.NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS == 0.0
    fixed_db = await mastering._narration_clip_correlation_db(
        clip, master_pcm, 20.0, start, end, tmp_path, index=0
    )
    assert fixed_db is not None
    assert fixed_db == -12.48
    assert fixed_db >= mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB

    # Phase 2 — MUTATE: revert to the old padded-window statistic. Classifies
    # ABSENT against the CURRENT threshold — this genuinely turns RED.
    monkeypatch.setattr(mastering, "NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS", 0.5)
    defect_db = await mastering._narration_clip_correlation_db(
        clip, master_pcm, 20.0, start, end, tmp_path, index=1
    )
    assert defect_db is not None
    assert defect_db == -32.94
    assert defect_db < mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB, (
        "expected the old padded-window statistic to false-negative this genuinely "
        "present clip against the CURRENT -25.0 dB threshold, not just the retired one"
    )
