"""Unit tests for the per-segment signalstats measurement (VC-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services import measure, renderer

# A realistic ``signalstats,metadata=print`` stderr fragment (two frames).
_SIGNALSTATS_DUMP = """\
frame:0    pts:0       pts_time:0
lavfi.signalstats.YMIN=16
lavfi.signalstats.YMAX=235
lavfi.signalstats.YAVG=120.0
lavfi.signalstats.YDIF=1.0
lavfi.signalstats.SATAVG=8.0
frame:1    pts:512     pts_time:0.017
lavfi.signalstats.YMIN=20
lavfi.signalstats.YMAX=225
lavfi.signalstats.YAVG=122.0
lavfi.signalstats.YDIF=3.0
lavfi.signalstats.SATAVG=10.0
"""


def test_parse_signalstats_averages_across_frames() -> None:
    stats = measure.parse_signalstats(_SIGNALSTATS_DUMP)
    assert stats["YMIN"] == 18.0  # (16+20)/2
    assert stats["YMAX"] == 230.0  # (235+225)/2
    assert stats["YAVG"] == 121.0
    assert stats["YDIF"] == 2.0
    assert stats["SATAVG"] == 9.0


def test_parse_signalstats_empty_is_empty_dict() -> None:
    assert measure.parse_signalstats("") == {}
    assert measure.parse_signalstats("garbage with no metrics") == {}


def test_parse_signalstats_ignores_non_numeric() -> None:
    stats = measure.parse_signalstats("lavfi.signalstats.YAVG=N/A\nlavfi.signalstats.YMIN=16")
    assert "YAVG" not in stats
    assert stats["YMIN"] == 16.0


def test_parse_signalstats_per_frame_handles_real_ffmpeg_prefix() -> None:
    # Regression: real ffmpeg emits the frame marker MID-LINE, prefixed with the
    # filter tag ``[Parsed_metadata_N @ 0x…] frame:N …`` — not at the line start.
    # A ``^``-anchored marker regex matched zero frames and silently collapsed
    # every per-frame pass to ONE whole-file aggregate (the per-frame banding +
    # dead-frame gates were quietly averaging). This locks the fix: each frame is
    # parsed separately, so a single blank frame's spread survives to the gate.
    dump = (
        "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMIN=16\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMAX=235\n"
        "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMIN=16\n"
        "[Parsed_metadata_1 @ 0x1] lavfi.signalstats.YMAX=16\n"
    )
    frames = measure.parse_signalstats_per_frame(dump)
    assert len(frames) == 2  # two distinct frames, NOT one collapsed aggregate
    assert frames[0]["YMAX"] - frames[0]["YMIN"] == 219.0  # rich frame
    assert frames[1]["YMAX"] - frames[1]["YMIN"] == 0.0  # blank frame survives


@pytest.mark.asyncio
async def test_measure_shot_stats_runs_through_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """It must exec via ``renderer._run`` (the concurrency gate) with
    ``capture_stderr=True`` and parse the returned stderr."""
    seen: dict[str, object] = {}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen["cmd"] = cmd
        seen["capture_stderr"] = capture_stderr
        return _SIGNALSTATS_DUMP

    monkeypatch.setattr(renderer, "_run", fake_run)
    stats = await measure.measure_shot_stats(Path("/tmp/seg.mp4"))
    assert stats["YDIF"] == 2.0
    assert seen["capture_stderr"] is True
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "ffmpeg"
    assert "signalstats,metadata=print" in cmd


@pytest.mark.asyncio
async def test_measure_shot_stats_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ffmpeg failure degrades to ``{}`` — a measurement hiccup never fails a job."""

    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("ffmpeg exit=1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await measure.measure_shot_stats(Path("/tmp/seg.mp4")) == {}


# ---------------------------------------------------------------------------
# S20 — timed per-frame parsing (pts_time survives the frame:N split so a
# per-shot bucketer can place each frame's banding on the film timeline).
# ---------------------------------------------------------------------------

# frame:0 carries a pts_time header; frame:1 deliberately omits pts_time.
_TIMED_DUMP = """\
frame:0    pts:0       pts_time:0.0
lavfi.signalstats.YMIN=16
lavfi.signalstats.YMAX=235
lavfi.signalstats.YDIF=1.0
frame:1    pts:512
lavfi.signalstats.YMIN=20
lavfi.signalstats.YMAX=225
lavfi.signalstats.YDIF=3.0
"""


def test_block_pts_time_parses_and_degrades() -> None:
    assert measure._block_pts_time("pts_time:1.5\nlavfi.signalstats.YMIN=16") == 1.5
    assert measure._block_pts_time("lavfi.signalstats.YMIN=16") is None  # no marker


def test_parse_signalstats_per_frame_timed_captures_pts_time() -> None:
    timed = measure.parse_signalstats_per_frame_timed(_TIMED_DUMP)
    assert len(timed) == 2
    assert timed[0][0] == 0.0  # pts_time survived the frame:N split
    assert timed[0][1]["YMIN"] == 16.0
    assert timed[1][0] is None  # block carried no pts_time marker → None
    assert timed[1][1]["YDIF"] == 3.0


def test_parse_signalstats_per_frame_timed_single_frame_degrade() -> None:
    """No frame markers but metrics present → one aggregate frame, pts_time None."""
    timed = measure.parse_signalstats_per_frame_timed(
        "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\n"
    )
    assert len(timed) == 1
    assert timed[0][0] is None
    assert timed[0][1]["YMAX"] == 235.0


def test_parse_signalstats_per_frame_timed_empty_is_empty_list() -> None:
    assert measure.parse_signalstats_per_frame_timed("") == []
    assert measure.parse_signalstats_per_frame_timed("garbage no metrics") == []


@pytest.mark.asyncio
async def test_measure_shot_frames_timed_runs_through_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors ``measure_shot_frames``: exec via the gate with the SAME discarded
    signalstats cmd, but returns ``(pts_time, metrics)`` tuples."""
    seen: dict[str, object] = {}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen["cmd"] = cmd
        seen["capture_stderr"] = capture_stderr
        return _TIMED_DUMP

    monkeypatch.setattr(renderer, "_run", fake_run)
    timed = await measure.measure_shot_frames_timed(Path("/tmp/seg.mp4"))
    assert [pts for pts, _ in timed] == [0.0, None]
    assert seen["capture_stderr"] is True
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert "signalstats,metadata=print" in cmd


@pytest.mark.asyncio
async def test_measure_shot_frames_timed_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("ffmpeg exit=1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await measure.measure_shot_frames_timed(Path("/tmp/seg.mp4")) == []
