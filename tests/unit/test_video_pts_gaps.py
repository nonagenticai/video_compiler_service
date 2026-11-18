"""``mastering.video_pts_gaps`` — the pts-interval half of ``cfr_ok``.

Delivered master 01a11032 (2026-10-06) reported ``cfr_ok: true`` with 952 frames
in 953 slots: frame 607 (pts 20.233333) -> frame 608 (pts 20.300000) is 66.667 ms,
one empty frame slot at the keyframe opening the last ``concat -c copy`` segment.
Every packet declared ``duration=512``, so the declared-duration scan
``framerate_coherence`` falls back on could not see it. These pin the reading that
can. The real-ffmpeg proof is ``tests/render_harness/test_video_pts_gap_render.py``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from src.services import mastering, renderer

#: The mp4 video timebase of that master: one 30 fps frame is 512 ticks.
_TB = Fraction(1, 15360)
_FRAME = 512


def _grid(frames: int, *, holes_after: tuple[int, ...] = ()) -> list[int]:
    """``frames`` pts on the 30 fps grid, with one EMPTY slot after each listed frame."""
    pts, tick = [], 0
    for index in range(frames):
        pts.append(tick)
        tick += _FRAME * (2 if index in holes_after else 1)
    return pts


def _decode_order(pts: list[int]) -> list[int]:
    """``pts`` as an I/P-then-B encoder emits them: each anchor before the B-frame
    it follows in presentation (0, 2, 1, 4, 3, ...)."""
    out = pts[:1]
    for i in range(1, len(pts), 2):
        out.extend(reversed(pts[i : i + 2]))
    return out


def test_the_delivered_masters_hole_is_one_gap_at_its_position() -> None:
    pts = _grid(952, holes_after=(607,))
    # The film's own numbers, to the microsecond ffprobe prints.
    assert [round(float(p * _TB), 6) for p in pts[606:610]] == [
        20.2,
        20.233333,
        20.3,
        20.333333,
    ]

    count, positions = mastering.video_pts_gaps(pts, _TB)

    assert count == 1
    assert positions == [{"at_seconds": 20.233333, "interval_ms": 66.667}]


def test_a_regular_timeline_has_no_gaps() -> None:
    assert mastering.video_pts_gaps(_grid(1369), _TB) == (0, [])


def test_b_frame_decode_order_is_not_a_gap() -> None:
    """Packets arrive in DECODE order (the film's own head: pts 0, 1024, 512, ...).
    Unsorted, every reordered pair is a negative or double interval."""
    decode_order = _decode_order(_grid(952))
    assert decode_order[:3] == [0, 1024, 512]
    assert mastering.video_pts_gaps(decode_order, _TB) == (0, [])

    assert mastering.video_pts_gaps(_decode_order(_grid(952, holes_after=(607,))), _TB)[0] == 1


def test_timebase_rounding_is_under_the_half_frame_bar_and_a_half_frame_is_at_it() -> None:
    # Matroska's 1 ms timebase renders 30 fps as 33/33/34 ms: jitter, not a gap.
    mkv = [round(n * 1000 / 30) for n in range(300)]
    assert mastering.video_pts_gaps(mkv, Fraction(1, 1000)) == (0, [])
    # Strictly MORE than half a frame off nominal: 768 is exactly +half (clean),
    # 769 is over it.
    assert mastering.video_pts_gaps([0, 512, 1024, 1792, 2304], _TB)[0] == 0
    assert mastering.video_pts_gaps([0, 512, 1024, 1793, 2305], _TB)[0] == 1


def test_a_crowded_slot_is_irregular_too() -> None:
    """A frame arriving a slot EARLY (two frames in one slot) breaks cadence the
    same as one arriving late."""
    count, positions = mastering.video_pts_gaps([0, 512, 1024, 1124, 1636, 2148], _TB)
    assert count == 1
    assert positions[0]["at_seconds"] == round(1024 / 15360, 6)


def test_positions_are_capped_and_the_count_is_not() -> None:
    pts = _grid(400, holes_after=tuple(range(10, 400, 20)))
    count, positions = mastering.video_pts_gaps(pts, _TB)
    assert count == 20
    assert len(positions) == mastering._VIDEO_PTS_GAP_POSITIONS_MAX


def test_fewer_than_two_frames_has_no_interval() -> None:
    assert mastering.video_pts_gaps([], _TB) == (0, [])
    assert mastering.video_pts_gaps([0], _TB) == (0, [])


def _probe_json(pts: list[int]) -> str:
    return json.dumps({"streams": [{"time_base": "1/15360"}], "packets": [{"pts": p} for p in pts]})


@pytest.mark.asyncio
async def test_the_report_scans_the_whole_file_in_one_packet_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen.append(cmd)
        return _probe_json(_grid(952, holes_after=(607,)))

    monkeypatch.setattr(renderer, "_run", fake_run)

    assert await mastering.video_pts_gap_report(Path("/tmp/x.mp4")) == (
        1,
        [{"at_seconds": 20.233333, "interval_ms": 66.667}],
    )
    (cmd,) = seen
    assert cmd[0] == "ffprobe"
    # The hole sat at 20.2 s; the old fallback read only the first 10 s.
    assert "-read_intervals" not in cmd
    assert cmd[cmd.index("-select_streams") + 1] == "v:0"
    assert cmd[cmd.index("-show_entries") + 1] == "stream=time_base:packet=pts"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_output",
    [
        "",  # ffprobe printed nothing
        "{}",  # no stream at all
        '{"streams": [], "packets": []}',  # audio-only file
        '{"streams": [{"time_base": "1/15360"}], "packets": []}',
        '{"streams": [{"time_base": "1/15360"}], "packets": [{"pts": 0}]}',
        '{"streams": [{"time_base": "1/15360"}], "packets": [{"dts": 0}, {"dts": 512}]}',
    ],
)
async def test_an_unreadable_timeline_is_unmeasured_never_zero(
    monkeypatch: pytest.MonkeyPatch, probe_output: str
) -> None:
    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return probe_output

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await mastering.video_pts_gap_report(Path("/tmp/x.mp4")) == (None, None)


@pytest.mark.asyncio
async def test_a_raising_probe_is_unmeasured_and_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("ffprobe exit=1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await mastering.video_pts_gap_report(Path("/tmp/x.mp4")) == (None, None)
