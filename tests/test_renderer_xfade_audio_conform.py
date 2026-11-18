"""4c: ``_render_xfade_group`` must conform each audio input to its declared
VIDEO length before ``acrossfade``, not feed the raw stream straight in.

Why this matters: ``acrossfade`` takes no offset — its overlap always lands at
the literal end of its left input and the literal start of its right input, on
whatever length that stream actually is. ``xfade``'s ``offset=`` is computed
from ``lengths[i]`` (the probed VIDEO stream duration). If a segment's audio
stream length differs from ``lengths[i]`` even slightly (an ALAC frame
boundary, an encoder priming/remainder sample), the two joins land at
different points in the timeline and the gap compounds across every
transition in the group.

Bench (wave 8, real ffmpeg, synthetic audio): declared video lengths 4.0/6.0s,
real audio stream lengths 4.08/5.95s, d=1.0. Unconformed ``acrossfade`` output
duration: 9.030s (0.03s off; the audio join lands 3.08s in where the video
join lands at 3.0s). Conforming each input with
``atrim=end=<L>,apad=whole_dur=<L>`` first lands the output at exactly 9.000s.

These are pure filtergraph-STRING assertions (``_run`` is mocked) — the real
mux is proven by ``tests/render_harness/test_speed_xfade_render.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import Transition
from src.config import get_settings
from src.services import renderer


def _install(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured["cmds"].append(cmd)
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    return captured


async def test_render_xfade_group_conforms_each_audio_input_before_the_crossfade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install(monkeypatch)
    paths = [tmp_path / "s0.mp4", tmp_path / "s1.mp4", tmp_path / "s2.mp4"]
    lengths = [4.0, 6.0, 3.0]
    transitions = [
        None,
        Transition(type="fade", duration=1.0),
        Transition(type="fade", duration=0.5),
    ]
    running = await renderer._render_xfade_group(
        paths, lengths, transitions, tmp_path / "out.mp4", get_settings()
    )
    fc = " ".join(captured["cmds"][-1])
    # Every raw input is trimmed AND padded to its own declared video length —
    # both directions, so a stream on EITHER side of `length` still lands on it —
    # and only THEN ramped and placed on the timeline.
    assert "[0:a]atrim=end=4.000000,apad=whole_dur=4.000000,afade=t=out" in fc
    assert "[1:a]atrim=end=6.000000,apad=whole_dur=6.000000,afade=t=in" in fc
    assert "[2:a]atrim=end=3.000000,apad=whole_dur=3.000000,afade=t=in" in fc
    # The fades sit exactly on the xfade windows: shot 1 starts at 4-1=3 s, shot 2
    # at 3+6-0.5=8.5 s (the same offsets xfade is given), sample-exact.
    assert (
        "[0:a]atrim=end=4.000000,apad=whole_dur=4.000000,afade=t=out:curve=tri:st=3.000000:d=1.000000[ax_p0]"
        in fc
    )
    assert (
        "[1:a]atrim=end=6.000000,apad=whole_dur=6.000000,afade=t=in:curve=tri:st=0:d=1.000000,"
        "afade=t=out:curve=tri:st=5.500000:d=0.500000,adelay=delays=144000S:all=1[ax_p1]"
    ) in fc
    assert (
        "[2:a]atrim=end=3.000000,apad=whole_dur=3.000000,afade=t=in:curve=tri:st=0:d=0.500000,"
        "adelay=delays=408000S:all=1[ax_p2]"
    ) in fc
    # Six decimals, not three: the offset is a frame count over the fps, and
    # ``xfade`` rounds what it is given onto the stream timebase — 12.167 for
    # 365/30 lands one frame late (see test_audio_timestamp_holes.py).
    assert "xfade=transition=fade:duration=1.000000:offset=3.000000" in fc
    assert "xfade=transition=fade:duration=0.500000:offset=8.500000" in fc
    # The mix consumes the CONFORMED labels only, never a raw `[i:a]`, and there
    # is no acrossfade (it truncates/drops audio depending on thread timing).
    assert "[ax_p0][ax_p1][ax_p2]amix=inputs=3:duration=longest:normalize=0[ax]" in fc
    assert "acrossfade=" not in fc
    # Composite duration is still lengths-minus-overlaps, unaffected by conform.
    assert running == pytest.approx(4.0 + 6.0 - 1.0 + 3.0 - 0.5)


# --------------------------------------------------------------------------
# The audio-bus offsets ARE the rendered join offsets (off the frame grid too)
# --------------------------------------------------------------------------

_XFADE_ARG_RE = re.compile(r"xfade=transition=\w+:duration=([0-9.]+):offset=([0-9.]+)")

# 2.517 s = 75.51 frames and 3.041 s = 91.23 frames at 30 fps: neither is on the
# grid, so ``running - d`` is off it at every join and the snap actually moves it.
_OFF_GRID_LENGTHS = [2.517, 3.041, 2.517]
_OFF_GRID_TRANSITIONS: list[Transition | None] = [
    None,
    Transition(type="fade", duration=0.5),
    Transition(type="fade", duration=0.5),
]


async def test_bus_offsets_equal_the_rendered_xfade_offsets_off_the_frame_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_bus_offsets_xfade`` used to compute ``start = prev_start + prev_len - d``
    with an UNSNAPPED ``d``, while ``_render_xfade_group`` put both on the frame
    grid — so a per-clip gain/mute window sat up to half a frame off the join it
    was meant to hug (2.017 / 4.558 here, against a picture at 61/30 / 137/30)."""
    captured = _install(monkeypatch)
    settings = get_settings()
    fps = settings.render_fps
    assert fps == 30  # the frame counts below are 30 fps arithmetic
    paths = [tmp_path / f"s{i}.mp4" for i in range(3)]
    running = await renderer._render_xfade_group(
        paths, _OFF_GRID_LENGTHS, _OFF_GRID_TRANSITIONS, tmp_path / "out.mp4", settings
    )
    rendered = _XFADE_ARG_RE.findall(" ".join(captured["cmds"][-1]))
    assert len(rendered) == 2

    offsets = renderer._bus_offsets_xfade(_OFF_GRID_LENGTHS, _OFF_GRID_TRANSITIONS, fps)

    # The renderer's own numbers, pinned so this cannot pass by both sides agreeing
    # on the OLD unsnapped values (which would print 2.017000 / 4.558000).
    assert rendered == [("0.500000", "2.033333"), ("0.500000", "4.566667")]
    # ...and the bus starts are those exact offsets, as the renderer printed them.
    assert [f"{start:.6f}" for start, _end in offsets[1:]] == [off for _d, off in rendered]
    # Exactly, not approximately: on the grid, and each shot its own length long.
    assert offsets == [
        (0.0, 2.517),
        (61 / 30, 61 / 30 + 3.041),
        (137 / 30, 137 / 30 + 2.517),
    ]
    # The last shot ends where the group does.
    assert offsets[-1][1] == running


def test_bus_offsets_restart_each_group_at_the_previous_groups_rendered_end() -> None:
    """Mixed hard cut + crossfade, off the grid: the second group starts at the
    first group's composite length — the value ``_render_xfade_group`` returns —
    and its in-group join is snapped relative to THAT group, as rendered."""
    fps = 30
    lengths = [2.517, 3.041, 2.517, 3.041]
    transitions = [None, Transition(duration=0.5), None, Transition(duration=0.5)]
    joins = renderer._xfade_group_joins(lengths[:2], transitions[:2], fps)
    group_end = joins[-1][1] + lengths[1]
    assert joins == [(0.0, 0.0), (0.5, 61 / 30)]

    offsets = renderer._bus_offsets_xfade(lengths, transitions, fps)

    assert offsets[:2] == [(0.0, 2.517), (61 / 30, group_end)]
    assert offsets[2] == (group_end, group_end + 2.517)
    assert offsets[3] == (group_end + 61 / 30, group_end + 61 / 30 + 3.041)
