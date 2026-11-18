"""``Shot.clip_out_seconds`` — the source-timeline OUT-point.

Before this field the only in-point control was ``clip_lead_seconds`` (+
``clip_anchor``) and the out-point was merely IMPLIED by ``seconds``. That is
enough while one shot uses one source, and pathological for the music-cut
product, where many shots are cut from ONE uploaded recording and differ only in
which span of it they show.

Two properties are asserted here without touching ffmpeg (the real-encode proof
that the delivered segment really is bounded lives in
``tests/render_harness/test_clip_out_point.py``):

- ``None`` (the default) emits EXACTLY the args it did before the field existed —
  no ``-t``, no declared span. That is the byte-identical guarantee the field
  description makes, and it is the only thing standing between this feature and
  every existing compile.
- a set out-point produces an INPUT-side ``-t`` (before ``-i``), because an
  output-side bound would let ffmpeg decode real frames past the out-point and
  show content the caller excluded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.schemas import Shot
from src.services import renderer

# NOT a module-level ``pytestmark``: the two Shot-validator tests at the bottom are
# synchronous, and a blanket asyncio mark makes pytest-asyncio warn on every one of
# them ("marked with '@pytest.mark.asyncio' but it is not an async function").
_asyncio = pytest.mark.asyncio

_CLIP_SECONDS = 100.0


@pytest.fixture
def _probe_100s(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every clip probed in this module is a 100s file."""

    async def _fake_dur(_: Path) -> float:
        return _CLIP_SECONDS

    monkeypatch.setattr(renderer, "_ffprobe_duration", _fake_dur)


@_asyncio
async def test_out_point_bounds_the_source_window(_probe_100s: None) -> None:
    """lead=4.0, out=7.5 ⇒ ``-ss 4.000 -t 3.500`` — exactly 4.0s-7.5s."""
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        0.0,
        10.0,
        clip_anchor="start",
        clip_lead_seconds=4.0,
        clip_out_seconds=7.5,
    )
    assert args == ["-ss", "4.000", "-t", "3.500"]
    assert offset == pytest.approx(4.0)
    assert span == pytest.approx(3.5)
    # These args are appended BEFORE the ``-i`` by render_shot, so the ``-t`` is
    # an INPUT option (caps what is read) rather than an output trim.
    assert args[args.index("-t") + 1] == "3.500"


@_asyncio
async def test_out_point_is_not_clamped_by_the_shot_duration(_probe_100s: None) -> None:
    """The in-point is taken verbatim in window mode.

    Without an out-point a lead is clamped to ``clip_len - shot_duration`` so the
    remaining clip still covers the shot. A caller that states BOTH ends has
    already decided how long the window is; re-applying that clamp would slide
    the window silently and show the wrong content. Here the 3.5s window is far
    shorter than the 10s shot and must survive intact (the freeze pad fills the
    rest downstream).
    """
    _args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        0.0,
        10.0,
        clip_anchor="start",
        clip_lead_seconds=95.0,
        clip_out_seconds=98.5,
    )
    assert offset == pytest.approx(95.0)
    assert span == pytest.approx(3.5)


@_asyncio
async def test_out_point_past_eof_clamps_to_the_file(_probe_100s: None) -> None:
    """An out-point past the end of the file means "to the end", not an error."""
    _args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 10.0, clip_lead_seconds=90.0, clip_out_seconds=500.0
    )
    assert offset == pytest.approx(90.0)
    assert span == pytest.approx(10.0)


@_asyncio
async def test_end_anchor_takes_the_tail_of_the_window(_probe_100s: None) -> None:
    """``clip_anchor='end'`` under a window is the tail OF THE WINDOW.

    out=40, shot=6 ⇒ the in-point moves to 34.0, never before the stated lead.
    """
    _args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 6.0, clip_anchor="end", clip_lead_seconds=10.0, clip_out_seconds=40.0
    )
    assert offset == pytest.approx(34.0)
    assert span == pytest.approx(6.0)


@_asyncio
async def test_end_anchor_never_seeks_before_the_stated_lead(_probe_100s: None) -> None:
    """A shot longer than the window keeps the window's own in-point."""
    _args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 30.0, clip_anchor="end", clip_lead_seconds=10.0, clip_out_seconds=14.0
    )
    assert offset == pytest.approx(10.0)
    assert span == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("anchor", "lead", "fraction"),
    [("start", 0.0, 0.0), ("start", 4.0, 0.0), ("end", 0.0, 0.0), ("start", 0.0, 0.5)],
)
@_asyncio
async def test_none_out_point_is_a_no_op(
    _probe_100s: None, anchor: str, lead: float, fraction: float
) -> None:
    """``clip_out_seconds=None`` never emits ``-t`` and never declares a span.

    Parametrised over all three legacy in-point modes so the guarantee covers
    every path a pre-existing compile can take, not just the default one.
    """
    args, _offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        fraction,
        10.0,
        clip_anchor=anchor,
        clip_lead_seconds=lead,
        clip_out_seconds=None,
    )
    assert "-t" not in args
    assert span is None


def test_out_point_must_follow_the_in_point() -> None:
    """An inverted/empty window is a 422 at the boundary, not a silent full clip."""
    for out in (3.5, 4.0):
        with pytest.raises(ValidationError, match="clip_out_seconds must be greater"):
            Shot(
                id="a",
                seconds=5.0,
                video_url="http://h/x.mp4",
                clip_lead_seconds=4.0,
                clip_out_seconds=out,
            )


def test_valid_window_is_accepted_and_defaults_to_none() -> None:
    assert Shot(id="a", seconds=5.0, video_url="http://h/x.mp4").clip_out_seconds is None
    shot = Shot(
        id="a", seconds=5.0, video_url="http://h/x.mp4", clip_lead_seconds=4.0, clip_out_seconds=7.5
    )
    assert shot.clip_out_seconds == pytest.approx(7.5)
