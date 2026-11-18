"""``_clip_seek_args`` must floor a window's in-point onto the SOURCE frame grid.

The argv-level half of the input-side twin of the ``setpts=PTS-STARTPTS`` fix
(#174). ``-ss`` is an accurate seek: ffmpeg starts decoding at the first frame
whose timestamp is >= the requested time, so a mid-frame request silently drops
up to one source frame at the head while the ``-t`` derived from that same
request still stops at the requested out-point. On shot 12 of the live 19-shot
reel — ``out_seconds 10.010 == source duration 10.010``, i.e. ZERO tail slack —
``-ss 8.010`` on a 23.976 fps source landed 0.0397s before the next real frame
(8.04972, source frame 193) and the shot shipped 42 frames against a plan of 43.
That was the reel's whole remaining deficit after #174 took it from -5 to -1.

The real-encode proof lives in ``tests/render_harness/test_seek_frame_grid.py``;
this module pins the emitted args, the millisecond ORDERING, and — most
importantly — that an unknown frame rate emits byte-identical args.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services import renderer

pytestmark = pytest.mark.asyncio

# 24000/1001, the rate the live reel's source footage carried.
_NTSC = 24000.0 / 1001.0


@pytest.fixture
def _probe_10s(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10.010s file: exactly 240 frames at 23.976 fps, so an out-point of
    10.010 sits at EOF with no tail slack — the production shape."""

    async def _fake_dur(_: Path) -> float:
        return 10.010

    monkeypatch.setattr(renderer, "_ffprobe_duration", _fake_dur)


@pytest.fixture
def _no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rate probe must not run when the caller declared a rate."""

    async def _boom(_: Path) -> float | None:
        raise AssertionError("_ffprobe_frame_rate called despite a declared source_fps")

    monkeypatch.setattr(renderer, "_ffprobe_frame_rate", _boom)


async def test_declared_fps_floors_the_in_point_onto_the_source_grid(
    _probe_10s: None, _no_probe: None
) -> None:
    """THE regression, at argv level. 8.010 is 192.048 source frames in, so frame
    192 sits at 8.008 and the requested time skips it. Floor the SEEK to 8.008 and
    leave the span at the planned 2.000: the window slides back by the 2 ms it
    recovered instead of growing. Standalone repro: 42 frames -> 43."""
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        0.0,
        2.0,
        clip_lead_seconds=8.010,
        clip_out_seconds=10.010,
        source_fps=_NTSC,
    )
    assert args == ["-ss", "8.008", "-t", "2.000"]
    assert offset == pytest.approx(8.008)
    assert span == pytest.approx(2.000)


async def test_unknown_fps_emits_byte_identical_args(_probe_10s: None) -> None:
    """No declared rate AND an unprobeable file ⇒ exactly the args this function
    emitted before the snap existed. A snap onto a GUESSED rate would move the
    in-point by an arbitrary amount — strictly worse than the bug."""

    async def _unknown(_: Path) -> float | None:
        return None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(renderer, "_ffprobe_frame_rate", _unknown)
        args, offset, span = await renderer._clip_seek_args(
            Path("/x.mp4"), 0.0, 2.0, clip_lead_seconds=8.010, clip_out_seconds=10.010
        )
    assert args == ["-ss", "8.010", "-t", "2.000"]
    assert offset == pytest.approx(8.010)
    assert span == pytest.approx(2.000)


@pytest.mark.parametrize("bad_fps", [0.0, -24.0])
async def test_non_positive_fps_is_treated_as_unknown(_probe_10s: None, bad_fps: float) -> None:
    """A zero/negative rate is a broken probe, not a grid. It must fall through to
    the probe and then to the no-snap path, never to a division blow-up."""

    async def _unknown(_: Path) -> float | None:
        return None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(renderer, "_ffprobe_frame_rate", _unknown)
        args, _offset, _span = await renderer._clip_seek_args(
            Path("/x.mp4"),
            0.0,
            2.0,
            clip_lead_seconds=8.010,
            clip_out_seconds=10.010,
            source_fps=bad_fps,
        )
    assert args == ["-ss", "8.010", "-t", "2.000"]


async def test_probed_rate_is_used_when_the_shot_declares_none(_probe_10s: None) -> None:
    """The production path: the cut planner states musical in/out times and
    leaves ``target_fps`` unset, so without the probe fallback the snap would be
    structurally inert on exactly the renders it was written for."""

    async def _ntsc(_: Path) -> float | None:
        return _NTSC

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(renderer, "_ffprobe_frame_rate", _ntsc)
        args, _offset, _span = await renderer._clip_seek_args(
            Path("/x.mp4"), 0.0, 2.0, clip_lead_seconds=8.010, clip_out_seconds=10.010
        )
    assert args == ["-ss", "8.008", "-t", "2.000"]


async def test_ms_grid_is_floored_not_rounded(_probe_10s: None, _no_probe: None) -> None:
    """ORDERING. Frame 193 sits at 8.0497083s. The emitted arg is ``f"{x:.3f}"``,
    which ROUNDS — 8.050, i.e. AFTER the frame, which reinstates the whole bug on
    the half of frame boundaries that round up. Flooring to the millisecond below
    is always <= the true frame time, and cannot reach the previous frame while
    the frame period exceeds 1 ms (true for every rate under 1000 fps)."""
    args, offset, _span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        0.0,
        1.0,
        clip_lead_seconds=8.060,
        clip_out_seconds=10.010,
        source_fps=_NTSC,
    )
    assert args[:2] == ["-ss", "8.049"], "rounded past the frame it was meant to include"
    assert offset == pytest.approx(8.049)


async def test_frame_aligned_in_point_does_not_move(_probe_10s: None, _no_probe: None) -> None:
    """1.001s is exactly 24 frames at 24000/1001. Float error must not push it
    back a whole frame (hence the epsilon in ``_snap_seek_to_source_grid``)."""
    args, _offset, _span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 2.0, clip_lead_seconds=1.001, clip_out_seconds=3.003, source_fps=_NTSC
    )
    assert args == ["-ss", "1.001", "-t", "2.002"]  # already aligned: nothing moves


async def test_zero_in_point_emits_no_seek_at_all(_probe_10s: None, _no_probe: None) -> None:
    """``in == 0.0`` is below the 0.1s noise floor, so there is no ``-ss`` to
    snap and the span is untouched — byte-identical to before the fix even with a
    rate declared."""
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 2.0, clip_lead_seconds=0.0, clip_out_seconds=2.0, source_fps=_NTSC
    )
    assert args == ["-t", "2.000"]
    assert offset == 0.0
    assert span == pytest.approx(2.0)


async def test_no_out_point_path_is_untouched(_probe_10s: None, _no_probe: None) -> None:
    """The snap is window-mode only. The legacy in-point modes (anchor/lead/
    fraction, no ``-t``) must keep their millisecond ROUNDING: they have no stated
    out-point, so there is always tail slack and nothing to recover."""
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), 0.0, 2.0, clip_lead_seconds=4.0006, source_fps=_NTSC
    )
    assert args == ["-ss", "4.001"]
    assert offset == pytest.approx(4.001)
    assert span is None


async def test_the_span_is_never_widened_by_the_snap(_probe_10s: None, _no_probe: None) -> None:
    """The measured trap. Deriving the span from the SNAPPED in-point (leaving the
    out-point fixed) widens it by the recovered delta, and that widened span
    becomes the segment's own output ``-t`` — so every seeked shot runs up to one
    frame LONG. Rendering 19 pseudo-shots proved it: the zero-slack shot was fixed
    but EIGHT of the other eighteen went one frame over plan. Pinned here at argv
    level because it is cheap: ``-t`` must equal ``out - requested_in`` exactly,
    whatever the seek did.
    """
    requested_in, out = 0.637, 0.981
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"),
        0.0,
        out - requested_in,
        clip_lead_seconds=requested_in,
        clip_out_seconds=out,
        source_fps=_NTSC,
    )
    assert args[1] == "0.625", "the seek must floor onto source frame 15 (0.62563s)"
    assert offset == pytest.approx(0.625)
    assert args[3] == "0.344"
    assert span == pytest.approx(out - requested_in)


def test_a_non_finite_source_fps_falls_back_instead_of_raising() -> None:
    """``inf``/``nan`` must take the unknown-rate path, not blow up the render.

    ``Shot.target_fps`` is declared ``gt=0`` with NO upper bound, so a JSON
    ``1e400`` reaches the renderer as ``inf``. ``inf > 0.0`` is True, so a bare
    positivity guard admits it and ``math.floor(offset * inf)`` raises
    ``OverflowError`` — which is not a bad frame count, it is a DEAD JOB.

    This is not hypothetical and it is not new: ``tests/test_cadence_nonfinite.py``
    exists because the same value once killed a whole job "with an opaque
    OverflowError before a single shot rendered", and ``cadence._coerce_fps`` was
    hardened then. This call site is new and inherited none of that, and an
    adversarial pass on this very change caught it against real ``render_shot``:
    unfixed `main` rendered 47 frames where the unguarded branch raised.

    ``nan`` already fails ``> 0.0`` by accident. Pinned here so it stays covered on
    purpose rather than by luck.
    """
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert renderer._snap_seek_to_source_grid(8.010, bad) == 8.01

    # And the guard must not have broken the paths that DO snap.
    assert renderer._snap_seek_to_source_grid(8.010, 24000 / 1001) == 8.008
    assert renderer._snap_seek_to_source_grid(8.010, None) == 8.01
    assert renderer._snap_seek_to_source_grid(8.010, 0.0) == 8.01
