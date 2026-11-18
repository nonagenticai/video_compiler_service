"""Arithmetic behind the audio-timestamp-hole fix (no ffmpeg).

Three pure pieces, each of which decided a real number on production master
01a10e19: where a segment ends (``_snap_to_frame_grid``), how long loudnorm's
input is (``_loudnorm_frame_align``), and how a hole is read off a packet list
(``mastering.audio_pts_gaps``). The real-ffmpeg proof is
``tests/render_harness/test_audio_timestamp_holes_render.py``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from structlog.testing import capture_logs

from src.api.schemas import Shot, Transition
from src.config import Settings, get_settings
from src.services import mastering, renderer

_FPS_VALUES = (24, 25, 30, 50, 60)


@pytest.mark.parametrize("fps", _FPS_VALUES)
@pytest.mark.parametrize("seconds", [0.004, 1.44, 2.50957, 8.44, 10.50957, 12.661305, 59.999])
@pytest.mark.parametrize("up", [True, False])
def test_snap_lands_on_a_whole_frame_within_one_frame(fps: int, seconds: float, up: bool) -> None:
    snapped = renderer._snap_to_frame_grid(seconds, fps, up=up)
    frames = snapped * fps
    assert abs(frames - round(frames)) < 1e-9, f"{snapped}s is {frames} frames at {fps} fps"
    assert round(frames) >= 1
    drift = snapped - seconds
    if round(seconds * fps) == 0 or math.floor(seconds * fps) == 0:
        return  # clamped up to the one-frame minimum
    assert abs(drift) < 1 / fps, f"moved {drift}s, more than one {fps} fps frame"
    assert drift >= -renderer._FRAME_GRID_TOLERANCE_S if up else drift <= 1e-12


def test_the_production_shots_snap_to_the_frame_counts_ffmpeg_rendered() -> None:
    """Run 01a10e19 at 30 fps: 12.661305 s of narration sat under 380 video frames
    (audio 5.4 ms SHORT -> the hole) and 10.50957 s over 315 (audio 9.6 ms LONG ->
    the stretched frame). Snapped up, both streams of each segment end together."""
    assert renderer._snap_to_frame_grid(12.661305, 30) == 380 / 30
    assert renderer._snap_to_frame_grid(10.50957, 30) == 316 / 30
    assert renderer._snap_to_frame_grid(8.44, 30) == 254 / 30


@pytest.mark.parametrize("fps", _FPS_VALUES)
@pytest.mark.parametrize("frames", [1, 3, 180, 315, 380, 1128])
def test_an_on_grid_duration_is_returned_unchanged(fps: int, frames: int) -> None:
    """Byte-identity for every shot that was already clean: the same float comes
    back in both directions, so its ``-t`` / ``apad`` tokens do not move."""
    on_grid = frames / fps
    assert renderer._snap_to_frame_grid(on_grid, fps) == on_grid
    assert renderer._snap_to_frame_grid(on_grid, fps, up=False) == on_grid


def test_a_probe_rounded_grid_value_does_not_gain_a_frame() -> None:
    """ffprobe prints 380/30 as 12.666667 — 0.33 us PAST the frame boundary. Read
    literally that is "off grid, round up": a whole extra frame on a clean shot."""
    assert renderer._snap_to_frame_grid(12.666667, 30) == 380 / 30
    assert renderer._snap_to_frame_grid(12.666666, 30, up=False) == 380 / 30


@pytest.mark.parametrize("speed", [0.5, 0.8, 1.25, 2.0])
def test_snap_is_taken_in_the_rendered_timeline(speed: float) -> None:
    """A retimed shot renders ``duration / speed`` long; THAT is what must be whole
    frames, and ``_apply_shot_speed`` must get back to it from the returned value."""
    fps = 30
    snapped = renderer._snap_to_frame_grid(7.2033, fps, speed=speed)
    rendered_frames = snapped / speed * fps
    assert abs(rendered_frames - round(rendered_frames)) < 1e-9
    assert 0 <= snapped / speed - 7.2033 / speed < 1 / fps


def test_snap_direction_follows_what_bounds_the_segment() -> None:
    assert renderer._snap_decided_lengths(
        2.50957, 2.50957, narrated=True, has_clip=False, fps=25, speed=1.0
    ) == (2.52, 2.50957)  # narrated: up, and the narration keeps ITS length
    assert renderer._snap_decided_lengths(
        2.50957, 2.50957, narrated=True, has_clip=True, fps=25, speed=1.0
    ) == (2.52, 2.50957)
    # An un-narrated card has nothing to outlast, so it is never "padded".
    assert renderer._snap_decided_lengths(
        2.03, 2.03, narrated=False, has_clip=False, fps=25, speed=1.0
    ) == (2.04, 2.04)
    # An un-narrated CLIP is decided later, inside the clip branch: untouched here.
    assert renderer._snap_decided_lengths(
        2.03, 2.03, narrated=False, has_clip=True, fps=25, speed=1.0
    ) == (2.03, 2.03)


@pytest.mark.parametrize(
    ("total", "target"),
    [
        (37.610875, "37.7"),  # the production bus: 376 frames + 10.875 ms
        (37.611, "37.7"),
        (29.247749, "29.3"),
        (9.68, "9.7"),
        (6.0, "6.0"),  # already whole frames: not pushed a frame further
        (37.7, "37.7"),
        (0.03, "0.1"),
    ],
)
def test_loudnorm_is_handed_whole_100ms_frames(total: float, target: str) -> None:
    frag = renderer._loudnorm_frame_align(total)
    assert frag == f"asetpts=N/SR/TB,atrim=end={target},apad=whole_dur={target},"
    assert float(target) >= total - 1e-9, "the pad may never fall short of the film"
    assert float(target) - total < renderer._LOUDNORM_FRAME_S + 1e-9


def test_alignment_sits_directly_before_loudnorm_in_both_passes() -> None:
    settings = get_settings().model_copy(update={"master_enabled": True})
    align = renderer._loudnorm_frame_align(37.611)
    for chain in (
        renderer._master_measure_filter(settings, 37.611),
        renderer._master_filter(settings, None, 37.611),
        renderer._master_filter(
            settings, {k: "-20" for k in renderer._MASTER_MEASURE_KEYS}, 37.611
        ),
    ):
        assert chain.count(align + "loudnorm=") == 1, chain
        assert chain.count("loudnorm=") == 1
    # No length known -> nothing added: the chain is the one that shipped before.
    assert renderer._loudnorm_frame_align(None) == ""
    assert "asetpts" not in renderer._master_filter(settings)
    assert "asetpts" not in renderer._master_measure_filter(settings)


# --- reading a hole off the packet list -------------------------------------------


def _aac_pts(n: int, holes: dict[int, int] | None = None) -> list[int]:
    """``n`` AAC packet timestamps; ``holes[i]`` extra samples before packet ``i``."""
    pts, t = [], -1024
    for i in range(n):
        t += (holes or {}).get(i, 0)
        pts.append(t)
        t += 1024
    return pts


def test_a_clean_stream_has_no_gaps() -> None:
    assert mastering.audio_pts_gaps(_aac_pts(1764), 48000) == (0, 0.0)


def test_the_production_master_reads_as_two_holes() -> None:
    """01a10e19: the packets before the holes were stamped 1282 and 5302 samples
    long instead of 1024 — 5.375 ms and 89.125 ms with no samples."""
    pts = _aac_pts(1760, {1089: 258, 1629: 4278})
    assert mastering.audio_pts_gaps(pts, 48000) == (2, 94.5)


def test_a_short_frame_and_one_sample_of_rounding_are_not_holes() -> None:
    # ALAC intermediates: a short last frame before each stream-copy join.
    alac = [0, 4096, 8192, 8192 + 672, 8192 + 672 + 4096, 8192 + 672 + 8192]
    assert mastering.audio_pts_gaps(alac, 48000) == (0, 0.0)
    assert mastering.audio_pts_gaps(_aac_pts(100, {50: 1}), 48000) == (0, 0.0)
    assert mastering.audio_pts_gaps(_aac_pts(100, {50: 2}), 48000) == (1, round(2000 / 48000, 3))


@pytest.mark.parametrize("pts", [[], [0], [0, 1024]])
def test_degenerate_packet_lists_do_not_raise(pts: list[int]) -> None:
    assert mastering.audio_pts_gaps(pts, 48000) == (0, 0.0)
    assert mastering.audio_pts_gaps([0, 1024, 2048], 0) == (0, 0.0)


# --- the two paths #242 did not reach: own-audio clips and crossfaded groups -------


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, *, probe_dur: float) -> list[list[str]]:
    """``render_shot`` / ``_render_xfade_group`` with every subprocess stubbed: the
    clip probes as ``probe_dur`` seconds and carries its own audio."""
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(cmd)
        return ""

    async def fake_probe(_: Path) -> float:
        return probe_dur

    async def fake_has_audio(_: Path) -> bool:
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        return Path(url)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return cmds


def _at_fps(fps: int) -> Settings:
    return get_settings().model_copy(update={"render_fps": fps})


async def _render_own_audio(
    tmp_path: Path, fps: int, *, seconds: float = 1.0, speed: float = 1.0
) -> float:
    async with httpx.AsyncClient() as http:
        return await renderer.render_shot(
            # No audio_url: the clip plays its OWN audio. ``seconds`` is the
            # placeholder a "play this recording" shot carries, never a cut length.
            Shot(id="rec", seconds=seconds, video_url="/clips/rec.mp4", speed=speed),
            tmp_path / "seg.mp4",
            _at_fps(fps),
            tmp_path,
            http,
        )


def _output_t(cmd: list[str]) -> str | None:
    """The OUTPUT-side ``-t`` (after the last ``-i``), or None when there is none."""
    tail = cmd[len(cmd) - cmd[::-1].index("-i") :]
    return tail[tail.index("-t") + 1] if "-t" in tail else None


@pytest.mark.parametrize(
    ("fps", "probed", "frames"),
    [
        (30, 12.345, 370),  # 370.35 frames -> DOWN: the recording has no 371st
        (25, 2.566667, 64),  # 64.17 frames
        (25, 2.59, 64),  # 64.75 frames: still down, never to the nearest
        (30, 12.666667, 380),  # a probe-rounded grid value stays where it is
        (24, 6.0, 144),
    ],
)
async def test_a_probed_own_audio_clip_is_bounded_on_the_frame_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fps: int, probed: float, frames: int
) -> None:
    """The clip's audio is cut to the sample and its picture to the frame, so the
    natural length of an own-audio clip is two different lengths. Bounded to
    ``floor(frames) / fps`` with the audio padded to exactly that, both streams of
    the segment end on the same instant and the stream-copy concat has no hole."""
    cmds = _fake_ffmpeg(monkeypatch, probe_dur=probed)
    reported = await _render_own_audio(tmp_path, fps)
    on_grid = frames / fps
    assert reported == on_grid
    cmd = cmds[-1]
    assert _output_t(cmd) == renderer._ffdur(on_grid), cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    # The audio is the clip's own ([0:a]) and is padded to the SAME bound: a clip
    # whose audio stream ends before its picture is the hole, an own-goal otherwise.
    assert re.search(r"\[0:a\][^;\[]*" + re.escape(renderer._pad_audio_to(on_grid)), graph), graph
    # ...and the picture is held to it too (a container that outlasts its frames).
    assert "tpad=stop_mode=clone:stop_duration=" in graph


async def test_a_probed_own_audio_clip_ignores_its_placeholder_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The contract the own-audio branch documents: a 12 s recording sent as
    ``seconds=1.0`` plays 12 s. The new bound is the CLIP's length, never the field."""
    _fake_ffmpeg(monkeypatch, probe_dur=12.0)
    assert await _render_own_audio(tmp_path, 30, seconds=1.0) == 12.0
    assert await _render_own_audio(tmp_path, 30, seconds=40.0) == 12.0


async def test_an_unprobeable_own_audio_clip_stays_unbounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No probed length -> nothing to snap. ``seconds`` is only the REPORTED
    fallback there; bounding the encode to it would cut a recording at its
    placeholder, so the segment keeps its natural length: no ``-t``, no pad."""
    cmds = _fake_ffmpeg(monkeypatch, probe_dur=0.0)
    assert await _render_own_audio(tmp_path, 30, seconds=1.03) == 1.03
    cmd = cmds[-1]
    assert "-t" not in cmd, cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "apad" not in graph
    assert "tpad" not in graph


@pytest.mark.parametrize("speed", [0.8, 1.25, 2.0])
async def test_a_retimed_own_audio_clip_is_bounded_in_the_rendered_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, speed: float
) -> None:
    fps = 30
    cmds = _fake_ffmpeg(monkeypatch, probe_dur=7.2033)
    reported = await _render_own_audio(tmp_path, fps, speed=speed)
    frames = reported * fps
    assert abs(frames - round(frames)) < 1e-9, f"{reported}s is {frames} frames"
    assert 0 <= 7.2033 / speed - reported < 1 / fps
    bound = _output_t(cmds[-1])
    assert bound is not None
    assert float(bound) == pytest.approx(reported, abs=1e-12)


_XFADE = re.compile(r"xfade=transition=\w+:duration=([0-9.]+):offset=([0-9.]+)")
_ADELAY = re.compile(r"adelay=delays=(\d+)S")


@pytest.mark.parametrize(
    ("fps", "requested"),
    [(30, 0.35), (30, 0.4), (25, 0.33), (25, 0.5), (24, 0.3), (60, 0.251), (30, 5.0)],
)
async def test_xfade_overlaps_and_offsets_land_on_the_frame_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fps: int, requested: float
) -> None:
    """A crossfaded group's audio is placed to the SAMPLE and its picture to the
    FRAME. ``xfade`` emits ``ceil(offset * fps)`` frames before each join, so an
    off-grid offset makes the group's picture longer than its audio — and the group
    file is stream-copied into the film like any other segment. On the grid, the
    audio delay and the picture offset are the same instant."""
    cmds = _fake_ffmpeg(monkeypatch, probe_dur=0.0)
    # Probed lengths of on-grid segments, as ffprobe prints them (6 decimals).
    frames = [95, 63, 37]
    lengths = [round(n / fps, 6) for n in frames]
    transitions = [None, *(Transition(type="fade", duration=requested) for _ in frames[1:])]
    paths = [tmp_path / f"s{i}.mp4" for i in range(len(frames))]
    running = await renderer._render_xfade_group(
        paths, lengths, transitions, tmp_path / "group.mp4", _at_fps(fps)
    )
    graph = cmds[-1][cmds[-1].index("-filter_complex") + 1]
    joins = [(float(d), float(o)) for d, o in _XFADE.findall(graph)]
    delays = [int(s) for s in _ADELAY.findall(graph)]
    assert len(joins) == len(delays) == len(frames) - 1
    half_sample = 0.5 / 48000
    picture_frames = 0
    for i, ((d, offset), delay) in enumerate(zip(joins, delays, strict=True), start=1):
        d_frames, offset_frames = d * fps, offset * fps
        assert abs(d_frames - round(d_frames)) < half_sample * fps, f"overlap {d}s off grid"
        assert abs(offset_frames - round(offset_frames)) < half_sample * fps, f"offset {offset}s"
        assert round(d_frames) >= 1
        # Never longer than asked by a frame, never longer than either neighbour.
        assert abs(d - requested) < 1 / fps or d < requested
        assert round(d_frames) <= min(frames[i - 1], frames[i])
        # The audio piece starts on the very sample the picture's join starts on.
        assert delay == round(offset_frames) * 48000 // fps
        picture_frames = round(offset_frames) + frames[i]
    # Picture and audio of the group are the same length, and that is what is returned.
    audio_seconds = delays[-1] / 48000 + lengths[-1]
    assert abs(audio_seconds - picture_frames / fps) < 1.5 / 48000
    assert abs(running - picture_frames / fps) < 1.5 / 48000


async def test_an_on_grid_xfade_keeps_its_requested_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cmds = _fake_ffmpeg(monkeypatch, probe_dur=0.0)
    running = await renderer._render_xfade_group(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"],
        [4.0, 6.0],
        [None, Transition(type="fade", duration=0.5)],
        tmp_path / "g.mp4",
        _at_fps(30),
    )
    graph = cmds[-1][cmds[-1].index("-filter_complex") + 1]
    assert [(float(d), float(o)) for d, o in _XFADE.findall(graph)] == [(0.5, 3.5)]
    assert running == 9.5


# ---------------------------------------------------------------------------
# audio_pts_gap_report: a master with no audio stream is UNMEASURED, quietly.
# ---------------------------------------------------------------------------


async def _gap_report(monkeypatch: pytest.MonkeyPatch, ffprobe_stdout: str) -> Any:
    async def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> str:
        assert cmd[0] == "ffprobe"
        return ffprobe_stdout

    monkeypatch.setattr(renderer, "_run", _fake_run)
    with capture_logs() as logs:
        result = await mastering.audio_pts_gap_report(Path("master.mp4"))
    return result, [entry["event"] for entry in logs]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ffprobe_stdout",
    ['{"packets": [], "streams": []}', '{"packets": []}', "{}"],
    ids=["empty-streams", "no-streams-key", "empty-object"],
)
async def test_gap_report_on_a_video_only_master_is_unmeasured_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, ffprobe_stdout: str
) -> None:
    """``ffprobe -select_streams a:0`` on a video-only file exits 0 with
    ``"streams": []``. That used to index ``[0]``, raise ``IndexError`` and log
    ``mastering.audio_pts_gaps.failed "list index out of range"`` on every
    video-only master — a warning about an input that is entirely expected."""
    result, events = await _gap_report(monkeypatch, ffprobe_stdout)
    assert result == (None, None)
    assert events == []


@pytest.mark.asyncio
async def test_gap_report_still_warns_on_a_real_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the guard must not swallow a genuinely unreadable probe."""
    result, events = await _gap_report(monkeypatch, "not json")
    assert result == (None, None)
    assert events == ["mastering.audio_pts_gaps.failed"]


@pytest.mark.asyncio
async def test_gap_report_measures_a_stream_that_is_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packets = ", ".join(f'{{"pts": {pts}}}' for pts in _aac_pts(100, {50: 2}))
    stdout = f'{{"packets": [{packets}], "streams": [{{"sample_rate": "48000", "time_base": "1/48000"}}]}}'
    result, events = await _gap_report(monkeypatch, stdout)
    assert result == (1, round(2000 / 48000, 3))
    assert events == []
