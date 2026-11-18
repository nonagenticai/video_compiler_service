"""Command/graph-string tests for per-clip speed + xfade transitions (WS-2).

ffmpeg/ffprobe + asset fetch are mocked: these assert the filtergraph SHAPE —
that ``speed`` injects ``setpts``/``atempo``, that a transition routes
``compile_shots`` through the re-encoding xfade join, and that the default (no
speed / no transition) path is byte-identical to the old fast concat. Real
muxing is proven by the render-harness suite (``tests/render_harness/``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import Shot, ShotAudio, Transition
from src.config import get_settings
from src.services import renderer
from src.services.renderer import (
    _apply_speed_to_vf,
    _atempo_chain,
    _bus_offsets_plain,
    _bus_offsets_xfade,
    _postclip_filter,
    _premaster_mute_filter,
    _ramped_mute_fragment,
    _setpts_speed_fragment,
    _solo_windows,
    _xfade_group_joins,
)


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


async def _noop_uniform_audio(segment_paths: object) -> None:
    return None


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, probe_dur: float = 3.0) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return probe_dur

    async def fake_has_audio(path: Path) -> bool:
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    # The concat audio-uniformity probe reads real ffprobe JSON; these fakes write
    # placeholder files, so it is stubbed like the other probes (proven for real in
    # tests/render_harness/test_mixed_depth_concat_render.py).
    monkeypatch.setattr(renderer, "_assert_uniform_segment_audio", _noop_uniform_audio)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


# --------------------------------------------------------------------------
# Pure speed helpers
# --------------------------------------------------------------------------


def test_setpts_speed_fragment() -> None:
    assert _setpts_speed_fragment(2.0) == "setpts=PTS/2"
    assert _setpts_speed_fragment(0.5) == "setpts=PTS/0.5"
    # 1.0 is a no-op — no filter at all so the default path is byte-identical.
    assert _setpts_speed_fragment(1.0) == ""


def test_apply_speed_to_vf_prepends_setpts() -> None:
    # setpts must come FIRST so the downstream fps filter resamples the frame
    # count; a trailing setpts would only re-stamp timestamps.
    assert _apply_speed_to_vf("scale=2:2,fps=25", 2.0) == "setpts=PTS/2,scale=2:2,fps=25"
    assert _apply_speed_to_vf("scale=2:2", 1.0) == "scale=2:2"


def test_atempo_chain_ranges() -> None:
    assert _atempo_chain(1.0) == ""
    assert _atempo_chain(2.0) == "atempo=2"
    assert _atempo_chain(0.5) == "atempo=0.5"
    # Factors outside a single atempo's [0.5, 2.0] window are chained.
    assert _atempo_chain(4.0) == "atempo=2,atempo=2"
    assert _atempo_chain(0.25) == "atempo=0.5,atempo=0.5"


def test_xfade_overlap_is_bounded_to_the_shorter_neighbours_whole_frames() -> None:
    def overlap(requested: float, prev_len: float, cur_len: float) -> float:
        joins = _xfade_group_joins(
            [prev_len, cur_len], [None, Transition(duration=requested)], fps=30
        )
        return joins[1][0]

    assert overlap(0.5, 1.0, 1.0) == 0.5
    # Clamped to the shorter adjacent clip — to its WHOLE frames (0.8 s = 24 frames),
    # never past it: a longer overlap would run off the clip (a black hole).
    assert overlap(2.0, 0.8, 1.5) == 24 / 30
    assert overlap(2.0, 0.79, 1.5) == 23 / 30
    # A sub-frame clip still yields the one-frame minimum xfade can express.
    assert overlap(1.0, 0.02, 0.02) == 1 / 30


# --------------------------------------------------------------------------
# render_shot: speed injects setpts + atempo
# --------------------------------------------------------------------------


async def test_render_shot_speed_injects_setpts_and_atempo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s1", seconds=2.0, video_url="http://minio/clip.mp4", speed=2.0)
    out_dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "setpts=PTS/2" in cmd  # video retimed
    assert "atempo=2" in cmd  # audio retimed
    # The segment now lasts source/speed; clip_dur=3.0 (fake probe) → 1.5.
    assert out_dur == pytest.approx(1.5)


async def test_render_shot_speed_one_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s1", seconds=2.0, video_url="http://minio/clip.mp4")  # speed defaults to 1.0
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    # ``setpts=PTS/<speed>`` is the speed retime; narrowed from a bare "setpts="
    # because the chain now always carries the frame-exactness rebase
    # ``setpts=PTS-STARTPTS``. The no-op guarantee is unchanged and still exact.
    assert "setpts=PTS/" not in cmd
    assert "atempo=" not in cmd


async def test_render_shot_speed_never_retimes_synthesised_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A shot whose primary audio is ``anullsrc`` gets NO ``atempo`` at any speed.

    ffmpeg's ``atempo`` does not map an all-zero buffer to an all-zero buffer: it
    emits a constant DC offset (~-0.011 / -39 dBFS on ffmpeg 4.4). Reproduce with:

        ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 \\
               -af atempo=1.25 -t 3 out.wav   # dc = -0.011, NOT silence

    That DC survives the stream-copy concat into the film bus, the ``loudnorm``
    master lifts it ~17 dB, and it also rides ``[nar_key]`` (the music ducker's
    sidechain key). On a music-only action reel — every shot muted, most shots
    retimed — it is the entire audible low end of the delivered file, and it read
    as "the source clip's audio survived the mute" in the render harness.
    Retiming silence is a no-op anyway (``anullsrc`` is infinite; the output
    ``-t`` sets the length), so the chain must simply not be emitted.

    ⚠️ EXPECTATION CHANGE: this test previously asserted ``atempo=0.5`` and
    ``[aout]`` for the image case, on the rationale that silent audio should
    still be routed through the filtergraph "so atempo can apply". That
    rationale was the bug. The VIDEO retime (``setpts``) is unchanged, and the
    segment length is unchanged (still bounded by the rescaled ``-t``).
    """
    # The muted case reaches the silence branch only on an ALL-MUTED request (the
    # music-only reel); in a mixed request it keeps mapping ``0:a`` and is muted
    # post-master, so it has real samples and its ``atempo`` is legitimate.
    for shot, all_muted in (
        (Shot(id="img", seconds=2.0, image_url="http://minio/slide.png", speed=0.5), False),
        (Shot(id="txt", seconds=2.0, description="a card", speed=2.0), False),
        (
            Shot(
                id="muted",
                seconds=2.0,
                video_url="http://minio/clip.mp4",
                speed=2.0,
                audio=ShotAudio(mute=True),
            ),
            True,
        ),
    ):
        captured = _install_fakes(monkeypatch)
        await renderer.render_shot(
            shot,
            tmp_path / f"{shot.id}.mp4",
            get_settings(),
            tmp_path,
            http,
            all_shots_muted=all_muted,
        )
        cmd = " ".join(captured["cmds"][-1])
        assert "anullsrc" in cmd, f"{shot.id}: expected the synthesised-silence branch"
        assert "setpts=PTS/" in cmd, f"{shot.id}: the VIDEO retime must be unaffected"
        assert "atempo" not in cmd, (
            f"{shot.id}: atempo is being applied to anullsrc silence — ffmpeg emits a "
            f"DC offset, not silence, so the segment ships an inaudible-looking thump "
            f"that loudnorm later amplifies. cmd={cmd}"
        )
        # The retimed length still comes from the output -t, not from atempo.
        assert "-t" in captured["cmds"][-1]


async def test_render_shot_speed_still_retimes_real_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The silence carve-out above must not disarm the retime for REAL samples:
    a narrated shot and an UNMUTED own-audio clip both still get ``atempo``."""
    captured = _install_fakes(monkeypatch)
    narrated = Shot(
        id="s1",
        seconds=2.0,
        image_url="http://minio/slide.png",
        audio_url="http://m/vo.mp3",
        speed=0.5,
    )
    await renderer.render_shot(narrated, tmp_path / "n.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "anullsrc" not in cmd
    assert "atempo=0.5" in cmd
    assert "[aout]" in cmd


# --------------------------------------------------------------------------
# render_shot: per-clip audio level (gain / mute / fade) injects volume/afade
# --------------------------------------------------------------------------


async def test_render_shot_audio_gain_does_not_bake_into_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Per-clip gain no longer bakes into the shot SEGMENT — the default-ON master
    would flatten a pre-master gain, so it moves to the POST-master timeline pass
    (see _postclip_filter). A gain-only shot's segment is IDENTICAL to an unedited
    one: no ``volume`` node, and nothing in its audio graph but the unconditional
    ``_SHOT_EDGE_FADE_S`` edge fade every segment now carries."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1", seconds=2.0, video_url="http://minio/clip.mp4", audio=ShotAudio(gain_db=-6)
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "volume=-6.0dB" not in cmd
    # The audio graph carries the edge fade and NOTHING else — the same graph an
    # unedited shot of the same length gets, so the gain is genuinely not baked.
    # (It is no longer a bare ``-map 0:a``: every segment is fade-terminated so the
    # sample-exact PCM concat never joins two non-zero samples.)
    assert f"afade=t=in:st=0:d={renderer._SHOT_EDGE_FADE_S}" in cmd
    assert "afade=t=out:st=2.992:d=0.008" in cmd
    unedited = Shot(id="s1", seconds=2.0, video_url="http://minio/clip.mp4")
    await renderer.render_shot(unedited, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    assert " ".join(captured["cmds"][-1]) == cmd


async def test_render_shot_audio_fade_injects_afade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=2.0,
        video_url="http://minio/clip.mp4",
        audio_url="http://minio/narration.wav",
        audio=ShotAudio(fade_in_s=0.5, fade_out_s=1.0),
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "afade=t=in:st=0:d=0.5" in cmd
    # fake_probe drives the narration duration to 3.0 → fade-out starts at
    # duration - fade_out_s = 2.0.
    assert "afade=t=out:st=2.0:d=1.0" in cmd


async def test_render_shot_audio_mute_does_not_bake_into_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Per-clip mute is not baked as a segment ``volume=0`` (same reason as gain);
    the level pass stays POST-master. What an ALL-MUTED request's segment DOES do
    is refuse to map the clip's own audio at all — the mute is realised by
    rendering silence, so it is exact even inside a crossfade and never reaches the
    bed's sidechain key."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s1", seconds=2.0, video_url="http://minio/clip.mp4", audio=ShotAudio(mute=True))
    await renderer.render_shot(
        shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http, all_shots_muted=True
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "volume=0" not in cmd
    assert "-map 0:a" not in cmd
    assert "anullsrc" in cmd
    # The audio graph holds the unconditional edge fade and nothing else — in
    # particular no ``volume`` node, which is the invariant under test.
    assert "volume" not in cmd
    assert "afade=t=in:st=0:d=0.008,afade=t=out:st=2.992:d=0.008[aout]" in cmd


async def test_render_shot_fade_out_anchors_to_clip_duration_for_own_audio_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Regression: a video shot with NO audio_url but whose clip carries its OWN
    audio enters the ``clip_has_audio`` branch, which sets ``trim=False`` and
    reassigns ``duration = clip_dur`` (real footage length). The primary-audio
    fade-out must anchor to that final pre-speed clip length, NOT the stale
    ``shot.seconds`` fallback. Before the fix, ``_apply_shot_audio`` was computed
    ahead of the branch chain while ``duration`` was still ``shot.seconds`` (5.0),
    so the fade started at ``st=3`` (5.0 - 2.0) instead of ``st=28`` (30.0 - 2.0),
    fading footage out 25s early.
    """
    # clip_dur (fake probe) = 30.0, well past shot.seconds = 5.0.
    captured = _install_fakes(monkeypatch, probe_dur=30.0)
    shot = Shot(
        id="s1",
        seconds=5.0,
        video_url="http://minio/clip.mp4",  # no audio_url → clip supplies its own audio
        audio=ShotAudio(fade_out_s=2.0),
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    # Anchored to the real clip length: 30.0 - 2.0 = 28.0.
    assert "afade=t=out:st=28.0:d=2.0" in cmd
    # NOT the stale shot.seconds fallback: 5.0 - 2.0 = 3.0.
    assert "st=3.0" not in cmd


# --------------------------------------------------------------------------
# Bus-offset helpers + _postclip_filter (pure) — the POST-master per-clip pass
# --------------------------------------------------------------------------


def test_bus_offsets_plain_is_cumsum() -> None:
    """Plain concat: each shot starts at the cumulative sum of prior lengths."""
    assert _bus_offsets_plain([4.0, 6.0, 7.0]) == [(0.0, 4.0), (4.0, 10.0), (10.0, 17.0)]


def test_bus_offsets_xfade_overlaps_compound() -> None:
    """xfade d=1 between all of [4,6,7] → [(0,4),(3,9),(8,15)]: the recurrence uses
    the PREVIOUS shot's length (start_k = start_{k-1} + L_{k-1} - d), not the
    current one."""
    transitions = [None, Transition(duration=1.0), Transition(duration=1.0)]
    assert _bus_offsets_xfade([4.0, 6.0, 7.0], transitions, 30) == [
        (0.0, 4.0),
        (3.0, 9.0),
        (8.0, 15.0),
    ]


def test_bus_offsets_xfade_hard_cut_starts_at_group_end() -> None:
    """A None (hard-cut) boundary starts flush against the previous group's
    composite end — no overlap — while a real transition still crossfades."""
    # [s0]--cut--[s1]--xfade0.5--[s2], lengths all 2.0.
    transitions = [None, None, Transition(duration=0.5)]
    assert _bus_offsets_xfade([2.0, 2.0, 2.0], transitions, 30) == [
        (0.0, 2.0),
        (2.0, 4.0),
        (3.5, 5.5),
    ]


def test_solo_windows_plain_concat_are_raw_offsets() -> None:
    """No transitions → nothing to trim; the solo window IS the raw offset."""
    shots = [Shot(id=f"s{i}", seconds=2.0) for i in range(3)]
    offsets = [(0.0, 4.0), (4.0, 10.0), (10.0, 17.0)]
    assert _solo_windows(shots, offsets) == offsets


def test_solo_windows_exclude_crossfade_overlap() -> None:
    """Each shot is trimmed out of the blend it shares with a transitioned
    neighbour, so an edit can never reach into a clip the user did not touch."""
    shots = [
        Shot(id="s0", seconds=4.0),
        Shot(id="s1", seconds=6.0, transition=Transition(duration=1.0)),
        Shot(id="s2", seconds=7.0, transition=Transition(duration=1.0)),
    ]
    # Raw bus (from _bus_offsets_xfade): [(0,4), (3,9), (8,15)].
    offsets = [(0.0, 4.0), (3.0, 9.0), (8.0, 15.0)]
    assert _solo_windows(shots, offsets) == [
        (0.0, 3.0),  # trail-trimmed by the 1s xfade out of s0
        (4.0, 8.0),  # lead- AND trail-trimmed
        (9.0, 15.0),  # lead-trimmed
    ]


def test_solo_windows_none_when_shot_fully_consumed_by_overlaps() -> None:
    """A shot swallowed by its own crossfades has no solo region → None (never an
    inverted/empty enable window)."""
    shots = [
        Shot(id="s0", seconds=2.0),
        Shot(id="s1", seconds=2.0, transition=Transition(duration=1.0)),
        Shot(id="s2", seconds=2.0, transition=Transition(duration=1.0)),
    ]
    # s1 spans [1,3) but both its 1s overlaps consume it entirely.
    offsets = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    assert _solo_windows(shots, offsets)[1] is None


def test_premaster_mute_skips_shot_with_no_solo_region() -> None:
    """A fully-overlapped shot emits NO clause even when it requests a mute."""
    shots = [
        Shot(id="s0", seconds=2.0),
        Shot(id="s1", seconds=2.0, transition=Transition(duration=1.0), audio=ShotAudio(mute=True)),
        Shot(id="s2", seconds=2.0, transition=Transition(duration=1.0)),
    ]
    assert _premaster_mute_filter(shots, [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]) == ""


def test_premaster_mute_uses_solo_window_not_raw_offset() -> None:
    """The emitted enable window is the SOLO region: a mute on s1 must NOT reach
    into s0's crossfade tail (raw window would be between(3,9); solo is (4,8))."""
    # This is a MIXED request (s0/s2 unmuted), so the mute is owned by the
    # NARRATION-BUS chain — render_shot still maps the clip audio, precisely so the
    # crossfade blends do not ramp an unmuted neighbour against silence. See
    # _all_shots_muted.
    shots = [
        Shot(id="s0", seconds=4.0),
        Shot(
            id="s1",
            seconds=6.0,
            audio_url="http://minio/vo.mp3",
            transition=Transition(duration=1.0),
            audio=ShotAudio(mute=True),
        ),
        Shot(id="s2", seconds=7.0, transition=Transition(duration=1.0)),
    ]
    frag = _premaster_mute_filter(shots, [(0.0, 4.0), (3.0, 9.0), (8.0, 15.0)])
    assert frag == _ramped_mute_fragment(4.0, 8.0)
    assert "3.000" not in frag  # never reaches back into s0's tail


def test_per_clip_chains_empty_when_no_gain_or_mute() -> None:
    """No shot requests gain/mute → "" (byte-identical no-op) from BOTH chains,
    even with offsets."""
    shots = [
        Shot(id="a", seconds=2.0, audio=ShotAudio(fade_in_s=0.5)),  # fade only
        Shot(id="b", seconds=2.0, audio=None),
    ]
    offsets = [(0.0, 2.0), (2.0, 4.0)]
    assert _postclip_filter(shots, offsets) == ""
    assert _premaster_mute_filter(shots, offsets) == ""
    # Empty offsets → no-op regardless of audio.
    muted = [Shot(id="a", seconds=2.0, audio=ShotAudio(mute=True))]
    assert _postclip_filter(muted, []) == ""
    assert _premaster_mute_filter(muted, []) == ""


def test_per_clip_chains_split_gain_and_mute_by_position() -> None:
    """gain and mute become time-ranged ``volume`` nodes keyed to each shot's bus
    window — but in DIFFERENT passes, and mute still wins over gain on the same
    shot. Fade-only shots contribute nothing to either.

    Shots ``a``/``b`` are unmuted, so this is a MIXED request and the narration-bus
    chain owns every mute in it — with or without an ``audio_url`` (``c`` narrated, ``d`` a
    silent card). Whether the muted shot happened to carry narration is not the
    question the mute turns on; the question is whether the whole request is muted,
    which is what lets render_shot silence at source instead. See
    ``_all_shots_muted``."""
    shots = [
        Shot(id="a", seconds=2.0, audio=None),
        Shot(id="b", seconds=2.0, audio=ShotAudio(gain_db=-6.0)),
        Shot(
            id="c",
            seconds=2.0,
            audio_url="http://minio/vo.mp3",
            audio=ShotAudio(mute=True, gain_db=-6.0),
        ),
        Shot(id="d", seconds=2.0, audio=ShotAudio(mute=True, gain_db=-6.0)),
    ]
    offsets = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    # Post-master: ONLY b's gain. c and d are muted, so their -6 dB is not emitted
    # here (mute wins) and, critically, no `volume=0` reaches the bus that carries
    # the music bed.
    assert _postclip_filter(shots, offsets) == "volume=-6.0dB:enable='between(t,2.000,4.000)'"
    # Pre-master narration bus: c and d's mutes, edge-ramped, and nothing else.
    assert _premaster_mute_filter(shots, offsets) == (
        f"{_ramped_mute_fragment(4.0, 6.0)},{_ramped_mute_fragment(6.0, 8.0)}"
    )


async def test_render_shot_default_audio_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """No audio block and an all-default ShotAudio produce the SAME command — the
    no-op invariant at the command-string layer.

    It is no longer "no afade at all": every segment now carries the unconditional
    ``_SHOT_EDGE_FADE_S`` edge fade (so the sample-exact PCM concat never joins two
    non-zero samples), and both of these shots must carry exactly that one and no
    ``volume`` node. The AUTHOR fade is still absent, which is what "default" means
    here — asserted by pinning the emitted durations to the edge fade."""
    captured = _install_fakes(monkeypatch)
    cmds: list[str] = []
    for audio in (None, ShotAudio()):
        shot = Shot(id="s1", seconds=2.0, video_url="http://minio/clip.mp4", audio=audio)
        await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
        cmd = " ".join(captured["cmds"][-1])
        assert "volume" not in cmd
        # ...then padded to the segment's own ``-t`` (the probed 3.0 s clip length, on
        # the frame grid) so the clip's audio and picture end together.
        assert (
            "afade=t=in:st=0:d=0.008,afade=t=out:st=2.992:d=0.008,apad=whole_dur=3.0[aout]" in cmd
        )
        cmds.append(cmd)
    assert cmds[0] == cmds[1]


# --------------------------------------------------------------------------
# compile_shots: transition routes through the xfade join
# --------------------------------------------------------------------------


async def test_compile_shots_transition_uses_xfade_not_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fakes(monkeypatch)
    shots = [
        Shot(id="s0", seconds=2.0, image_url="http://m/a.png"),
        Shot(
            id="s1",
            seconds=2.0,
            image_url="http://m/b.png",
            transition=Transition(type="fade", duration=0.5),
        ),
    ]
    await renderer.compile_shots(shots, get_settings())
    joined = [" ".join(c) for c in captured["cmds"]]
    assert any("xfade=transition=fade" in c for c in joined)
    # Audio crossfade = afade ramps summed by amix (acrossfade races; see renderer).
    assert any("amix=inputs=2:duration=longest:normalize=0" in c for c in joined)
    assert not any("acrossfade=" in c for c in joined)
    # The re-encoding xfade path replaces the concat demuxer for the video join,
    # so no ffmpeg invocation uses "-f concat".
    assert not any("concat" in c for c in captured["cmds"])


async def test_compile_shots_no_transition_keeps_concat_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fakes(monkeypatch)
    settings = get_settings().model_copy(update={"master_enabled": False})
    shots = [
        Shot(id="s0", seconds=2.0, image_url="http://m/a.png"),
        Shot(id="s1", seconds=2.0, image_url="http://m/b.png"),
    ]
    await renderer.compile_shots(shots, settings)
    # No transition → the fast stream-copy concat demuxer, no xfade re-encode.
    assert any(("concat" in c and "copy" in c) for c in captured["cmds"])
    assert not any("xfade" in " ".join(c) for c in captured["cmds"])


async def test_compile_shots_first_shot_transition_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transition on the FIRST shot has no preceding shot to dissolve from and
    must be ignored — the compile stays on the fast concat path."""
    captured = _install_fakes(monkeypatch)
    settings = get_settings().model_copy(update={"master_enabled": False})
    shots = [
        Shot(id="s0", seconds=2.0, image_url="http://m/a.png", transition=Transition()),
        Shot(id="s1", seconds=2.0, image_url="http://m/b.png"),
    ]
    await renderer.compile_shots(shots, settings)
    assert not any("xfade" in " ".join(c) for c in captured["cmds"])
