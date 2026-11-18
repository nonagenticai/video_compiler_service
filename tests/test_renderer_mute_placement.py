"""Where a per-clip ``audio.mute`` is realised — at SOURCE or on the NARRATION BUS.

``audio.mute`` has two correct implementations and the WHOLE REQUEST picks
between them (``renderer._all_shots_muted``):

* every shot muted AND source-silenceable (the music-only action reel) →
  source-level silence in ``render_shot``: no ``0:a`` map at all. Every crossfade is then
  silence-to-silence, the source never reaches the ducker's sidechain key, and
  the filter-side pass must stay OUT of it (a real render measured the delivered
  file at -91 dBFS when it did not).
* a mixed request (the narration products) → the segment keeps mapping its clip
  audio and an edge-ramped, time-ranged ``volume`` on the NARRATION bus does the
  muting, keyed to the shot's SOLO window. Silencing at source here would make an ``acrossfade``
  ramp an UNMUTED neighbour's audio against silence, damaging audio nobody
  edited — the exact regression this split exists to prevent
  (``tests/render_harness/test_audio_level_render.py``).

A shot with its own ``audio_url`` is NOT source-silenceable — ``render_shot`` maps
that narration in every branch without consulting the mute — so one narrated shot
sends the WHOLE request down the filter path. Cover both shapes: this file
once used ``_footage()`` exclusively, and that is how an all-muted narrated reel
came to ship its narration at full level.

These are argv-level and fast; the render harness proves the same split in
delivered samples but takes minutes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import MusicBed, Shot, ShotAudio, Transition
from src.config import get_settings
from src.services import renderer
from src.services.renderer import (
    _all_shots_muted,
    _postclip_filter,
    _premaster_mute_filter,
    _ramped_mute_fragment,
)
from tests.conftest import is_delivered_master_measure


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


async def _noop_uniform_audio(segment_paths: object) -> None:
    return None


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, probe_dur: float = 3.0) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return probe_dur

    async def fake_has_audio(path: Path) -> bool:
        # Drone/screen-recording footage always carries an audio stream — that is
        # what makes "which path owns the mute" a live question at all.
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
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


def _footage(shot_id: str, *, mute: bool) -> Shot:
    return Shot(
        id=shot_id,
        seconds=2.0,
        video_url=f"http://minio/{shot_id}.mp4",
        audio=ShotAudio(mute=True) if mute else None,
    )


def _narrated(shot_id: str, *, mute: bool, video: bool = True) -> Shot:
    """A shot carrying its OWN narration track. ``render_shot`` maps that track
    (``1:a``) in every branch WITHOUT consulting the mute, so this shape can not be
    silenced at source no matter what the rest of the request looks like."""
    return Shot(
        id=shot_id,
        seconds=2.0,
        video_url=f"http://minio/{shot_id}.mp4" if video else None,
        image_url=None if video else f"http://minio/{shot_id}.png",
        audio_url=f"http://minio/{shot_id}.mp3",
        audio=ShotAudio(mute=True) if mute else None,
    )


def _image(shot_id: str, *, mute: bool) -> Shot:
    return Shot(
        id=shot_id,
        seconds=2.0,
        image_url=f"http://minio/{shot_id}.png",
        audio=ShotAudio(mute=True) if mute else None,
    )


# --------------------------------------------------------------------------
# The property itself
# --------------------------------------------------------------------------


def test_all_shots_muted_is_a_whole_request_property() -> None:
    """True only when EVERY shot is muted. One unmuted shot — the shot whose audio
    a blend would damage — is enough to hand the mute back to the narration bus.
    """
    assert _all_shots_muted([_footage("a", mute=True), _footage("b", mute=True)]) is True
    assert _all_shots_muted([_footage("a", mute=True), _footage("b", mute=False)]) is False
    assert _all_shots_muted([_footage("a", mute=False)]) is False
    # A shot with an audio block that does not request a mute does not count.
    assert _all_shots_muted([Shot(id="a", seconds=2.0, audio=ShotAudio(gain_db=-6.0))]) is False
    # An empty request is not a music-only reel.
    assert _all_shots_muted([]) is False


def test_a_narrated_shot_is_never_source_silenceable() -> None:
    """ "Every shot muted" is not enough — every shot must also be SILENCEABLE at
    source. ``render_shot`` maps a shot's own ``audio_url`` as ``1:a`` in all three
    branches without ever consulting the mute, so a request of muted NARRATED shots
    silences nothing at source. Counting it as all-muted made the filter chain
    skip those shots as already-silent and the narration shipped at FULL level.
    """
    assert _all_shots_muted([_narrated("n0", mute=True), _narrated("n1", mute=True)]) is False
    # One narrated shot is enough to hand the WHOLE request back to the filter path.
    assert _all_shots_muted([_footage("s0", mute=True), _narrated("n0", mute=True)]) is False
    # An image/text card with no narration has no source audio to begin with —
    # it renders anullsrc either way, so it stays source-silenceable.
    assert _all_shots_muted([_footage("s0", mute=True), _image("i0", mute=True)]) is True


# --------------------------------------------------------------------------
# Branch 1 — the segment (render_shot)
# --------------------------------------------------------------------------


async def test_all_muted_request_silences_the_segment_at_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    await renderer.render_shot(
        _footage("s0", mute=True),
        tmp_path / "seg.mp4",
        get_settings(),
        tmp_path,
        http,
        all_shots_muted=True,
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "anullsrc" in cmd
    assert "0:a" not in cmd


async def test_mixed_request_keeps_mapping_the_clip_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """THE regression guard at argv level: the same muted shot in a request that
    also holds unmuted shots must still map ``0:a``. Source silence here makes the
    ``acrossfade`` into the next shot ramp THAT shot's audio against silence."""
    captured = _install_fakes(monkeypatch)
    await renderer.render_shot(
        _footage("s0", mute=True),
        tmp_path / "seg.mp4",
        get_settings(),
        tmp_path,
        http,
        all_shots_muted=False,
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "0:a" in cmd
    assert "anullsrc" not in cmd


# --------------------------------------------------------------------------
# Branch 2 — the post-master pass (_postclip_filter)
# --------------------------------------------------------------------------


def test_premaster_mutes_the_mixed_request_and_leaves_the_all_muted_one_alone() -> None:
    """The two halves must never both fire and never both abstain."""
    offsets = [(0.0, 2.0), (2.0, 4.0)]
    mixed = [_footage("s0", mute=True), _footage("s1", mute=False)]
    frag = _premaster_mute_filter(mixed, offsets)
    assert frag == _ramped_mute_fragment(0.0, 2.0)
    assert frag != ""

    all_muted = [_footage("s0", mute=True), _footage("s1", mute=True)]
    assert _premaster_mute_filter(all_muted, offsets) == ""
    # ...and the post-master pass no longer owns ANY mute, in either shape.
    assert _postclip_filter(mixed, offsets) == ""
    assert _postclip_filter(all_muted, offsets) == ""


def test_premaster_mutes_an_all_muted_request_of_narrated_shots() -> None:
    """The hole this file missed by only ever using ``_footage()``: every shot muted,
    but every shot narrated. Nothing can be silenced at source (the narration is
    mapped regardless), so the filter-side clauses are the ONLY mute there is and
    they MUST be emitted — skipping them shipped the narration at full level.
    """
    shots = [_narrated("n0", mute=True, video=False), _narrated("n1", mute=True)]
    assert _premaster_mute_filter(shots, [(0.0, 2.0), (2.0, 4.0)]) == (
        f"{_ramped_mute_fragment(0.0, 2.0)},{_ramped_mute_fragment(2.0, 4.0)}"
    )


async def test_the_mute_reaches_the_sidechain_key_and_never_the_bed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """WHERE the mute is spliced, as a property of the graph rather than a string
    position. Two defects lived in the old finished-bus placement and both are
    structural, so both are checked structurally:

    * the mute must land UPSTREAM of the ``asplit`` that mints ``[nar_key]`` — a
      muted shot that reaches the key still ducks the bed under itself;
    * it must land upstream of the bed ``amix`` too — downstream of that, a
      ``volume`` window zeroes the BED, not the shot.
    """
    captured: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        captured.append(cmd)
        if cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00")
        return ""

    async def fake_has_audio(path: Path) -> bool:
        return True

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    bed = tmp_path / "bed.mp3"
    bed.write_bytes(b"bed")
    mute = _ramped_mute_fragment(0.0, 2.0)

    await renderer.overlay_audio_bed(
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        4.0,
        (bed, MusicBed(url="http://m/b.mp3")),
        [],
        get_settings(),
        premaster_mute=mute,
    )
    graph = captured[-1][captured[-1].index("-filter_complex") + 1]
    assert mute in graph, "the mute must be in the bed pass' graph at all"
    assert graph.index(mute) < graph.index("asplit"), "a muted shot must not duck the bed"
    assert graph.index(mute) < graph.index("amix"), "the mute must never reach the bed"


async def test_a_muted_narrated_shot_maps_its_narration_regardless(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Why the predicate above must exclude narrated shots, proven at argv level:
    even when the caller insists the whole request is source-silenced, the narration
    input is still mapped. Source silencing simply does not exist for this shape."""
    captured = _install_fakes(monkeypatch)
    await renderer.render_shot(
        _narrated("n0", mute=True),
        tmp_path / "seg.mp4",
        get_settings(),
        tmp_path,
        http,
        all_shots_muted=True,
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "1:a" in cmd
    assert "anullsrc" not in cmd


def test_gain_and_mute_ride_their_own_positions() -> None:
    """Only MUTE is realised at source. A gain is not, so it is still emitted —
    the skip must be keyed to the mute, not to the request shape alone."""
    shots = [
        Shot(id="s0", seconds=2.0, video_url="http://m/a.mp4", audio=ShotAudio(mute=True)),
        Shot(id="s1", seconds=2.0, video_url="http://m/b.mp4", audio=ShotAudio(gain_db=-6.0)),
    ]
    # Not all-muted (s1 only asks for gain) → s0's mute rides the PRE-master bus and
    # s1's gain rides the post-master pass. Each position carries exactly its own
    # clause: a mute in the gain chain would zero the bed, a gain before loudnorm
    # would be flattened back to target.
    offsets = [(0.0, 2.0), (2.0, 4.0)]
    assert _postclip_filter(shots, offsets) == ("volume=-6.0dB:enable='between(t,2.000,4.000)'")
    assert _premaster_mute_filter(shots, offsets) == _ramped_mute_fragment(0.0, 2.0)


# --------------------------------------------------------------------------
# The two halves, joined by compile_shots
# --------------------------------------------------------------------------


async def test_compile_shots_threads_the_property_to_both_halves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end at argv level: exactly one of the two mechanisms fires per
    request shape. The shot cache is disabled so every segment's command is really
    emitted (a hit would silently make the assertions vacuous)."""
    settings = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "shot_cache_max_bytes": 0}
    )

    captured = _install_fakes(monkeypatch)
    await renderer.compile_shots([_footage("s0", mute=True), _footage("s1", mute=True)], settings)
    cmds = [" ".join(c) for c in captured["cmds"]]
    assert sum("anullsrc" in c for c in cmds) == 2, "every segment must be silent at source"
    assert not any("eval=frame" in c for c in cmds), (
        "no filter-side mute over an already-silent reel"
    )

    captured = _install_fakes(monkeypatch)
    await renderer.compile_shots([_footage("s0", mute=True), _footage("s1", mute=False)], settings)
    cmds = [" ".join(c) for c in captured["cmds"]]
    assert not any("anullsrc" in c for c in cmds), "a mixed request never silences at source"
    assert any(_ramped_mute_fragment(0.0, 3.0) in c for c in cmds), (
        "the mixed request's mute must reach an ffmpeg command"
    )


async def test_compile_shots_keeps_the_filter_side_mute_for_a_narrated_reel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reviewer's repro, end to end: an all-muted request whose shots are an
    IMAGE and a CLIP, both narrated. Neither can be silenced at source, so the
    narration-bus clauses must be emitted — the shape that previously delivered
    unattenuated narration over the bed."""
    settings = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "shot_cache_max_bytes": 0}
    )
    captured = _install_fakes(monkeypatch)
    await renderer.compile_shots(
        [_narrated("n0", mute=True, video=False), _narrated("n1", mute=True)], settings
    )
    cmds = [" ".join(c) for c in captured["cmds"]]
    assert not any("anullsrc" in c for c in cmds), "narration can not be silenced at source"
    assert any(_ramped_mute_fragment(0.0, 3.0) in c for c in cmds), (
        "the only real mute must still fire"
    )


async def test_all_muted_reel_skips_the_offset_probing_it_cannot_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``need_offsets`` must track when the per-clip chains actually emit. On a
    source-silenced reel it emits nothing, so probing every segment for bus offsets
    is N ffprobe calls spent on a guaranteed-empty filter."""
    settings = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "shot_cache_max_bytes": 0}
    )
    flags: list[object] = []
    join = renderer._join_segments

    async def spy(*args: Any, **kwargs: Any) -> Any:
        flags.append(kwargs.get("need_offsets"))
        return await join(*args, **kwargs)

    _install_fakes(monkeypatch)
    monkeypatch.setattr(renderer, "_join_segments", spy)

    await renderer.compile_shots([_footage("s0", mute=True), _footage("s1", mute=True)], settings)
    assert flags == [False], "source-silenced: no per-clip clause needs an offset"

    flags.clear()
    await renderer.compile_shots([_footage("s0", mute=True), _footage("s1", mute=False)], settings)
    assert flags == [True], "mixed: the volume=0 clause is keyed to the bus offset"

    flags.clear()
    await renderer.compile_shots([_narrated("n0", mute=True), _narrated("n1", mute=True)], settings)
    assert flags == [True], "all muted but narrated: still a filter-side mute"


async def test_shot_cache_key_separates_the_two_mute_paths() -> None:
    """The two paths produce DIFFERENT segment samples for the same shot, so they
    must not share a cache entry — while every non-all-muted job's key (the whole
    existing pod cache) stays exactly what it was."""
    settings = get_settings()
    muted = _footage("s0", mute=True)
    unmuted = _footage("s0", mute=False)

    assert renderer._shot_cache_key(muted, settings, 0.0) != renderer._shot_cache_key(
        muted, settings, 0.0, all_shots_muted=True
    )
    # An unmuted shot renders identically either way → same key, cache preserved.
    assert renderer._shot_cache_key(unmuted, settings, 0.0) == renderer._shot_cache_key(
        unmuted, settings, 0.0, all_shots_muted=True
    )


def test_premaster_solo_window_still_spares_a_blended_neighbour() -> None:
    """The reason the mixed path exists, restated at the filter layer: the muted
    shot's clause is keyed to its SOLO window, so it never reaches into the
    crossfade blend it shares with the unmuted neighbour."""
    shots = [
        Shot(id="s0", seconds=4.0, video_url="http://m/a.mp4"),
        Shot(
            id="s1",
            seconds=6.0,
            video_url="http://m/b.mp4",
            transition=Transition(duration=1.0),
            audio=ShotAudio(mute=True),
        ),
    ]
    frag = _premaster_mute_filter(shots, [(0.0, 4.0), (3.0, 9.0)])
    assert frag == _ramped_mute_fragment(4.0, 9.0)
    # The blend region [3,4) belongs to the UNMUTED neighbour and must be untouched:
    # the ramp starts at 4.000, not at the raw offset 3.000.
    assert "(t-4.000)" in frag and "(t-3.000)" not in frag
