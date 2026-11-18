"""Wave 13: four defects in the delivered A/V chain, each asserted as a PROPERTY
of the emitted ffmpeg commands rather than as a filter-string position.

Every assertion here was mutation-tested — the covered line was broken and the
test confirmed FAILING — because this project has had six tests survive deleting
the call site they covered.

* **C4** a post-master per-clip GAIN landed downstream of the master's limiter
  with nothing after it, so a boost clipped in the AAC encoder. Measured on a
  -1 dBFS bus with a +6 dB clause: peak +4.99 dBFS and 60,150 samples at or above
  full scale; with the limiter appended, peak -1.30 dBFS and ZERO clipped samples.
* **C7** two lossy AAC generations whenever a compile requested a per-clip gain:
  the bed/master pass encoded AAC, then the gain pass decoded and re-encoded it.
* **C8** the crossfade-group encode ran at the DELIVERY CRF cell. Its
  ``filter_complex`` spans the whole pair, so that is a second full generation
  over every frame of every shot in a transition.
* **C9** the audiobook gap concat butt-joined ``anullsrc`` silence against clips
  ending on a non-zero sample. Measured on 24 kHz mono clips: the join
  discontinuity fell from 0.05493 to 0.00446 (12.3x) and the last sample before
  the gap from -0.05493 to -0.00027.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import Shot, ShotAudio, Transition
from src.config import Settings, get_settings
from src.services import renderer
from src.services.renderer import (
    _SHOT_EDGE_FADE_S,
    _concat_edge_fade,
    _postclip_limiter,
    _postclip_raises_level,
)


def _settings(tmp_path: Path, **over: Any) -> Settings:
    return get_settings().model_copy(update={"work_dir": str(tmp_path), **over})


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_has_audio(path: Path) -> bool:
        return True

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    return cmds


def _audio_filter(cmd: list[str]) -> str:
    for flag in ("-filter_complex", "-af"):
        if flag in cmd:
            return cmd[cmd.index(flag) + 1]
    return ""


def _gain(gain_db: float) -> list[Shot]:
    return [
        Shot(id="s0", seconds=2.0, video_url="http://m/a.mp4"),
        Shot(id="s1", seconds=2.0, video_url="http://m/b.mp4", audio=ShotAudio(gain_db=gain_db)),
    ]


# ---------------------------------------------------------------------------
# C4 — the gain must not be the last thing the bus sees
# ---------------------------------------------------------------------------


def test_only_a_boost_arms_the_postclip_limiter() -> None:
    """The limiter is needed exactly when the pass can RAISE the bus. A cut or a
    mute can only lower it, so those compiles stay byte-identical to today —
    a limiter that armed on everything would be a change with no defect behind it.
    """
    assert _postclip_raises_level(_gain(6.0)) is True
    assert _postclip_raises_level(_gain(-6.0)) is False
    assert _postclip_raises_level([Shot(id="s", seconds=2.0, audio=ShotAudio(mute=True))]) is False
    assert _postclip_raises_level([Shot(id="s", seconds=2.0)]) is False
    # A muted shot that ALSO carries a boost cannot raise anything: the mute is
    # realised first and wins.
    muted_boost = [Shot(id="s", seconds=2.0, audio=ShotAudio(mute=True, gain_db=6.0))]
    assert _postclip_raises_level(muted_boost) is False


def test_the_postclip_ceiling_is_the_master_ceiling(tmp_path: Path) -> None:
    """DERIVED, not a second hardcoded constant. The module has had a limiter whose
    ceiling was disconnected from ``master_tp`` before (0.95, i.e. -0.45 dBFS,
    unrelated to any setting); a new one must not reintroduce that."""
    settings = _settings(tmp_path, master_tp=-3.0)
    frag = _postclip_limiter(settings)
    assert f"limit={renderer._ffnum(renderer._limiter_linear_limit(settings))}" in frag
    # ...and it must track the setting, not sit at a constant.
    assert _postclip_limiter(_settings(tmp_path, master_tp=-1.0)) != frag
    # `level` defaults to TRUE in ffmpeg and re-normalises back up to full scale,
    # which makes `limit=` bound nothing at all.
    assert "level=disabled" in frag


async def test_a_boost_is_limited_and_a_cut_is_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The property at command level: the chain handed to the per-clip pass ENDS in
    a limiter for a boost, and ends in the gain itself for a cut.

    Asserting "ends in" rather than "contains" is deliberate — a limiter anywhere
    but last does not bound the delivered peak, which is the whole defect
    (``_master_filter`` documents the same ordering trap on the master bus)."""
    settings = _settings(tmp_path, shot_cache_max_bytes=0)
    for gain_db, expect_limiter in ((6.0, True), (-6.0, False)):
        cmds = _capture(monkeypatch)
        await renderer._finalize_audio(
            tmp_path / "concat.mp4",
            tmp_path,
            4.0,
            None,
            [],
            settings,
            shots=_gain(gain_db),
            shot_offsets=[(0.0, 2.0), (2.0, 4.0)],
        )
        chains = [_audio_filter(c) for c in cmds if "volume=" in _audio_filter(c)]
        assert chains, f"gain {gain_db} must reach an ffmpeg command at all"
        tail = chains[-1]
        assert ("alimiter" in tail) is expect_limiter, tail
        if expect_limiter:
            assert tail.index("volume=") < tail.index("alimiter"), (
                "the limiter must come AFTER the gain, or it bounds nothing"
            )
            assert tail.rstrip("[aout]").endswith("level=disabled"), (
                f"nothing may follow the limiter: {tail}"
            )


# ---------------------------------------------------------------------------
# C7 — exactly one lossy generation, in every configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gain_db", "music", "grade"),
    [
        (6.0, False, ""),
        (6.0, True, ""),
        (6.0, False, "eq=contrast=1.1"),
        (0.0, False, ""),
        (0.0, True, ""),
    ],
)
async def test_exactly_one_aac_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gain_db: float, music: bool, grade: str
) -> None:
    """Count the AAC ENCODES across the whole tail, for every shape of compile.

    Two is the defect (bed/master encoded AAC, then the per-clip pass decoded that
    AAC and re-encoded AAC); zero would mean a deliverable left the service as
    lossless ALAC. Counting encodes is the property — which pass happens to own
    the encode is an implementation detail this test must not pin."""
    settings = _settings(tmp_path, shot_cache_max_bytes=0)
    bed = tmp_path / "bed.mp3"
    bed.write_bytes(b"bed")
    from src.api.schemas import MusicBed

    cmds = _capture(monkeypatch)
    await renderer._finalize_audio(
        tmp_path / "concat.mp4",
        tmp_path,
        4.0,
        (bed, MusicBed(url="http://m/b.mp3")) if music else None,
        [],
        settings,
        shots=_gain(gain_db),
        shot_offsets=[(0.0, 2.0), (2.0, 4.0)],
        grade_vf=grade,
    )
    aac = [c for c in cmds if "aac" in c and "-c:a" in c and c[c.index("-c:a") + 1] == "aac"]
    assert len(aac) == 1, f"expected ONE AAC encode, got {len(aac)}:\n" + "\n".join(
        " ".join(c) for c in aac
    )
    # ...and when an earlier pass exists it must have stayed LOSSLESS, or the one
    # AAC encode above is being fed an already-lossy bus.
    if gain_db > 0 and (music or grade):
        assert any("alac" in c for c in cmds), "the upstream pass must stay lossless"


# ---------------------------------------------------------------------------
# C8 — the crossfade group is a second generation, not a delivery encode
# ---------------------------------------------------------------------------


async def test_xfade_group_encodes_at_the_intermediate_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``xfade``'s ``filter_complex`` spans the ENTIRE pair, so every frame of both
    shots is re-encoded. At the delivery cell that re-quantises material generation
    1 already quantised at that exact cell — what ``render_intermediate_crf_delta``
    exists to prevent, happening on the deliverable.

    Asserted against ``mastering.encode_ladder`` rather than a literal CRF so the
    test cannot drift from the ladder, and with a non-zero delta configured so the
    two cells are actually distinguishable (at delta=0 they are equal by design and
    the assertion would be unfalsifiable)."""
    from src.services import mastering

    settings = _settings(tmp_path, render_intermediate_crf_delta=6)
    delivery, _ = mastering.encode_ladder(settings)
    intermediate, _ = mastering.encode_ladder(settings, intermediate=True)
    assert intermediate != delivery, "delta must be non-zero or this test proves nothing"

    cmds = _capture(monkeypatch)

    async def fake_probe(path: Path) -> float:
        return 2.0

    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    await renderer._render_xfade_group(
        [tmp_path / "a.mp4", tmp_path / "b.mp4"],
        [2.0, 2.0],
        [None, Transition(duration=0.5)],
        tmp_path / "group.mp4",
        settings,
    )
    cmd = cmds[-1]
    assert "xfade" in " ".join(cmd), "wrong command captured"
    assert cmd[cmd.index("-crf") + 1] == str(intermediate)
    assert cmd[cmd.index("-crf") + 1] != str(delivery)
    # Still concat-stream-copy compatible: the cell is the ONLY thing that moved.
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-fps_mode:v" in cmd and cmd[cmd.index("-fps_mode:v") + 1] == "cfr"


# ---------------------------------------------------------------------------
# C9 — the audiobook gap concat
# ---------------------------------------------------------------------------


def test_concat_edge_fade_ramps_both_ends_of_a_clip() -> None:
    """``concat`` butt-joins, so a clip's own edges are the only place the join can
    be landed on silence."""
    frag = _concat_edge_fade("[0:a]", "[e0]", 1.0)
    assert frag.startswith("[0:a]") and frag.endswith("[e0]")
    assert f"afade=t=in:st=0:d={_SHOT_EDGE_FADE_S}" in frag
    # The fade-out must start one ramp BEFORE the end, or it does not reach zero
    # by the join.
    assert f"afade=t=out:st={1.0 - _SHOT_EDGE_FADE_S:.6f}:d={_SHOT_EDGE_FADE_S}" in frag


def test_a_clip_too_short_for_two_ramps_keeps_its_label() -> None:
    """Overlapping ramps would dip the whole clip (the rule ``_apply_shot_audio``
    already applies), but the NODE must still exist: the concat inputs are
    index-aligned by label, so dropping it silently shifts a clip onto a gap."""
    for seconds in (0.0, _SHOT_EDGE_FADE_S, 2 * _SHOT_EDGE_FADE_S):
        frag = _concat_edge_fade("[4:a]", "[e2]", seconds)
        assert frag == "[4:a]anull[e2]", seconds
        assert "afade" not in frag


async def test_gap_silence_is_never_sized_from_a_fabricated_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback this replaces (``or (24000, 1)``) fabricated the nominal Kokoro
    shape from an UNMEASURED clip, so a 48 kHz stereo book got 24 kHz mono gaps
    from an assumption nothing in the output recorded. It must fail loudly instead.
    """
    _capture(monkeypatch)

    async def no_params(path: Path) -> tuple[int, int] | None:
        return None

    async def noop(paths: object) -> None:
        return None

    monkeypatch.setattr(renderer, "_probe_audio_params", no_params)
    monkeypatch.setattr(renderer, "_assert_uniform_audio", noop)
    with pytest.raises(renderer.FfmpegError, match="gap silence cannot be sized"):
        await renderer._concat_audio_filter(
            [tmp_path / "c0.wav", tmp_path / "c1.wav"], tmp_path / "out.wav", "wav", 250
        )


async def test_the_fades_are_indexed_by_input_not_by_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE off-by-one this shape invites: gap inputs are interleaved between the
    clips, so clip ``i`` is ffmpeg input ``2*i``. A fade built from the clip index
    would fade the GAP and leave the clip's edges raw — and the graph would still
    be valid, so only an input-index assertion can catch it."""
    cmds = _capture(monkeypatch)

    async def params(path: Path) -> tuple[int, int]:
        return 24000, 1

    async def dur(path: Path) -> float:
        return 0.5

    async def noop(paths: object) -> None:
        return None

    monkeypatch.setattr(renderer, "_probe_audio_params", params)
    monkeypatch.setattr(renderer, "_ffprobe_duration", dur)
    monkeypatch.setattr(renderer, "_assert_uniform_audio", noop)
    await renderer._concat_audio_filter(
        [tmp_path / f"c{i}.wav" for i in range(3)], tmp_path / "out.wav", "wav", 250
    )
    graph = _audio_filter(cmds[-1])
    # 3 clips + 2 gaps = inputs 0,1,2,3,4; the CLIPS are 0, 2, 4.
    assert graph.count("afade=t=in") == 3, graph
    for clip_input in ("[0:a]afade", "[2:a]afade", "[4:a]afade"):
        assert clip_input in graph, f"{clip_input} missing from {graph}"
    # The gaps are inputs 1 and 3 and must reach concat RAW — fading silence is
    # pointless, and a fade there means the indices are wrong.
    assert "[1:a]afade" not in graph and "[3:a]afade" not in graph
    assert "[e0][1:a][e1][3:a][e2]concat=n=5" in graph


# ---------------------------------------------------------------------------
# C5 — the mute envelope is a real ramp, and the maths is checked, not trusted
# ---------------------------------------------------------------------------


def test_the_mute_envelope_is_a_trapezoid_and_not_a_switch() -> None:
    """Evaluate the emitted ffmpeg expression in Python at the times that matter,
    rather than asserting on its text — the text is what a tautology looks like
    here, and an off-by-one in the envelope would be invisible to a string compare.

    ``volume``'s ``enable=`` toggles instantly, which is the click this replaces.
    The envelope must be 1 outside the window, exactly 0 across the hold, and
    strictly between at the ramp midpoints."""
    import re

    frag = renderer._ramped_mute_fragment(4.0, 9.0)
    # `aeval`, not `volume`: volume's expression support is eval=FRAME (one value
    # per ~1024-sample frame), measured as a 2-step staircase, not a ramp.
    assert frag.startswith("aeval=exprs="), frag
    assert "eval=frame" not in frag
    assert frag.count("val(") == 2 and frag.endswith(":c=same"), "one expr per stereo channel"

    envelope = re.search(r"val\(0\)\*\((.+?)\)'\|", frag)
    assert envelope is not None, frag
    expr = envelope.group(1)
    r = renderer._POSTCLIP_MUTE_RAMP_S

    def gain(t: float) -> float:
        return eval(  # noqa: S307 — evaluating OUR OWN generated expression, no input
            expr.replace("clip(", "_clip(").replace("min(", "min("),
            {"_clip": lambda x, lo, hi: max(lo, min(hi, x)), "min": min, "t": t},
        )

    # Tolerances only where binary floating point makes an exact 0/1 unreachable
    # ((4.01-4.0)/0.01 is 0.9999999999999787, not 1.0). Every bound below is a
    # value the DEFECT would miss by ~1.0, not by 1e-14.
    assert gain(0.0) == 1.0, "before the window the bus is untouched"
    assert gain(4.0) == 1.0, "the ramp STARTS at the solo window edge"
    assert gain(4.0 + r) < 1e-9, "fully muted one ramp in"
    assert gain(6.5) == 0.0, "the hold is exact digital silence"
    assert gain(9.0 - r) < 1e-9
    assert gain(9.0) > 1.0 - 1e-9, "back to full level by the far edge"
    assert gain(12.0) == 1.0, "after the window the bus is untouched"
    assert 0.0 < gain(4.0 + r / 2) < 1.0, "the ramp midpoint must be PARTIAL, not switched"
    assert 0.0 < gain(9.0 - r / 2) < 1.0


def test_a_short_solo_window_clamps_the_ramp_instead_of_overlapping() -> None:
    """Two 10 ms ramps inside a short window would overlap and dip the middle
    without ever reaching silence. The ramp is clamped to a quarter of the window,
    so it always reaches zero and always holds.

    ⚠️ HONEST SCOPE: this clamp is UNREACHABLE from the only production caller.
    ``_solo_windows`` never returns a window shorter than ``_MIN_SOLO_WINDOW_S``
    (0.05 s), whose quarter (12.5 ms) already exceeds the 10 ms ramp — so in the
    live pipeline the ramp is always the full 10 ms. It is a guard on this
    helper's own contract, tested by calling it directly, and it is documented
    that way rather than dressed up as a live path."""
    assert renderer._MIN_SOLO_WINDOW_S / 4 > renderer._POSTCLIP_MUTE_RAMP_S, (
        "if this ever inverts, the clamp becomes live and needs a pipeline test"
    )
    frag = renderer._ramped_mute_fragment(0.0, 0.02)
    assert "/0.0100" not in frag, "a 20 ms window cannot carry the full 10 ms ramp twice"
    assert "/0.0050" in frag, "clamped to window/4"
    # Degenerate windows produce no node at all rather than an invalid expression.
    assert renderer._ramped_mute_fragment(1.0, 1.0) == ""
    assert renderer._ramped_mute_fragment(2.0, 1.0) == ""
