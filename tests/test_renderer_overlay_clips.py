"""Alpha overlay clips: animated graphics composited ON TOP of the shot picture.

ffmpeg/ffprobe + asset fetch are mocked — these assert the ARGV and the
filtergraph shape, which is where every failure mode of this feature lives.

THE REGRESSION THIS FILE EXISTS FOR
-----------------------------------
The same VP9-alpha WebM decodes differently depending on which decoder ffmpeg
picks, and the wrong one fails SILENTLY (no warning, no non-zero exit) — measured
on this project's own builds::

    ffmpeg's NATIVE ``vp9`` (the default!)  first pixel fd 00 00 ff  → alpha 255, DISCARDED
    ``-c:v libvpx-vp9``                     first pixel fd 00 00 00  → alpha 0, correct

So the compositor would lay an opaque black box over the drone footage and the
render would "succeed". ``-c:v`` is a PER-INPUT option, so it is not enough for
the tokens to be somewhere in the argv — they must sit immediately before the
overlay's own ``-i``. Every assertion here is on argv ORDER for that reason.

Second trap, for anyone tempted to validate these assets: a CORRECT VP9-alpha
file reports ``pix_fmt=yuv420p``, not ``yuva420p`` — the alpha rides in WebM
``BlockAdditional`` side data (container tag ``alpha_mode=1``). A "is it
yuva420p?" check fails on every good build.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from src.api.schemas import AccentCue, OverlayClip, Shot, SplitScreen
from src.config import get_settings
from src.services import renderer

_OVL = "http://minio/lower-third.webm"
_LOGO = "http://minio/logo-bug.webm"


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    codec: str = "vp9",
    unfetchable: set[str] | None = None,
    audio_dur: float = 3.0,
) -> dict[str, Any]:
    """Fake ``_run`` / probes / fetch. ``codec`` is what the overlay codec probe
    reports — the gate that decides whether the alpha decoder is forced."""
    captured: dict[str, Any] = {"cmds": []}
    bad = unfetchable or set()

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
            return ""
        # The only ffprobe that reaches the real ``_run`` from this path is the
        # overlay codec probe (duration/has-audio are patched below).
        return f"{codec}\n"

    async def fake_probe(path: Path) -> float:
        return audio_dur

    async def fake_has_audio(path: Path) -> bool:
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url in bad:
            raise httpx.HTTPError(f"boom: {url}")
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


def _ffmpeg_cmd(captured: dict[str, Any]) -> list[str]:
    """The last real ffmpeg argv (skipping the ffprobe calls)."""
    return [c for c in captured["cmds"] if c[0] == "ffmpeg"][-1]


def _graph(cmd: list[str]) -> str:
    """The ``-filter_complex`` value, or the ``-vf`` value when on the fast path."""
    for flag in ("-filter_complex", "-vf"):
        if flag in cmd:
            return cmd[cmd.index(flag) + 1]
    raise AssertionError(f"no filter args in {cmd}")


def _input_paths(cmd: list[str]) -> list[str]:
    """Every ``-i`` operand, in input-index order."""
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-i"]


# ---------------------------------------------------------------------------
# THE decoder trap — argv ORDER, not mere presence
# ---------------------------------------------------------------------------


async def test_alpha_decoder_sits_immediately_before_the_overlay_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """``-c:v libvpx-vp9`` must be the two tokens DIRECTLY preceding the overlay's
    ``-i``. Anywhere else and ffmpeg applies it to a different input (or to the
    output), the native vp9 decoder handles the overlay, and the alpha is silently
    thrown away — an opaque box over the footage, exit code 0."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=4.0,
        video_url="http://minio/drone.mp4",
        overlay_clips=[OverlayClip(url=_OVL)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)

    ovl_path = next(p for p in _input_paths(cmd) if p.endswith("-ovl0.webm"))
    at = cmd.index(ovl_path)
    assert cmd[at - 3 : at] == ["-c:v", "libvpx-vp9", "-i"], (
        f"decoder must immediately precede the overlay -i; got {cmd[at - 4 : at + 1]}"
    )
    # ...and nowhere else: a stray forced decoder would hit another input.
    assert cmd.count("libvpx-vp9") == 1
    # The forced INPUT decoder must not leak to the output: the last ``-c:v`` in
    # the argv is the delivery encoder, set after every input.
    assert cmd[len(cmd) - cmd[::-1].index("-c:v")] == "libx264"


async def test_non_vp9_overlay_gets_no_forced_decoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """``libvpx-vp9`` can ONLY decode VP9 — forcing it on a PNG/ProRes overlay is a
    hard ffmpeg error, which would fail the whole shot. The decoder is therefore
    probe-gated, and a non-VP9 overlay is still composited, just with ffmpeg's own
    decoder (it has no alpha-in-side-data problem to begin with)."""
    captured = _install_fakes(monkeypatch, codec="png")
    shot = Shot(
        id="s1",
        seconds=4.0,
        video_url="http://minio/drone.mp4",
        overlay_clips=[OverlayClip(url=_OVL)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)
    assert "libvpx-vp9" not in cmd
    assert any(p.endswith("-ovl0.webm") for p in _input_paths(cmd))
    assert "overlay=" in _graph(cmd)


# ---------------------------------------------------------------------------
# The empty default must not move a single byte of argv
# ---------------------------------------------------------------------------


def _branch_shots(branch: str) -> tuple[Shot, Shot]:
    """One shot that OMITS ``overlay_clips`` and one that sends it empty, on the
    named visual branch. Built explicitly rather than by ``**kwargs`` splat so the
    keyword types stay checkable."""
    if branch == "clip":
        return (
            Shot(id="s1", seconds=4.0, video_url="http://minio/drone.mp4"),
            Shot(id="s1", seconds=4.0, video_url="http://minio/drone.mp4", overlay_clips=[]),
        )
    if branch == "image":
        return (
            Shot(id="s1", seconds=4.0, image_url="http://minio/slide.png"),
            Shot(id="s1", seconds=4.0, image_url="http://minio/slide.png", overlay_clips=[]),
        )
    return (
        Shot(id="s1", seconds=4.0, description="A text card"),
        Shot(id="s1", seconds=4.0, description="A text card", overlay_clips=[]),
    )


@pytest.mark.parametrize("branch", ["clip", "image", "text"])
async def test_empty_overlay_clips_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    http: httpx.AsyncClient,
    branch: str,
) -> None:
    """Not merely "no overlay filter": the ENTIRE argv of a shot that omits the
    field and one that sends ``overlay_clips=[]`` must be identical, on every
    visual branch. That is the no-op invariant the whole fleet's existing renders
    (and the shot render cache) depend on."""
    captured = _install_fakes(monkeypatch)
    without, explicit = _branch_shots(branch)

    await renderer.render_shot(without, tmp_path / "a.mp4", get_settings(), tmp_path, http)
    base = _ffmpeg_cmd(captured)[:-1]
    await renderer.render_shot(explicit, tmp_path / "b.mp4", get_settings(), tmp_path, http)
    same = _ffmpeg_cmd(captured)[:-1]

    assert same == base
    joined = " ".join(base)
    for token in ("libvpx-vp9", "yuva420p", "overlay=", "ovb0", "ovl0"):
        assert token not in joined


async def test_empty_overlay_clips_keeps_the_text_card_vf_fast_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """An empty overlay list must not itself trigger any overlay compositing.

    The text-card branch's plain ``-vf`` path existed for the no-audio-processing
    case, but 3b8d1e7 made the per-shot edge fade MANDATORY (every shot now bakes
    an ``afade`` to kill the PCM-concat click), so ``audio_frag`` is non-empty for
    any shot with duration > 2*``_SHOT_EDGE_FADE_S`` and the plain ``-vf`` path is
    no longer reachable at all — this shot now runs through ``-filter_complex``
    like every other audio-bearing shot. What an empty overlay list must still
    guarantee is what this test actually checks: no overlay compositing tokens
    leak into the graph."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s1", seconds=3.0, description="A text card", overlay_clips=[])
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)
    assert "-vf" not in cmd
    assert "-filter_complex" in cmd
    joined = " ".join(cmd)
    for token in ("libvpx-vp9", "yuva420p", "overlay=", "ovb0", "ovl0"):
        assert token not in joined


# ---------------------------------------------------------------------------
# Fast path OFF whenever an overlay is present
# ---------------------------------------------------------------------------


async def test_overlay_disables_the_vf_fast_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A bare ``-vf`` cannot reference a second input at all, so an overlay MUST
    route the shot onto ``-filter_complex``. It does so by construction: the layer
    chain makes the graph multi-chain (``;``), which is exactly the condition
    ``_video_audio_map_args`` already tests."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=3.0,
        description="A text card",
        overlay_clips=[OverlayClip(url=_OVL)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)
    assert "-vf" not in cmd
    assert "-filter_complex" in cmd
    graph = _graph(cmd)
    assert ";" in graph
    # The base chain is closed into a link and re-joined with the layer.
    assert "[ovb0];" in graph
    assert "[ovb0][ovl0]overlay=" in graph
    # The final chain stays label-less so the enclosing [0:v]…[v] wrap still closes.
    # The mandatory per-shot edge fade (3b8d1e7) now appends an audio segment
    # after ";[N:a]...[aout]", so the video chain's [v] no longer ends the whole
    # -filter_complex string — assert the boundary directly instead of the tail.
    assert ":eval=frame:repeatlast=0[v];" in graph


# ---------------------------------------------------------------------------
# Input indices with split + sfx + overlay all in play
# ---------------------------------------------------------------------------


async def test_input_indices_survive_split_plus_overlay_plus_sfx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """0=primary, 1=narration, 2=split second source, 3=overlay, 4=SFX.

    The overlay ``-i`` is appended BEFORE ``_video_audio_map_args`` appends the SFX
    inputs, so the SFX keep the indices the audio graph labels them with. Append it
    afterwards and every ``[N:a]`` in the amix silently points at the wrong stream —
    which is why this asserts the labels, not just the input count."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=4.0,
        video_url="http://minio/drone.mp4",
        audio_url="http://minio/vo.wav",
        split=SplitScreen(layout="h", second_source_url="http://minio/second.mp4"),
        sfx=[AccentCue(url="http://minio/ding.wav", at_seconds=1.0)],
        overlay_clips=[OverlayClip(url=_OVL)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)

    paths = _input_paths(cmd)
    assert len(paths) == 5
    assert paths[1].endswith("-audio")
    assert paths[2].endswith("-split.mp4")
    assert paths[3].endswith("-ovl0.webm")
    assert paths[4].endswith("-sfx0")

    graph = _graph(cmd)
    assert "[2:v]" in graph  # split half
    assert "[3:v]" in graph  # overlay layer
    assert "[4:a]" in graph  # accent SFX — the label that silently breaks if we reorder
    assert "[1:a]" in graph  # narration


async def test_two_overlays_take_consecutive_indices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Two overlays stack: each closes the running chain and re-joins, so the
    second composites on top of the first."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=4.0,
        image_url="http://minio/slide.png",
        overlay_clips=[OverlayClip(url=_OVL), OverlayClip(url=_LOGO, x="W-w-64", y="64")],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)
    graph = _graph(cmd)
    assert "[2:v]" in graph and "[3:v]" in graph
    assert graph.index("[ovb0][ovl0]overlay=") < graph.index("[ovb1][ovl1]overlay=")
    assert "overlay=x=W-w-64:y=64:eval=frame:repeatlast=0" in graph
    assert cmd.count("libvpx-vp9") == 2


# ---------------------------------------------------------------------------
# retime_with_speed
# ---------------------------------------------------------------------------


async def test_retime_with_speed_divides_start_and_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The overlay is a SEPARATE input and is never touched by
    ``_apply_shot_speed``'s ``setpts`` (which only retimes the primary chain), so a
    cue authored on the SOURCE timeline must be divided by ``speed`` here. Get this
    wrong and the lower-third drifts off the frame it was authored for."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=8.0,
        video_url="http://minio/drone.mp4",
        audio_url="http://minio/vo.wav",
        speed=2.0,
        overlay_clips=[OverlayClip(url=_OVL, start_seconds=4.0, duration_seconds=2.0)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    graph = _graph(_ffmpeg_cmd(captured))
    assert "trim=duration=1" in graph
    assert "setpts=PTS-STARTPTS+2/TB" in graph
    # The PRIMARY still carries its own retime — the two are independent.
    assert "setpts=PTS/2" in graph


async def test_retime_with_speed_false_leaves_the_times_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Opting out pins the overlay to RENDERED wall-clock time instead — a logo bug
    that must appear 4s into the delivered segment however the picture was retimed."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=8.0,
        video_url="http://minio/drone.mp4",
        audio_url="http://minio/vo.wav",
        speed=2.0,
        overlay_clips=[
            OverlayClip(url=_OVL, start_seconds=4.0, duration_seconds=2.0, retime_with_speed=False)
        ],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    graph = _graph(_ffmpeg_cmd(captured))
    assert "trim=duration=2" in graph
    assert "setpts=PTS-STARTPTS+4/TB" in graph


async def test_speed_one_is_identical_under_either_retime_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """At speed=1.0 the flag cannot matter — a divide by 1 is the identity."""
    captured = _install_fakes(monkeypatch)
    graphs: list[str] = []
    for flag in (True, False):
        shot = Shot(
            id="s1",
            seconds=4.0,
            image_url="http://minio/slide.png",
            overlay_clips=[
                OverlayClip(url=_OVL, start_seconds=1.5, retime_with_speed=flag),
            ],
        )
        await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
        graphs.append(_graph(_ffmpeg_cmd(captured)))
    assert graphs[0] == graphs[1]
    assert "setpts=PTS-STARTPTS+1.5/TB" in graphs[0]


# ---------------------------------------------------------------------------
# Layer chain shape
# ---------------------------------------------------------------------------


async def test_layer_chain_carries_alpha_scales_and_dims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """``format=yuva420p`` is what materialises the alpha plane in the graph (the
    forced decoder only makes it available); ``fit`` scales inside the canvas
    without padding — the alpha is the framing; ``colorchannelmixer=aa`` SCALES the
    authored alpha rather than replacing it, so soft edges stay soft."""
    captured = _install_fakes(monkeypatch)
    settings = get_settings()
    shot = Shot(
        id="s1",
        seconds=4.0,
        image_url="http://minio/slide.png",
        overlay_clips=[OverlayClip(url=_OVL, opacity=0.5, x="(W-w)/2", y="H-h-80")],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    graph = _graph(_ffmpeg_cmd(captured))
    assert (
        f"scale={settings.render_width}:{settings.render_height}"
        ":force_original_aspect_ratio=decrease" in graph
    )
    assert "format=yuva420p" in graph
    assert "colorchannelmixer=aa=0.5" in graph
    assert "overlay=x=(W-w)/2:y=H-h-80:eval=frame:repeatlast=0" in graph
    # No pad: padding an overlay would paint opaque bars over the footage.
    assert "pad=" not in graph.split("[ovl0]")[0].split("[2:v]")[-1]


async def test_native_scale_and_full_opacity_emit_no_extra_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """``scale='native'`` composites at the authored pixel size (no resize), and
    ``opacity=1.0`` (the default) is the identity, so neither emits a
    colorchannelmixer. It DOES still emit a same-size ``scale=iw:ih`` node with
    the derived colour matrix ahead of ``format=yuva420p`` — that conversion is
    an RGBA->YUV colour operation through swscale regardless of resize, so it
    needs the matrix exactly like the 'fit' branch's real resize does (see
    ``renderer._overlay_clip_chain``)."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=4.0,
        image_url="http://minio/slide.png",
        overlay_clips=[OverlayClip(url=_OVL, scale="native")],
    )
    settings = get_settings()
    await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    graph = _graph(_ffmpeg_cmd(captured))
    expected_matrix = renderer._sws_flags(settings)
    assert (
        f"[2:v]scale=iw:ih:{expected_matrix},format=yuva420p,setpts=PTS-STARTPTS+0/TB[ovl0]"
        in graph
    )
    assert "colorchannelmixer" not in graph


# ---------------------------------------------------------------------------
# Degradation + fetch dedupe
# ---------------------------------------------------------------------------


async def test_unfetchable_overlay_drops_the_overlay_not_the_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """An expired presigned URL / 404 on a DECORATION must never fail a compile —
    the shot renders without it, exactly like an unfetchable SFX cue."""
    captured = _install_fakes(monkeypatch, unfetchable={_OVL})
    shot = Shot(
        id="s1",
        seconds=4.0,
        image_url="http://minio/slide.png",
        overlay_clips=[OverlayClip(url=_OVL)],
    )
    duration = await renderer.render_shot(
        shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http
    )
    assert duration > 0
    cmd = _ffmpeg_cmd(captured)
    joined = " ".join(cmd)
    assert "overlay=" not in joined
    assert "libvpx-vp9" not in joined
    assert len(_input_paths(cmd)) == 2  # image + synthesised silence only


async def test_one_bad_overlay_keeps_the_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Degradation is per-overlay, and the surviving one still gets index 2."""
    captured = _install_fakes(monkeypatch, unfetchable={_OVL})
    shot = Shot(
        id="s1",
        seconds=4.0,
        image_url="http://minio/slide.png",
        overlay_clips=[OverlayClip(url=_OVL), OverlayClip(url=_LOGO)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = _ffmpeg_cmd(captured)
    graph = _graph(cmd)
    assert "[2:v]" in graph
    assert "[3:v]" not in graph
    assert "[ovb1]" not in graph
    assert cmd.count("libvpx-vp9") == 1


async def test_overlay_reused_across_shots_downloads_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The whole point of routing overlays through ``_fetch_asset``: a logo bug is
    by definition the SAME url on every shot of the film."""
    downloads: list[str] = []

    async def counting_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        downloads.append(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    _install_fakes(monkeypatch)
    monkeypatch.setattr(renderer, "_fetch", counting_fetch)
    cache = renderer.SourceFetchCache(tmp_path)
    for i in range(3):
        shot = Shot(
            id=f"s{i}",
            seconds=2.0,
            image_url="http://minio/slide.png",
            overlay_clips=[OverlayClip(url=_LOGO)],
        )
        await renderer.render_shot(
            shot,
            tmp_path / f"seg{i}.mp4",
            get_settings(),
            tmp_path,
            http,
            source_cache=cache,
        )
    await cache.aclose()
    assert downloads.count(_LOGO) == 1


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


def test_more_than_two_overlays_is_rejected() -> None:
    """Two is the real product shape (a lower-third + a logo bug); each extra one
    is another decoded VP9 input and another full-canvas overlay pass per frame."""
    with pytest.raises(ValidationError):
        Shot(
            id="s1",
            seconds=2.0,
            overlay_clips=[OverlayClip(url=_OVL), OverlayClip(url=_LOGO), OverlayClip(url=_OVL)],
        )


def test_renderer_cap_matches_the_schema_cap() -> None:
    """The renderer keeps its own bound so ``render_shot`` is safe when a ``Shot``
    is built directly (tests, the thumbnail path) rather than via the API."""
    from src.api import schemas

    assert renderer._OVERLAY_CLIP_MAX_PER_SHOT == schemas._MAX_OVERLAY_CLIPS


@pytest.mark.parametrize(
    "expr",
    [
        "0:eval=init",  # a second filter OPTION
        "0,drawtext=text=pwn",  # a whole extra FILTER in the chain
        "0[x];[x]null",  # a new graph chain / link label
        "0'",  # quote breakout
        "0\\",  # escape breakout
    ],
)
def test_overlay_position_expressions_cannot_break_out_of_the_filtergraph(expr: str) -> None:
    """``x``/``y`` land INLINE in a ``-filter_complex``. They are genuinely
    expressions (``(W-w)/2``), so they cannot be locked down to digits — but ``:``
    ``,`` ``;`` ``[`` ``]`` and quotes are all filtergraph metacharacters and are
    refused at the boundary, where it is cheap."""
    with pytest.raises(ValidationError):
        OverlayClip(url=_OVL, x=expr)
    with pytest.raises(ValidationError):
        OverlayClip(url=_OVL, y=expr)


def test_ordinary_position_expressions_are_accepted() -> None:
    """The real ones must still pass — a regex that rejects these is useless."""
    for expr in ["0", "64", "(W-w)/2", "H-h-80", "main_w-overlay_w-24", "w*0.05"]:
        assert OverlayClip(url=_OVL, x=expr, y=expr).x == expr
