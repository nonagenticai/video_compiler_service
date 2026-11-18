"""Command/graph-string + schema tests for the VC-5 finish profile (letterbox).

ffmpeg is NOT invoked: these assert the filtergraph SHAPE — that an absent
``finish`` leaves every filter string EXACTLY as it was (the byte-identical
no-op invariant), and that ``letterbox_color`` overrides the pad/canvas colour
at every site that paints bars (contain scale/pad, scroll, synthetic text-card
source) while the crop-based ``cover`` fit — which has no bars — is untouched.
The delivered pixels are proven by ``tests/render_harness/test_letterbox_render.py``.

A ``broadcast_safe`` legal-luma clamp was designed and CUT (a pre-encode lutyuv
cannot guarantee a decoded 16..235 master — libx264 ringing around text puts the
illegal values back, worse at higher CRF). There is deliberately no test for it:
the feature does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from src.api.routes.compile import _settings_for_finish
from src.api.schemas import CompileRequest, FinishProfile, Shot
from src.config import Settings, get_settings
from src.services import renderer
from src.services.renderer import _scale_pad_filter, _scroll_filter

_BRAND = "#101820"

# One shot of every visual kind — every chain the pad/canvas colour can reach.
_SHOT_KINDS = [
    Shot(id="clip", seconds=2.0, video_url="clip.mp4"),
    Shot(id="image", seconds=2.0, image_url="img.png"),
    Shot(id="scroll", seconds=2.0, image_url="page.png", fit="scroll"),
    Shot(id="card", seconds=2.0, description="Hello"),
]


def _branded() -> Settings:
    return get_settings().model_copy(update={"render_letterbox_color": _BRAND})


# --------------------------------------------------------------------------
# Absent finish ⇒ byte-identical filter strings (the hard requirement)
# --------------------------------------------------------------------------


def test_default_scale_pad_filter_is_the_exact_pre_vc5_string() -> None:
    """Pin the DEFAULT contain chain literally: any drift (a swapped colour, a
    stray filter) breaks the byte-identical no-op invariant.

    The expected string now carries the frame-exactness rebase
    (``setpts=PTS-STARTPTS``, appended by ``_cadence_fragment``). That is a
    deliberate UPDATE of what the correct chain is, not a relaxation — the
    pre-fix chain delivered one frame short on any mid-frame ``-ss`` (a real reel
    shipped 685 frames against a 690-frame plan). Still a full equality on the
    whole string."""
    s = get_settings()
    w, h = s.render_width, s.render_height
    # swscale flags are pinned deliberately (lanczos: +9.8% retained detail on
    # a real 1920->2560 upscale vs the bicubic default); this guard still pins
    # the EXACT string, it just pins the new one.
    assert _scale_pad_filter(s, caption=None) == (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos+accurate_rnd+full_chroma_int"
        f":in_color_matrix=bt709:out_color_matrix=bt709,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={s.render_bg_color},"
        f"setsar=1,fps={s.render_fps},setpts=PTS-STARTPTS,format=yuv420p"
    )


def test_default_scroll_filter_keeps_the_service_bg_colour() -> None:
    s = get_settings()
    assert f"color={s.render_bg_color}" in _scroll_filter(s, duration=8.0)


@pytest.mark.parametrize("shot", _SHOT_KINDS, ids=lambda s: s.id)
async def test_default_chain_keeps_the_service_bg_colour_for_any_shot_type(
    shot: Shot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No finish profile ⇒ not one filter string moves, on any shot type."""
    s = get_settings()
    cmd = " ".join(await _shot_cmd(shot, s, tmp_path, monkeypatch))
    assert _BRAND not in cmd
    assert s.render_bg_color in cmd


# --------------------------------------------------------------------------
# letterbox_color overrides the pad / canvas colour
# --------------------------------------------------------------------------


def test_letterbox_color_overrides_the_pad_colour() -> None:
    out = _scale_pad_filter(_branded(), caption=None, fit="contain")
    assert f"color={_BRAND}" in out
    assert get_settings().render_bg_color not in out


def test_letterbox_color_overrides_the_scroll_pad_colour() -> None:
    assert f"color={_BRAND}" in _scroll_filter(_branded(), duration=5.0)


@pytest.mark.parametrize("shot", _SHOT_KINDS, ids=lambda s: s.id)
async def test_letterbox_color_reaches_every_shot_chain(
    shot: Shot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clip, image, scroll (pad ``color=``) and text card (lavfi ``color=c=``):
    the brand colour replaces the service bg everywhere bars/canvas are painted."""
    cmd = " ".join(await _shot_cmd(shot, _branded(), tmp_path, monkeypatch))
    assert f"color={_BRAND}" in cmd or f"color=c={_BRAND}" in cmd
    assert get_settings().render_bg_color not in cmd


def test_cover_fit_has_no_pad_so_letterbox_colour_is_moot() -> None:
    """Cover fills the canvas (crop, no pad) — the brand colour has nothing to
    paint, and must not smuggle a colour= option into the chain."""
    assert "color=" not in _scale_pad_filter(_branded(), caption=None, fit="cover")


# --------------------------------------------------------------------------
# Request → settings plumbing (_settings_for_finish)
# --------------------------------------------------------------------------


def test_settings_for_finish_none_returns_settings_unchanged() -> None:
    s = get_settings()
    assert _settings_for_finish(s, None) is s


def test_settings_for_finish_all_default_profile_is_a_no_op() -> None:
    s = get_settings()
    assert _settings_for_finish(s, FinishProfile()) is s


def test_settings_for_finish_folds_the_letterbox_colour() -> None:
    out = _settings_for_finish(get_settings(), FinishProfile(letterbox_color=_BRAND))
    assert out.render_letterbox_color == _BRAND


# --------------------------------------------------------------------------
# FinishProfile schema validation
# --------------------------------------------------------------------------


def test_finish_profile_default_is_inert() -> None:
    assert FinishProfile().letterbox_color is None


def test_compile_request_finish_defaults_to_none() -> None:
    assert CompileRequest(shots=[Shot(id="a", seconds=1.0)]).finish is None


def test_compile_request_accepts_a_finish_profile() -> None:
    body = CompileRequest(
        shots=[Shot(id="a", seconds=1.0)], finish=FinishProfile(letterbox_color=_BRAND)
    )
    assert body.finish is not None
    assert body.finish.letterbox_color == _BRAND


@pytest.mark.parametrize("value", ["#101820", "#ABCDEF", "#abcdef", "#101820ff"])
def test_finish_profile_accepts_valid_hex(value: str) -> None:
    assert FinishProfile(letterbox_color=value).letterbox_color == value


@pytest.mark.parametrize(
    "value",
    [
        "101820",  # no '#'
        "#10182",  # too short
        "#1018200",  # 7 digits
        "#GGGGGG",  # not hex
        "navy",  # colour name — not accepted, the value is interpolated
        "0x101820",  # ffmpeg 0x form — not accepted; keep ONE shape
        # Filtergraph-injection attempts: the pattern is the guard, since this
        # value lands inside the pad `color=` option.
        "#101820,drawbox=t=fill",
        "#101820:x=0",
        "",
    ],
)
def test_finish_profile_rejects_bad_hex(value: str) -> None:
    with pytest.raises(ValidationError):
        FinishProfile(letterbox_color=value)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


async def _shot_cmd(
    shot: Shot, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Render one shot with ffmpeg/ffprobe/fetch mocked out; return the ffmpeg argv.

    The synthetic ``color=`` text-card source is assembled INLINE in
    ``render_shot`` (not in a standalone helper), so the command string is the
    only seam that shows it. Same fake-installer shape as
    ``tests/test_renderer_speed_xfade.py``.
    """
    captured: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured.append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return 3.0

    async def fake_has_audio(path: Path) -> bool:
        return True

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        return Path(url)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    # The ENCODE command, not merely the first _run: a clip shot now runs a
    # read-only ffprobe of its source's video-stream duration first (see
    # renderer._ffprobe_source_duration), and this helper's callers assert on
    # filter strings, which only the ffmpeg invocation carries.
    return next(cmd for cmd in captured if cmd[0] == "ffmpeg")
