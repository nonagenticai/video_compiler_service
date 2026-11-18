"""A/V quality wave 3: production CFR pin + universal cadence re-assertion.

Root cause: ``cfr_ok=false`` on ~46% of masters, live-verified over 7 days.
``-fps_mode:v cfr`` was gated behind the ``deterministic`` render tag, and
production compiles always pass ``deterministic=False`` — so no encode site
in the service actually pinned CFR. Separately, ``_video_codec_args`` emitted
no ``-r``, and only 2 of its 6 callers appended one by hand; the text/drawtext
shot branch (``concat_shots`` is ``-c copy``, so per-segment drift survives
into the master) was one of the 4 that did not.

Fix: hoist ``-fps_mode:v cfr`` out of the ``deterministic`` branch (software/libx264
path only — meaningless on the hardware encoders) and make ``-r render_fps``
unconditional inside ``_video_codec_args`` itself, so every one of its 6
callers — including the text branch — gets both without having to ask.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer


def _settings(**overrides: object) -> Any:
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# 3a — the CFR pin must reach production (deterministic=False)
# ---------------------------------------------------------------------------


def test_cfr_pin_applies_in_production_non_deterministic_mode() -> None:
    """The headline fix: a plain production call (deterministic=False, the
    only mode production ever uses) must still carry -fps_mode:v cfr on the
    software encoder path."""
    args = renderer._video_codec_args(_settings())
    assert "-fps_mode:v" in args
    assert args[args.index("-fps_mode:v") + 1] == "cfr"


def test_cfr_pin_stays_off_hardware_encoders() -> None:
    """-fps_mode:v is documented as meaningless on videotoolbox/nvenc; the pin
    must stay scoped to the software branch it always lived in."""
    for encoder in ("h264_videotoolbox", "h264_nvenc"):
        settings = _settings(render_encoder=encoder, render_hw_bitrate="9M")
        args = renderer._video_codec_args(settings, deterministic=False)
        assert "-fps_mode:v" not in args
        args_det = renderer._video_codec_args(settings, deterministic=True)
        assert "-fps_mode:v" not in args_det


def test_deterministic_still_adds_threads_and_bitexact_only() -> None:
    """deterministic=True still adds -threads 1 / -flags +bitexact on the
    software path; -fps_mode:v is no longer conditional on it (it is always on
    that path now), so this only checks the tokens that ARE still gated."""
    off = renderer._video_codec_args(_settings(), deterministic=False)
    on = renderer._video_codec_args(_settings(), deterministic=True)
    assert "-threads" not in off
    assert "-threads" in on and on[on.index("-threads") + 1] == "1"
    assert "-flags" not in off
    assert on[on.index("-flags") + 1] == "+bitexact"
    # the CFR pin is present in BOTH — it is no longer the deterministic flag.
    assert "-fps_mode:v" in off and "-fps_mode:v" in on


# ---------------------------------------------------------------------------
# 3b — -r render_fps is now unconditional from _video_codec_args
# ---------------------------------------------------------------------------


def test_video_codec_args_always_emits_render_fps() -> None:
    settings = _settings(render_fps=24)
    for deterministic in (False, True):
        args = renderer._video_codec_args(settings, deterministic=deterministic)
        assert "-r" in args, args
        assert args[args.index("-r") + 1] == "24"


def test_video_codec_args_emits_render_fps_on_hardware_path_too() -> None:
    settings = _settings(render_encoder="h264_videotoolbox", render_hw_bitrate="9M", render_fps=25)
    args = renderer._video_codec_args(settings)
    assert args[args.index("-r") + 1] == "25"


def test_grade_video_args_does_not_duplicate_r_flag() -> None:
    """_grade_video_args used to append its OWN -r after _video_codec_args;
    now the helper is the single source, so -r must appear exactly once."""
    args = renderer._grade_video_args("eq=contrast=1.1", _settings())
    assert args.count("-r") == 1


# ---------------------------------------------------------------------------
# 3c — the text/drawtext shot branch reaches the same cadence pin
# ---------------------------------------------------------------------------


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str]) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return 3.0

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
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


@pytest.mark.asyncio
async def test_text_card_segment_encode_gets_cadence_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The drawtext/text-card branch (no visual asset at all) builds no
    _scale_pad_filter and previously reached ffmpeg with no fps= filter and
    no -r — the exact shape that lets avg_frame_rate/r_frame_rate disagree at
    the concat join. It must now carry -r render_fps from the encode ladder."""
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="txt1", seconds=3.0, description="Cadence must hold")
    settings = get_settings().model_copy(update={"render_fps": 30})
    await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    cmd = captured["cmds"][-1]
    assert "drawtext" in " ".join(cmd)  # confirms we hit the text branch
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == "30"
    assert "-fps_mode:v" in cmd
    assert cmd[cmd.index("-fps_mode:v") + 1] == "cfr"


def test_copy_branch_reasserts_the_colour_triplet() -> None:
    """⛔ THE DEFECT THIS GUARDS — measured on three delivered masters (2026-09-13).

    Each carried ``color_space=bt709`` with ``color_primaries`` and
    ``color_transfer`` BOTH ``unknown``, which is the whole reason
    ``colorspace_ok`` read false on 16/16 masters. The tags are not lost at the
    encode: ``_video_codec_args`` emits all three, a bench with its verbatim args
    tags correctly, and a ``-f concat -c copy`` join preserves them. They are lost
    at the LAST pass to touch video — this one — whose ``-c:v copy`` map rides the
    same invocation as an active audio ``-filter_complex`` and depends on the
    muxer propagating codecpar. Production runs ffmpeg 7.1.5; 4.4 propagates
    intact, so the behaviour is version-dependent and must not be relied on.

    The flags are legal alongside ``-c:v copy`` — verified against real ffmpeg:
    an entirely untagged input re-muxed with them probes as a full bt709 triplet,
    with no re-encode.
    """
    args = renderer._grade_video_args("", _settings())
    assert "-c:v" in args and "copy" in args, "the no-grade path must stay a stream copy"
    for flag in ("-color_primaries", "-color_trc", "-colorspace"):
        assert flag in args, f"{flag} missing — the delivered master ships untagged"
    # Still a copy: re-tagging must not have turned this into a re-encode.
    assert "libx264" not in args
    assert "-crf" not in args
