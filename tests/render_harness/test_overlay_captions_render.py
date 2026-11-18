"""Real-ffmpeg render-harness tests for OVERLAY (soft mov_text) captions.

Sibling of ``test_captions_render.py`` (burned-in). Overlay mode does NOT burn
the cues into the picture — it muxes them as a selectable ``mov_text`` soft
subtitle track via a follow-on ``-c copy`` remux (no video re-encode). These
drive an ACTUAL whole-timeline ``compile_shots`` and interrogate the DELIVERED
mp4: overlay adds exactly one ``mov_text`` stream and leaves the caption band
UNBURNED; burned_in adds NO soft track; and no-captions is byte-identical.

╔══════════════════════════════════════════════════════════════════════════╗
║ REVERT-PROOF: ``test_overlay_muxes_soft_track_no_burn`` MUST FAIL if the  ║
║ ``_mux_soft_captions`` call is dropped from ``compile_shots`` — with no    ║
║ remux the delivered file carries ZERO subtitle streams, so the            ║
║ subtitle-codecs assertion (== ["mov_text"]) collapses.                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Reuses the burned-in module's clip/frame/band helpers so the two suites share
one pixel-delta convention. MinIO/storage is NOT exercised; ``compile_shots``
returns a local path.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.api.schemas import CaptionCue, Shot, ShotCaptions
from src.services import renderer
from tests.render_harness.harness import requires_filters
from tests.render_harness.test_captions_render import (
    _ABSENT,
    _BAND_Y0,
    _H,
    _band_change,
    _frame_rgb_at,
    _settings,
    _solid_clip,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
    # Presence of the binary is not capability: a Homebrew/macOS ffmpeg is on
    # PATH but built without libass/libfreetype, so this filter is absent and
    # the render dies with a bare `exit=8: Filter not found`. Skip with a reason
    # instead. See tests/render_harness/harness.py::requires_filters.
    requires_filters("ass"),
]


async def _subtitle_codecs(path: Path) -> list[str]:
    """ffprobe the delivered file's subtitle streams; return their codec names.

    ``renderer._run`` returns captured stdout as a str (default, no
    ``capture_stderr``) — the same wrapper every harness probe reuses — so the
    csv output is parsed directly, one codec per non-empty line."""
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


async def _compile(
    shots: list[Shot],
    tmp_path: Path,
    *,
    captions: list[ShotCaptions] | None = None,
    captions_render: str = "burned_in",
) -> Path:
    """Run a full whole-timeline compile and return the delivered file."""
    final_path, _, _, _, _ = await renderer.compile_shots(
        shots,
        _settings(tmp_path),
        captions=captions,
        captions_render=captions_render,
    )
    return final_path


async def test_overlay_muxes_soft_track_no_burn(tmp_path: Path) -> None:
    """OVERLAY = SOFT TRACK, NO BURN. Two shots, one cue on shot 0 (film
    0.2..0.8), compiled with ``captions_render="overlay"``. The delivered file
    carries exactly one ``mov_text`` soft subtitle track, and the caption band
    at film 0.5 is NOT burned (matches the caption-less control within noise).
    REVERT-PROOF: drop ``_mux_soft_captions`` and the mov_text stream vanishes."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(
            shot_index=0, cues=[CaptionCue(text="HELLO OVERLAY WORLD", start=0.2, end=0.8)]
        )
    ]
    overlay = await _compile(shots, tmp_path / "ov", captions=caps, captions_render="overlay")
    control = await _compile(shots, tmp_path / "ctl", captions=None)

    # Exactly one selectable soft subtitle track, codec mov_text.
    assert await _subtitle_codecs(overlay) == ["mov_text"], "overlay did not mux one mov_text track"

    # The picture is NOT burned: the caption band at the cue's film time agrees
    # with the caption-less control (only re-encode/remux noise).
    fo = await _frame_rgb_at(overlay, 0.5, tmp_path / "fo.rgb")
    fk = await _frame_rgb_at(control, 0.5, tmp_path / "fk.rgb")
    assert _band_change(fo, fk, _BAND_Y0, _H) < _ABSENT, "overlay burned the caption into the band"


async def test_omitted_captions_render_defaults_to_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEFAULT = OVERLAY. Owner rule: subtitles are NEVER burned into the picture
    unless a caller asks for it. A compile with cues and NO ``captions_render``
    must (a) deliver exactly one ``mov_text`` soft track and (b) never put the
    libass ``ass=`` burn filter into ANY ffmpeg command it runs. FAIL-BEFORE: with
    the old ``"burned_in"`` default the burn filter appears and the delivered file
    has zero subtitle streams."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(shot_index=0, cues=[CaptionCue(text="DEFAULT CAPTION", start=0.2, end=0.8)])
    ]
    cmds: list[list[str]] = []
    real_run = renderer._run

    async def _recording_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(list(cmd))
        return await real_run(cmd, capture_stderr=capture_stderr)

    monkeypatch.setattr(renderer, "_run", _recording_run)
    final_path, _, _, _, _ = await renderer.compile_shots(
        shots, _settings(tmp_path / "def"), captions=caps
    )

    burn_cmds = [c for c in cmds if any("ass=filename=" in arg for arg in c)]
    assert burn_cmds == [], "omitted captions_render burned the captions into the picture"
    assert await _subtitle_codecs(final_path) == ["mov_text"], (
        "omitted captions_render did not mux a mov_text soft track"
    )


async def test_burned_in_has_no_soft_track(tmp_path: Path) -> None:
    """BURNED_IN = NO SOFT TRACK. Same shots+cue, ``captions_render="burned_in"``
    → the cue is baked into the picture and the delivered file carries ZERO
    subtitle streams (nothing to toggle)."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(shot_index=0, cues=[CaptionCue(text="BURNED CAPTION", start=0.2, end=0.8)])
    ]
    burned = await _compile(shots, tmp_path / "bn", captions=caps, captions_render="burned_in")
    control = await _compile(shots, tmp_path / "ctl", captions=None)

    assert await _subtitle_codecs(burned) == [], "burned_in leaked a soft subtitle track"

    # And it really is burned into the band (vs the caption-less control).
    fb = await _frame_rgb_at(burned, 0.5, tmp_path / "fb.rgb")
    fk = await _frame_rgb_at(control, 0.5, tmp_path / "fk.rgb")
    assert _band_change(fb, fk, _BAND_Y0, _H) > _ABSENT, "burned_in did not burn the caption"


async def test_no_captions_no_soft_track_byte_identical(tmp_path: Path) -> None:
    """NO CAPTIONS = NO SOFT TRACK, BYTE-IDENTICAL. ``captions=None`` never muxes
    a soft track regardless of ``captions_render`` (the mode is ignored when
    there are no cues), and the delivered file is sha256-identical to a plain
    compile — the overlay feature is invisible when unused. Mirrors
    ``test_captions_render.test_no_captions_is_byte_identical``."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    plain = await _compile(shots, tmp_path / "plain", captions=None)
    # captions=None + overlay mode must still take the plain path (mode ignored).
    overlay_off = await _compile(
        shots, tmp_path / "ovoff", captions=None, captions_render="overlay"
    )

    assert await _subtitle_codecs(plain) == [], "a caption-less compile grew a soft track"
    assert await _subtitle_codecs(overlay_off) == [], "overlay-with-no-captions muxed a soft track"
    assert (
        hashlib.sha256(plain.read_bytes()).hexdigest()
        == hashlib.sha256(overlay_off.read_bytes()).hexdigest()
    ), "captions=None + overlay perturbed the -c copy fast path"
