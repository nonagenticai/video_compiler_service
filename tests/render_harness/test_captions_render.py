"""Real-ffmpeg render-harness tests for burned-in captions (VC-6).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_grade_render.py`` / ``test_directed_footage_render.py``. These drive an
ACTUAL whole-timeline ``compile_shots`` with captions and interrogate the
DELIVERED mp4's pixels: the string tests in ``tests/test_captions_ass.py`` prove
the ASS document shape; THESE prove the caption is really burned into the frame,
at the right FILM time (offset + ÷speed), that a caption-less compile stays
byte-identical, and that a grade + captions still cost ONE combined re-encode.

╔══════════════════════════════════════════════════════════════════════════╗
║ REVERT-PROOF: ``test_caption_is_burned_in`` (a), ``…offset_and_speed`` (c) ║
║ and ``…grade_and_captions_one_pass`` (d) MUST FAIL if ``captions_vf`` is   ║
║ dropped from the combined whole-film ``-vf`` (or the caption re-encode is  ║
║ skipped). With no ass filter applied the captioned render == the control   ║
║ in the caption band, so every band-delta assertion collapses to noise and  ║
║ fails. That is the whole point of this file.                               ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━ API ASSUMPTION (reconcile with the implementer) ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Captions ride the SAME whole-film grade re-encode via a new keyword-only
``captions: list[ShotCaptions] | None = None`` on ``renderer.compile_shots``.
``captions=None`` (or ``[]``) leaves the ``-c copy`` fast path byte-identical.
Only ``_compile`` below (a single reconciliation point) constructs that call.
Shot bus offsets are NOT exposed by ``compile_shots``; case (c) reconstructs
them from the known clip durations it feeds (shot 0 renders 2.0s at speed 1.0,
so shot 1's on-screen bus window starts at film t=2.0).

MinIO/storage is NOT exercised: ``compile_shots`` returns a local path.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.api.schemas import (
    CaptionCue,
    CaptionWord,
    FinishProfile,
    GradeSpec,
    Shot,
    ShotCaptions,
)
from src.config import Settings, get_settings
from src.services import renderer
from src.services.measure import measure_shot_stats
from tests.render_harness.harness import requires_filters

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

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}
# Bottom third of the frame — the caption band. A solid-colour source keeps the
# re-encode noise here near zero so the caption's changed pixels stand alone.
_BAND_Y0 = int(_H * 0.66)
# A caption line paints hundreds+ of near-white outline pixels; solid-source
# re-encode noise in the same band is a handful. 300/150 sit safely between.
_PRESENT = 300
_ABSENT = 150


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={**_SMALL, "master_enabled": True, "work_dir": str(tmp_path)}
    )


async def _compile(
    shots: list[Shot],
    tmp_path: Path,
    *,
    finish: FinishProfile | None = None,
    captions: list[ShotCaptions] | None = None,
    caption_mood: str | None = None,
    karaoke: bool = False,
    captions_render: str = "burned_in",
) -> Path:
    """Run a full whole-timeline compile and return the delivered file.

    Single reconciliation point for the API assumption at the top of the module:
    captions ride ``compile_shots(..., captions=...)`` on the whole-film pass.
    This module tests the BURN, so it requests ``captions_render="burned_in"``
    explicitly — the service default is ``"overlay"`` (never burned by default).
    """
    final_path, _, _, _, _ = await renderer.compile_shots(
        shots,
        _settings(tmp_path),
        finish=finish,
        captions=captions,
        captions_render=captions_render,
        caption_mood=caption_mood,
        captions_karaoke=karaoke,
    )
    return final_path


async def _solid_clip(path: Path, *, color: str = "0x808080", duration: float = 1.0) -> Path:
    """A solid-colour clip (+ a quiet sine) — a clean background whose re-encode
    is near-lossless, so a caption's pixels are the only material change in the
    band. (``make_test_clip`` uses moving colour bars, too busy for a clean
    caption-vs-control pixel delta.)"""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={_W}x{_H}:d={duration}:r={_FPS}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def _colour_clip(path: Path, *, duration: float = 1.0) -> Path:
    """A colour-bars clip (testsrc) — has chroma for a grade's SATAVG to move."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={_W}x{_H}:rate={_FPS}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def _frame_rgb_at(path: Path, t: float, out: Path) -> bytes:
    """Decode the frame at film time ``t`` of ``path`` to raw rgb24 bytes."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(out),
        ]
    )
    return out.read_bytes()


def _band_change(a: bytes, b: bytes, y0: int, y1: int) -> int:
    """Count rgb24 bytes that changed by >40 between two _Wx_H frames, over rows
    ``[y0, y1)`` (a horizontal band). A burned caption lights up hundreds+; a
    caption-free band changes only by re-encode noise."""
    n = 0
    lo = y0 * _W * 3
    hi = min(y1, _H) * _W * 3
    hi = min(hi, len(a), len(b))
    for i in range(lo, hi):
        if abs(a[i] - b[i]) > 40:
            n += 1
    return n


def _strong_grade() -> FinishProfile:
    """A strong, deterministic grade (contrast + saturation + split-tone); grain/
    glow left at identity so the picture delta is attributable and the render
    stays deterministic. Mirrors ``test_grade_render._strong_grade``."""
    return FinishProfile(
        grade=GradeSpec(
            enabled=True,
            contrast=1.4,
            saturation=1.35,
            shadow_tint="#0B1437",
            highlight_tint="#EA580C",
            grade_intensity=1.0,
            broadcast_safe=False,
        )
    )


async def test_caption_is_burned_in(tmp_path: Path) -> None:
    """(a) VISIBLE BURN. Two shots, one caption cue on shot 0 (film 0.2..0.8):
    the delivered frame at the cue's film time differs from the caption-less
    control in the caption band, while a frame OUTSIDE the cue (shot 1) does not.
    REVERT-PROOF: drop ``captions_vf`` and the band delta collapses to noise."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(
            shot_index=0, cues=[CaptionCue(text="HELLO CAPTION WORLD", start=0.2, end=0.8)]
        )
    ]
    captioned = await _compile(shots, tmp_path / "cap", captions=caps)
    control = await _compile(shots, tmp_path / "ctl", captions=None)

    # At the cue's film time (0.5) the caption band differs materially…
    fc = await _frame_rgb_at(captioned, 0.5, tmp_path / "fc.rgb")
    fk = await _frame_rgb_at(control, 0.5, tmp_path / "fk.rgb")
    assert _band_change(fc, fk, _BAND_Y0, _H) > _PRESENT, "caption not visible in the band"
    # …and the change is localised to the band: the top of the frame is untouched.
    assert _band_change(fc, fk, 0, int(_H * 0.25)) < _ABSENT, "caption bled outside its band"

    # Away from any cue (shot 1, film 1.5) the two renders agree in the band.
    fc2 = await _frame_rgb_at(captioned, 1.5, tmp_path / "fc2.rgb")
    fk2 = await _frame_rgb_at(control, 1.5, tmp_path / "fk2.rgb")
    assert _band_change(fc2, fk2, _BAND_Y0, _H) < _ABSENT, (
        "control differs where there is no caption"
    )


async def test_no_captions_is_byte_identical(tmp_path: Path) -> None:
    """(b) BYTE-IDENTITY. ``captions=None`` and ``captions=[]`` both skip the
    caption re-encode (empty ASS ⇒ ``-c copy`` fast path), so the delivered files
    are sha256-identical — the caption feature is invisible when unused.
    Mirrors ``test_grade_render.test_no_grade_is_byte_identical``."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    plain = await _compile(shots, tmp_path / "plain", captions=None)
    empty = await _compile(shots, tmp_path / "empty", captions=[])
    assert (
        hashlib.sha256(plain.read_bytes()).hexdigest()
        == hashlib.sha256(empty.read_bytes()).hexdigest()
    ), "an empty captions list perturbed the -c copy fast path"


async def test_caption_placement_honours_offset_and_speed(tmp_path: Path) -> None:
    """(c) OFFSET + SPEED PLACEMENT. Shot 0 (2.0s, speed 1.0) then shot 1 (2.0s
    source at speed 2.0 ⇒ a 1.0s segment), so shot 1's bus window is film
    2.0..3.0 (reconstructed from the fed durations, since compile_shots doesn't
    expose offsets). A cue on shot 1 at shot-local 0.2..1.0 maps by ÷speed to
    film 2.0+0.2/2=2.1 .. 2.0+1.0/2=2.5.

    This PROVES offset + ÷speed, not a naive ``seg_start + cue.start`` sum: a
    naive mapping would place the cue at film 2.2..3.0, so it would be ABSENT at
    2.1 and PRESENT at 2.7. We assert the opposite of the naive prediction at the
    discriminating frame 2.7."""
    src = await _solid_clip(tmp_path / "src.mp4", duration=2.0)
    shots = [
        Shot(id="s0", seconds=2.0, video_url=str(src)),
        Shot(id="s1", seconds=2.0, video_url=str(src), speed=2.0),
    ]
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="SPEED CAPTION", start=0.2, end=1.0)])]
    captioned = await _compile(shots, tmp_path / "cap", captions=caps)
    control = await _compile(shots, tmp_path / "ctl", captions=None)

    async def band_delta(t: float, tag: str) -> int:
        a = await _frame_rgb_at(captioned, t, tmp_path / f"c_{tag}.rgb")
        b = await _frame_rgb_at(control, t, tmp_path / f"k_{tag}.rgb")
        return _band_change(a, b, _BAND_Y0, _H)

    # PRESENT inside the ÷speed window (2.1..2.5): the caption renders at all.
    assert await band_delta(2.3, "in") > _PRESENT, "caption missing inside its ÷speed window"
    # ABSENT well before shot 2 (shot 0 has no caption).
    assert await band_delta(1.0, "pre") < _ABSENT, "caption leaked into shot 0"
    # ABSENT past the ÷speed end (2.5). A naive offset+sum mapping would still be
    # showing it here (its window would run to 3.0) — this frame is the proof.
    assert await band_delta(2.7, "post") < _ABSENT, (
        "caption still visible past its ÷speed end — offset was summed, not divided by speed"
    )


async def test_karaoke_burns_a_valid_track(tmp_path: Path) -> None:
    """(e) KARAOKE RENDERS. A cue carrying per-word ``words`` with ``karaoke=True``
    (burned_in) produces a real, playable mp4 whose caption band lights up — i.e.
    the ``{\\k}`` ASS document is accepted by libass/ffmpeg and burns, exactly like
    a phrase caption. (The ``\\k`` timecode content itself is asserted in the unit
    tests; this proves the karaoke document is renderable end-to-end.)"""
    src = await _solid_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(
            shot_index=0,
            cues=[
                CaptionCue(
                    text="HELLO KARAOKE WORLD",
                    start=0.2,
                    end=0.9,
                    words=[
                        CaptionWord(text="HELLO", start=0.2, end=0.45),
                        CaptionWord(text="KARAOKE", start=0.45, end=0.7),
                        CaptionWord(text="WORLD", start=0.7, end=0.9),
                    ],
                )
            ],
        )
    ]
    karaoke = await _compile(shots, tmp_path / "kar", captions=caps, karaoke=True)
    control = await _compile(shots, tmp_path / "ctl", captions=None)

    fc = await _frame_rgb_at(karaoke, 0.5, tmp_path / "fc.rgb")
    fk = await _frame_rgb_at(control, 0.5, tmp_path / "fk.rgb")
    assert _band_change(fc, fk, _BAND_Y0, _H) > _PRESENT, "karaoke caption not visible in the band"
    # localised to the band — no bleed to the top of the frame.
    assert _band_change(fc, fk, 0, int(_H * 0.25)) < _ABSENT, (
        "karaoke caption bled outside its band"
    )


async def test_grade_and_captions_survive_one_pass(tmp_path: Path) -> None:
    """(d) GRADE + CAPTIONS, ONE RE-ENCODE. A compile carrying BOTH a finish
    grade and a caption cue must land BOTH on the delivered mp4 in a single
    whole-film pass: the grade's SATAVG shift still holds vs an ungraded control,
    AND the caption is still visible vs the graded-but-caption-less render."""
    src = await _colour_clip(tmp_path / "src.mp4", duration=1.0)
    shots = [
        Shot(id="s0", seconds=1.0, video_url=str(src)),
        Shot(id="s1", seconds=1.0, video_url=str(src)),
    ]
    caps = [
        ShotCaptions(shot_index=0, cues=[CaptionCue(text="GRADED CAPTION", start=0.2, end=0.8)])
    ]
    grade = _strong_grade()
    control = await _compile(shots, tmp_path / "ctl", finish=None, captions=None)
    graded_only = await _compile(shots, tmp_path / "go", finish=grade, captions=None)
    graded_capt = await _compile(shots, tmp_path / "gc", finish=grade, captions=caps)

    # The grade survived the combined pass: saturation/split-tone moved SATAVG.
    cs = await measure_shot_stats(control)
    gs = await measure_shot_stats(graded_capt)
    assert cs and gs, "signalstats readback failed on a delivered file"
    assert abs(gs["SATAVG"] - cs["SATAVG"]) > 3.0, (
        f"grade did not survive the caption pass: control={cs['SATAVG']:.2f} "
        f"graded+captioned={gs['SATAVG']:.2f}"
    )

    # The caption survived the same pass: the band differs from the grade-only
    # render (identical grade + encoding, so the delta is the caption alone).
    fc = await _frame_rgb_at(graded_capt, 0.5, tmp_path / "fc.rgb")
    fg = await _frame_rgb_at(graded_only, 0.5, tmp_path / "fg.rgb")
    assert _band_change(fc, fg, _BAND_Y0, _H) > _PRESENT, "caption lost when a grade was present"
