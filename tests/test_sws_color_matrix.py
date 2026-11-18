"""The swscale in/out colour matrix must match the colr atom the encode tags.

Before this fix, no ``scale=`` node in the filtergraph declared a colour matrix
at all, so swscale silently ran its BT.601 default converting RGB-origin
content that passes through a ``scale=`` node (generated stills, slide PNGs,
fitted overlays — ``_scale_pad_filter``/``_split_filter``/``_scroll_filter``/
``_overlay_clip_chain(scale="fit")``) to yuv420p, while
``mastering.colorspace_tag_args`` tagged the resulting stream bt709 (the
fleet default). Measured through the repo's verbatim chain, decoded back
honouring the tag, vs the original RGB (mean per-pixel RGB distance): red
patch 22.16 -> 3.17 (7.0x), teal 18.38 -> 2.48 (7.4x), orange 8.52 -> 2.12
(4.0x), navy 5.79 -> 1.26 (4.6x), red text (thin/AA) 6.81 -> 4.34 (1.6x).

⚠️ This fix does NOT cover drawtext/drawbox/color= (captions, badges,
text-card backgrounds). Those draw straight onto an already-YUV frame via
ffmpeg's ``drawutils``, which hardcodes BT.601 and never traverses a
``scale=`` node — verified on ffmpeg 4.4:
``drawbox=color=red:t=fill,format=yuv420p`` -> Y=0x51 identically with and
without a trailing ``scale=...:out_color_matrix=bt709``. Captions and
text-card bodies are NOT colour-corrected by this change.

Two properties are proven here with real ffmpeg (no mocks):

1. The matrix name is DERIVED from ``render_colorspace`` via the same
   ``mastering.colorspace_tag_args`` accessor the encoder uses to write the
   tag — never hardcoded — so a job that retargets colourspace stays correct.
2. Declaring the matrix is a byte-identical NO-OP on already-yuv420p footage
   (screen capture, never RGB-origin) — the control that makes the fix safe to
   ship unconditionally.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.services import renderer

pytestmark = [
    pytest.mark.render,
]

_FFMPEG_MISSING = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# 1. Derivation — NOT hardcoded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("colorspace", "expected_matrix"),
    [
        ("bt709", "bt709"),
        ("bt601", "bt601"),  # tag is "smpte170m" -> swscale wants "bt601"
        ("bt2020", "bt2020"),  # tag is "bt2020nc" -> swscale wants "bt2020"
        ("srgb", "bt709"),  # srgb's -colorspace tag is also "bt709"
    ],
)
def test_sws_color_matrix_derives_from_render_colorspace(
    colorspace: str, expected_matrix: str
) -> None:
    """Mutate the mapping (or hardcode a single value) and this goes red for
    every colorspace except whichever one was hardcoded."""
    settings = _settings(render_colorspace=colorspace)
    assert renderer._sws_color_matrix(settings) == expected_matrix


def test_sws_color_matrix_tracks_colorspace_tag_args_not_a_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derivation reads the LAST element of ``colorspace_tag_args`` (the
    ``-colorspace`` value) — proven by monkeypatching that accessor and
    checking the matrix output follows it, rather than asserting a fixed
    string that a hardcoded implementation could also satisfy."""
    from src.services import mastering

    def _fake_tag_args(_settings: Settings) -> list[str]:
        return [
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "bt2020-10",
            "-colorspace",
            "bt2020nc",
        ]

    monkeypatch.setattr(mastering, "colorspace_tag_args", _fake_tag_args)
    assert renderer._sws_color_matrix(_settings()) == "bt2020"


def test_sws_color_matrix_falls_back_to_bt709_when_tagging_disabled() -> None:
    """An unset/disabled render_colorspace still needs SOME matrix for swscale
    (it has no "off") — bt709 is ffmpeg's own default output tag, so this is a
    safe fallback, not a hidden hardcode of the common case."""
    assert renderer._sws_color_matrix(_settings(render_colorspace="")) == "bt709"
    assert renderer._sws_color_matrix(_settings(render_colorspace="not-a-real-value")) == "bt709"


def test_sws_flags_embeds_the_derived_matrix_both_directions() -> None:
    settings = _settings(render_colorspace="bt2020")
    flags = renderer._sws_flags(settings)
    assert "in_color_matrix=bt2020" in flags
    assert "out_color_matrix=bt2020" in flags
    assert flags.startswith(renderer._SWS_FLAGS)


def test_scale_pad_filter_declares_the_matrix_on_its_scale_node() -> None:
    """End-to-end (still pure-python): the actual filtergraph fragment a shot
    renders through carries the matrix, not just the helper function."""
    settings = _settings(render_colorspace="bt601")
    chain = renderer._scale_pad_filter(settings, None)
    assert "in_color_matrix=bt601" in chain
    assert "out_color_matrix=bt601" in chain


def test_overlay_clip_chain_native_scale_still_declares_the_matrix() -> None:
    """``scale='native'`` skips the resize, but ``format=yuva420p`` right after
    it is still an RGBA->YUV conversion through swscale — so a 'native' overlay
    needs a same-size ``scale=iw:ih`` node carrying the matrix, exactly like
    the 'fit' branch's real resize does. Mutate this away (drop the ``else``
    branch) and this test goes red."""
    from src.api.schemas import OverlayClip

    settings = _settings(render_colorspace="bt601")
    clip = OverlayClip(url="file:///tmp/logo.webm", scale="native")
    chain = renderer._overlay_clip_chain(3, clip, settings)
    assert "scale=iw:ih" in chain
    assert "in_color_matrix=bt601" in chain
    assert "out_color_matrix=bt601" in chain
    # The matrix scale node must precede the format conversion it protects.
    assert chain.index("scale=iw:ih") < chain.index("format=yuva420p")


def test_thumbnail_scale_declares_the_matrix() -> None:
    """``thumbnail.py``'s poster-retarget ``scale=`` node must carry the same
    swscale flags + derived matrix as every other scale node in the compiler
    (renderer._sws_flags) — it was previously a bare, unflagged ``scale=W:H``.
    Mutate the f-string back to bare ``scale={out_w}:{out_h}`` and this goes red."""
    import inspect

    from src.services import thumbnail

    src_text = inspect.getsource(thumbnail.extract_candidates)
    assert "_sws_flags(settings)" in src_text


# ---------------------------------------------------------------------------
# 2. No-op on already-yuv420p footage — real ffmpeg, verbatim repo chain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(_FFMPEG_MISSING, reason="real-ffmpeg no-op proof requires ffmpeg + ffprobe")
async def test_colour_matrix_is_noop_on_yuv420p_footage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Screen-capture-style footage (already yuv420p, never RGB-origin) must
    render byte-identically whether or not the scale node declares the colour
    matrix — this is what makes it safe to add the matrix unconditionally
    instead of only for RGB-origin sources.

    Uses ``_scale_pad_filter`` verbatim (the same function every footage/image
    shot renders through), diffing only whether ``_sws_flags`` carries the
    matrix, then runs both through real ffmpeg and compares raw decoded frame
    bytes."""
    settings = _settings(render_width=64, render_height=64, render_fps=10)
    src = tmp_path / "footage.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(src),
        ]
    )

    with_matrix_vf = renderer._scale_pad_filter(settings, None)
    monkeypatch.setattr(renderer, "_sws_flags", lambda _s: renderer._SWS_FLAGS)
    without_matrix_vf = renderer._scale_pad_filter(settings, None)
    monkeypatch.undo()

    assert "in_color_matrix=" in with_matrix_vf
    assert "in_color_matrix=" not in without_matrix_vf

    async def _decode_raw(vf: str, dest: Path) -> bytes:
        await renderer._run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                vf,
                "-frames:v",
                "5",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rawvideo",
                str(dest),
            ]
        )
        return dest.read_bytes()

    raw_with = await _decode_raw(with_matrix_vf, tmp_path / "with.raw")
    raw_without = await _decode_raw(without_matrix_vf, tmp_path / "without.raw")
    assert len(raw_with) > 0
    assert (
        hashlib.md5(raw_with, usedforsecurity=False).hexdigest()
        == hashlib.md5(raw_without, usedforsecurity=False).hexdigest()
    )
