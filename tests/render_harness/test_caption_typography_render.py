"""Real-libass proof for the request's caption typography (font, colours, box).

``tests/test_captions_ass.py`` proves the ``Style: Cap`` line; these prove what
libass actually DRAWS from it, through the same two functions the renderer uses
(``build_film_ass`` + ``captions_vf_fragment``) and a real ``ass`` filter:

* the embedded Inter is what gets drawn — the frame differs from the default
  DejaVu render AND from the same style rendered without the service's
  ``fontsdir`` (where libass silently falls back to a system font);
* a box colour draws ONE continuous box per cue — every pixel of the box's
  padding rows and padding columns is the same colour. The default per-line
  plate (BorderStyle=3) overlaps itself between lines and between karaoke words
  and paints darker stripes there; that is the defect this must not bring back.

Rendered in RGB (``format=rgb24`` before ``ass``), so no chroma subsampling
blurs the colours being compared.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.api.schemas import CaptionCue, CaptionWord, ShotCaptions
from src.config import Settings, get_settings
from src.services.captions_ass import (
    CaptionLook,
    build_film_ass,
    caption_look_from_request,
    captions_vf_fragment,
)
from tests.render_harness.harness import requires_filters

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on runner"),
    requires_filters("ass"),
]

_W, _H = 960, 540
_BG = (0x2B, 0xB6, 0xA3)  # brand teal: the box must read against it
_BOX = (0x0B, 0x1F, 0x3A)  # brand navy


def _settings() -> Settings:
    return get_settings().model_copy(update={"render_width": _W, "render_height": _H})


def _cue(karaoke_words: bool) -> list[ShotCaptions]:
    text = "Users can set graduated autonomy levels for 30+ capabilities, ranging from"
    words = None
    if karaoke_words:
        words = [
            CaptionWord(text=w, start=i * 0.1, end=(i + 1) * 0.1)
            for i, w in enumerate(text.split()[:9])
        ]
    return [
        ShotCaptions(shot_index=0, cues=[CaptionCue(text=text, start=0.0, end=2.0, words=words)])
    ]


def _render(
    tmp_path: Path,
    look: CaptionLook | None,
    *,
    karaoke: bool = False,
    fontsdir: bool = True,
    name: str = "f",
) -> list[bytes]:
    """One RGB frame at t=0.5s as rows of bytes."""
    ass = build_film_ass(
        _cue(karaoke), [(0.0, 3.0)], [1.0], _settings(), karaoke=karaoke, look=look
    )
    ass_path = tmp_path / f"{name}.ass"
    ass_path.write_text(ass, encoding="utf-8")
    frag = captions_vf_fragment(ass_path, look.fontsdir if (look and fontsdir) else None)
    exe = shutil.which("ffmpeg")
    assert exe is not None
    colour = "0x{:02X}{:02X}{:02X}".format(*_BG)
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell, resolved exe
        [
            exe,
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={colour}:s={_W}x{_H}:d=1:r=10",
            "-vf",
            f"format=rgb24,{frag}",
            "-ss",
            "0.5",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    raw = proc.stdout
    assert len(raw) == _W * _H * 3
    return [raw[y * _W * 3 : (y + 1) * _W * 3] for y in range(_H)]


def _px(rows: list[bytes], x: int, y: int) -> tuple[int, int, int]:
    r = rows[y]
    return r[3 * x], r[3 * x + 1], r[3 * x + 2]


def _changed(a: list[bytes], b: list[bytes]) -> int:
    return sum(
        1
        for y in range(_H)
        for x in range(_W)
        if a[y][3 * x : 3 * x + 3] != b[y][3 * x : 3 * x + 3]
    )


def _near(p: tuple[int, ...], q: tuple[int, ...], tol: int = 3) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(p, q, strict=True))


def _box_bounds(rows: list[bytes]) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) inclusive bounds of every pixel that is not the background."""
    xs, ys = [], []
    for y in range(_H):
        for x in range(_W):
            if not _near(_px(rows, x, y), _BG, 2):
                xs.append(x)
                ys.append(y)
    assert xs, "nothing was drawn"
    return min(xs), min(ys), max(xs), max(ys)


def test_inter_is_the_font_libass_draws(tmp_path: Path) -> None:
    inter = caption_look_from_request("Inter", None, None, None)
    drawn = _render(tmp_path, inter, name="inter")
    default = _render(tmp_path, None, name="default")
    no_fontsdir = _render(tmp_path, inter, fontsdir=False, name="nofonts")
    # Glyph shapes differ over thousands of pixels; two renders of the same
    # style are identical (checked below), so this is not noise.
    assert _changed(drawn, default) > 2000
    assert _changed(drawn, no_fontsdir) > 2000
    assert _changed(drawn, _render(tmp_path, inter, name="inter2")) == 0


@pytest.mark.parametrize("karaoke", [False, True])
def test_box_colour_draws_one_continuous_box_per_cue(tmp_path: Path, karaoke: bool) -> None:
    look = caption_look_from_request("Inter", "#F8FAFC", "#0B1F3A", 0.7)
    rows = _render(tmp_path, look, karaoke=karaoke, name=f"box{karaoke}")
    x0, y0, x1, y1 = _box_bounds(rows)
    # Inside the padding (font 24px at 540p -> padding 6px), off every glyph.
    inset = 2
    edge = [_px(rows, x, y0 + inset) for x in range(x0 + inset, x1 - inset + 1)]
    edge += [_px(rows, x, y1 - inset) for x in range(x0 + inset, x1 - inset + 1)]
    edge += [_px(rows, x0 + inset, y) for y in range(y0 + inset, y1 - inset + 1)]
    edge += [_px(rows, x1 - inset, y) for y in range(y0 + inset, y1 - inset + 1)]
    ref = edge[0]
    off = [p for p in edge if not _near(p, ref)]
    assert not off, f"box is not one uniform rectangle: {len(off)} px differ from {ref}"
    # ...and it is the requested navy at 70% over the teal background.
    expect = tuple(round(0.7 * b + 0.3 * g) for b, g in zip(_BOX, _BG, strict=True))
    assert _near(ref, expect, 4), (ref, expect)
    # Two wrapped lines (non-karaoke) sit inside the ONE box: it is taller than a line.
    if not karaoke:
        assert y1 - y0 > 2 * 24
