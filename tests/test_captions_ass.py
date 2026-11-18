"""Pure-unit tests for the burned-in caption ASS builders (VC-6).

ffmpeg is NOT invoked: these assert the ``build_film_ass`` offset/speed/clamp
MATH and the ``captions_vf_fragment`` filter shape directly on the emitted ASS
document string. Real muxing/burn-in is proven by
``tests/render_harness/test_captions_render.py`` (the sibling of
``test_renderer_directed.py`` ↔ ``test_directed_footage_render.py``).

The film-bus mapping under test (from ``captions_ass.build_film_ass``): for shot
``k`` with bus window ``shot_offsets[k] = (seg_start, seg_end)`` and playback
``speeds[k]`` ::

    film_start = seg_start + cue.start / speed
    film_end   = min(seg_start + cue.end / speed, seg_end)

so a shot-local cue is retimed by ÷speed and clamped to the shot's on-screen
window — NOT a naive ``seg_start + cue.start`` sum.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.api.schemas import CaptionCue, CaptionWord, ShotCaptions
from src.config import Settings, get_settings
from src.services.captions_ass import (
    CAPTION_FONT_FAMILIES,
    CAPTION_FONTS_DIR,
    CaptionStyle,
    build_film_ass,
    caption_look_from_request,
    captions_vf_fragment,
    resolve_caption_style,
)

# ASS centisecond timecodes round to 0.01s; allow a hair over half a centi.
_TS_TOL = 0.02


def _settings(**overrides: object) -> Settings:
    """Real Settings with known caption/render fields; overridable per test."""
    return get_settings().model_copy(
        update={"render_width": 1280, "render_height": 720, **overrides}
    )


def _dialogue_lines(ass: str) -> list[str]:
    return [line for line in ass.splitlines() if line.startswith("Dialogue:")]


def _tc_to_seconds(tc: str) -> float:
    """Parse an ASS ``H:MM:SS.cc`` timecode to seconds."""
    hours, minutes, rest = tc.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def _dialogue_times(line: str) -> tuple[float, float]:
    # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
    parts = line.split(",", 9)
    return _tc_to_seconds(parts[1]), _tc_to_seconds(parts[2])


def _dialogue_text(line: str) -> str:
    return line.split(",", 9)[9]


# --------------------------------------------------------------------------
# offset + speed math (the core of the feature)
# --------------------------------------------------------------------------


def test_cue_maps_to_shot_bus_offset() -> None:
    """A cue in shot index 1 (bus window 3.0..7.0) with speed 1.0 maps its
    shot-local 0.5..1.5 straight onto film 3.5..4.5 (seg_start + cue time)."""
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="hi", start=0.5, end=1.5)])]
    ass = build_film_ass(caps, [(0.0, 3.0), (3.0, 7.0)], [1.0, 1.0], _settings())
    lines = _dialogue_lines(ass)
    assert len(lines) == 1, "expected exactly one Dialogue for one surviving cue"
    start, end = _dialogue_times(lines[0])
    assert start == pytest.approx(3.5, abs=_TS_TOL)
    assert end == pytest.approx(4.5, abs=_TS_TOL)


def test_speed_divides_shot_local_time() -> None:
    """speed=2.0 on shot 1 compresses its bus window, so the shot-local cue is
    retimed by ÷speed: film 3.0+0.5/2=3.25 .. 3.0+1.5/2=3.75. This proves the
    mapping divides by speed rather than summing raw shot-local seconds."""
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="hi", start=0.5, end=1.5)])]
    ass = build_film_ass(caps, [(0.0, 3.0), (3.0, 7.0)], [1.0, 2.0], _settings())
    start, end = _dialogue_times(_dialogue_lines(ass)[0])
    assert start == pytest.approx(3.25, abs=_TS_TOL)
    assert end == pytest.approx(3.75, abs=_TS_TOL)


def test_end_is_clamped_to_shot_window() -> None:
    """A cue whose ÷speed end overruns the shot's on-screen window is clamped to
    ``shot_offsets[k][1]`` — the caption cannot outlast the shot it belongs to."""
    # shot 1 window is only 3.0..4.0 (1s on screen); cue end 2.0 → 3.0+2.0=5.0,
    # clamped down to 4.0. Start 3.0+0.5=3.5 is inside the window and untouched.
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="hi", start=0.5, end=2.0)])]
    ass = build_film_ass(caps, [(0.0, 3.0), (3.0, 4.0)], [1.0, 1.0], _settings())
    start, end = _dialogue_times(_dialogue_lines(ass)[0])
    assert start == pytest.approx(3.5, abs=_TS_TOL)
    assert end == pytest.approx(4.0, abs=_TS_TOL)


# --------------------------------------------------------------------------
# empty / no-op inputs → "" (keeps the -c copy fast path byte-identical)
# --------------------------------------------------------------------------


def test_no_shot_captions_returns_empty_document() -> None:
    assert build_film_ass([], [(0.0, 1.0)], [1.0], _settings()) == ""


def test_shot_captions_with_no_cues_returns_empty_document() -> None:
    caps = [ShotCaptions(shot_index=0, cues=[])]
    assert build_film_ass(caps, [(0.0, 1.0)], [1.0], _settings()) == ""


# --------------------------------------------------------------------------
# two-line wrap
# --------------------------------------------------------------------------


def test_long_text_wraps_to_at_most_two_lines() -> None:
    """A cue longer than ``caption_max_line_chars`` wraps to <=2 lines joined by
    the literal ASS newline ``\\N`` at the max-line-chars boundary."""
    settings = _settings(caption_max_line_chars=20)
    caps = [
        ShotCaptions(
            shot_index=0,
            cues=[CaptionCue(text="hello there my good friends", start=0.0, end=2.0)],
        )
    ]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], settings)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    assert r"\N" in text, "a long caption did not wrap"
    assert text.count(r"\N") == 1, "caption wrapped to more than two lines"
    # each rendered line respects the max-line-chars budget (greedy fill).
    first_line = text.split(r"\N")[0]
    assert len(first_line) <= 20


def test_overlong_cue_keeps_every_word_and_relies_on_libass_wrap() -> None:
    """A cue far longer than 2x the line budget must NOT drop any word (captions are
    script-faithful) — overflow is handled by libass smart-wrap (WrapStyle 0), which
    re-wraps an over-long line inside the frame instead of clipping off-canvas."""
    settings = _settings(caption_max_line_chars=20)
    words = [f"word{i}" for i in range(24)]  # ~160 chars, well past 2x20
    caps = [
        ShotCaptions(
            shot_index=0,
            cues=[CaptionCue(text=" ".join(words), start=0.0, end=2.0)],
        )
    ]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], settings)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    for word in words:
        assert word in text, f"caption dropped {word!r} — not script-faithful"
    # The Script Info must let libass re-wrap overflow inside the margin box.
    assert "WrapStyle: 0" in ass


def test_short_text_is_not_wrapped() -> None:
    settings = _settings(caption_max_line_chars=20)
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="Hi there", start=0.0, end=1.0)])]
    ass = build_film_ass(caps, [(0.0, 5.0)], [1.0], settings)
    assert r"\N" not in _dialogue_text(_dialogue_lines(ass)[0])


# --------------------------------------------------------------------------
# document header + vf fragment
# --------------------------------------------------------------------------


def test_script_info_playres_matches_settings() -> None:
    """[Script Info] carries PlayResX/PlayResY equal to the render canvas so
    libass positions captions in the delivered frame's pixel space."""
    settings = _settings(render_width=1280, render_height=720)
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    ass = build_film_ass(caps, [(0.0, 2.0)], [1.0], settings)
    assert "[Script Info]" in ass
    assert "PlayResX: 1280" in ass
    assert "PlayResY: 720" in ass


def test_vf_fragment_references_the_ass_path() -> None:
    p = Path("/tmp/x/captions.ass")
    frag = captions_vf_fragment(p)
    assert "ass=" in frag
    assert str(p) in frag


# --------------------------------------------------------------------------
# per-mood style (resolve_caption_style + the Cap style line)
# --------------------------------------------------------------------------


def _style_line(ass: str) -> str:
    return next(line for line in ass.splitlines() if line.startswith("Style: Cap,"))


def test_resolve_caption_style_known_alias_and_unknown() -> None:
    """Known moods resolve to a style; aliases fold onto the canonical key;
    empty/None/unknown ⇒ None (⇒ the caller uses settings defaults)."""
    assert isinstance(resolve_caption_style("energetic"), CaptionStyle)
    # alias 'upbeat' → 'energetic' → same style object as the canonical key.
    assert resolve_caption_style("Upbeat") == resolve_caption_style("energetic")
    assert resolve_caption_style(None) is None
    assert resolve_caption_style("") is None
    assert resolve_caption_style("no-such-mood") is None


def test_style_none_is_byte_identical_cap_line() -> None:
    """``style=None`` must emit the EXACT default ``Style: Cap`` line (render_width=1280,
    render_height=720 ⇒ font 32, margin_v 43, margin_h round(1280*20/1920)=13; primary
    white, secondary &H00808080 dim-grey, outline black, bold 0, outline 2). If the
    un-stub drifted this line, this test bites.

    margin_h is render_width-scaled the same way margin_v is render_height-scaled
    (20/1920 keeps the default 1920-wide delivery byte-identical at 20px) — see
    ``test_margin_h_scales_with_render_width`` for the scaling proof.

    SecondaryColour is the dim mid-grey default (ASS &HAABBGGRR = &H00808080), NOT the
    old bright-red &H000000FF — see ``test_karaoke_default_secondary_is_grey_not_red``.
    """
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    # Pinned on the bare-outline look (caption_box_enabled=False); the default
    # backplate look is pinned by test_default_style_emits_semi_opaque_backplate.
    ass = build_film_ass(caps, [(0.0, 2.0)], [1.0], _settings(caption_box_enabled=False))
    assert _style_line(ass) == (
        "Style: Cap,DejaVu Sans,32,&H00FFFFFF,&H00808080,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,13,13,43,1"
    )


def _style_fields(ass: str) -> dict[str, str]:
    """The Cap style line keyed by the ``[V4+ Styles]`` Format header."""
    fmt = next(line for line in ass.splitlines() if line.startswith("Format: Name,"))
    names = [n.strip() for n in fmt.removeprefix("Format:").split(",")]
    values = _style_line(ass).removeprefix("Style:").strip().split(",")
    return dict(zip(names, values, strict=True))


@pytest.mark.parametrize(
    ("width", "height", "font", "padding"),
    [(1920, 1080, "49", "12"), (3840, 2160, "97", "24")],
)
def test_default_style_emits_semi_opaque_backplate(
    width: int, height: int, font: str, padding: str
) -> None:
    """The DEFAULT burned-in look puts a semi-opaque dark box behind every line.

    Showcase spike run 01a0fd4d-9948 (13s frame) burned white text with a 2px
    outline straight onto dense app text and it was illegible. libass
    ``BorderStyle=3`` draws an opaque box per line filled with OutlineColour and
    padded by ``Outline`` px; the plate must be dark, partly transparent, and its
    padding must scale with the font so 1080p and 4k keep the same proportions.
    """
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    settings = _settings(render_width=width, render_height=height)
    fields = _style_fields(build_film_ass(caps, [(0.0, 2.0)], [1.0], settings))

    assert fields["BorderStyle"] == "3"
    assert fields["Fontsize"] == font
    assert fields["Outline"] == padding  # 0.25 x font size, scales with height
    for key in ("OutlineColour", "BackColour"):
        colour = fields[key]  # &HAABBGGRR
        alpha, rgb = int(colour[2:4], 16), colour[4:]
        assert 0x20 <= alpha <= 0x90, f"{key} {colour} must be semi-opaque"
        assert rgb == "000000", f"{key} {colour} must be a dark plate"


def test_backplate_survives_a_mood_style() -> None:
    """A mood (karaoke colours, bold, outline_px) restyles the text but must not
    strip the backplate: the plate is what keeps it legible over footage."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    ass = build_film_ass(
        caps, [(0.0, 2.0)], [1.0], _settings(), style=resolve_caption_style("energetic")
    )
    fields = _style_fields(ass)
    assert fields["BorderStyle"] == "3"
    assert fields["OutlineColour"] == fields["BackColour"] == "&H50000000"
    assert fields["PrimaryColour"] == "&H0000E5FF"


def test_margin_h_is_noop_at_1920_wide() -> None:
    """render_width=1920 (the pre-scaling default) must reproduce the literal
    margin_h=20 exactly."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    settings = _settings(render_width=1920, render_height=1080)
    ass = build_film_ass(caps, [(0.0, 2.0)], [1.0], settings)
    assert _style_line(ass).split(",")[-3] == "20"


def test_margin_h_scales_with_render_width() -> None:
    """Doubling render_width to 3840 (2x 1920) must double margin_h: 20 -> 40."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.0, end=1.0)])]
    settings = _settings(render_width=3840, render_height=2160)
    ass = build_film_ass(caps, [(0.0, 2.0)], [1.0], settings)
    assert _style_line(ass).split(",")[-3] == "40"


def test_mood_changes_style_line_not_dialogue_timing() -> None:
    """A mood style overrides the Cap style (colour/size/weight) but leaves the
    Dialogue timings untouched — styling and timing are independent."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hi", start=0.5, end=1.5)])]
    # Outline thickness is a glyph-outline concept, so pin the bare-outline mode.
    no_box = _settings(caption_box_enabled=False)
    plain = build_film_ass(caps, [(0.0, 3.0)], [1.0], no_box)
    styled = build_film_ass(
        caps, [(0.0, 3.0)], [1.0], no_box, style=resolve_caption_style("energetic")
    )
    # style line differs (energetic ⇒ amber primary, bold, larger font 36, outline 3)
    assert _style_line(plain) != _style_line(styled)
    assert "&H0000E5FF" in _style_line(styled)  # amber-yellow highlight colour
    assert ",36," in _style_line(styled)  # 720 * 0.050
    # bold flag flips to -1 and the outline thickens to 3 for energetic.
    assert _style_line(styled).endswith(",-1,0,0,0,100,100,0,0,1,3,0,2,13,13,43,1")
    # …but the Dialogue timings are byte-identical to the plain build.
    assert _dialogue_times(_dialogue_lines(styled)[0]) == _dialogue_times(_dialogue_lines(plain)[0])


# --------------------------------------------------------------------------
# karaoke ({\k}) word-highlight — burned_in only
# --------------------------------------------------------------------------


def test_karaoke_emits_k_tags_summing_to_cue_window() -> None:
    """A cue with words rendered with ``karaoke=True`` emits ``{\\k<cs>}`` chunks
    whose centisecond durations (word durations + inter-word gaps) sum to the
    cue's on-screen window."""
    cue = CaptionCue(
        text="hello world",
        start=0.0,
        end=1.0,
        words=[
            CaptionWord(text="hello", start=0.0, end=0.4),
            CaptionWord(text="world", start=0.5, end=1.0),
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings(), karaoke=True)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    assert "hello" in text and "world" in text
    ks = [int(v) for v in re.findall(r"\\k(\d+)", text)]
    # hello 40cs, gap 10cs, world 50cs → 100cs == the 1.0s cue window.
    assert sum(ks) == 100, f"karaoke \\k durations {ks} do not span the cue window"


def test_karaoke_retimes_by_speed_and_clamps_to_window() -> None:
    """Word times ride the SAME ÷speed retime + ``seg_end`` clamp as the cue."""
    # speed 2.0: shot-local 0..2 halves to a 0..1 film window; word end past the
    # shot window clamps to seg_end.
    cue = CaptionCue(
        text="quick brown",
        start=0.0,
        end=2.0,
        words=[
            CaptionWord(text="quick", start=0.0, end=0.8),  # ÷2 → 0.0..0.4 → 40cs
            CaptionWord(text="brown", start=1.0, end=3.0),  # ÷2 → 0.5..1.5, clamp end→1.0
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    # shot window is only 0..1.0 on the bus, so film_end clamps to 1.0.
    ass = build_film_ass(caps, [(0.0, 1.0)], [2.0], _settings(), karaoke=True)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    ks = [int(v) for v in re.findall(r"\\k(\d+)", text)]
    # quick 40, gap 10, brown 50 (0.5..1.0 after clamp) → 100cs == the clamped window.
    assert sum(ks) == 100, f"{ks}"


def test_karaoke_escapes_word_text_before_k_braces() -> None:
    """A stray brace in a WORD is neutralised (→ paren) while the ``\\k`` override
    braces survive — proving the escape runs on the word BEFORE the tag is added."""
    cue = CaptionCue(
        text="a {b} c",
        start=0.0,
        end=0.9,
        words=[
            CaptionWord(text="a", start=0.0, end=0.3),
            CaptionWord(text="{b}", start=0.3, end=0.6),
            CaptionWord(text="c", start=0.6, end=0.9),
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings(), karaoke=True)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    assert "(b)" in text, "word brace not neutralised — a stray brace would open a bogus tag"
    assert "{b}" not in text
    assert r"\k" in text, "the \\k override braces were destroyed by escaping"


def test_karaoke_falls_back_to_plain_when_cue_has_no_words() -> None:
    """``karaoke=True`` on a cue that carries no ``words`` renders the plain phrase
    (no ``\\k``) — karaoke degrades gracefully to phrase captions."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="no words here", start=0.0, end=1.0)])]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings(), karaoke=True)
    text = _dialogue_text(_dialogue_lines(ass)[0])
    assert r"\k" not in text
    assert "no words here" in text


def test_words_ignored_when_karaoke_off() -> None:
    """A cue WITH words but ``karaoke=False`` (the default) renders the plain
    phrase — words are inert unless karaoke is explicitly requested."""
    cue = CaptionCue(
        text="hello world",
        start=0.0,
        end=1.0,
        words=[
            CaptionWord(text="hello", start=0.0, end=0.5),
            CaptionWord(text="world", start=0.5, end=1.0),
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    with_words = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings())
    plain_caps = [
        ShotCaptions(shot_index=0, cues=[CaptionCue(text="hello world", start=0.0, end=1.0)])
    ]
    without = build_film_ass(plain_caps, [(0.0, 10.0)], [1.0], _settings())
    assert r"\k" not in _dialogue_text(_dialogue_lines(with_words)[0])
    # identical to the same cue without any words field
    assert _dialogue_lines(with_words) == _dialogue_lines(without)


def _secondary_colour(ass: str) -> str:
    # Style: Cap,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,...
    return _style_line(ass).split(",")[4]


def test_karaoke_default_secondary_is_grey_not_red() -> None:
    r"""With ``karaoke=True`` and ``style=None`` (no/unknown mood), the ``\k`` sweep
    draws un-highlighted words in SecondaryColour, so the default must be a dim,
    unobtrusive grey — NOT bright red.

    REVERT NOTE: this test MUST FAIL if the default SecondaryColour reverts to the
    old red ``&H000000FF`` (ASS &HAABBGGRR = opaque bright red), which flashed every
    not-yet-swept word red until the highlight reached it.
    """
    cue = CaptionCue(
        text="hello world",
        start=0.0,
        end=1.0,
        words=[
            CaptionWord(text="hello", start=0.0, end=0.5),
            CaptionWord(text="world", start=0.5, end=1.0),
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings(), style=None, karaoke=True)
    secondary = _secondary_colour(ass)
    assert secondary == "&H00808080", f"karaoke default SecondaryColour is {secondary}"
    assert secondary != "&H000000FF", "un-swept karaoke words must not render bright red"


def test_karaoke_secondary_comes_from_settings_and_override_changes_it() -> None:
    r"""The un-swept karaoke SecondaryColour default is sourced from
    ``settings.caption_secondary`` (not a hardcoded literal), so overriding that
    setting changes the emitted SecondaryColour.

    REVERT NOTE: this test MUST FAIL if the fallback reverts to the hardcoded
    ``"&H00808080"`` literal — the override would then be ignored and the emitted
    SecondaryColour would stay grey regardless of ``caption_secondary``.
    """
    cue = CaptionCue(
        text="hello world",
        start=0.0,
        end=1.0,
        words=[
            CaptionWord(text="hello", start=0.0, end=0.5),
            CaptionWord(text="world", start=0.5, end=1.0),
        ],
    )
    caps = [ShotCaptions(shot_index=0, cues=[cue])]
    # default from settings.caption_secondary (dim grey)
    default_ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings(), style=None, karaoke=True)
    assert _secondary_colour(default_ass) == "&H00808080"
    # overriding the setting flows straight through to the Style line
    override_ass = build_film_ass(
        caps,
        [(0.0, 10.0)],
        [1.0],
        _settings(caption_secondary="&H0000AAFF"),
        style=None,
        karaoke=True,
    )
    assert _secondary_colour(override_ass) == "&H0000AAFF", (
        "settings.caption_secondary override did not reach the emitted SecondaryColour "
        "— the fallback is still a hardcoded literal"
    )


def test_non_karaoke_dialogue_unaffected_by_secondary_default() -> None:
    """SecondaryColour only matters on the ``\\k`` path; a non-karaoke render's
    Dialogue output is byte-identical regardless of the SecondaryColour default —
    the grey change touches only the Style line, never the burned-in Dialogue text."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="hello world", start=0.0, end=1.0)])]
    ass = build_film_ass(caps, [(0.0, 10.0)], [1.0], _settings())  # karaoke off (default)
    # the non-karaoke Dialogue carries no colour override — no \k, no SecondaryColour.
    text = _dialogue_text(_dialogue_lines(ass)[0])
    assert r"\k" not in text
    assert text == "hello world"


# --------------------------------------------------------------------------
# request caption typography (CompileRequest.caption_font_family /
# caption_text_colour / caption_box_colour / caption_box_opacity -> CaptionLook)
# --------------------------------------------------------------------------

_TWO_LINE_CUE = [
    ShotCaptions(
        shot_index=0,
        cues=[
            CaptionCue(
                text="Users can set graduated autonomy levels for 30+ capabilities",
                start=0.0,
                end=2.0,
                words=[
                    CaptionWord(text="Users", start=0.0, end=0.4),
                    CaptionWord(text="can", start=0.4, end=0.8),
                    CaptionWord(text="set", start=0.8, end=1.2),
                ],
            )
        ],
    )
]


@pytest.mark.parametrize("mood", [None, "energetic", "calm"])
@pytest.mark.parametrize("karaoke", [False, True])
@pytest.mark.parametrize("box", [True, False])
def test_request_without_typography_fields_is_byte_identical(
    mood: str | None, karaoke: bool, box: bool
) -> None:
    """No typography field on the request => no look => the SAME ASS document and
    the SAME ``ass=`` fragment as before the fields existed."""
    look = caption_look_from_request(None, None, None, None)
    assert look is None
    settings = _settings(caption_box_enabled=box)
    style = resolve_caption_style(mood)
    before = build_film_ass(_TWO_LINE_CUE, [(0.0, 3.0)], [1.0], settings, style, karaoke)
    after = build_film_ass(_TWO_LINE_CUE, [(0.0, 3.0)], [1.0], settings, style, karaoke, look=look)
    assert after == before
    p = Path("/tmp/x/captions.ass")
    assert captions_vf_fragment(p, None) == captions_vf_fragment(p) == f"ass=filename='{p}'"


def test_request_typography_sets_the_cap_style() -> None:
    """Inter + brand text colour + brand box colour/opacity land on the style
    line: Fontname, PrimaryColour, BackColour, and ONE box per cue."""
    look = caption_look_from_request("Inter", "#F8FAFC", "#0B1F3A", 0.85)
    settings = _settings(render_width=1920, render_height=1080)
    fields = _style_fields(build_film_ass(_TWO_LINE_CUE, [(0.0, 3.0)], [1.0], settings, look=look))
    assert fields["Fontname"] == "Inter SemiBold"
    assert fields["Bold"] == "0"
    assert fields["PrimaryColour"] == "&H00FCFAF8"  # #F8FAFC as &HAABBGGRR
    # BorderStyle 4: one box around the whole event in BackColour; the glyph
    # stroke (OutlineColour) is fully transparent; Outline is the box padding.
    assert fields["BorderStyle"] == "4"
    assert fields["BackColour"] == "&H263A1F0B"  # alpha round(0.15*255)=0x26, #0B1F3A
    assert fields["OutlineColour"] == "&HFF000000"
    assert fields["Outline"] == "12"  # 0.25 x 49px, same padding as the default plate
    assert fields["Fontsize"] == "49"  # size untouched


def test_bold_mood_selects_the_inter_bold_face() -> None:
    look = caption_look_from_request("Inter", None, None, None)
    ass = build_film_ass(
        _TWO_LINE_CUE,
        [(0.0, 3.0)],
        [1.0],
        _settings(),
        resolve_caption_style("energetic"),
        look=look,
    )
    fields = _style_fields(ass)
    assert (fields["Fontname"], fields["Bold"]) == ("Inter", "-1")
    # A font-only look keeps today's per-line plate (no box field was sent).
    assert fields["BorderStyle"] == "3"
    assert fields["PrimaryColour"] == "&H0000E5FF"  # the mood's colour survives


def test_box_colour_alone_keeps_the_default_opacity_and_opacity_alone_is_black() -> None:
    default_alpha = _settings().caption_box_colour[2:4]
    colour_only = _style_fields(
        build_film_ass(
            _TWO_LINE_CUE,
            [(0.0, 3.0)],
            [1.0],
            _settings(),
            look=caption_look_from_request(None, None, "#0B1F3A", None),
        )
    )
    assert colour_only["BackColour"] == f"&H{default_alpha}3A1F0B"
    assert colour_only["BorderStyle"] == "4"
    assert colour_only["Fontname"] == "DejaVu Sans"
    opacity_only = _style_fields(
        build_film_ass(
            _TWO_LINE_CUE,
            [(0.0, 3.0)],
            [1.0],
            _settings(caption_box_enabled=False),
            look=caption_look_from_request(None, None, None, 1.0),
        )
    )
    assert opacity_only["BackColour"] == "&H00000000"
    assert opacity_only["BorderStyle"] == "4"


def test_embedded_family_adds_fontsdir_with_only_font_files() -> None:
    """libass loads EVERY file in fontsdir as a font, so it holds fonts only; the
    OFL licence ships beside it."""
    look = caption_look_from_request("Inter", None, None, None)
    assert look is not None and look.fontsdir == CAPTION_FONTS_DIR
    p = Path("/tmp/x/captions.ass")
    assert captions_vf_fragment(p, look.fontsdir) == (
        f"ass=filename='{p}':fontsdir='{CAPTION_FONTS_DIR}'"
    )
    assert sorted(f.name for f in CAPTION_FONTS_DIR.iterdir()) == [
        "Inter-Bold.ttf",
        "Inter-SemiBold.ttf",
    ]
    for f in CAPTION_FONTS_DIR.iterdir():
        assert f.read_bytes()[:4] == b"\x00\x01\x00\x00"  # TrueType outlines
    assert "SIL Open Font License" in (CAPTION_FONTS_DIR.parent / "OFL-Inter.txt").read_text()
    # A system family needs no fontsdir.
    dejavu = caption_look_from_request("DejaVu Sans", None, None, None)
    assert dejavu is not None and dejavu.fontsdir is None


def test_schema_font_allowlist_is_the_embedded_set() -> None:
    import typing

    from src.api.schemas import CompileRequest

    ann = CompileRequest.model_fields["caption_font_family"].annotation
    literal = next(a for a in typing.get_args(ann) if typing.get_origin(a) is typing.Literal)
    assert typing.get_args(literal) == CAPTION_FONT_FAMILIES


_SHOT = {"id": "s1", "description": "d", "seconds": 1.0}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("caption_text_colour", "#FFF"),
        ("caption_text_colour", "white"),
        ("caption_text_colour", "#GGGGGG"),
        ("caption_text_colour", "F8FAFC"),
        ("caption_box_colour", "#0B1F3A80"),
        ("caption_box_colour", "rgb(0,0,0)"),
        ("caption_font_family", "Arial"),
        ("caption_font_family", "inter"),
        ("caption_box_opacity", 1.5),
        ("caption_box_opacity", -0.1),
    ],
)
def test_schema_rejects_bad_caption_typography(field: str, value: object) -> None:
    from pydantic import ValidationError

    from src.api.schemas import CompileRequest

    with pytest.raises(ValidationError):
        CompileRequest.model_validate({"shots": [_SHOT], field: value})


def test_schema_accepts_brand_caption_typography() -> None:
    from src.api.schemas import CompileRequest

    req = CompileRequest.model_validate(
        {
            "shots": [_SHOT],
            "caption_font_family": "Inter",
            "caption_text_colour": "#F8FAFC",
            "caption_box_colour": "#0b1f3a",
            "caption_box_opacity": 0.85,
        }
    )
    assert (req.caption_font_family, req.caption_text_colour) == ("Inter", "#F8FAFC")
    absent = CompileRequest.model_validate({"shots": [_SHOT]})
    assert (
        absent.caption_font_family,
        absent.caption_text_colour,
        absent.caption_box_colour,
        absent.caption_box_opacity,
    ) == (None, None, None, None)


@pytest.mark.parametrize("family", [None, "Inter"])
def test_renderer_burn_fragment_carries_the_look(tmp_path: Path, family: str | None) -> None:
    """``renderer._resolve_captions_vf`` writes the look into the ASS file it burns
    and hands libass the fontsdir only for an embedded family."""
    from src.api.schemas import Shot
    from src.services.renderer import _resolve_captions_vf

    look = caption_look_from_request(family, None, None, None)
    frag = _resolve_captions_vf(
        _TWO_LINE_CUE,
        [Shot(id="s1", description="d", seconds=3.0)],
        [(0.0, 3.0)],
        _settings(),
        tmp_path,
        "burned_in",
        caption_look=look,
    )
    ass_path = tmp_path / "captions.ass"
    if family is None:
        assert frag == f"ass=filename='{ass_path}'"
        assert _style_fields(ass_path.read_text())["Fontname"] == "DejaVu Sans"
    else:
        assert frag == f"ass=filename='{ass_path}':fontsdir='{CAPTION_FONTS_DIR}'"
        assert _style_fields(ass_path.read_text())["Fontname"] == "Inter SemiBold"
