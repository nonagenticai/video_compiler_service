"""Burned-in word/phrase-timed captions (VC-6) as a libass ASS subtitle track.

Pure string builders — no I/O. ``build_film_ass`` maps each shot's shot-local
cues onto the FILM bus (applying the shot's ``speed`` retime and clamping to the
shot's on-screen window from ``shot_offsets``) and emits one ASS document with a
single ``Cap`` style and one ``Dialogue`` per surviving cue. The caller writes
the returned text to a file under the ASCII-safe work root and feeds
``captions_vf_fragment`` into the SAME whole-film grade re-encode, so captions
never add a second encode and an empty document keeps the ``-c copy`` fast path
byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.schemas import CaptionCue, ShotCaptions
    from src.config import Settings


@dataclass(frozen=True)
class CaptionStyle:
    """Per-video caption look overrides (all optional; ``None`` ⇒ ``Settings`` default).

    Resolved from the run-level mood by :func:`resolve_caption_style` so the caption
    styling tracks the same mood token that already drives the colour grade. Every
    field left ``None`` falls through to the corresponding ``settings.caption_*``
    default, so an all-``None`` style (or ``None`` itself) is byte-identical to the
    pre-mood output. Colours are ASS ``&HAABBGGRR`` strings.
    """

    primary: str | None = None  # fill / karaoke-highlighted colour
    secondary: str | None = None  # karaoke un-highlighted colour (PrimaryColour before sweep)
    outline_colour: str | None = None
    font_size_frac: float | None = None
    outline_px: int | None = None
    bold: bool | None = None
    margin_v_frac: float | None = None


# Mood → caption look. Keys mirror the orchestrator colorist's mood vocabulary so a
# caption's look reads with the grade's look. An unknown/absent mood ⇒ ``None`` ⇒
# the plain ``settings.caption_*`` defaults (byte-identical to before this feature).
_CAPTION_STYLE_BY_MOOD: dict[str, CaptionStyle] = {
    "energetic": CaptionStyle(
        primary="&H0000E5FF",  # amber-yellow highlight
        secondary="&H00BBBBBB",
        font_size_frac=0.050,
        outline_px=3,
        bold=True,
    ),
    "confident": CaptionStyle(
        primary="&H00FFFFFF",
        secondary="&H00AAAAAA",
        outline_px=3,
        bold=True,
    ),
    "warm": CaptionStyle(
        primary="&H00A0D8FF",  # soft warm amber
        secondary="&H00889AAA",
        bold=True,
    ),
    "calm": CaptionStyle(
        primary="&H00FFFFFF",
        secondary="&H00999999",
        font_size_frac=0.042,
        outline_px=2,
    ),
    "moody": CaptionStyle(
        primary="&H00E0E0E0",
        secondary="&H00707070",
        outline_px=3,
    ),
    "neutral": CaptionStyle(),  # explicit no-op: mood recognised, look = defaults
}

# Fonts shipped WITH the service for burned-in captions, handed to libass via the
# ``ass`` filter's ``fontsdir`` (libass loads every file in that directory as a
# memory font, so the directory holds font files only — the OFL licence sits one
# level up). Inter 4.1 static TTFs from the official rsms/inter release
# (Inter-4.1.zip, extras/ttf/), SIL Open Font License 1.1: see
# ``src/assets/fonts/OFL-Inter.txt``. They match the film cards' typography
# (video_production_service's film kit sets Inter 600/700).
CAPTION_FONTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "caption"

# Caption font families a request may name (``CompileRequest.caption_font_family``)
# -> (ASS Fontname for regular weight, ASS Fontname for bold, needs fontsdir).
# Inter's regular caption weight is SemiBold (600): the SemiBold TTF's legacy
# family name is "Inter SemiBold", while "Inter" + Bold=-1 selects Inter-Bold
# (700). Measured with ffmpeg 7.1.5 / libass 0.17.3 (the image's Debian trixie
# build): "Inter" with Bold=0 does NOT resolve to the SemiBold face.
_CAPTION_FONT_FAMILIES: dict[str, tuple[str, str, bool]] = {
    "DejaVu Sans": ("DejaVu Sans", "DejaVu Sans", False),
    "Inter": ("Inter SemiBold", "Inter", True),
}
CAPTION_FONT_FAMILIES: tuple[str, ...] = tuple(_CAPTION_FONT_FAMILIES)


def hex_to_ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    """``#RRGGBB`` (+ ASS alpha, 0 = opaque .. 255 = transparent) -> ``&HAABBGGRR``."""
    rgb = hex_rgb.removeprefix("#")
    if len(rgb) != 6:
        raise ValueError(f"expected #RRGGBB, got {hex_rgb!r}")
    int(rgb, 16)  # raises on a non-hex digit
    rr, gg, bb = rgb[0:2], rgb[2:4], rgb[4:6]
    if not 0 <= alpha <= 255:
        raise ValueError(f"ASS alpha out of range: {alpha}")
    return f"&H{alpha:02X}{bb}{gg}{rr}".upper()


@dataclass(frozen=True)
class CaptionLook:
    """Per-request caption typography (all optional; ``None`` => today's look).

    Built from the ``CompileRequest.caption_*`` typography fields by
    :func:`caption_look_from_request`. Applied ON TOP of the mood style: the
    request names the brand's font and colours, the mood still sets size/weight.

    ``box_colour``/``box_opacity``: when EITHER is set the cue gets ONE continuous
    box (libass ``BorderStyle=4``: a single box around the whole event, filled with
    BackColour and padded by ``Outline``) instead of the default per-line boxes
    (``BorderStyle=3``), whose semi-opaque boxes overlap between lines and between
    karaoke words and draw darker stripes. OutlineColour goes fully transparent in
    that mode, because BorderStyle=4 also strokes every glyph in OutlineColour.
    """

    font_family: str | None = None
    text_colour: str | None = None  # #RRGGBB
    box_colour: str | None = None  # #RRGGBB
    box_opacity: float | None = None  # 0 (clear) .. 1 (solid)

    @property
    def is_default(self) -> bool:
        return (
            self.font_family is None
            and self.text_colour is None
            and self.box_colour is None
            and self.box_opacity is None
        )

    @property
    def fontsdir(self) -> Path | None:
        """The directory libass must load fonts from, or ``None`` (system fonts)."""
        if self.font_family is None:
            return None
        return CAPTION_FONTS_DIR if _CAPTION_FONT_FAMILIES[self.font_family][2] else None


def caption_look_from_request(
    font_family: str | None,
    text_colour: str | None,
    box_colour: str | None,
    box_opacity: float | None,
) -> CaptionLook | None:
    """The :class:`CaptionLook` for a request's typography fields, or ``None``
    when none is set (=> byte-identical to before these fields existed)."""
    look = CaptionLook(font_family, text_colour, box_colour, box_opacity)
    if look.is_default:
        return None
    if font_family is not None and font_family not in _CAPTION_FONT_FAMILIES:
        raise ValueError(f"unknown caption font family {font_family!r}")
    return look


# Aliases onto the canonical keys above (lower-cased on lookup).
_CAPTION_MOOD_ALIASES: dict[str, str] = {
    "upbeat": "energetic",
    "exciting": "energetic",
    "high-energy": "energetic",
    "serious": "confident",
    "professional": "confident",
    "bold": "confident",
    "friendly": "warm",
    "inviting": "warm",
    "relaxed": "calm",
    "gentle": "calm",
    "soft": "calm",
    "dramatic": "moody",
    "dark": "moody",
    "somber": "moody",
}


def resolve_caption_style(mood: str | None) -> CaptionStyle | None:
    """Map a run-level mood token to a :class:`CaptionStyle`, or ``None`` when the
    mood is empty/unrecognised (⇒ the caller renders with ``settings.caption_*``
    defaults, byte-identical to before per-mood styling)."""
    if not mood:
        return None
    key = mood.strip().lower()
    key = _CAPTION_MOOD_ALIASES.get(key, key)
    return _CAPTION_STYLE_BY_MOOD.get(key)


def _fmt_ts(seconds: float) -> str:
    """Format a film-bus time as an ASS timecode ``H:MM:SS.cc`` (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    centis = round(seconds * 100)
    hours, rem = divmod(centis, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """Neutralise ASS override-block delimiters in user text.

    ``{``/``}`` open/close an ASS override block, so a stray brace in caption
    text would swallow the following characters as a (bogus) tag. Map them to
    parentheses. Real newlines are collapsed by the word-based wrap in ``_wrap``.
    """
    return text.replace("{", "(").replace("}", ")")


def _wrap(text: str, max_chars: int, newline: str = r"\N") -> str:
    """Greedily wrap ``text`` into at most two lines joined by ``newline`` (the
    literal ASS newline ``\\N`` by default; SRT passes a real ``"\\n"``).
    Splitting on whitespace also collapses any real newlines. Returns ``""`` for
    blank text."""
    words = text.split()
    if not words:
        return ""
    line1: list[str] = []
    split_at = len(words)
    for i, word in enumerate(words):
        candidate = " ".join([*line1, word])
        if line1 and len(candidate) > max_chars:
            split_at = i
            break
        line1.append(word)
    rest = words[split_at:]
    if not rest:
        return " ".join(line1)
    return " ".join(line1) + newline + " ".join(rest)


def _iter_film_cues(
    shot_captions: list[ShotCaptions],
    shot_offsets: list[tuple[float, float]],
    speeds: list[float],
    settings: Settings,
) -> Iterator[tuple[float, float, str]]:
    """Yield ``(film_start, film_end, raw_text)`` for each surviving cue, mapping
    each shot's shot-local cue times onto the film bus (offset + ÷speed, clamped
    to the shot window). For shot ``k`` with bus window
    ``shot_offsets[k] = (seg_start, seg_end)`` and playback ``speeds[k]``::

        film_start = seg_start + cue.start / speed
        film_end   = min(seg_start + cue.end / speed, seg_end)

    Cues with an empty clamped window (``film_end <= film_start``) or blank text
    are skipped. Text is the RAW cue text — callers apply their own
    escaping/wrapping. ``settings`` is accepted for signature symmetry with the
    callers (the shared mapping does not itself need it). Thin projection of
    :func:`_iter_film_cues_full` so both the phrase and karaoke paths share ONE
    copy of the offset/÷speed/clamp math (no drift)."""
    del settings  # kept for signature symmetry; mapping needs no settings field
    for film_start, film_end, cue, *_ in _iter_film_cues_full(shot_captions, shot_offsets, speeds):
        yield film_start, film_end, cue.text


def _iter_film_cues_full(
    shot_captions: list[ShotCaptions],
    shot_offsets: list[tuple[float, float]],
    speeds: list[float],
) -> Iterator[tuple[float, float, CaptionCue, float, float, float]]:
    """Like :func:`_iter_film_cues` but also yields the source ``cue`` and its
    shot's ``(seg_start, seg_end, speed)`` so a caller (karaoke) can map the cue's
    per-word times onto the film bus with the SAME offset/÷speed/clamp math.

    Survival rules and their ORDER are identical to :func:`_iter_film_cues`
    (empty clamped window, then blank text), so the set of surviving cues — and
    thus the non-karaoke output — is byte-identical."""
    n_shots = len(shot_offsets)
    for shot_caption in shot_captions:
        k = shot_caption.shot_index
        if k < 0 or k >= n_shots or k >= len(speeds):
            continue
        seg_start, seg_end = shot_offsets[k]
        speed = speeds[k]
        if speed <= 0:
            speed = 1.0
        for cue in shot_caption.cues:
            film_start = seg_start + cue.start / speed
            film_end = min(seg_start + cue.end / speed, seg_end)
            if film_end <= film_start:
                continue
            if not cue.text.split():
                continue
            yield film_start, film_end, cue, seg_start, seg_end, speed


def _karaoke_text(
    cue: CaptionCue,
    film_start: float,
    film_end: float,
    seg_start: float,
    seg_end: float,
    speed: float,
) -> str:
    """Render a cue's ``words`` as ASS ``{\\k<cs>}word`` karaoke chunks, or ``""``
    when the cue has no words.

    Each word's shot-local ``[start, end)`` is mapped onto the film bus with the
    SAME ``seg_start + t/speed`` (clamped to ``[film_start, film_end]``) math as
    the cue window, so word highlight can't drift from the cue timing. ``\\k``
    durations are centiseconds RELATIVE to the Dialogue start: a leading/inter-word
    gap is emitted as a text-less ``{\\k<gap>}`` that just advances the karaoke
    clock. The word text is ASS-escaped BEFORE the ``\\k`` braces are prepended, so
    a stray brace in a word can't open a bogus override block. No explicit ``\\N``
    wrap — libass ``WrapStyle 0`` re-wraps an over-long line inside the margin box
    while preserving the ``\\k`` tags."""
    words = cue.words or []
    chunks: list[str] = []
    prev = film_start
    for word in words:
        wfs = seg_start + word.start / speed
        wfe = min(seg_start + word.end / speed, seg_end)
        wfs = min(max(wfs, film_start), film_end)
        wfe = min(max(wfe, wfs), film_end)
        gap_cs = round((wfs - prev) * 100)
        dur_cs = round((wfe - wfs) * 100)
        if gap_cs > 0:
            chunks.append(f"{{\\k{gap_cs}}}")
        chunks.append(f"{{\\k{max(dur_cs, 0)}}}{_escape(word.text)} ")
        prev = wfe
    return "".join(chunks).strip()


def build_film_ass(
    shot_captions: list[ShotCaptions],
    shot_offsets: list[tuple[float, float]],
    speeds: list[float],
    settings: Settings,
    style: CaptionStyle | None = None,
    karaoke: bool = False,
    look: CaptionLook | None = None,
) -> str:
    """Build the whole-film ASS caption document, or ``""`` when no cue survives.

    For shot ``k`` (addressed by ``ShotCaptions.shot_index``) with bus window
    ``shot_offsets[k] = (seg_start, seg_end)`` and playback ``speeds[k]``::

        film_start = seg_start + cue.start / speed
        film_end   = min(seg_start + cue.end / speed, seg_end)

    A cue whose clamped window is empty (``film_end <= film_start``) or whose
    text is blank is skipped.

    ``style`` overrides the settings-derived ``Cap`` style per field (``None`` ⇒
    the ``settings.caption_*`` default); ``style is None`` is byte-identical to the
    pre-mood output. ``karaoke=True`` renders each cue's ``words`` as ``{\\k}``
    word-highlight chunks (falling back to plain text when a cue has no words).
    ``look`` applies the request's caption typography (font family, text colour,
    one continuous box per cue); ``None`` is byte-identical to before it existed.
    """
    dialogues: list[str] = []
    for film_start, film_end, cue, seg_start, seg_end, speed in _iter_film_cues_full(
        shot_captions, shot_offsets, speeds
    ):
        if karaoke and cue.words:
            text = _karaoke_text(cue, film_start, film_end, seg_start, seg_end, speed)
        else:
            text = _wrap(_escape(cue.text), settings.caption_max_line_chars)
        if not text:
            continue
        dialogues.append(
            f"Dialogue: 0,{_fmt_ts(film_start)},{_fmt_ts(film_end)},Cap,,0,0,0,,{text}"
        )
    if not dialogues:
        return ""

    font_size_frac = (
        style.font_size_frac
        if style is not None and style.font_size_frac is not None
        else settings.caption_font_size_frac
    )
    margin_v_frac = (
        style.margin_v_frac
        if style is not None and style.margin_v_frac is not None
        else settings.caption_margin_v_frac
    )
    outline_px = (
        style.outline_px
        if style is not None and style.outline_px is not None
        else settings.caption_outline_px
    )
    primary = style.primary if style is not None and style.primary else settings.caption_primary
    # SecondaryColour is only meaningful on the karaoke (\k) path, where un-swept
    # words are drawn in it; it is inert for non-karaoke renders. The default was
    # &H000000FF (ASS &HAABBGGRR = opaque bright RED), which flashed every
    # not-yet-highlighted word red. Fall back to opaque dim mid-grey instead.
    secondary = (
        style.secondary if style is not None and style.secondary else settings.caption_secondary
    )
    outline_colour = (
        style.outline_colour
        if style is not None and style.outline_colour
        else settings.caption_outline_colour
    )
    is_bold = bool(style is not None and style.bold)
    bold = "-1" if is_bold else "0"
    font_name = settings.caption_font_family
    if look is not None and look.font_family is not None:
        regular, bold_name, _ = _CAPTION_FONT_FAMILIES[look.font_family]
        font_name = bold_name if is_bold else regular
    if look is not None and look.text_colour is not None:
        primary = hex_to_ass_colour(look.text_colour)

    font_size = round(settings.render_height * font_size_frac)
    margin_v = round(settings.render_height * margin_v_frac)
    # Scaled off render_width the same way margin_v scales off render_height.
    # 20/1920 keeps the default 1920x1080 delivery byte-identical (20px) while
    # fixing the ~0.5% gutter a fixed 20px produced at 3840px wide (now ~1.04%,
    # matching the ~1.85%-at-1080px-wide / ~0.5%-at-3840px-wide disparity the
    # fixed value produced).
    margin_h = round(settings.render_width * (20 / 1920))
    # Backplate (default ON). A thin outline alone is illegible over busy UI
    # footage, so each line gets a semi-opaque box: libass BorderStyle=3 draws an
    # opaque box per line, filled with OutlineColour and extending ``Outline`` px
    # past the glyphs — so in this mode OutlineColour IS the plate colour and
    # Outline IS the padding (BackColour is set to the same plate colour so a
    # renderer that fills from BackColour draws the same box). The padding is a
    # fraction of the font size, so the plate keeps its proportions at any
    # render_height (1080p and 4k alike). Mood ``outline_px``/``outline_colour``
    # describe a glyph outline, which this mode does not draw.
    if settings.caption_box_enabled:
        border_style = 3
        outline_colour = settings.caption_box_colour
        back_colour = settings.caption_box_colour
        outline_px = max(1, round(font_size * settings.caption_box_padding_frac))
    else:
        border_style = 1
        back_colour = "&H00000000"
    if look is not None and (look.box_colour is not None or look.box_opacity is not None):
        # ONE box per cue (see CaptionLook). Unset colour/opacity fall back to the
        # default plate's black / alpha so either field alone is meaningful.
        default_alpha = int(settings.caption_box_colour[2:4], 16)
        alpha = (
            round((1 - look.box_opacity) * 255) if look.box_opacity is not None else default_alpha
        )
        border_style = 4
        back_colour = hex_to_ass_colour(look.box_colour or "#000000", alpha)
        outline_colour = "&HFF000000"
        outline_px = max(1, round(font_size * settings.caption_box_padding_frac))
    style_line = (
        f"Style: Cap,{font_name},{font_size},"
        f"{primary},{secondary},{outline_colour},"
        f"{back_colour},{bold},0,0,0,100,100,0,0,{border_style},{outline_px},0,2,"
        f"{margin_h},{margin_h},{margin_v},1"
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {settings.render_width}",
        f"PlayResY: {settings.render_height}",
        # WrapStyle 0 (smart wrap) honours our explicit \N line breaks for the
        # normal 2-line layout, but also re-wraps any line that still overflows the
        # MarginL/MarginR box onto further lines — so an over-long phrase degrades
        # to an extra line inside the frame instead of clipping off both edges.
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        # ⛔ Explicit, never omitted: with NO ``YCbCr Matrix`` line libass
        # treats the script as legacy ``TV.601`` and newer libass/ffmpeg (0.17.5
        # with ffmpeg 9) honour that and re-map every colour through BT.601 —
        # measured: the 70% navy caption box came out (30,85,95), not the
        # requested (21,76,90). ``None`` = the colours ARE the RGB values we
        # wrote; older libass, which ignored the header, renders the same.
        "YCbCr Matrix: None",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding"
        ),
        style_line,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *dialogues,
        "",
    ]
    return "\n".join(lines)


def _fmt_srt_ts(seconds: float) -> str:
    """Format a film-bus time as an SRT timecode ``HH:MM:SS,mmm`` (milliseconds)."""
    if seconds < 0:
        seconds = 0.0
    millis = round(seconds * 1000)
    hours, rem = divmod(millis, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_film_srt(
    shot_captions: list[ShotCaptions],
    shot_offsets: list[tuple[float, float]],
    speeds: list[float],
    settings: Settings,
) -> str:
    """Build a whole-film SRT document (soft/overlay captions), or ``""`` when no
    cue survives. Reuses ``_iter_film_cues`` so the timing matches
    ``build_film_ass`` exactly; text is wrapped to <=2 lines with real newlines
    and is NOT ASS-escaped (SRT has no override-block syntax)."""
    blocks: list[str] = []
    idx = 0
    for film_start, film_end, raw in _iter_film_cues(shot_captions, shot_offsets, speeds, settings):
        text = _wrap(raw, settings.caption_max_line_chars, newline="\n")
        if not text:
            continue
        idx += 1
        blocks.append(f"{idx}\n{_fmt_srt_ts(film_start)} --> {_fmt_srt_ts(film_end)}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def captions_vf_fragment(ass_path: Path, fontsdir: Path | None = None) -> str:
    """Return the ``ass=`` filtergraph fragment for ``ass_path``.

    The path lives under the controlled, ASCII/no-colon work root, so no heavy
    filtergraph escaping is needed; single-quoting guards against spaces.
    ``fontsdir`` (the service's shipped caption fonts, see ``CAPTION_FONTS_DIR``)
    is added only when the request named an embedded family, so a request without
    one keeps the exact pre-existing fragment.
    """
    if fontsdir is None:
        return f"ass=filename='{ass_path}'"
    return f"ass=filename='{ass_path}':fontsdir='{fontsdir}'"
