"""Pure-unit tests for the soft/overlay caption SRT builder (overlay mode).

ffmpeg is NOT invoked: these assert the ``build_film_srt`` offset/speed/clamp
MATH and the SRT timecode/format directly on the emitted SRT document string.
``build_film_srt`` reuses ``captions_ass._iter_film_cues`` — the SAME film-bus
mapping as ``build_film_ass`` — so a word-timed cue renders identically whether
burned (ASS) or muxed as a soft ``mov_text`` track.

The film-bus mapping under test: for shot ``k`` with bus window
``shot_offsets[k] = (seg_start, seg_end)`` and playback ``speeds[k]`` ::

    film_start = seg_start + cue.start / speed
    film_end   = min(seg_start + cue.end / speed, seg_end)

so a shot-local cue is retimed by ÷speed and clamped to the shot's on-screen
window — NOT a naive ``seg_start + cue.start`` sum.
"""

from __future__ import annotations

from src.api.schemas import CaptionCue, ShotCaptions
from src.config import Settings, get_settings
from src.services.captions_ass import _fmt_srt_ts, build_film_srt


def _settings(**overrides: object) -> Settings:
    """Real Settings with known caption/render fields; overridable per test."""
    return get_settings().model_copy(
        update={"render_width": 1280, "render_height": 720, **overrides}
    )


def _blocks(srt: str) -> list[str]:
    """Split an SRT document into its cue blocks (dropping the trailing newline)."""
    return srt.rstrip("\n").split("\n\n") if srt else []


# --------------------------------------------------------------------------
# absolute mapping + ÷speed + clamp (the core of the feature)
# --------------------------------------------------------------------------


def test_cue_maps_to_bus_offset_and_divides_by_speed() -> None:
    """A cue on shot 1 (bus window 2.0..3.0, speed 2.0, shot-local 0.2..1.0) maps
    to film 2.0+0.2/2=2.1 .. 2.0+1.0/2=2.5 → ``00:00:02,100 --> 00:00:02,500``.

    This proves offset + ÷speed, NOT a naive ``seg_start + cue.start`` sum (which
    would wrongly yield ``00:00:02,200 --> 00:00:03,000``)."""
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="hi", start=0.2, end=1.0)])]
    srt = build_film_srt(caps, [(0.0, 2.0), (2.0, 3.0)], [1.0, 2.0], _settings())
    blocks = _blocks(srt)
    assert len(blocks) == 1
    assert "00:00:02,100 --> 00:00:02,500" in blocks[0]
    # a naive seg_start+cue.start impl would emit these — must be absent.
    assert "00:00:02,200" not in srt
    assert "00:00:03,000" not in srt


def test_end_is_clamped_to_shot_window() -> None:
    """A cue whose ÷speed end overruns the shot's on-screen window is clamped to
    ``shot_offsets[k][1]`` — the caption cannot outlast the shot it belongs to."""
    # shot 1 window is only 3.0..4.0; cue end 2.0 → 3.0+2.0=5.0, clamped to 4.0.
    caps = [ShotCaptions(shot_index=1, cues=[CaptionCue(text="hi", start=0.5, end=2.0)])]
    srt = build_film_srt(caps, [(0.0, 3.0), (3.0, 4.0)], [1.0, 1.0], _settings())
    assert "00:00:03,500 --> 00:00:04,000" in srt


# --------------------------------------------------------------------------
# timecode format
# --------------------------------------------------------------------------


def test_fmt_srt_ts_formats_hms_millis() -> None:
    assert _fmt_srt_ts(2.1) == "00:00:02,100"
    assert _fmt_srt_ts(3661.5) == "01:01:01,500"


def test_fmt_srt_ts_clamps_negative_to_zero() -> None:
    assert _fmt_srt_ts(-5.0) == "00:00:00,000"


# --------------------------------------------------------------------------
# empty / no-op inputs → "" (overlay track omitted, no soft-sub muxed)
# --------------------------------------------------------------------------


def test_no_shot_captions_returns_empty_document() -> None:
    assert build_film_srt([], [(0.0, 1.0)], [1.0], _settings()) == ""


def test_shot_captions_with_no_cues_returns_empty_document() -> None:
    caps = [ShotCaptions(shot_index=0, cues=[])]
    assert build_film_srt(caps, [(0.0, 1.0)], [1.0], _settings()) == ""


def test_all_blank_text_returns_empty_document() -> None:
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="   ", start=0.0, end=1.0)])]
    assert build_film_srt(caps, [(0.0, 2.0)], [1.0], _settings()) == ""


# --------------------------------------------------------------------------
# index increments (1-based, per emitted cue)
# --------------------------------------------------------------------------


def test_index_increments_per_cue() -> None:
    caps = [
        ShotCaptions(
            shot_index=0,
            cues=[
                CaptionCue(text="first", start=0.0, end=1.0),
                CaptionCue(text="second", start=1.0, end=2.0),
            ],
        )
    ]
    srt = build_film_srt(caps, [(0.0, 5.0)], [1.0], _settings())
    blocks = _blocks(srt)
    assert len(blocks) == 2
    assert blocks[0].startswith("1\n")
    assert blocks[1].startswith("2\n")
    assert srt.endswith("\n")


# --------------------------------------------------------------------------
# SRT is NOT ASS-escaped (no override-block syntax to neutralise)
# --------------------------------------------------------------------------


def test_srt_text_is_not_ass_escaped() -> None:
    """A brace ``{`` survives verbatim in SRT (ASS maps it to ``(``)."""
    caps = [ShotCaptions(shot_index=0, cues=[CaptionCue(text="a {tag} b", start=0.0, end=1.0)])]
    srt = build_film_srt(caps, [(0.0, 5.0)], [1.0], _settings())
    assert "{tag}" in srt
    assert "(tag)" not in srt


# --------------------------------------------------------------------------
# ≤2-line wrap uses a REAL newline (not the ASS literal \N)
# --------------------------------------------------------------------------


def test_long_text_wraps_with_real_newline() -> None:
    settings = _settings(caption_max_line_chars=20)
    caps = [
        ShotCaptions(
            shot_index=0,
            cues=[CaptionCue(text="hello there my good friends", start=0.0, end=2.0)],
        )
    ]
    srt = build_film_srt(caps, [(0.0, 10.0)], [1.0], settings)
    block = _blocks(srt)[0]
    # block is "1\n<start> --> <end>\n<line1>\n<line2>" — the text carries a real
    # newline between the two wrapped lines, and NEVER the ASS literal "\N".
    text = block.split("\n", 2)[2]
    assert "\n" in text, "a long caption did not wrap onto a second line"
    assert r"\N" not in srt, "SRT must use a real newline, not the ASS literal \\N"
    first_line = text.split("\n")[0]
    assert len(first_line) <= 20
