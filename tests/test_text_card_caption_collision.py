"""The text-card body must not overlap its own caption.

``renderer.py`` centres the text-card body (``y=(h-text_h)/2``) and the
caption (``y=h-text_h-margin_v``) as two INDEPENDENT ``drawtext`` filters in
the same chain — neither has any idea the other exists. A 3-4 line caption
plus a long wrapped body can overlap with no guard against it.

The fix reserves the caption's own footprint (``_caption_top_px``) when a
caption is present and centers the body in the remaining band above it
(``_text_card_body_y``), instead of across the whole frame. Absent a
caption the body keeps ffmpeg's own dynamic ``(h-text_h)/2`` — a real font
metric — so that case is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import Settings, get_settings
from src.services import renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


async def _render_text_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shot: Shot, settings: Settings
) -> list[str]:
    captured: dict[str, list[str]] = {}

    async def fake_run(cmd: list[str]) -> str:
        if cmd[0] == "ffmpeg":
            captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    return captured["cmd"]


# ---------------------------------------------------------------------------
# Pure geometry — the collision math itself.
# ---------------------------------------------------------------------------


def test_body_y_is_unchanged_dynamic_expr_without_a_caption() -> None:
    """No caption -> byte-identical to the pre-fix behaviour: ffmpeg's own
    dynamic text_h, not a Python-side numeric estimate."""
    settings = _settings()
    assert renderer._text_card_body_y(None, "short body", 56, settings) == "(h-text_h)/2"
    assert renderer._text_card_body_y("", "short body", 56, settings) == "(h-text_h)/2"


def test_body_y_is_a_fixed_number_above_the_caption_when_present() -> None:
    settings = _settings()
    caption = "A caption long enough to wrap onto two or three separate lines here"
    body = "line one\nline two\nline three"
    y = renderer._text_card_body_y(caption, body, 56, settings)
    assert y != "(h-text_h)/2"
    y_val = int(y)
    caption_top = renderer._caption_top_px(caption, settings)
    body_block_h = renderer._text_block_height_px(
        body, 56, renderer._TEXT_CARD_BODY_LINE_SPACING_PX
    )
    # The body's bottom edge (numeric estimate) must not cross into the
    # reserved caption band.
    assert y_val + body_block_h <= caption_top


def test_body_y_clamps_to_zero_and_warns_when_body_exceeds_the_reserved_band() -> None:
    """Force the actual overlap-impossible branch: a caption that eats nearly
    the whole frame plus a many-line body whose footprint exceeds what is
    left. Before the fix this silently fell through the SAME
    ``max(0, (caption_top - body_block_h) // 2)`` arithmetic as the benign
    case (still returning 0, but with no signal anything went wrong). The
    real behaviour worth asserting is the WARNING — that is what makes this
    non-tautological versus the benign-regime test above, which never
    reaches this branch at all."""
    settings = _settings()
    # A caption long enough to wrap across many lines, eating most of the
    # frame height and leaving a tiny reserved band above it.
    huge_caption = " ".join(["word"] * 400)
    caption_top = renderer._caption_top_px(huge_caption, settings)
    assert caption_top < settings.render_height * 0.3  # sanity: band really is tiny

    # A body whose own multi-line footprint clearly exceeds that tiny band.
    body = "\n".join(["a long body line of text"] * 10)
    body_fontsize = round(settings.render_height * renderer._TEXT_CARD_BODY_FONTSIZE_FRAC)
    body_block_h = renderer._text_block_height_px(
        body, body_fontsize, renderer._TEXT_CARD_BODY_LINE_SPACING_PX
    )
    assert body_block_h > caption_top  # sanity: we are actually in the clamp branch

    logged: list[dict[str, object]] = []
    orig_warning = renderer.log.warning

    def fake_warning(event: str, **kw: object) -> None:
        logged.append({"event": event, **kw})

    renderer.log.warning = fake_warning  # type: ignore[method-assign]
    try:
        y = renderer._text_card_body_y(huge_caption, body, body_fontsize, settings)
    finally:
        renderer.log.warning = orig_warning  # type: ignore[method-assign]

    assert y == "0"
    assert any(entry["event"] == "text_card_body_exceeds_reserved_band" for entry in logged)


def test_caption_top_px_reserves_more_room_for_more_caption_lines() -> None:
    settings = _settings()
    short_top = renderer._caption_top_px("one short line", settings)
    long_caption = " ".join(["word"] * 40)  # wraps to several lines
    long_top = renderer._caption_top_px(long_caption, settings)
    assert long_top < short_top < settings.render_height


def test_caption_top_px_is_full_height_without_a_caption() -> None:
    settings = _settings()
    assert renderer._caption_top_px(None, settings) == settings.render_height
    assert renderer._caption_top_px("", settings) == settings.render_height


# ---------------------------------------------------------------------------
# End-to-end through render_shot (ffmpeg mocked) — the real -vf string.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_card_with_long_caption_and_long_body_do_not_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings()
    long_description = (
        "This is a long description that will wrap across several lines on the "
        "text card canvas because it contains many words describing the scene "
        "in full sentences rather than a short title fragment."
    )
    long_caption = (
        "And this is the verbatim narration caption line for this beat, which is "
        "also long enough to wrap across multiple lines at the bottom of frame."
    )
    shot = Shot(id="cc1", seconds=4.0, description=long_description, caption=long_caption)
    cmd = await _render_text_card(monkeypatch, tmp_path, shot, settings)
    joined = " ".join(cmd)

    # Body no longer centers across the whole frame once a caption is present.
    assert "y=(h-text_h)/2" not in joined

    # Recompute what the body's fixed y SHOULD be and confirm it appears
    # verbatim in the emitted filter — this fails if the collision logic is
    # bypassed or its numbers drift from what render_shot actually emits.
    body_fontsize = round(settings.render_height * renderer._TEXT_CARD_BODY_FONTSIZE_FRAC)
    body_text = renderer._wrap_for_canvas(
        long_description, fontsize=body_fontsize, canvas_width=settings.render_width
    )
    expected_y = renderer._text_card_body_y(long_caption, body_text, body_fontsize, settings)
    assert f":y={expected_y}" in joined

    # And the geometric invariant: body's reserved band stays clear of the
    # caption's reserved band.
    caption_top = renderer._caption_top_px(long_caption, settings)
    body_block_h = renderer._text_block_height_px(
        body_text, body_fontsize, renderer._TEXT_CARD_BODY_LINE_SPACING_PX
    )
    assert int(expected_y) + body_block_h <= caption_top


@pytest.mark.asyncio
async def test_text_card_without_caption_keeps_dynamic_centering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression guard: a shot with no caption must still use ffmpeg's own
    dynamic centering, proving the fix is additive, not a general rewrite."""
    settings = _settings()
    shot = Shot(id="cc2", seconds=2.0, description="Hello world")
    cmd = await _render_text_card(monkeypatch, tmp_path, shot, settings)
    joined = " ".join(cmd)
    assert "y=(h-text_h)/2" in joined
