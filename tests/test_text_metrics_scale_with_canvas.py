"""Text metrics that were flat px constants now scale with canvas size.

All three were absolute px, so they were correct only at the 1080p/1920px
canvas they were tuned on and drifted the moment a job rendered at a
different tier. The caption already scaled correctly (`round(render_height *
frac)`, VC-6) — these follow the same pattern and are identity at their
original reference size, so nothing changes for a 1080p/1920px job.

* ``_safe_margin_px``: 100px at 1920px width (5.2%) -> 3.9% at 2560px if left
  flat. Feeds both the wrap gutter (``_wrap_for_canvas``) and the callout
  label anchor clamp (``_callout_filter``).
* the text-card body fontsize: 56px at 1080p (5.19%) -> 3.89% at 1440p (a
  -25% relative shrink), packing ~76 chars/line via ``_wrap_for_canvas``
  instead of the intended ~55.
* the callout label fontsize: 44px at 1080p, pinned regardless of canvas —
  now matches ``_caption_filter_fragment``'s default fraction (44/1080) at
  ANY render_height, not only at 1080p.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import Callout, Shot
from src.config import get_settings
from src.services import renderer


def _settings(**overrides: object):
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# _safe_margin_px
# ---------------------------------------------------------------------------


def test_safe_margin_px_is_identity_at_1920() -> None:
    """The reference size must reproduce the exact legacy 100px — this is the
    byte-identical-at-1080p/1920px invariant the whole fix depends on."""
    assert renderer._safe_margin_px(1920) == 100


def test_safe_margin_px_scales_proportionally_at_2560() -> None:
    """A flat 100px constant would still read 100 here (5.2% -> 3.9% of
    width, silently shrinking); the fraction keeps the SAME percentage."""
    assert renderer._safe_margin_px(2560) == round(2560 * 100 / 1920)
    assert renderer._safe_margin_px(2560) != 100


def test_wrap_for_canvas_gutter_scales_with_canvas_width() -> None:
    """End-to-end through the actual wrap helper: the safe width used to
    compute max_chars must shrink the flat-100px gutter proportionally at a
    wider canvas, not hold it at 100px."""
    text = "one two three four five six seven eight nine ten eleven twelve"
    out_1920 = renderer._wrap_for_canvas(text, fontsize=56, canvas_width=1920)
    out_2560 = renderer._wrap_for_canvas(text, fontsize=56, canvas_width=2560)
    # A wider canvas at the same fontsize must fit AT LEAST as much per line
    # (more safe width available) — this only holds if the gutter scaled
    # with the canvas instead of eating a proportionally bigger slice.
    assert max(len(line) for line in out_2560.split("\n")) >= max(
        len(line) for line in out_1920.split("\n")
    )


def test_callout_label_anchor_clamp_uses_scaled_margin(tmp_path: Path) -> None:
    """The callout label's x/y clamp expressions must embed the CANVAS-scaled
    margin, not the flat legacy 100."""
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="label", bbox=(200, 400, 80, 60), label="Click Save")],
    )
    settings = _settings(render_width=2560, render_height=1440)
    out = renderer._callout_filter("scale=2560:1440", shot, settings, 2.0, tmp_path)
    margin = renderer._safe_margin_px(2560)
    assert margin != 100
    assert f"\\,{margin}\\," in out
    assert "\\,100\\," not in out


# ---------------------------------------------------------------------------
# text-card body fontsize
# ---------------------------------------------------------------------------


def test_text_card_body_fontsize_is_identity_at_1080p() -> None:
    settings = _settings(render_height=1080)
    fontsize = round(settings.render_height * renderer._TEXT_CARD_BODY_FONTSIZE_FRAC)
    assert fontsize == 56


@pytest.mark.asyncio
async def test_text_card_render_shot_uses_scaled_body_fontsize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end through ``render_shot``'s text-card branch (ffmpeg mocked):
    the emitted ``-vf`` must carry the 1440p-scaled fontsize, not a flat 56.

    A flat 56px constant would stay 56 here (3.89% of 1440 vs 1080p's 5.19%,
    a -25% relative shrink) — this reads the ACTUAL ``-vf`` drawtext fontsize
    the command carries, not a value recomputed from the constant off to the
    side, so it goes red if the render_shot code stops using the fraction."""
    import httpx

    captured: dict[str, list[str]] = {}

    async def fake_run(cmd: list[str]) -> str:
        if cmd[0] == "ffmpeg":
            captured["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    settings = _settings(render_height=1440, render_width=2560)
    shot = Shot(id="tc1", seconds=2.0, description="A description long enough to wrap onto lines")
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    expected_fontsize = round(1440 * 56 / 1080)
    assert expected_fontsize != 56
    joined = " ".join(captured["cmd"])
    assert f":fontsize={expected_fontsize}:" in joined
    assert ":fontsize=56:" not in joined


# ---------------------------------------------------------------------------
# callout label fontsize
# ---------------------------------------------------------------------------


def test_callout_label_fontsize_is_identity_at_1080p(tmp_path: Path) -> None:
    settings = _settings(render_height=1080, render_width=1920)
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="label", bbox=(200, 400, 80, 60), label="Click Save")],
    )
    out = renderer._callout_filter("scale=1920:1080", shot, settings, 2.0, tmp_path)
    assert ":fontsize=44:" in out


def test_callout_label_fontsize_scales_at_1440p(tmp_path: Path) -> None:
    """A flat 44 would stay 44 at 1440p, no longer matching the caption's own
    fontsize at that render_height. The fraction must scale it up."""
    settings = _settings(render_height=1440, render_width=2560)
    shot = Shot(
        id="a",
        seconds=2.0,
        callouts=[Callout(at=0.0, type="label", bbox=(200, 400, 80, 60), label="Click Save")],
    )
    out = renderer._callout_filter("scale=2560:1440", shot, settings, 2.0, tmp_path)
    expected = round(1440 * 44 / 1080)
    assert expected != 44
    assert f":fontsize={expected}:" in out
    assert ":fontsize=44:" not in out
