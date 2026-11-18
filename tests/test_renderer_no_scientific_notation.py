"""A sub-millisecond duration must never reach ffmpeg as ``4.2e-05``.

A real ``lesson_v1`` render died outright on this, mid-``lesson_compile``::

    [Parsed_tpad_6] Unable to parse option value "4.2e-05" as duration
    Error applying option 'stop_duration' to filter 'tpad': Invalid argument
    ffmpeg exit=234

``_ffnum`` was ``f"{value:g}"``, and ``%g`` switches to scientific notation below
1e-4. Every ``AV_OPT_TYPE_DURATION`` option (``tpad:stop_duration``,
``trim:duration``, ``afade:d``, ``color:d``) and the CLI ``-t`` are parsed by
``av_parse_time``, which rejects an exponent — so the render failed after the
work was already done and the money already spent.

⭐ EVERY assertion here reads the EMITTED ARGUMENT STRING, never the input float.
A test that checked "the pad is small" would have passed against the bug: the
input was always a fine float, and the defect lived entirely in the formatting.
So each test greps the produced fragment for an ``e``-exponent and pins the exact
replacement token.
"""

from __future__ import annotations

import re

import pytest

from src.api.schemas import ShotAudio
from src.config import Settings, get_settings
from src.services.renderer import (
    _COVER_PAD_MARGIN_SECONDS,
    _apply_shot_audio,
    _ffdur,
    _ffnum,
    _freeze_pad_suffix,
)

# The literal shape from the production failure: a 12.0s narration line against a
# clip that probed 42 microseconds shorter.
_CRASH_DURATION = 12.0
_CRASH_CLIP_DUR = 12.0 - 4.2e-05

# An exponent in an ffmpeg numeric argument, e.g. ``4.2e-05`` or ``1e+20``. Matched
# against a whole emitted fragment, so it is anchored on a digit to avoid colliding
# with the ``e`` in filter names (``afade``, ``scale``, ``setpts``, ``zoompan``).
_EXPONENT = re.compile(r"\d[eE][+-]?\d")


def _settings() -> Settings:
    return get_settings()


def _assert_no_exponent(fragment: str) -> None:
    """The gate every test in this file runs: no exponent anywhere in the string."""
    found = _EXPONENT.search(fragment)
    assert found is None, (
        f"scientific notation {found.group(0)!r} emitted into an ffmpeg argument; "
        f"av_parse_time rejects it and the render dies with exit=234: {fragment!r}"
    )


# ---------------------------------------------------------------------------
# The formatters themselves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [4.2e-05, 1e-05, 9.9e-05, 1e-09, -4.2e-05, 1e20, 1e-12],
)
def test_ffnum_never_emits_an_exponent(value: float) -> None:
    """``_ffnum`` is exponent-free at BOTH ends of ``%g``'s plain range."""
    _assert_no_exponent(_ffnum(value))


def test_ffnum_renders_the_crash_value_as_plain_decimal() -> None:
    """The exact value from the production log, pinned to its replacement token."""
    assert _ffnum(4.2e-05) == "0.000042"
    assert _ffnum(1e-09) == "0.000000001"
    assert _ffnum(1e-12) == "0"  # underflows ffmpeg's own resolution -> plain zero


def test_ffnum_is_byte_identical_inside_percent_g_range() -> None:
    """The fix must not churn any argument that was already fine — every existing
    emitted string keeps its exact ``%g`` spelling."""
    assert _ffnum(30.0) == "30"
    assert _ffnum(0.03) == "0.03"
    assert _ffnum(2.5) == "2.5"
    assert _ffnum(0.0) == "0"
    assert _ffnum(0.0001) == "0.0001"  # the boundary %g still renders plainly


def test_ffdur_keeps_full_precision_unlike_percent_g() -> None:
    """``_ffdur`` exists because ``%g`` would truncate a segment-terminating ``-t``
    to 6 significant digits, moving the bound by ~0.2 ms."""
    assert _ffdur(123.456789) == "123.456789"
    assert _ffnum(123.456789) == "123.457"  # why _ffdur is a separate formatter
    assert _ffdur(4.2e-05) == "0.000042"
    _assert_no_exponent(_ffdur(4.2e-05))


# ---------------------------------------------------------------------------
# The real emit site that crashed
# ---------------------------------------------------------------------------
def test_freeze_pad_skips_the_degenerate_sub_frame_pad() -> None:
    """A 42µs pad is below one output frame, so ``tpad`` could not clone a frame for
    it even if ffmpeg accepted the argument. It is omitted entirely rather than
    emitted as a degenerate filter."""
    suffix = _freeze_pad_suffix(True, _CRASH_CLIP_DUR, _CRASH_DURATION, _settings())
    _assert_no_exponent(suffix)
    assert "stop_duration" not in suffix, (
        "a sub-frame freeze pad must be skipped, not emitted: " + repr(suffix)
    )
    assert suffix == ""


def test_freeze_pad_keeps_the_coverage_clone_when_bounded() -> None:
    """Skipping the degenerate pad must NOT drop the terminal coverage clone — that
    is what holds the picture out to the output-side ``-t``. It must also be
    FINITE, never ``stop=-1``: this exact shape (a sub-frame, ~42µs gap) is what
    livelocked a real render (#232) once a second, finite audio input joined the
    same graph — see ``tests/render_harness/test_freeze_pad_bounded_livelock.py``.

    The clone's ``stop_duration`` is ceiling'd at the WHOLE segment
    ``_CRASH_DURATION``, not the (probe-derived, lying-header-vulnerable) gap —
    see ``_bounded_cover_pad``'s docstring."""
    suffix = _freeze_pad_suffix(True, _CRASH_CLIP_DUR, _CRASH_DURATION, _settings(), bounded=True)
    _assert_no_exponent(suffix)
    assert "stop=-1" not in suffix
    ceiling = _CRASH_DURATION + _COVER_PAD_MARGIN_SECONDS
    assert suffix == f",fps=30,tpad=stop_mode=clone:stop_duration={_ffnum(ceiling)}"


def test_freeze_pad_still_emits_a_real_pad() -> None:
    """The epsilon must not swallow a pad that genuinely has work to do."""
    suffix = _freeze_pad_suffix(True, 4.0, 9.0, _settings())
    _assert_no_exponent(suffix)
    assert suffix.startswith(",fps=30,tpad=stop_mode=clone:stop_duration=5")


def test_freeze_pad_emits_a_pad_just_above_one_frame() -> None:
    """The floor sits at one output frame, and a pad just over it survives — so the
    guard is a frame-scale epsilon, not a blanket suppression of small pads."""
    settings = _settings()
    one_frame = 1.0 / settings.render_fps
    duration = 9.0
    suffix = _freeze_pad_suffix(True, duration - one_frame * 1.5, duration, settings)
    _assert_no_exponent(suffix)
    assert "stop_duration" in suffix

    just_under = _freeze_pad_suffix(True, duration - one_frame * 0.5, duration, settings)
    assert just_under == ""


# ---------------------------------------------------------------------------
# The sibling duration options that were bare f-strings (same parser, same crash)
# ---------------------------------------------------------------------------
def test_shot_audio_fade_emits_no_exponent() -> None:
    """``ShotAudio.fade_*_s`` is bounded ``[0, 30]`` with no exclusive lower bound, so
    a sub-100µs fade is a legal author value that used to interpolate as ``4.2e-05``
    into ``afade``'s DURATION options."""
    audio = ShotAudio(fade_in_s=4.2e-05, fade_out_s=4.2e-05)
    chain = _apply_shot_audio(audio, 10.0)
    _assert_no_exponent(chain)
    assert "afade=t=in:st=0:d=0.000042" in chain
    assert "d=0.000042" in chain.split("afade=t=out:")[1]


def test_shot_audio_fade_is_unchanged_for_ordinary_values() -> None:
    """The bare-float sites were changed for the exponent only — an ordinary fade
    keeps its exact previous spelling (``repr``, not ``%g``)."""
    chain = _apply_shot_audio(ShotAudio(fade_in_s=1.5, fade_out_s=2.0), 10.0)
    assert "afade=t=in:st=0:d=1.5" in chain
    assert "afade=t=out:st=8.0:d=2.0" in chain
