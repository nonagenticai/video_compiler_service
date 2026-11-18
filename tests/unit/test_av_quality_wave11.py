"""A/V quality wave 11 — removing a wasted lossy generation from the picture.

⭐ THE FINDING. The delivered picture passed through THREE x264 generations, and
every one of them called ``renderer._video_codec_args`` with the SAME delivery
CRF cell:

  1. the per-segment encode (from raw frames — the only generation that is a
     first encode),
  2. the picture-polish re-encode (``_deband_segment`` / ``_regrade_segment`` /
     ``_polish_segment``),
  3. the whole-film grade or music-bed pass.

Generations 2 and 3 re-quantise material x264 had ALREADY quantised at that
exact cell. Re-encoding at CRF 18 something that is already a CRF 18 encode does
not "hold" quality — x264 re-decides every coefficient and throws detail away a
second time. Fix: an explicit ``intermediate`` cell (``render_crf`` minus
``render_intermediate_crf_delta``, default 6, floored at 0) for the encodes whose
output is a TEMP file, leaving ``render_crf`` to own the DELIVERY cell alone.

⛔ WHAT WAS DELIBERATELY *NOT* CHANGED, and why the delivery passes are excluded.
``apply_finish_grade`` and ``overlay_audio_bed`` ARE the final encode: their
output is returned straight out of ``_apply_final_audio`` as the deliverable
(only audio passes are applied afterwards). Marking them ``intermediate`` would
hand every delivered master a ~2x bitrate with no picture to gain — the input to
that pass is already a delivery-cell encode, so the extra bits buy nothing that
was not already lost upstream. Generation 3 keeps ``render_crf`` by the same rule
that gives generation 2 the lower cell: the cell belongs to what the output IS,
not to where in the chain it sits.

Argv assertions only. An argv assertion proves a flag is EMITTED, never that
libx264 HONOURED it.
"""

from __future__ import annotations

import inspect

import pytest

from src.config import Settings, get_settings
from src.services import renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


def _crf(args: list[str]) -> str:
    return args[args.index("-crf") + 1]


# ---------------------------------------------------------------------------
# The cell itself
# ---------------------------------------------------------------------------


def test_intermediate_args_carry_a_lower_crf_than_delivery() -> None:
    """THE HEADLINE."""
    settings = _settings(render_crf=18, render_intermediate_crf_delta=6)
    assert _crf(renderer._video_codec_args(settings)) == "18"
    assert _crf(renderer._video_codec_args(settings, intermediate=True)) == "12"


def test_intermediate_changes_only_the_rate_control_tokens() -> None:
    """The colour stamp, the CFR pin, the GOP and the pixel format must be
    IDENTICAL on both cells. A debanded segment is concatenated (stream-copy)
    alongside segments that were never debanded, so any divergence in those
    tokens would break ``concat_shots``' copy path — a far worse defect than the
    generational loss this change removes."""
    settings = _settings(render_crf=18, render_intermediate_crf_delta=6)
    delivery = renderer._video_codec_args(settings)
    intermediate = renderer._video_codec_args(settings, intermediate=True)

    def _without_crf(args: list[str]) -> list[str]:
        i = args.index("-crf")
        return args[:i] + args[i + 2 :]

    assert _without_crf(delivery) == _without_crf(intermediate)


def test_intermediate_is_opt_in_so_every_unmarked_call_site_is_unchanged() -> None:
    """``intermediate`` defaults to False and is keyword-only, so no existing
    call site silently changed cell and no positional argument can drift into
    it."""
    params = inspect.signature(renderer._video_codec_args).parameters
    assert params["intermediate"].default is False
    assert params["intermediate"].kind is inspect.Parameter.KEYWORD_ONLY


def test_delta_zero_makes_intermediate_byte_identical_to_delivery() -> None:
    """The rollback contract. ``render_intermediate_crf_delta=0`` must restore the
    previous behaviour EXACTLY, so this ships as a reversible knob rather than a
    one-way change to the encode."""
    settings = _settings(render_crf=18, render_intermediate_crf_delta=0)
    assert renderer._video_codec_args(settings) == renderer._video_codec_args(
        settings, intermediate=True
    )


@pytest.mark.parametrize("encoder", ["h264_videotoolbox", "h264_nvenc"])
def test_intermediate_is_a_no_op_on_the_hardware_branches(encoder: str) -> None:
    """The HW encoders are bitrate-driven and ignore ``-crf`` entirely, so there
    is no cell to lower. Pinned as a documented no-op rather than left to look
    like an oversight — and pinned so nobody "completes" it with an invented
    ``render_hw_bitrate`` multiplier that no measurement supports."""
    settings = _settings(render_encoder=encoder, render_crf=18, render_intermediate_crf_delta=6)
    assert renderer._video_codec_args(settings) == renderer._video_codec_args(
        settings, intermediate=True
    )


# ---------------------------------------------------------------------------
# The wiring — WHICH encodes take the intermediate cell
# ---------------------------------------------------------------------------

#: Segment-level re-encodes: their output is a temp file that a later pass
#: re-encodes (grade / bed) or that ships only as picture polish.
#:
#: ``_render_xfade_group`` joined this list in wave 13. It qualifies under the
#: SAME rule the module docstring states -- "the cell belongs to what the output
#: IS, not to where in the chain it sits": its output is a temp segment that the
#: concat pass copies, not the deliverable. Before wave 13 it called
#: ``_video_codec_args(settings, deterministic)`` with no ``intermediate``, and
#: because the xfade ``filter_complex`` spans the ENTIRE pair rather than just
#: the transition window, every frame of every shot participating in a
#: transition was decoded from a delivery-cell segment and re-quantised at that
#: same cell -- exactly the wasted generation wave 11 existed to remove, still
#: happening on any film that uses transitions.
_INTERMEDIATE_ENCODERS = (
    "_deband_segment",
    "_regrade_segment",
    "_polish_segment",
    "_render_xfade_group",
)

#: The DELIVERY encode. Its output is returned as the deliverable.
_DELIVERY_ENCODERS = ("apply_finish_grade", "overlay_audio_bed")


@pytest.mark.parametrize("name", _INTERMEDIATE_ENCODERS)
def test_segment_re_encodes_use_the_intermediate_cell(name: str) -> None:
    src = inspect.getsource(getattr(renderer, name))
    assert "_video_codec_args(settings, deterministic, intermediate=True)" in src


@pytest.mark.parametrize("name", _DELIVERY_ENCODERS)
def test_the_delivery_encode_keeps_render_crf(name: str) -> None:
    """⛔ The inverse assertion, and it is the load-bearing one: it is what stops
    a later session from "finishing the job" by marking the final pass
    intermediate too and doubling every delivered file for nothing."""
    src = inspect.getsource(getattr(renderer, name))
    assert "intermediate=True" not in src


def test_exactly_the_inventoried_encode_sites_take_the_intermediate_cell() -> None:
    """A count, so a NEW re-encode site added later cannot quietly inherit the
    delivery cell (or the intermediate one) without this test being updated and
    the choice being made deliberately."""
    src = inspect.getsource(renderer)
    assert src.count("intermediate=True)") == len(_INTERMEDIATE_ENCODERS)
