"""A/V quality wave 10 — the defects a fresh sweep of ``origin/main`` found
after the loudness theories closed (the newest master measured -14.2 LUFS,
LRA 1.9 LU, true peak -1.2 dBTP, A/V gap -0.008 s: clean).

Every item here is an argv/filtergraph assertion. Where a claim is about what
ffmpeg actually DOES with the flag, the proof lives in
``tests/render_harness/test_av_quality_wave10_render.py`` — an argv assertion
proves a flag is EMITTED, never that it is HONOURED.

3a — colour tags. ``ffprobe`` of delivered masters returns ``color_space=bt709``
     with ``color_transfer``/``color_primaries`` BOTH ``unknown``, i.e. the whole
     100%-false ``colorspace_ok`` fleet rate. The brief's premise (that only the
     matrix is declared) is WRONG against the real code — ``colorspace_tag_args``
     has always emitted all three. Measured on the PRODUCTION ffmpeg (7.1.5,
     ``python:3.14-slim`` + Debian trixie ``ffmpeg``) with this repo's verbatim
     ``_video_codec_args``: only the matrix lands; trc and primaries are dropped
     at the ENCODE. ffmpeg >= 5 propagates the filtergraph frame's (unspecified)
     colour properties onto the encoder context and clobbers the output options.
     On ffmpeg 4.4 the same command tags all three, which is why this never
     showed up locally. Fix: stamp the SPS VUI with ``-bsf:v h264_metadata``.

3b — bed-loop seams dip ~3 dB. ``_timeline_crossfade_mix`` used ``afade`` at its
     default ``curve=tri`` (equal GAIN) for every seam. ``_bed_loop_filter`` joins
     ONE asset to ITSELF at a different offset — decorrelated — so an equal-gain
     sum is -3 dB at the seam centre: a periodic duck under the music once per
     ``asset_seconds``. Fix: a ``curve`` PARAMETER, ``qsin`` (equal POWER) at loop
     seams only; segment joins keep ``tri``.

3c — an author fade LONGER than its shot silently removed BOTH fades. The
     ``st = duration - fade_out; if st >= 0`` branch emitted nothing, and the
     implicit ``_SHOT_EDGE_FADE_S`` had already been overwritten by the author
     value — so the segment boundary landed on an arbitrary non-zero sample: the
     step-discontinuity click the edge fade exists to remove, reintroduced by
     asking for MORE fade. Fix: clamp to ``duration``, never drop. ``fade_in``
     likewise.

3d — ``master_normalization_type`` read-back was blind. PASS 2's ``loudnorm``
     carried no ``print_format``; af_loudnorm defaults to ``print_format=none``,
     which suppresses the end-of-run summary ``_NORMALIZATION_TYPE_RE`` searches
     for, so the field was ``None`` on every row ever recorded. An observability
     hole, NOT a fallback to dynamic.

3e — the second hardcoded limiter. ``overlay_audio_bed``'s ``master_enabled=False``
     branch still carried ``alimiter=limit=0.95``; wave 8 derived only the master
     chain and the per-shot SFX mix from ``master_tp``.

3f — ``_BED_LOOP_MAX_COPIES`` was dead. ``copies > MAX`` returned the OVERSIZED
     count instead of the ``(0, 0.0)`` no-plan its own docstring promises, so the
     caller opened that many ``-i`` decoders in a 2 Gi pod (a 20 s bed under a
     30 min film = 90 inputs).

3g (single cadence conform site) is deliberately NOT here — wave 8 measured that
those sites all read ONE ``settings.render_fps``, so it is tidying, not a defect.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.api.schemas import ShotAudio
from src.config import get_settings
from src.services import mastering, renderer


def _settings(**overrides: object) -> Any:
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# 3a — the colour triplet must be STAMPED, not merely declared
# ---------------------------------------------------------------------------


def test_colorspace_bsf_args_emits_the_full_h264_vui_triplet() -> None:
    """bt709 => VUI code 1 for all three. h264_metadata takes NUMBERS only; it
    rejects ``colour_primaries=bt709`` outright, so the ffmpeg colour NAMES in
    ``_COLORSPACE_TRIPLETS`` must be mapped, not passed through."""
    args = mastering.colorspace_bsf_args(_settings(render_colorspace="bt709"))
    assert args == [
        "-bsf:v",
        "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
    ]


def test_colorspace_bsf_args_tracks_the_configured_colorspace() -> None:
    """The stamp is DERIVED from the same ``_COLORSPACE_TRIPLETS`` row the
    ``-colorspace`` tag comes from — a job retargeted away from bt709 gets its
    own codes, never a hardcoded 1/1/1."""
    args = mastering.colorspace_bsf_args(_settings(render_colorspace="bt2020"))
    # bt2020 primaries = 9, bt2020-10 transfer = 14, bt2020nc matrix = 9.
    assert args[1] == (
        "h264_metadata=colour_primaries=9:transfer_characteristics=14:matrix_coefficients=9"
    )
    # srgb shares bt709 primaries/matrix but carries the iec61966-2-1 transfer (13).
    srgb = mastering.colorspace_bsf_args(_settings(render_colorspace="srgb"))
    assert srgb[1] == (
        "h264_metadata=colour_primaries=1:transfer_characteristics=13:matrix_coefficients=1"
    )


def test_colorspace_bsf_args_empty_when_colorspace_disabled() -> None:
    """No tag, no stamp — the disabled path must stay byte-identical."""
    assert mastering.colorspace_bsf_args(_settings(render_colorspace="none")) == []
    assert mastering.colorspace_tag_args(_settings(render_colorspace="none")) == []


@pytest.mark.parametrize("encoder", ["libx264", "h264_videotoolbox", "h264_nvenc"])
def test_video_codec_args_stamps_the_vui_on_every_encoder_branch(encoder: str) -> None:
    """THE HEADLINE. The stamp rides ``_video_codec_args`` — the single place
    every encode site in the service goes through — so no call site can forget
    it, and the hardware branches (which have no filtergraph of their own to
    hang a ``setparams`` on) get it too."""
    args = renderer._video_codec_args(_settings(render_encoder=encoder))
    assert "-bsf:v" in args
    assert args[args.index("-bsf:v") + 1] == (
        "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"
    )
    # The declarative tag is NOT removed — it is what a `-c:v copy` remux and the
    # mp4 `colr` atom read, and it is correct on ffmpeg 4.4 (PR CI is on 6.1.1).
    assert "-color_trc" in args and "-color_primaries" in args and "-colorspace" in args


def test_video_codec_args_omits_the_stamp_when_colorspace_disabled() -> None:
    assert "-bsf:v" not in renderer._video_codec_args(_settings(render_colorspace="off"))


# ---------------------------------------------------------------------------
# 3b — equal-POWER at loop seams, equal-GAIN everywhere else
# ---------------------------------------------------------------------------


def test_bed_loop_seams_use_equal_power_qsin() -> None:
    """Both sides of a loop seam are the SAME asset at different offsets, i.e.
    decorrelated. `tri` (equal gain) sums to 0.707 = -3 dB at the seam centre."""
    graph = renderer._bed_loop_filter(2, 3, 0.25, "bed", 4.0)
    assert graph.count("afade=t=in:curve=qsin") == 2  # every seam but the head
    assert graph.count("afade=t=out:curve=qsin") == 2  # every seam but the tail
    assert "curve=tri" not in graph


def test_timeline_crossfade_mix_defaults_to_equal_gain_tri() -> None:
    """SCOPE GUARD. ``_timeline_crossfade_mix``'s other caller joins two DIFFERENT
    segments across an xfade; equal-gain is correct there, so the curve is a
    PARAMETER and the default must not move."""
    parts = renderer._timeline_crossfade_mix([("[0:a]", 4.0, 0.0, 1.0, 0.0)], "ax")
    assert "afade=t=out:curve=tri" in parts[0]
    assert "qsin" not in "".join(parts)


# ---------------------------------------------------------------------------
# 3c — an over-long author fade is CLAMPED, never dropped
# ---------------------------------------------------------------------------


def test_author_fade_out_longer_than_the_shot_is_clamped_not_dropped() -> None:
    """The defect: a 5 s ``fade_out_s`` on a 3 s shot produced NO fade at all —
    not even the implicit 8 ms edge fade, which the author value had already
    overwritten — leaving the segment boundary on an arbitrary non-zero sample."""
    chain = renderer._apply_shot_audio(ShotAudio(fade_out_s=5.0), 3.0)
    assert "afade=t=out" in chain, "an over-long fade_out must still emit a fade"
    assert "afade=t=out:st=0.0:d=3.0" in chain


def test_author_fade_in_longer_than_the_shot_is_clamped() -> None:
    """``fade_in`` was unclamped too: it would ramp past the end of the segment
    and leave the whole shot fading up from nothing."""
    chain = renderer._apply_shot_audio(ShotAudio(fade_in_s=9.0), 3.0)
    assert "afade=t=in:st=0:d=3.0" in chain


def test_normal_author_fades_are_unchanged_by_the_clamp() -> None:
    """Regression guard: a fade that fits is untouched."""
    chain = renderer._apply_shot_audio(ShotAudio(fade_in_s=0.5, fade_out_s=1.0), 6.0)
    assert "afade=t=in:st=0:d=0.5" in chain
    assert "afade=t=out:st=5.0:d=1.0" in chain


# ---------------------------------------------------------------------------
# 3d — pass 2 must PRINT the summary the read-back parses
# ---------------------------------------------------------------------------


def test_pass2_loudnorm_requests_the_summary_the_readback_parses() -> None:
    """``_NORMALIZATION_TYPE_RE`` searches stderr for ``Normalization Type:``.
    af_loudnorm defaults to ``print_format=none``, so without this the line never
    existed and ``master_normalization_type`` was None on EVERY row."""
    measured = {
        "input_i": "-23.0",
        "input_tp": "-5.0",
        "input_lra": "7.0",
        "input_thresh": "-33.0",
    }
    chain = renderer._loudnorm_filter(_settings(), measured)
    assert "linear=true" in chain
    assert "print_format=summary" in chain
    # The single-pass fail-open branch prints too — it is the branch whose mode
    # we most need to see.
    assert "print_format=summary" in renderer._loudnorm_filter(_settings(), None)


def test_pass1_measure_filter_still_prints_json() -> None:
    """SCOPE GUARD. Pass 1's output is PARSED AS JSON — summary there would break
    the measurement that makes linear mode possible at all."""
    measure = renderer._master_measure_filter(_settings())
    assert "print_format=json" in measure
    assert "print_format=summary" not in measure


def test_readback_parses_a_real_summary_shape() -> None:
    """The regex against the literal block ffmpeg prints (captured verbatim from
    a local ffmpeg run of the pass-2 command)."""
    stderr = (
        "[Parsed_loudnorm_1 @ 0x1] \n"
        "Input Integrated:    -3.0 LUFS\n"
        "Output Integrated:   -32.8 LUFS\n"
        "Output True Peak:    -32.1 dBTP\n"
        "Normalization Type:   Dynamic\n"
        "Target Offset:        +0.0 LU\n"
    )
    assert renderer._record_master_normalization_type(stderr, requested_linear=True) == "dynamic"


# ---------------------------------------------------------------------------
# 3e — the second hardcoded limiter
# ---------------------------------------------------------------------------


def test_master_off_bed_mix_limiter_is_derived_from_master_tp() -> None:
    """This is the LAST limiter a ``master_enabled=False`` render sees, so its
    ceiling IS the delivered one. It must track the one configured ceiling."""
    import inspect

    source = inspect.getsource(renderer.overlay_audio_bed)
    assert "alimiter=limit=0.95" not in source
    assert "_limiter_linear_limit(settings)" in source


@pytest.mark.parametrize("master_tp", [-1.0, -2.0, -6.0])
def test_bed_mix_limiter_moves_with_the_configured_ceiling(master_tp: float) -> None:
    """Derived, not coincidentally equal: change ``master_tp`` and the ceiling
    must follow it, 0.3 dB below."""
    settings = _settings(master_tp=master_tp)
    expected = 10.0 ** ((master_tp - renderer._LIMITER_TP_HEADROOM_DB) / 20.0)
    assert renderer._limiter_linear_limit(settings) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 3f — the bed-loop copy cap must actually cap
# ---------------------------------------------------------------------------


def test_bed_loop_plan_falls_back_when_copies_exceed_the_cap() -> None:
    """A 20 s bed under a 30 min film wanted 90 separate ``-i`` decoders in a
    2 Gi pod. The docstring's contract for over-cap is ``(0, 0.0)`` — no plan,
    caller keeps ``-stream_loop -1``. It used to return the oversized count."""
    copies, xfade = renderer._bed_loop_plan(asset_seconds=20.0, needed=1800.0)
    assert (copies, xfade) == (0, 0.0)
    assert copies <= renderer._BED_LOOP_MAX_COPIES


def test_bed_loop_plan_still_plans_right_up_to_the_cap() -> None:
    """The cap must not be off-by-one: a plan needing exactly MAX copies is
    still a plan."""
    xfade = renderer._BED_LOOP_XFADE_S
    asset = 10.0
    needed = renderer._BED_LOOP_MAX_COPIES * (asset - xfade) + xfade
    copies, got_xfade = renderer._bed_loop_plan(asset_seconds=asset, needed=needed)
    assert copies == renderer._BED_LOOP_MAX_COPIES
    assert got_xfade == pytest.approx(xfade)
    # One hair more than the cap can cover falls back instead of over-opening.
    over, _ = renderer._bed_loop_plan(asset_seconds=asset, needed=needed + 1.0)
    assert over == 0
