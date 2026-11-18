"""Pure-unit tests for the renderer escape + wrap helpers — no ffmpeg needed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx
import pytest
import structlog

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer
from src.services.renderer import (
    _AVG_GLYPH_WIDTH_RATIO,
    _build_shot_timeline,
    _caption_filter_fragment,
    _clip_seek_fractions,
    _escape_drawtext,
    _freeze_pad_suffix,
    _safe_margin_px,
    _scale_pad_filter,
    _scroll_filter,
    _video_codec_args,
    _wrap_for_canvas,
    probe_dimensions,
)


def test_build_shot_timeline_zips_shots_to_probed_offsets() -> None:
    """Each shot is keyed to its index-aligned [start, end) bus offset; a shot with
    no line_id emits line_id=None. Pure — no ffmpeg."""
    shots = [
        Shot(id="s0", line_id="L0", seconds=2.0),
        Shot(id="s1", line_id="L1", seconds=3.0),
        Shot(id="s2", seconds=1.5),  # no line_id
    ]
    offsets = [(0.0, 2.0), (2.0, 5.0), (5.0, 6.5)]
    timeline = _build_shot_timeline(shots, offsets)
    assert timeline == [
        {"line_id": "L0", "start": 0.0, "end": 2.0},
        {"line_id": "L1", "start": 2.0, "end": 5.0},
        {"line_id": None, "start": 5.0, "end": 6.5},
    ]
    # Values are plain floats (JSON-serializable), not numpy/other.
    assert all(isinstance(e["start"], float) and isinstance(e["end"], float) for e in timeline)


def test_build_shot_timeline_empty_offsets_yields_empty() -> None:
    """With no offsets (probe-free path never reached) the zip is empty."""
    assert _build_shot_timeline([Shot(id="s0", seconds=1.0)], []) == []


def test_video_codec_args_carry_settings_crf_and_preset() -> None:
    """The libx264 args must reflect the configured quality knobs, not the old
    hardcoded ``-preset veryfast`` with libx264's default CRF."""
    settings = get_settings().model_copy(update={"render_crf": 18, "render_preset": "slow"})
    args = _video_codec_args(settings)
    assert args[args.index("-preset") + 1] == "slow"
    assert args[args.index("-crf") + 1] == "18"
    assert "veryfast" not in args
    assert args[:2] == ["-c:v", "libx264"]


def test_video_codec_args_default_encoder_is_libx264() -> None:
    """Unset render_encoder keeps the software path (no behaviour change)."""
    args = _video_codec_args(get_settings())
    assert args[:2] == ["-c:v", "libx264"]
    assert "-preset" in args and "-crf" in args


@pytest.mark.parametrize("encoder", ["h264_videotoolbox", "h264_nvenc"])
def test_video_codec_args_hardware_encoder_is_bitrate_driven(encoder: str) -> None:
    """A hardware encoder selects bitrate (-b:v) and drops the libx264-only
    -preset/-crf knobs, while keeping yuv420p so concat stream-copy still works."""
    settings = get_settings().model_copy(
        update={"render_encoder": encoder, "render_hw_bitrate": "9M"}
    )
    args = _video_codec_args(settings)
    assert args[:2] == ["-c:v", encoder]
    assert args[args.index("-b:v") + 1] == "9M"
    assert "-preset" not in args
    assert "-crf" not in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_video_codec_args_deterministic_tokens_only_on_software_path() -> None:
    """The deterministic tokens (`-threads 1`, `-fps_mode:v cfr`, `-flags +bitexact`)
    are bit-reproducibility knobs that only mean something for libx264. On a HW
    encoder they are meaningless (and HW output isn't bit-reproducible), so they
    must be omitted there while libx264 carries them."""
    settings = get_settings()
    sw = _video_codec_args(settings, deterministic=True)
    assert "-threads" in sw and sw[sw.index("-threads") + 1] == "1"
    assert "-fps_mode:v" in sw and sw[sw.index("-fps_mode:v") + 1] == "cfr"
    assert sw[sw.index("-flags") + 1] == "+bitexact"

    hw_settings = settings.model_copy(
        update={"render_encoder": "h264_videotoolbox", "render_hw_bitrate": "9M"}
    )
    hw = _video_codec_args(hw_settings, deterministic=True)
    assert "-threads" not in hw
    assert "-fps_mode:v" not in hw
    assert "-flags" not in hw
    assert hw[:2] == ["-c:v", "h264_videotoolbox"]


def test_video_codec_args_single_encode_ladder_call_preserves_crf_vbv() -> None:
    """Collapsing the double `encode_ladder` call to one must not change the
    resolved crf (libx264 -crf) — the value is unchanged by the refactor."""
    settings = get_settings().model_copy(update={"render_crf": 21})
    off = _video_codec_args(settings)
    det = _video_codec_args(settings, deterministic=True)
    assert off[off.index("-crf") + 1] == "21"
    # deterministic changes only the appended tokens, never the crf/preset.
    assert det[det.index("-crf") + 1] == "21"
    assert off[: off.index("-crf") + 2] == det[: det.index("-crf") + 2]


def test_clip_seek_fractions_spreads_reused_footage_clips() -> None:
    """A footage clip reused N times yields 0, 1/N, … so the promo pans through
    the page; images and single-use clips keep the natural start (0)."""
    shots = [
        Shot(id="a", seconds=5, video_url="clipA"),
        Shot(id="b", seconds=5, image_url="img1"),
        Shot(id="c", seconds=5, video_url="clipA"),
        Shot(id="d", seconds=5, video_url="clipB"),  # single use → 0
        Shot(id="e", seconds=5, video_url="clipA"),
    ]
    assert _clip_seek_fractions(shots) == [0.0, 0.0, 1 / 3, 0.0, 2 / 3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("anchor", "lead", "fraction", "expected_ss"),
    [
        # anchor 'end' → show the last `shot_duration` secs (skip the load-in head).
        ("end", 0.0, 0.0, "89.900"),  # 100s clip, 10s shot → 90 - 0.10
        # lead-trim (anchor 'start') skips the white load-in, plays forward.
        ("start", 4.0, 0.0, "4.000"),
        # lead bounded so the clip still covers the shot duration.
        ("start", 200.0, 0.0, "90.000"),
        # no anchor/lead + no fraction → no seek.
        ("start", 0.0, 0.0, None),
        # fraction still works when anchor/lead unset (reused-clip panning).
        ("start", 0.0, 0.5, "45.000"),
    ],
)
async def test_clip_seek_args_anchor_and_lead(
    monkeypatch: pytest.MonkeyPatch,
    anchor: str,
    lead: float,
    fraction: float,
    expected_ss: str | None,
) -> None:
    async def _fake_dur(_: Path) -> float:
        return 100.0

    monkeypatch.setattr(renderer, "_ffprobe_duration", _fake_dur)
    args, offset, span = await renderer._clip_seek_args(
        Path("/x.mp4"), fraction, 10.0, clip_anchor=anchor, clip_lead_seconds=lead
    )
    assert args == (["-ss", expected_ss] if expected_ss else [])
    # The reported offset is exactly the value handed to ffmpeg (0.0 when no seek).
    assert offset == (float(expected_ss) if expected_ss else 0.0)
    # No out-point ⇒ no input-side -t and no declared span (byte-identical path).
    assert span is None


def test_clip_seek_fractions_all_zero_without_reuse() -> None:
    """No repeated footage clip ⇒ every shot keeps the clip's natural start."""
    shots = [
        Shot(id="a", seconds=4, video_url="only"),
        Shot(id="b", seconds=4, image_url="img"),
    ]
    assert _clip_seek_fractions(shots) == [0.0, 0.0]


def test_escape_drawtext_handles_special_chars() -> None:
    raw = "Hello: world's [test], 100% \\path"
    out = _escape_drawtext(raw)
    assert ":" not in out.replace(r"\:", "")
    assert "'" not in out.replace(r"\'", "")
    assert "[" not in out.replace(r"\[", "")
    assert "]" not in out.replace(r"\]", "")
    assert "%" not in out.replace(r"\%", "")
    assert "," not in out.replace(r"\,", "")


def test_escape_drawtext_passes_plain_text_through() -> None:
    raw = "Hello world"
    assert _escape_drawtext(raw) == raw


def test_escape_drawtext_collapses_real_newlines() -> None:
    """Regression: LLM markdown paragraphs broke ffmpeg drawtext parsing (exit=8).

    Real newlines in the input must NEVER reach the filter value -- the
    drawtext parser treats them as filter-argument separators. Collapse
    all whitespace runs into single spaces so the caption renders as one
    visual line, matching what the on-screen text actually shows anyway.
    """
    raw = "Para one.\n\nPara two.\n\tIndented line.\r\nFinal."
    out = _escape_drawtext(raw)
    assert "\n" not in out
    assert "\r" not in out
    assert "\t" not in out
    assert out == "Para one. Para two. Indented line. Final."


def test_escape_drawtext_handles_realworld_llm_critique() -> None:
    """The actual failing input from website_review_compile run e3b3f3cc."""
    raw = (
        "The website audit reveals significant deficiencies.\n\n"
        "From a technical standpoint, the site is vulnerable.\n\n"
        "Action items: **Security**, **Usability**, **Content**."
    )
    out = _escape_drawtext(raw)
    # No raw newlines survive, all special chars escaped, value is one line.
    assert "\n" not in out
    assert r"\," in out  # commas escaped
    assert r"\*" not in out  # asterisk needs no escape (passes through)
    # Single-quote inside an apostrophe-free string still safe.
    assert " " in out


@pytest.mark.asyncio
async def test_probe_dimensions_parses_ffprobe_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffprobe ``csv=s=x:p=0`` emits ``WIDTHxHEIGHT`` for the v:0 stream."""

    async def fake_run(cmd: list[str]) -> str:
        assert cmd[0] == "ffprobe"
        assert "v:0" in cmd  # primary video stream only
        return "3840x2160\n"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await probe_dimensions(renderer.Path("/tmp/out.mp4")) == (3840, 2160)


@pytest.mark.asyncio
async def test_probe_dimensions_returns_zeros_on_unparsable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-conforming probe output degrades to (0, 0) — never raises."""

    async def fake_run(cmd: list[str]) -> str:
        return "N/A"

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await probe_dimensions(renderer.Path("/tmp/out.mp4")) == (0, 0)


@pytest.mark.asyncio
async def test_probe_output_duration_parses_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffprobe gives the actual encoded duration of the delivered master.

    The probe now asks for the ``v:0`` stream duration alongside the container's
    (``-of json``) so an audio-padded SOURCE is measured on its picture — see
    ``_resolve_probed_duration``. On a delivered master the two agree, so this
    number is unchanged.
    """

    async def fake_run(cmd: list[str]) -> str:
        assert cmd[0] == "ffprobe"
        return '{"streams": [{"duration": "263.512000"}], "format": {"duration": "263.512000"}}'

    monkeypatch.setattr(renderer, "_run", fake_run)
    assert await renderer.probe_output_duration(renderer.Path("/tmp/out.mp4")) == 263.512


@pytest.mark.asyncio
async def test_probe_output_duration_zero_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe failure degrades to 0.0 (unknown) — never fails the render."""

    async def boom(cmd: list[str]) -> str:
        raise renderer.FfmpegError("ffprobe exited 1")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await renderer.probe_output_duration(renderer.Path("/tmp/out.mp4")) == 0.0


# ---------------------------------------------------------------------------
# _wrap_for_canvas — the 2026-05-31 "words flow off the screen" fix.
# ---------------------------------------------------------------------------


def test_wrap_for_canvas_keeps_short_line_on_one_row() -> None:
    """A short title at the body fontsize must NOT get split — no false wraps."""
    out = _wrap_for_canvas("Hello world", fontsize=56, canvas_width=1920)
    assert "\n" not in out
    assert out == "Hello world"


def test_wrap_for_canvas_splits_long_script_line() -> None:
    """The actual user-visible defect: a 50-char script line at the old fs=64
    extended ~1760 px past the 1920-px canvas, gettin clipped on both sides.
    With wrapping it must come back as multiple lines that each fit."""
    long_line = (
        "We all know the cycle: the procrastination, the complex workflows, "
        "and the endless excuses about needing a dedicated video team."
    )
    out = _wrap_for_canvas(long_line, fontsize=56, canvas_width=1920)
    assert "\n" in out, "Expected the long sentence to wrap to multiple lines"
    # No single output line exceeds the computed max chars, with a small
    # tolerance for break_long_words=False overshoots on long tokens.
    safe_width = 1920 - 2 * _safe_margin_px(1920)
    max_chars = int(safe_width / (56 * _AVG_GLYPH_WIDTH_RATIO))
    for line in out.split("\n"):
        assert len(line) <= max_chars + 8, line


def test_wrap_for_canvas_collapses_internal_whitespace() -> None:
    """LLM output sometimes carries newlines / tabs between words — collapse
    them BEFORE wrapping so a wrapped line never has stray whitespace runs."""
    out = _wrap_for_canvas("alpha\n\tbeta   gamma", fontsize=48, canvas_width=1920)
    assert "\t" not in out
    assert "  " not in out


def test_wrap_for_canvas_does_not_break_urls_or_hyphenated_tokens() -> None:
    """The CTA line in a w2v promo carries the verbatim URL `w2v.nonagentic.ai`.
    break_long_words=False keeps it intact even if it slightly exceeds the
    estimated max line length, because splitting it would break the spoken /
    visual contract that verbatim items must surface as-is."""
    out = _wrap_for_canvas(
        "Open w2v.nonagentic.ai and ship a polished video this afternoon.",
        fontsize=56,
        canvas_width=1920,
    )
    assert "w2v.nonagentic.ai" in out
    # The URL is on exactly one line, not split mid-token.
    assert all("w2v.nonagentic.ai" in line or "w2v" not in line for line in out.split("\n"))


def test_wrap_for_canvas_handles_empty_input() -> None:
    assert _wrap_for_canvas("", fontsize=56, canvas_width=1920) == ""
    assert _wrap_for_canvas("   \n\t ", fontsize=56, canvas_width=1920) == ""


def test_wrap_for_canvas_narrow_canvas_still_makes_progress() -> None:
    """A 9:16 mobile canvas (1080-wide) requires more aggressive wrapping
    than the 1920-wide default. Verify the helper adapts."""
    out_wide = _wrap_for_canvas(
        "Turn your ideas into professional content instantly.",
        fontsize=56,
        canvas_width=1920,
    )
    out_narrow = _wrap_for_canvas(
        "Turn your ideas into professional content instantly.",
        fontsize=56,
        canvas_width=1080,
    )
    assert out_narrow.count("\n") >= out_wide.count("\n")


# ---------------------------------------------------------------------------
# _scale_pad_filter fit modes — the AI-image pillarbox fix.
# ---------------------------------------------------------------------------


def test_scale_pad_filter_contain_pads_and_never_crops() -> None:
    """Default contain fits-inside and pads with the bg colour — no crop, so
    framed footage / slide text is never clipped."""
    settings = get_settings()
    out = _scale_pad_filter(settings, caption=None, fit="contain")
    assert "force_original_aspect_ratio=decrease" in out
    assert f"pad={settings.render_width}:{settings.render_height}" in out
    assert f"color={settings.render_bg_color}" in out
    assert "crop=" not in out


def test_scale_pad_filter_default_is_contain() -> None:
    """Omitting fit must keep the existing contain behaviour (back-compat)."""
    settings = get_settings()
    assert _scale_pad_filter(settings, caption=None) == _scale_pad_filter(
        settings, caption=None, fit="contain"
    )


def test_scale_pad_filter_cover_fills_frame_with_crop_no_pad() -> None:
    """Cover scales-to-fill + center-crops so the canvas is full (no navy
    bars) — for atmospheric AI-image backgrounds."""
    settings = get_settings()
    out = _scale_pad_filter(settings, caption=None, fit="cover")
    assert "force_original_aspect_ratio=increase" in out
    assert f"crop={settings.render_width}:{settings.render_height}" in out
    assert "pad=" not in out


def test_scale_pad_filter_cover_keeps_caption_fragment(tmp_path) -> None:
    """A cover shot can still burn in a caption — the fragment appends after
    the scale/crop chain."""
    settings = get_settings()
    out = _scale_pad_filter(settings, caption="Hello", work_dir=tmp_path, shot_id="s1", fit="cover")
    assert "crop=" in out
    assert "drawtext" in out


def test_scroll_filter_fits_width_and_pans_top_to_bottom() -> None:
    """Scroll fits a tall page to the canvas width and pans a full-width window
    from the top to the bottom over the shot duration (y travels with t)."""
    settings = get_settings()
    out = _scroll_filter(settings, duration=8.0)
    w, h = settings.render_width, settings.render_height
    assert f"scale={w}:-2" in out  # fit to canvas width, keep aspect
    assert f"crop={w}:{h}:0:" in out  # full-width window, x pinned to 0
    assert "ih-" in out and "t/8.000" in out  # y pans with time over the duration
    assert f"pad={w}:max(ih" in out  # short pages padded so crop never fails
    # commas inside the crop/pad expressions are escaped for the filtergraph
    assert "\\," in out


def test_scroll_filter_keeps_caption_fragment(tmp_path) -> None:
    settings = get_settings()
    out = _scroll_filter(settings, duration=5.0, caption="Hi", work_dir=tmp_path, shot_id="s1")
    assert "crop=" in out and "drawtext" in out


def test_caption_filter_fragment_is_a_noop_at_1080p(tmp_path) -> None:
    """render_height=1080 (the default) must reproduce the pre-scaling literal
    values exactly: fontsize=44, y=h-text_h-60."""
    settings = get_settings().model_copy(update={"render_width": 1920, "render_height": 1080})
    out = _caption_filter_fragment("Hello", settings, tmp_path, "s1")
    assert "fontsize=44" in out
    assert "y=h-text_h-60" in out


def test_caption_filter_fragment_scales_with_render_height(tmp_path) -> None:
    """Doubling render_height to 2160 (2x 1080) must double both the font size
    and the bottom margin: 44 -> 88, 60 -> 120."""
    settings = get_settings().model_copy(update={"render_width": 3840, "render_height": 2160})
    out = _caption_filter_fragment("Hello", settings, tmp_path, "s1")
    assert "fontsize=88" in out
    assert "y=h-text_h-120" in out


async def test_probe_dimensions_returns_zero_and_warns_on_ffprobe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ffprobe must keep the best-effort contract — return (0, 0) and
    never raise — but it must no longer do so silently: a WARNING is emitted so
    the diagnostic gap (an unexplained 0x0 reported to the orchestrator) is
    surfaced in the logs with the path and the underlying error."""

    async def boom(_cmd: list[str]) -> str:
        raise renderer.FfmpegError("ffprobe exit=1")

    monkeypatch.setattr(renderer, "_run", boom)

    with structlog.testing.capture_logs() as logs:
        dims = await probe_dimensions(Path("/tmp/does-not-exist.mp4"))

    assert dims == (0, 0)
    warnings = [e for e in logs if e["log_level"] == "warning"]
    assert any(e["event"] == "probe_dimensions.ffprobe_failed" for e in warnings)
    entry = next(e for e in warnings if e["event"] == "probe_dimensions.ffprobe_failed")
    assert entry["path"] == "/tmp/does-not-exist.mp4"
    assert "ffprobe exit=1" in entry["error"]


# --- _freeze_pad_suffix (Step B freeze-pad cap) -----------------------------


def _stop_duration(suffix: str) -> str:
    """Extract the ``tpad`` stop_duration token from a freeze-pad suffix."""
    marker = "stop_duration="
    start = suffix.index(marker) + len(marker)
    return suffix[start : suffix.index(",", start)]


def test_freeze_pad_suffix_empty_when_clip_fills_narration() -> None:
    # clip_dur >= duration ⇒ no pad, byte-identical empty return regardless of cap.
    settings = get_settings()
    assert _freeze_pad_suffix(True, 9.0, 9.0, settings) == ""
    assert _freeze_pad_suffix(True, 9.0, 9.0, settings, pad_cap=8.0) == ""
    # Not narration-driven ⇒ no pad either.
    assert _freeze_pad_suffix(False, 4.0, 9.0, settings) == ""


def test_freeze_pad_suffix_uncapped_default_ramps_over_whole_tail() -> None:
    settings = get_settings()
    suffix = _freeze_pad_suffix(True, 4.0, 9.0, settings)
    assert suffix.startswith(f",fps={settings.render_fps},tpad=stop_mode=clone:stop_duration=")
    assert "zoompan=" in suffix
    # tpad fills to the FULL narration length (9 - 4 = 5s), so the video always
    # covers the audio — no black, no truncated audio.
    assert _stop_duration(suffix) == "5"


def test_freeze_pad_cap_bounds_the_drift_but_not_the_tpad_fill() -> None:
    # A valid cap (clip_dur < cap < duration) changes the DRIFT ramp but leaves the
    # tpad clone-to-full-duration untouched: the pad still fills the audio, only the
    # drift settles at the ceiling instead of easing over the whole over-length tail.
    settings = get_settings()
    uncapped = _freeze_pad_suffix(True, 4.0, 9.0, settings)
    capped = _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=5.5)
    assert capped != uncapped  # the drift ramp moved
    assert _stop_duration(capped) == _stop_duration(uncapped) == "5"  # audio fill unchanged


def test_freeze_pad_cap_is_a_noop_when_out_of_range_or_equal_to_duration() -> None:
    # A cap >= duration, <= clip_dur, or exactly at the duration is meaningless and
    # falls back to the uncapped ramp — byte-identical to the default.
    settings = get_settings()
    baseline = _freeze_pad_suffix(True, 4.0, 9.0, settings)
    assert _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=9.0) == baseline  # == duration
    assert _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=12.0) == baseline  # > duration
    assert _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=3.0) == baseline  # <= clip_dur
    assert _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=None) == baseline


def test_freeze_pad_suffix_speed_1x_is_byte_identical() -> None:
    # speed=1.0 must scale every emitted time by 1 → character-identical to the
    # default (no-speed) suffix, both uncapped and capped. This is the invariant
    # that keeps the default render path byte-identical.
    settings = get_settings()
    assert _freeze_pad_suffix(True, 4.0, 9.0, settings, speed=1.0) == _freeze_pad_suffix(
        True, 4.0, 9.0, settings
    )
    assert _freeze_pad_suffix(
        True, 4.0, 9.0, settings, pad_cap=5.5, speed=1.0
    ) == _freeze_pad_suffix(True, 4.0, 9.0, settings, pad_cap=5.5)


def test_freeze_pad_suffix_speed_scales_emitted_times_into_post_retime_timeline() -> None:
    # The tpad runs DOWNSTREAM of setpts=PTS/speed, so its stop_duration must be in
    # the POST-retime timeline: (duration - clip_dur)/speed. At speed=0.5 (slow-mo)
    # the 5s real fill becomes a 10s fill so the frozen tail reaches the retimed
    # -t=duration/speed bound (the A/V-mismatch fix); at speed=2.0 it halves.
    settings = get_settings()
    assert _stop_duration(_freeze_pad_suffix(True, 4.0, 9.0, settings, speed=0.5)) == "10"
    assert _stop_duration(_freeze_pad_suffix(True, 4.0, 9.0, settings, speed=2.0)) == "2.5"


# ---------------------------------------------------------------------------
# _fetch RAM cap (max_fetch_bytes) — safe_get mocked, no network
# ---------------------------------------------------------------------------


# `_fetch` now streams through `w2v_common.net.safe_get_to_path` instead of buffering a
# `safe_get` response and handing `resp.content` to `_write_atomic`. These two tests drive
# the REAL helper over an `httpx.MockTransport` rather than a fake `safe_get`, because the
# properties that matter here are properties of the streaming terminal, not of a stub:
# the cap has to bite MID-STREAM, and `dest` has to stay absent when it does. A fake that
# simply raises would assert neither — it would pass against an implementation that wrote
# the body first and checked afterwards, which is the exact OOM this change removes.
#
# `_TEST_HOST` is a public literal IP, so `validate_public_url` admits it without any
# allow_hosts plumbing and no DNS lookup happens; MockTransport answers before a socket
# is ever opened.
_TEST_HOST = "93.184.216.34"


def _mock_body_client(body: bytes) -> httpx.AsyncClient:
    """An `httpx.AsyncClient` that serves ``body`` for any request, chunked.

    The chunking is load-bearing: the real cap is counted per decoded chunk as the
    body streams, so a single-shot body would let a post-hoc size check pass this
    test. Splitting the body forces the breach to be detected before the last byte
    arrives.
    """

    async def _stream() -> AsyncIterator[bytes]:
        for i in range(0, len(body), 8):
            yield body[i : i + 8]

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_stream())

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


async def test_fetch_rejects_oversize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A body over ``max_fetch_bytes`` aborts the transfer and leaves NOTHING behind.

    Proves three things at once: the setting is threaded down to the streaming
    helper (a 100-byte body under a 10-byte cap must fail — under the old 512 MiB
    default it would not), ``ResponseTooLargeError`` propagates out of ``_fetch``
    unchanged so ``_BAD_ASSET_ERRORS`` degrades this one shot, and ``dest`` does not
    exist afterwards. The last one is the per-source-dedupe invariant: N shots share
    one fetched file, so a half-written asset left at ``dest`` would be handed to
    ffmpeg by the NEXT shot and silently ship a short film.
    """
    small = get_settings().model_copy(update={"max_fetch_bytes": 10})
    monkeypatch.setattr(renderer, "get_settings", lambda: small)

    dest = tmp_path / "asset.bin"
    async with _mock_body_client(b"x" * 100) as http:
        with pytest.raises(renderer.ResponseTooLargeError):
            await renderer._fetch(f"http://{_TEST_HOST}/clip.mp4", dest, http)

    assert not dest.exists(), "an over-cap fetch must publish nothing at dest"
    assert list(tmp_path.iterdir()) == [], "the partial temp file must be cleaned up too"


async def test_fetch_within_cap_writes_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A body under the cap lands at ``dest`` byte-for-byte, and ``_fetch`` returns it.

    ``dest`` is deliberately inside a directory that does NOT exist yet:
    ``safe_get_to_path`` writes its temp file as a SIBLING of ``dest`` and does not
    mkdir, so ``_fetch`` owns creating the parent. Without that, the first fetch into
    a fresh scratch dir fails with an OSError that ``_try_fetch`` swallows as a "bad
    asset" — a good shot degraded by a missing directory.
    """
    settings = get_settings().model_copy(update={"max_fetch_bytes": 1024})
    monkeypatch.setattr(renderer, "get_settings", lambda: settings)
    body = b"hello-shot-asset-streamed-in-chunks"

    dest = tmp_path / "fresh-scratch" / "asset.bin"
    async with _mock_body_client(body) as http:
        out = await renderer._fetch(f"http://{_TEST_HOST}/clip.mp4", dest, http)

    assert out == dest
    assert dest.read_bytes() == body
    assert list(dest.parent.glob("*.part-*")) == [], "the temp file must be renamed, not left"


# --- _fetch local-input confinement (point-of-use) ---------------------------
#
# `allow_local_file_inputs` is a SUBMIT-time policy and is enforced at the API trust
# boundary (`_guard_local_file_inputs` on /api/compile + /api/concat-audio, the prefix
# check on /api/thumbnails) — see test_compile_route.py / test_thumbnail_route.py.
# `_fetch` does NOT re-evaluate it: doing so applied a caller-input policy to refs that
# never came from a caller (an in-process `compile_shots` staging its own files), which
# silently dropped their audio out of the delivered mp4.
#
# What `_fetch` DOES enforce, and what these tests pin, is trust-level-independent:
# traversal tokens are always refused, and when the caller passes a `local_root` (i.e.
# the ref IS caller-supplied and was admitted by the boundary) the path is confined to
# it. These call `_fetch` DIRECTLY on purpose: a guard shielded by an upstream
# validator cannot be proven through the public route.


async def test_fetch_confines_caller_supplied_path_to_local_root(tmp_path: Path) -> None:
    """`local_root` given: a path INSIDE it resolves and is returned (both URL shapes)."""
    asset = tmp_path / "job-1" / "clip.mp4"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x00\x00\x00\x20ftypisom")

    bare = await renderer._fetch(
        str(asset), tmp_path / "dest", cast(httpx.AsyncClient, None), local_root=tmp_path
    )
    via_uri = await renderer._fetch(
        f"file://{asset}", tmp_path / "dest", cast(httpx.AsyncClient, None), local_root=tmp_path
    )
    assert bare == asset.resolve() == via_uri


async def test_fetch_rejects_caller_supplied_path_outside_local_root(tmp_path: Path) -> None:
    """Opting into local inputs is not a licence to read the pod: outside is refused."""
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    for url in (str(outside), f"file://{outside}", "/etc/passwd"):
        with pytest.raises(renderer.LocalFileInputError):
            await renderer._fetch(
                url, root / "dest", cast(httpx.AsyncClient, None), local_root=root
            )


async def test_fetch_containment_is_by_segment_not_string_prefix(tmp_path: Path) -> None:
    """``<root>-evil/x`` shares root's STRING prefix but not its segments.

    A ``str(resolved).startswith(str(root))`` containment check passes this path.
    ``Path.parents`` does not — which is the whole point of the check.
    """
    root = tmp_path / "work"
    root.mkdir()
    sibling = tmp_path / "work-evil"
    sibling.mkdir()
    asset = sibling / "clip.mp4"
    asset.write_bytes(b"x")
    assert str(asset).startswith(str(root))  # the string-prefix check WOULD pass
    with pytest.raises(renderer.LocalFileInputError):
        await renderer._fetch(
            str(asset), root / "dest", cast(httpx.AsyncClient, None), local_root=root
        )


async def test_fetch_resolves_symlinks_before_containment(tmp_path: Path) -> None:
    """A symlink planted INSIDE the root that points out of it must not be a bypass:
    the path is resolved FIRST, then contained."""
    root = tmp_path / "work"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("pod-local secret")
    link = root / "innocent.mp4"
    link.symlink_to(secret)
    with pytest.raises(renderer.LocalFileInputError):
        await renderer._fetch(
            str(link), root / "dest", cast(httpx.AsyncClient, None), local_root=root
        )


@pytest.mark.parametrize(
    "suffix",
    [
        "../../etc/passwd",  # literal traversal
        "%2e%2e%2fetc/passwd",  # percent-encoded ../
        "%2E%2E%2Fetc/passwd",  # case-variant of the same
        "%252e%252e%252fetc/passwd",  # double-encoded
        "..%5cetc%5cpasswd",  # encoded backslash separator
    ],
)
@pytest.mark.parametrize("confined", [True, False])
async def test_fetch_rejects_traversal_and_encoded_separators(
    tmp_path: Path, suffix: str, confined: bool
) -> None:
    """Traversal is REFUSED, never normalised/decoded — with OR without a
    ``local_root``, and even when the path is nominally rooted inside it. This is the
    one rule that needs no trust context, so it applies to internal refs too."""
    with pytest.raises(renderer.LocalFileInputError):
        await renderer._fetch(
            f"{tmp_path}/{suffix}",
            tmp_path / "dest",
            cast(httpx.AsyncClient, None),
            local_root=tmp_path if confined else None,
        )


async def test_fetch_accepts_internally_staged_path_without_local_root(tmp_path: Path) -> None:
    """REGRESSION GUARD. An in-process caller (the render harness, a fixture staged by
    the service itself) passes a local path with no ``local_root``. Re-applying the
    submit-time flag here refused it, `_try_fetch` degraded it away, and the delivered
    mp4 came out with NO AUDIO (-inf LUFS) — 22 render-harness failures whose only
    trace was a generic `shot_asset_unfetchable` line. It must resolve."""
    asset = tmp_path / "src.mp4"
    asset.write_bytes(b"\x00\x00\x00\x20ftypisom")
    assert not get_settings().allow_local_file_inputs  # prod default, and irrelevant here
    got = await renderer._fetch(str(asset), tmp_path / "dest", cast(httpx.AsyncClient, None))
    assert got == asset


async def test_try_fetch_does_not_degrade_a_refused_local_input(tmp_path: Path) -> None:
    """A refused local input must FAIL the compile, not degrade one shot.

    ``_try_fetch`` degrades genuine bad-asset conditions (404, expired presign,
    transport error). A confinement/traversal refusal is not one: it means a bug in
    what staged the path or a bypass attempt, and degrading it ships an mp4 silently
    missing content. So ``LocalFileInputError`` must stay OUT of ``_BAD_ASSET_ERRORS``
    (and out of ``ValueError``)."""
    assert not issubclass(renderer.LocalFileInputError, renderer._BAD_ASSET_ERRORS)
    with pytest.raises(renderer.LocalFileInputError):
        await renderer._try_fetch(
            "/etc/passwd", tmp_path / "dest", cast(httpx.AsyncClient, None), local_root=tmp_path
        )


# ---------------------------------------------------------------------------
# Probed duration: PER VIDEO STREAM, not per container
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"duration": "6.000000"}, 6.0),
        ({"duration": "N/A"}, 0.0),  # what ffprobe writes when it is not carried
        ({}, 0.0),
        ({"duration": None}, 0.0),
        ({"duration": "0.000000"}, 0.0),
        (None, 0.0),
    ],
)
def test_duration_field(node: object, expected: float) -> None:
    """PURE. Anything that is not a positive number means 'not carried' (0.0),
    which is what routes the caller to the other scope."""
    assert renderer._duration_field(node) == expected


@pytest.mark.parametrize(
    ("stream", "container", "expected"),
    [
        # THE DEFECT: 6s of video muxed with 12s of audio. format=duration is the
        # max over streams and reports the AUDIO length.
        (6.0, 12.0, 6.0),
        # No v:0 stream at all (an audio-only input: the audiobook concat, a
        # narration track) -> the container is the only measurement there is.
        (0.0, 12.0, 12.0),
        # Matroska/WebM carries no stream-level duration -> same fallback.
        (0.0, 8.5, 8.5),
        # A stream header claiming to outlast the file it lives in is lying.
        (30.0, 12.0, 12.0),
        # Both unknown stays unknown; every caller's `or 0.0` handles it.
        (0.0, 0.0, 0.0),
        # An unprobeable container must not veto a measured stream.
        (6.0, 0.0, 6.0),
    ],
)
def test_resolve_probed_duration(stream: float, container: float, expected: float) -> None:
    assert renderer._resolve_probed_duration(stream, container) == expected


async def test_audio_padded_clip_is_measured_on_its_video_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 6s picture muxed with 12s of audio must measure 6s, not 12s.

    ``format=duration`` is the MAXIMUM over every stream in the container by
    construction, so this exact file probes as 12.000 — and that is the number
    ``_ffprobe_duration`` used to return.
    """

    async def _fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        assert cmd[0] == "ffprobe"
        # Exactly what ffprobe emits for this file under the flags we pass.
        return '{"streams": [{"duration": "6.000000"}], "format": {"duration": "12.000000"}}'

    monkeypatch.setattr(renderer, "_run", _fake_run)

    assert await renderer._ffprobe_duration(Path("/padded.mp4")) == 6.0
    assert await renderer._playable_clip_duration(Path("/padded.mp4"), 0.0, None) == 6.0
    # The same number drives the seek arithmetic: anchor='end' takes the settled
    # tail of the PICTURE. Off the container length it seeks to 12-4=8.0 and
    # would run 4s past the last decodable video frame.
    _args, offset, _span = await renderer._clip_seek_args(
        Path("/padded.mp4"), 0.0, 4.0, clip_anchor="end"
    )
    assert offset == 1.900  # (6.0 - 4.0) - 0.10, not (12.0 - 4.0) - 0.10


async def test_audio_only_input_still_measures_its_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An audio-only input (the audiobook concat, a narration track) has NO v:0
    stream, so ffprobe returns an empty ``streams`` list. Its number must be
    unchanged — this path must not become 0.0."""

    async def _fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return '{"streams": [], "format": {"duration": "42.500000"}}'

    monkeypatch.setattr(renderer, "_run", _fake_run)
    assert await renderer._ffprobe_duration(Path("/narration.wav")) == 42.5


async def test_unparseable_probe_payload_still_degrades_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-existing contract: a payload we cannot read is 0.0, not a raise.
    (A non-zero ffprobe EXIT still raises FfmpegError from ``_run``.)"""

    async def _fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return "not json at all"

    monkeypatch.setattr(renderer, "_run", _fake_run)
    assert await renderer._ffprobe_duration(Path("/broken.mp4")) == 0.0


async def test_audio_padded_clip_still_freeze_pads_under_its_narration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE PRODUCT DEFECT this fix closes.

    ``_freeze_pad_suffix`` gates on ``0.0 < clip_dur < duration``. Fed the
    container's 12.0 for a clip whose PICTURE is 6s, a 10s narration line reads as
    'this clip already fills its line', so NO ``tpad`` clone is emitted — the
    picture ends at 6s while the output-side ``-t 10`` keeps the segment at the
    planned 10s. The delivered container duration then AGREES with the plan, which
    is precisely why nothing downstream ever noticed.
    """

    async def _fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return '{"streams": [{"duration": "6.000000"}], "format": {"duration": "12.000000"}}'

    monkeypatch.setattr(renderer, "_run", _fake_run)

    clip_dur = await renderer._playable_clip_duration(Path("/padded.mp4"), 0.0, None)
    suffix = renderer._freeze_pad_suffix(True, clip_dur, 10.0, get_settings())
    assert "tpad=stop_mode=clone" in suffix, (
        "no freeze pad emitted: the clip was measured on the container "
        "(audio) length, so its 4s picture gap under the narration is invisible"
    )
    assert "stop_duration=4" in suffix
