"""Shot-list to MP4 via ffmpeg.

Each shot is rendered to a uniform segment (same resolution / fps / pixel
format, H.264 video + AAC 48 kHz stereo audio — silent when the shot has no
narration). Uniform segments let the final join use the fast ``concat``
demuxer with ``-c copy``.

A shot's visual is, in precedence order: ``video_url`` (a pre-recorded clip),
``image_url`` (a still held for the duration), or a solid card with the
``description`` text drawn on it. When ``audio_url`` is set the segment is
stretched to the narration length so the voice-over always plays in full;
otherwise the requested ``seconds`` is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import math
import re
import secrets
import shlex
import shutil
import textwrap
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import structlog
from w2v_common.net import ResponseTooLargeError, SSRFError, safe_get_to_path

from src.api.schemas import (
    AccentCue,
    AmbientCue,
    AudioClip,
    Callout,
    Chapter,
    ColorAdjust,
    CursorKey,
    FinishProfile,
    FocusKey,
    GradeSpec,
    MusicBed,
    OverlayClip,
    Shot,
    ShotAudio,
    ShotCaptions,
    SoundCue,
    SplitScreen,
    TextOverlay,
    Transition,
)
from src.config import Settings, get_settings
from src.observability import metrics

if TYPE_CHECKING:
    from src.services.captions_ass import CaptionLook


def _endpoint_netloc(endpoint: str | None) -> str | None:
    """Normalize a MinIO endpoint setting to a bare ``host[:port]`` netloc."""
    if not endpoint:
        return None
    ep = endpoint.strip()
    if "://" in ep:
        ep = ep.split("://", 1)[1]
    ep = ep.split("/", 1)[0]  # drop any path
    if "@" in ep:
        ep = ep.rsplit("@", 1)[-1]  # drop any userinfo
    return ep.lower() or None


def _minio_allow_hosts() -> set[str]:
    """Our own MinIO endpoint netloc(s) — presigned shot/clip URLs at these
    internal hosts are trusted past the SSRF guard. Everything else, including a
    redirect off the trusted host, is still validated per-hop."""
    netloc = _endpoint_netloc(get_settings().minio_endpoint)
    return {netloc} if netloc else set()


# Heuristic char-advance ratio for a proportional sans font at integer fontsize.
# Helvetica/DejaVu averages ≈0.50 em wide per glyph; we use 0.55 to err toward
# wrapping a touch sooner (smaller-than-actual line width estimate) so long
# punctuation-heavy lines don't kiss the safe edge.
_AVG_GLYPH_WIDTH_RATIO = 0.55
# Safe-area margin as a FRACTION of canvas width; matches the 100 px gutters
# title-card generators in the fleet leave for logo placement and burn-in
# subtitles at 1920px width, so wrapped drawtext lands inside the same frame.
# A fraction (not a flat px count) so the margin — and everything measured
# against it (the wrap gutter, the callout label clamp) — stays proportional
# above/below the 1920px canvas this was tuned on, instead of shrinking to
# 3.9% of width at 2560px while still reading as "the same 100px margin".
_SAFE_MARGIN_FRAC = 100 / 1920
# The text-card body's fontsize as a fraction of render_height; identity at
# 1080p (56px). Previously a flat 56px pinned to 1080p: at 1440p that stayed
# 56px (3.89% of height vs 1080p's 5.19%, a -25% relative shrink) and
# `_wrap_for_canvas` packed ~76 chars/line instead of ~55, following the same
# render_height-fraction pattern the caption already uses (VC-6 /
# `_caption_filter_fragment`).
_TEXT_CARD_BODY_FONTSIZE_FRAC = 56 / 1080
# The text-card body drawtext's ``line_spacing=`` option — pulled out to a name
# (rather than a second inline literal) so the caption/body collision-avoidance
# math (`_text_block_height_px` calls in the text-card branch) uses the EXACT
# gap the body filter actually renders with.
_TEXT_CARD_BODY_LINE_SPACING_PX = 14
# drawtext's real rendered glyph box (``text_h``) exceeds the nominal
# ``fontsize=`` value passed to it — fonts routinely carry ascent+descent
# beyond their point size. 1.2 is the conventional typographic single-line-
# height multiplier (not a measurement of this specific font/renderer), used
# here because ``_text_block_height_px`` is Python-side geometry that can
# never call back into drawtext for its actual metric (see that function's
# docstring). Without it, BOTH sides of the collision-avoidance arithmetic in
# ``_text_card_body_y`` underestimate: the reserved caption band is too
# short, and the body's own footprint is too short, so a "no overlap"
# clamp result can still visually overlap.
_DRAWTEXT_LINE_HEIGHT_FACTOR = 1.2
# The callout label caption's fontsize as a fraction of render_height;
# identity at 1080p (44px) — matches `_caption_filter_fragment`'s DEFAULT
# `caption_drawtext_font_size_frac` (also 44/1080) at any render_height, not
# only at 1080p as the flat-px constant this replaces did.
_CALLOUT_LABEL_FONTSIZE_FRAC = 44 / 1080


def _safe_margin_px(canvas_width: int) -> int:
    """Safe-area gutter in px for a canvas of ``canvas_width`` — identity
    (100px) at the 1920px canvas this was originally tuned on."""
    return round(canvas_width * _SAFE_MARGIN_FRAC)


log = structlog.get_logger(__name__)

_FETCH_TIMEOUT_SECONDS = 120.0
# ``-b:a`` is NOT optional here. Left unset, ffmpeg's NATIVE aac encoder picks its
# own rate (~126 kbps measured) and narration is re-encoded 2-3 times before
# delivery — per-shot encode, the master pass (which is ``-c:v copy`` but a full
# audio decode/re-encode), and ``apply_postclip_audio`` whenever any shot carries
# gain or mute. Measured on a real narration clip, one generation vs the source:
# default 39.39 dB SNR at 125.8 kbps, ``160k`` 44.54 dB, ``192k`` **50.14 dB** at
# 169.1 kbps. +10.75 dB of headroom against generational loss for +43 kbps, on a
# bus whose video track is ~840 kbps — the bits are noise next to the picture.
_AUDIO_CODEC_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]

# ---------------------------------------------------------------------------
# INTERMEDIATE (per-shot segment) audio — PCM, not AAC.
# ---------------------------------------------------------------------------
# Segments are joined by the concat DEMUXER at ``-c copy`` (``concat_shots``),
# which cannot trim codec delay. Every AAC segment carries ~1024 samples of
# encoder delay plus trailing padding, and a stream copy keeps BOTH, so each cut
# inserts a hole and shifts everything after it. Measured on 3x 1.000 s
# phase-continuous sine segments encoded with the args above and joined with the
# real ``concat_shots`` command:
#
#   AAC-in-mp4  -c copy : decoded 3.072000 s (+72.0 ms, +24.0 ms/segment),
#                         a 1153-sample (24.02 ms) near-silent run at EVERY
#                         internal join.
#   pcm_s16le-in-mov    : decoded 3.000000 s (0 drift), no interior silence.
#   alac-in-mp4 -c copy : decoded 3.000000 s (0 drift), no interior silence.
#
# Production telemetry agrees from the other direction: median
# ``av_gap_seconds`` = 0.097 s ~= 4 cuts x 24 ms. The audible shape is a DROPOUT
# (stutter/hole), not a click.
#
# So the intermediate is lossless PCM and AAC is applied exactly ONCE, at the
# final mux (``overlay_audio_bed`` / ``master_audio`` / ``apply_postclip_audio``
# / ``_encode_delivery_audio``). That also REMOVES 2-3 lossy AAC generations —
# per-shot encode, the master pass, and the post-clip pass all used to re-encode
# already-AAC audio.
#
# WHY ALAC AND NOT RAW PCM. Both are lossless and both measure sample-exact
# (bench row 3 above), but ``pcm_s16le`` is not a codec an MP4 muxer will carry
# on this ffmpeg, which would force every segment and both concat intermediates
# into a MOV container. ALAC is a native ffmpeg encoder (no external library, so
# it is in the slim runtime image), is carried by MP4, and still takes
# ``-movflags +faststart`` — so the container, the file extensions, and the
# ``-c:a copy`` in ``_deband_segment`` / ``_regrade_segment`` /
# ``apply_finish_grade`` are all EXACTLY as they were. It is also ~2.6x smaller
# than raw PCM on measured material (73.9 KB vs 582.5 KB for the same 3 s bench
# join), which halves the extra scratch this change costs.
#
# ``-ar``/``-ac`` are deliberately IDENTICAL to the AAC args: segments have
# always been 48 kHz stereo (``_A48`` normalises every input to that before the
# mix), so nothing is up- or down-mixed by this change — in particular no
# ``aformat=channel_layouts=stereo`` is introduced anywhere new. The
# mono-preserving audiobook/ACX path (``_concat_audio_copy``, WAV) is a different
# pipeline and is untouched.
#
# ⛔ ``-sample_fmt s32p`` IS LOAD-BEARING — without it the film has no narration.
# ALAC takes its BIT DEPTH from whatever sample format the filter graph hands the
# encoder: an ``s16`` input (a 24 kHz pcm_s16le narration WAV, bare-mapped or
# through ``aresample``) negotiates ``s16p`` -> 16-bit ALAC, while a FLOAT graph
# (``amix``/``alimiter`` on an SFX shot, an AAC clip's own ``fltp`` audio)
# negotiates ``s32p`` -> 24-bit ALAC. ``concat_shots`` stream-copies, so the
# joined stream carries ONE extradata (the first segment's) and every segment of
# the other depth is decoded with the wrong frame layout. Measured on the prod
# image (ffmpeg 7.1.5), real ``say`` speech at -22.7 dBFS / centroid 2.5 kHz:
#
#   16-bit narration after a 24-bit SFX segment : -124.6 dBFS (gone) and
#                                                 -8.2 dBFS, centroid 8.5 kHz,
#                                                 flatness 0.91 (broadband noise)
#   24-bit SFX segment after 16-bit narration   : -6.1 dBFS, centroid 11.5 kHz,
#                                                 flatness 0.82
#
# That is the delivered-master symptom of run 01a09f82 (website_review,
# 2026-09-14): silence where the narration was, plus ~10 kHz noise bursts.
# Pinning ``s32p`` makes EVERY segment 24-bit ALAC regardless of the graph that
# fed it (16-bit content is carried exactly — lossless, just wider), and
# ``concat_shots`` now refuses to stream-copy segments whose audio params differ.
_SEGMENT_AUDIO_SAMPLE_FMT = "s32p"
_SEGMENT_AUDIO_CODEC_ARGS = [
    "-c:a",
    "alac",
    "-sample_fmt",
    _SEGMENT_AUDIO_SAMPLE_FMT,
    "-ar",
    "48000",
    "-ac",
    "2",
]

# Container for every per-shot segment and for the concat/xfade intermediates.
# UNCHANGED at ``.mp4`` — ALAC is chosen precisely so it does not have to move.
# Named as a constant so the segment container and ``_SEGMENT_AUDIO_CODEC_ARGS``
# stay visibly coupled: a future codec that MP4 cannot carry must change both.
_SEGMENT_EXT = ".mp4"


def _segment_audio_codec_args(deterministic: bool = False) -> list[str]:
    """Per-shot/intermediate audio args: lossless 48 kHz stereo ALAC.

    ``deterministic`` takes no extra tokens here — a LOSSLESS encoder has no
    rate-control or psychoacoustic decisions to vary, so the samples are
    bit-reproducible by construction, and the container/writing-app tag is
    already handled by the global ``-fflags +bitexact`` from
    ``_bitexact_input_args``. The parameter exists so every call site reads the
    same as its ``_video_codec_args`` neighbour."""
    del deterministic
    return list(_SEGMENT_AUDIO_CODEC_ARGS)


def _video_codec_args(
    settings: Settings, deterministic: bool = False, *, intermediate: bool = False
) -> list[str]:
    """Per-shot video-encode args for the configured encoder.

    ``intermediate=True`` marks an encode whose output is a TEMP file that a
    later pass re-encodes or that ships only as picture polish — today the three
    segment re-encodes (``_deband_segment``, ``_regrade_segment``,
    ``_polish_segment``). It lowers the CRF cell by
    ``render_intermediate_crf_delta`` (default 6) and drops VBV; see
    ``mastering.encode_ladder`` for the full reasoning and the measurement it
    rests on. Everything else about the args — the colour stamp, the CFR pin,
    the GOP, the deterministic tokens — is IDENTICAL, so an intermediate stays
    concat-stream-copy compatible with a delivery-cell segment beside it.

    ⚠️ It is a NO-OP on the hardware encoder branches. ``h264_videotoolbox`` /
    ``h264_nvenc`` are bitrate-driven and ignore ``-crf`` entirely, so there is
    no cell to lower; giving them a second ``render_hw_bitrate``-derived knob
    would be an unvalidated constant on a path this fleet does not currently
    run. Documented rather than silently half-applied.

    ⛔ The DELIVERED master encode must NOT pass ``intermediate=True``.
    ``render_crf`` is the delivery cell and owns it alone. Note that
    ``apply_finish_grade`` and ``overlay_audio_bed`` ARE that final encode —
    their output is returned straight out of ``_apply_final_audio`` as the
    deliverable (only audio is touched after them) — which is why neither is an
    intermediate despite both being a third generation.

    ``render_encoder`` selects the encode path (default ``libx264``):

    - ``libx264`` — CPU/software. Settings-driven quality (``render_preset`` +
      ``render_crf``). Replaces the old fixed ``-preset veryfast`` / default-CRF
      constant, which was visibly soft on slide/screen text at 2K/4K.
    - ``h264_videotoolbox`` / ``h264_nvenc`` — hardware encoders (Apple media
      engine / NVIDIA NVENC). These ignore ``-preset``/``-crf`` and are
      bitrate-driven (``render_hw_bitrate``); only valid on a deployment whose
      host actually exposes that encoder. This is the offload seam: flip
      ``RENDER_ENCODER`` in gitops env, no code change.

    All paths keep ``-pix_fmt yuv420p`` so the uniform per-shot encode stays
    concat-stream-copy compatible (``concat_shots``).

    The encode/delivery master (VC-2, contract C11) adds — additively — a BT.709
    ``colr`` atom to BOTH branches (``colorspace_tag_args``) and threads the
    CRF/VBV ladder cell (``encode_ladder``). With default knobs the ladder cell
    is exactly ``render_crf`` with no VBV, so the ONLY change versus the
    pre-master args is the additive colorspace tag — the encode stays
    param-identical. VBV ``-maxrate``/``-bufsize`` appear only when configured.

    ``-r render_fps`` (VC-3 cadence) is emitted UNCONDITIONALLY, on both
    branches, so every one of this helper's callers re-asserts a constant
    output frame rate at the point of encode — this is the single source now;
    a caller must NOT also append its own ``-r`` (that would duplicate the
    flag on the ffmpeg command line).

    ``deterministic`` (per-job ``deterministic`` render tag, default off) still
    appends ``-threads 1 -flags +bitexact`` **only on the libx264/software
    path** so the encode is bit-reproducible across runs: x264 multithreading
    picks slice boundaries by thread-scheduling order (run-to-run noise), and
    ``+bitexact`` drops the encoder's version stamp. The HW encoders
    (``h264_videotoolbox``/``h264_nvenc``) are NOT bit-reproducible and
    ``-threads`` is meaningless there, so those tokens are intentionally
    omitted on that branch.

    ``-fps_mode:v cfr`` (also VC-3, the actual CFR *pin* rather than the
    declared rate) is emitted UNCONDITIONALLY on the libx264/software path — it
    used to be gated behind ``deterministic``, which meant production (which
    always renders with ``deterministic=False``) never got it, and is exactly
    why ``cfr_ok`` was false on ~46% of masters. It stays OFF the hardware
    branch: the docstring's own words were "meaningless" there
    (videotoolbox/nvenc don't consult it), so the pin stays scoped to the
    encoder branch it always belonged to.

    ⛔ It is spelled ``-fps_mode:v``, NEVER the old global ``-vsync``.
    ``-vsync`` was deprecated in ffmpeg 5.1 and is REJECTED by current ffmpeg
    (Homebrew 9.x: every render failed); ffmpeg 6.1 (CI runner apt) and 7.1
    (the image) still accepted it, so a base-image bump would have silently
    broken every production render. ``-fps_mode`` exists since 5.1 — every
    supported runtime (6.1 / 7.1 / 9.x) has it — so there is no version gate.
    Same semantics (``cfr`` = ``-vsync cfr``); it is a per-output-stream
    OUTPUT option, so it must precede the output path (it does: these args
    are spliced before it), and ``:v`` scopes it to the video stream.

    ``-g``/``-keyint_min`` (VC-4, 3c) are also emitted UNCONDITIONALLY on both
    branches, from this single place, so a scrub on the delivered master (played
    directly off a presigned MinIO URL, no proxy transcode) never has to decode
    more than 2s back to the prior keyframe.
    """
    from src.services import mastering

    color = mastering.colorspace_tag_args(settings)
    # ⛔ ``colorspace_tag_args`` alone is NOT enough on the production ffmpeg.
    # Measured on 7.1.5 with these verbatim args: only ``color_space`` lands;
    # ``color_transfer`` and ``color_primaries`` both come back ``unknown`` — the
    # entire 100%-false ``colorspace_ok`` fleet rate, and it happens HERE at the
    # encode, not at a downstream ``-c:v copy`` (a correctly-tagged file was
    # measured to keep all three through both). The bitstream filter stamps the
    # SPS VUI directly and is version-independent. See
    # ``mastering.colorspace_bsf_args`` for the full measurement.
    color_bsf = mastering.colorspace_bsf_args(settings)
    crf, vbv = mastering.encode_ladder(settings, intermediate=intermediate)
    encoder = settings.render_encoder
    fps_args = ["-r", str(settings.render_fps)]
    # VC-4 (3c): explicit GOP so a keyframe lands at least every 2s (2x fps
    # frames apart) instead of x264's ~250-frame (~10s at 30fps) default. The UI
    # scrubs the delivered master directly off a presigned MinIO URL — no proxy
    # transcode — so a keyframe-less 10s span meant seeking back into it had to
    # decode from the PRIOR keyframe. Centralised here (the same single place
    # ``-r render_fps`` was centralised in #218) so no call site duplicates it.
    gop_args = ["-g", str(settings.render_fps * 2), "-keyint_min", str(settings.render_fps)]
    if encoder in ("h264_videotoolbox", "h264_nvenc"):
        # HW encoders are not bit-reproducible; deterministic/-fps_mode tokens
        # intentionally omitted (meaningless on this branch).
        args = [
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            *color,
            "-b:v",
            settings.render_hw_bitrate,
            *vbv,
            *fps_args,
            *gop_args,
            *color_bsf,
        ]
    else:
        # Default / explicit software path.
        args = [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            *color,
            "-preset",
            settings.render_preset,
            "-crf",
            str(crf),
            *vbv,
            "-fps_mode:v",
            "cfr",
            *fps_args,
            *gop_args,
            *color_bsf,
        ]
        if deterministic:
            args += ["-threads", "1", "-flags", "+bitexact"]
    return args


def _audio_codec_args(deterministic: bool = False) -> list[str]:
    """AAC 48 kHz stereo output args, plus ``-flags:a +bitexact`` when
    ``deterministic`` so the audio encode carries no version stamp. Returns the
    ``_AUDIO_CODEC_ARGS`` tokens UNCHANGED when off, so every prod caller's
    command stays byte-identical."""
    if deterministic:
        return [*_AUDIO_CODEC_ARGS, "-flags:a", "+bitexact"]
    return list(_AUDIO_CODEC_ARGS)


def _bitexact_input_args(deterministic: bool = False) -> list[str]:
    """Global ``-fflags +bitexact`` (placed before ``-i``) for a deterministic
    mux: strips ffmpeg's wall-clock ``creation_time`` and the ``encoder``/writing-
    app version tag from EVERY container in the chain (per-shot encode, concat,
    xfade, grade/master/caption remux), so the same input yields byte-identical
    output. Empty (no new tokens) when off — prod stays byte-identical to today."""
    return ["-fflags", "+bitexact"] if deterministic else []


# Output-codec args for the audiobook concat path, keyed by AudioConcatRequest
# output_format. WAV stays PCM (lossless, fast); MP3 re-encodes at VBR q2.
_AUDIOBOOK_OUTPUT_CODECS: dict[str, list[str]] = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    # m4b = AAC in an MP4/iTunes container (the audiobook delivery format). The
    # ``ipod`` muxer writes an m4b that carries ID3v2/Nero chapter markers; 128k
    # AAC is the ACX-accepted spoken-word bitrate.
    "m4b": ["-c:a", "aac", "-b:a", "128k", "-f", "ipod"],
}


def _audiobook_codec_args(output_format: str, deterministic: bool = False) -> list[str]:
    """Audiobook output-codec args, plus ``-flags:a +bitexact`` when
    ``deterministic`` so the encode carries no version stamp (drops the LAME
    version tag on the MP3/libmp3lame path; WAV/PCM is already sample-exact and
    relies on the global ``+bitexact`` before ``-i`` to strip its container tag).
    Returns the ``_AUDIOBOOK_OUTPUT_CODECS`` tokens UNCHANGED when off, so every
    prod caller's command stays byte-identical."""
    codecs = _AUDIOBOOK_OUTPUT_CODECS[output_format]
    if deterministic:
        return [*codecs, "-flags:a", "+bitexact"]
    return list(codecs)


class FfmpegError(RuntimeError):
    pass


class RetryableRenderError(RuntimeError):
    """A TRANSIENT render failure — a subprocess timeout or a signal-kill (OOM=-9) under
    memory pressure, NOT a deterministic encode error. The caller stamps the job
    ``interrupted`` (re-runnable) rather than terminal ``failed`` so a resend/reaper
    re-runs it; the poison-attempt + lifetime caps still bound a truly-wedged job."""


# ---------------------------------------------------------------------------
# Per-pod ffmpeg concurrency gate (the exit=-9 OOM fix)
# ---------------------------------------------------------------------------
# Shots within one compile now render CONCURRENTLY (compile_shots fans them
# out, bounded by a local gate), and multiple compile jobs also interleave on
# the event loop, so without a cap their combined full-resolution ffmpeg
# decode/encode buffers OOM-kill ffmpeg (exit=-9) — the failure observed under
# load. This is the global ceiling: it bounds total ffmpeg/ffprobe subprocesses
# in flight per pod across ALL shots and ALL requests, and so must stay in
# lockstep with the pod memory limit (~1GiB per concurrent encode).
#
# An asyncio (not threading) Semaphore is correct because ``_run`` is async and
# awaited from the event loop. It is cached against the running loop and rebuilt
# when the loop changes: an asyncio.Semaphore binds to the loop it is first used
# on and raises if reused across loops, so each test's fresh event loop must get
# its own. Gating ``_run`` (rather than just the encodes) is deadlock-free
# because calls are sequential per request and ``_run`` never nests inside
# ``_run``; ffprobe is sub-second so the occasional slot it borrows is noise.
_DEFAULT_MAX_CONCURRENT_FFMPEG = 2
_ffmpeg_max_concurrent = _DEFAULT_MAX_CONCURRENT_FFMPEG
_ffmpeg_semaphore: asyncio.Semaphore | None = None
_ffmpeg_semaphore_loop: asyncio.AbstractEventLoop | None = None


def configure_ffmpeg_concurrency(max_concurrent: int) -> None:
    """Set the per-pod concurrent-ffmpeg cap (called once at app startup).

    Stores the size and drops any cached semaphore so the next ``_run`` rebuilds
    it at the configured size inside the running event loop. Safe to call with
    no running loop (e.g. ``create_app`` at import time) — no semaphore is
    constructed here.
    """
    global _ffmpeg_max_concurrent, _ffmpeg_semaphore, _ffmpeg_semaphore_loop
    _ffmpeg_max_concurrent = max(1, max_concurrent)
    _ffmpeg_semaphore = None
    _ffmpeg_semaphore_loop = None


def _get_ffmpeg_semaphore() -> asyncio.Semaphore:
    """Return the per-loop ffmpeg gate, building it on first use in this loop."""
    global _ffmpeg_semaphore, _ffmpeg_semaphore_loop
    loop = asyncio.get_running_loop()
    if _ffmpeg_semaphore is None or _ffmpeg_semaphore_loop is not loop:
        _ffmpeg_semaphore = asyncio.Semaphore(_ffmpeg_max_concurrent)
        _ffmpeg_semaphore_loop = loop
    return _ffmpeg_semaphore


def _escape_drawtext(text: str) -> str:
    """Escape characters that are special inside ffmpeg's ``drawtext`` value.

    See https://ffmpeg.org/ffmpeg-filters.html#Notes-on-filtergraph-escaping.

    LLM-generated captions (orchestrator's website_review_compile + similar)
    contain real newlines from multi-paragraph markdown. ffmpeg's drawtext
    filter parser treats unescaped newlines as a filter-arg break and aborts
    with exit=8 ("Unable to parse option value"). drawtext's ``\\n`` literal
    escape is unreliable across builds, and the 64px single-line layout
    used by the orchestrator render path does not word-wrap paragraphs
    legibly anyway. Pragmatic fix: collapse every run of whitespace
    (newlines, tabs, multiple spaces) into a single space BEFORE the
    per-character escapes — that is what the rendered card actually shows
    in any case.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return (
        text.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace(",", r"\,")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


async def _run(cmd: list[str], *, capture_stderr: bool = False) -> str:
    log.debug("ffmpeg.exec", cmd=" ".join(shlex.quote(c) for c in cmd))
    # Hold the gate only around the exec + I/O drain (the memory-heavy part);
    # the cheap error formatting below must not occupy a slot.
    async with _get_ffmpeg_semaphore():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=get_settings().ffmpeg_subprocess_timeout_seconds,
            )
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()  # reap the child while STILL holding the semaphore
            raise
        except TimeoutError as exc:
            # A single ffmpeg step stalled (transient — DB/disk/CPU pressure), NOT the
            # whole-job work_timeout. Distinct from the outer asyncio.wait_for so the
            # caller re-runs rather than terminally failing with a misleading message.
            proc.kill()
            await proc.wait()
            raise RetryableRenderError(
                f"{cmd[0]} exceeded {get_settings().ffmpeg_subprocess_timeout_seconds}s "
                "subprocess timeout"
            ) from exc
    rc = proc.returncode
    if rc != 0:
        # ffmpeg's stderr layout: ~1.2 KiB of `configuration: ...` banner,
        # then the actual error message at the END. Take the tail so the
        # real error survives the truncation. Investigated 2026-05-28 on
        # website_review run 47fde1d9 where exit=234 was unknowable because
        # the previous `[:1000]` head-slice cut off mid-config-banner.
        err = stderr.decode(errors="replace")
        tail = err[-1500:] if len(err) > 1500 else err
        if rc is not None and rc < 0:
            # Killed by a signal (OOM=-9 under memory pressure, etc.) — a transient
            # RESOURCE failure, not a deterministic encode error. Retryable.
            raise RetryableRenderError(f"{cmd[0]} killed by signal {-rc}: ...{tail}")
        raise FfmpegError(f"{cmd[0]} exit={rc}: ...{tail}")
    # ffmpeg's measurement filters (loudnorm print_format=json, silencedetect)
    # write their stats to STDERR, not stdout — the read-only loudness measure
    # pass opts into ``capture_stderr`` to read them. Default False keeps every
    # existing caller returning stdout, byte-for-byte unchanged.
    if capture_stderr:
        return stderr.decode(errors="replace")
    return stdout.decode(errors="replace")


async def _discard_stream(reader: asyncio.StreamReader) -> None:
    """Read ``reader`` to EOF, discarding everything (pipe teardown helper)."""
    while await reader.read(1 << 16):
        pass


async def _stream_raw_stdout(
    cmd: list[str],
    consume: Callable[[asyncio.StreamReader], Awaitable[None]],
    *,
    budget_seconds: float,
) -> bool:
    """Sibling of ``_run`` for a caller that STREAMS binary stdout (a sequence
    of PGM frames) instead of buffering it: ``consume`` reads ``proc.stdout``
    incrementally, so a long film is never held in memory at once. Same
    semaphore, same cancellation/signal/exit-code handling as ``_run``.

    ``budget_seconds`` (capped at the fleet subprocess timeout) is a SOFT wall
    bound: on expiry the child is killed and this returns ``False`` — whatever
    ``consume`` already processed stands, the caller decides what an incomplete
    pass means. Returns ``True`` when ``consume`` finished and ffmpeg exited 0.

    Keeps the async-hardening invariant: every ffmpeg/ffprobe subprocess in
    this service goes through ``_run`` or this function, never a bare
    ``create_subprocess_exec`` elsewhere. Used by the banding spatial term's
    single-pass frame extraction (``measure.measure_spatial_factors_streamed``)."""
    log.debug("ffmpeg.exec_stream", cmd=" ".join(shlex.quote(c) for c in cmd))
    timeout = min(budget_seconds, get_settings().ffmpeg_subprocess_timeout_seconds)
    async with _get_ffmpeg_semaphore():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if proc.stdout is None or proc.stderr is None:  # PIPE was requested; never None
            proc.kill()
            await proc.wait()
            raise FfmpegError(f"{cmd[0]}: no stdout/stderr pipe")
        stderr_task = asyncio.ensure_future(proc.stderr.read())
        completed = False
        try:
            await asyncio.wait_for(consume(proc.stdout), timeout=timeout)
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            completed = True
        except TimeoutError:
            return False
        finally:
            # EVERY exit path — budget expiry, cancellation, or ANY exception out
            # of ``consume`` (a malformed frame, a measure error) — must kill and
            # reap the child while STILL holding the semaphore. Otherwise ffmpeg
            # stays alive blocked on a full stdout pipe and the stderr reader leaks.
            if proc.returncode is None:
                proc.kill()
                # asyncio's Process.wait() also waits for the pipes to close; a
                # stdout nobody reads (its StreamReader paused at the buffer limit)
                # never reaches EOF, so DRAIN it alongside the reap — else the
                # wait deadlocks on exactly the abandoned stream this path handles.
                await asyncio.gather(_discard_stream(proc.stdout), proc.wait())
            if not completed:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stderr_task
        stderr = await stderr_task
    rc = proc.returncode
    if rc != 0:
        err = stderr.decode(errors="replace")
        tail = err[-1500:] if len(err) > 1500 else err
        if rc is not None and rc < 0:
            raise RetryableRenderError(f"{cmd[0]} killed by signal {-rc}: ...{tail}")
        raise FfmpegError(f"{cmd[0]} exit={rc}: ...{tail}")
    return True


def _float_field(raw: object) -> float | None:
    """PURE: an ffprobe scalar as a float, or ``None`` when absent/``N/A``.

    Distinct from :func:`_duration_field`, which folds "absent" and "zero" onto
    the same ``0.0`` because no DURATION is meaningfully zero. A packet TIMESTAMP
    of exactly ``0.0`` is the first frame and must not be discarded, so this one
    keeps the two apart. NEVER raises."""
    if isinstance(raw, bool) or not isinstance(raw, str | int | float):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# How far back from the end of a long container the packet-timestamp fallback
# reads. Only the LAST video packet decides the picture length, so a full-stream
# header walk is wasted work on a 20-minute master; 15s is comfortably more than
# one keyframe interval at every cadence this fleet encodes (the seek lands on the
# keyframe BEFORE this mark, so the window is always ≥ this).
_PACKET_SCAN_TAIL_SECONDS = 15.0


def _duration_field(node: object) -> float:
    """PURE: an ffprobe node's ``duration`` as a positive float, else ``0.0``.

    ``0.0`` means "not carried" — ffprobe writes ``N/A`` (or omits the key) on
    every container that has no duration for that scope. NEVER raises."""
    if not isinstance(node, dict):
        return 0.0
    raw = node.get("duration")
    if not isinstance(raw, str | int | float):
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return value if value > 0.0 else 0.0


def _resolve_probed_duration(stream_seconds: float, container_seconds: float) -> float:
    """PURE: combine the ``v:0`` stream duration and the container duration.

    WHY the stream wins: ``format=duration`` is the MAXIMUM over every stream in
    the container by construction, so a clip whose AUDIO outruns its VIDEO probes
    as the AUDIO length. That number is not cosmetic — it is what
    ``_playable_clip_duration`` reports, and the freeze-pad guard tests
    ``0.0 < clip_dur < duration``. Fed 12.0 for a 6s picture under a 10s
    narration line, an audio-padded source reads as "already fills its
    narration", NO ``tpad`` clone is emitted, and the picture ends 4s early while
    the output-side ``-t`` keeps the segment at the planned length — so the
    delivered container duration AGREES with the plan and nothing downstream can
    see the gap. ``_clip_seek_args`` has the same dependency: ``clip_anchor='end'``
    derives its seek from ``clip_len - shot_duration`` and would land past the
    last decodable video frame.

    Absent (``0.0``) on either side falls back to the other, which keeps every
    non-video input on exactly its previous number: an audio-only file (the
    audiobook concat, ``_resolve_shot_assets``) has no ``v:0`` stream at all, and
    Matroska/WebM carries no stream-level ``duration``. The container bounds its
    streams by construction, so a stream header claiming to outlast the file it
    lives in is lying and is clamped."""
    if stream_seconds <= 0.0:
        return container_seconds
    if container_seconds <= 0.0:
        return stream_seconds
    return min(stream_seconds, container_seconds)


def _packet_end_seconds(packets: object) -> float:
    """PURE: the END of the last video packet in an ffprobe ``packets`` payload.

    ``max(pts_time + duration_time)`` over the packets — the MEASURED end of the
    picture, taken from the timestamps the container actually carries.

    ⚠️ NOT ``nb_frames / fps``, and that is the whole point. Real sources in this
    fleet are VFR (an observed capture reports ``avg_frame_rate`` 5.96 against an
    ``r_frame_rate`` of 30), so any frame-count-over-rate derivation is wrong by
    the ratio between those two numbers — it would under-report a 6s picture as
    1.2s and over-pad every shot. Timestamps are correct for CFR and VFR alike.

    ``max`` rather than "the last element" because ffprobe emits packets in
    STORAGE order, which for B-frames is not presentation order. A missing or
    unparseable ``duration_time`` contributes its ``pts_time`` alone (an
    underestimate of one frame, never an overestimate). ``0.0`` when the payload
    carries no usable packet — which the caller reads as "unmeasured" and falls
    back on, never as "zero-length picture"."""
    if not isinstance(packets, list):
        return 0.0
    end = 0.0
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        pts = _float_field(packet.get("pts_time"))
        if pts is None:
            continue
        end = max(end, pts + (_float_field(packet.get("duration_time")) or 0.0))
    return end


async def _video_stream_packet_end(path: Path, container_seconds: float) -> float:
    """The v:0 picture length of ``path``, measured from PACKET TIMESTAMPS.

    The fallback for a video stream that carries no ``duration`` tag, which is
    NOT a rare shape: Matroska/WebM never writes one (verified — an mkv muxed
    from a 6s picture under a 12s narration reports ``streams[0] == {}`` and
    ``format.duration == 12.023``), and neither do several fragmented-MP4 and
    piped writers. On that shape :func:`_resolve_probed_duration` has nothing to
    resolve and falls back to the container, and ``format=duration`` is the
    MAXIMUM over every stream by construction — so the audio length comes back as
    the picture length and the freeze-pad guard is fed the very number #202
    stopped it being fed on MP4.

    Cost is paid ONLY on that shape (a tagged stream never reaches here), and it
    is a header walk, not a decode. ``-read_intervals`` restricts it to the last
    ~15s when the container is long enough for that to matter; an interval that
    yields nothing (a seek that lands past the last video packet) retries the
    whole stream rather than silently reporting a short picture.

    ``0.0`` on any probe/parse failure — the caller then keeps the pre-existing
    container fallback, so this can only ever make the estimate MORE video-
    specific, never fail a render that used to work."""
    attempts: list[list[str]] = []
    if container_seconds > _PACKET_SCAN_TAIL_SECONDS * 2:
        attempts.append(
            ["-read_intervals", f"{container_seconds - _PACKET_SCAN_TAIL_SECONDS:.3f}%"]
        )
    attempts.append([])
    for args in attempts:
        try:
            out = await _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    *args,
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "packet=pts_time,duration_time",
                    "-of",
                    "json",
                    str(path),
                ]
            )
            payload = json.loads(out)
        except Exception as exc:  # noqa: BLE001 — best-effort; container fallback stands
            log.warning(
                "renderer.video_stream_packet_end.failed",
                path=str(path),
                read_intervals=bool(args),
                error=str(exc),
            )
            continue
        if not isinstance(payload, dict):
            continue
        end = _packet_end_seconds(payload.get("packets"))
        if end > 0.0:
            return end
    return 0.0


async def _ffprobe_duration(path: Path) -> float:
    """The duration of ``path`` in seconds, measured PER VIDEO STREAM.

    ONE ffprobe asking for both the ``v:0`` stream's own duration and the
    container's, resolved by :func:`_resolve_probed_duration`. Deliberately a
    single subprocess: a second probe here would double the ffmpeg-gate traffic
    on the hot per-shot path.

    THE ONE EXCEPTION, and it is the hole #202 left open: a v:0 stream that
    EXISTS but carries no ``duration`` tag. #202 made the stream win over the
    container, but on that shape there is no stream number to win with, so the
    container's max-over-all-streams (i.e. the audio) came back as the picture
    length anyway. :func:`_video_stream_packet_end` measures it from the packet
    timestamps instead. Distinguishing the two ``0.0``s matters: an audio-only
    input (the audiobook concat) has NO v:0 stream at all, so ``streams`` is
    empty and the container remains exactly right for it — only a stream that is
    present-but-untagged takes the second probe.

    Contract is otherwise unchanged: a non-zero ffprobe exit still raises
    ``FfmpegError`` (that is what keeps an unreadable source loud), and an
    unparseable payload still degrades to ``0.0``."""
    out = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(out)
    except ValueError:
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    streams = payload.get("streams")
    stream = streams[0] if isinstance(streams, list) and streams else None
    stream_seconds = _duration_field(stream)
    container_seconds = _duration_field(payload.get("format"))
    # A v:0 stream that EXISTS but is untagged: measure the picture from its packet
    # timestamps rather than inheriting the container's max-over-all-streams. Guarded
    # on ``stream is not None`` so an audio-only input (no v:0 at all) never pays for
    # a probe that could only come back empty.
    if stream is not None and stream_seconds <= 0.0:
        stream_seconds = await _video_stream_packet_end(path, container_seconds)
    return _resolve_probed_duration(stream_seconds, container_seconds)


async def probe_output_duration(path: Path) -> float:
    """ffprobe the ACTUAL encoded duration of the output container.

    Reported alongside the declared/nominal ``total_seconds`` so the orchestrator
    can flag a truncated or short-rendered file (a material gap between the two).
    Reporting-only metadata: returns 0.0 on ANY probe/parse failure so it never
    fails an otherwise-good render (the orchestrator treats 0.0 as "unknown").
    """
    try:
        return await _ffprobe_duration(path)
    except Exception as exc:  # noqa: BLE001 — best-effort metadata, never fail the render
        log.warning("probe_output_duration.ffprobe_failed", path=str(path), error=str(exc))
        return 0.0


async def probe_dimensions(path: Path) -> tuple[int, int]:
    """ffprobe the primary video stream's encoded (width, height).

    Used to report the ACTUAL output resolution back to the orchestrator (so the
    e2e post-promote check can assert the 2K/4K tier landed) rather than echoing
    the requested settings. Returns (0, 0) on ANY probe/parse failure (a bad
    ffprobe exit raises FfmpegError) — this is reporting-only metadata and must
    never fail an otherwise-good render.
    """
    try:
        out = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                str(path),
            ]
        )
        w_str, h_str = out.strip().split("x", 1)
        return int(w_str), int(h_str)
    except Exception as exc:  # noqa: BLE001 — best-effort metadata, never fail the render
        log.warning("probe_dimensions.ffprobe_failed", path=str(path), error=str(exc))
        return 0, 0


async def _audio_stream_seconds(path: Path) -> float | None:
    """The first audio stream's own duration, or ``None`` when it cannot be read.

    Per STREAM, not the container: the container is the max over streams, and a
    film whose picture outlasts its audio by a frame would report the picture.
    Best-effort — never raises (the one caller fails open on ``None``)."""
    try:
        out = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "csv=p=0",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort; caller fails open
        log.warning("audio_stream_seconds.ffprobe_failed", path=str(path), error=str(exc))
        return None
    value = _float_field(out.strip().split("\n", 1)[0].strip(",") or None)
    return value if value is not None and value > 0 else None


async def _has_audio_stream(path: Path) -> bool:
    out = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-select_streams",
            "a",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        return bool(json.loads(out).get("streams"))
    except ValueError, AttributeError:
        return False


async def _probe_video_codec(path: Path) -> str | None:
    """The ``codec_name`` of ``path``'s first video stream, or ``None`` if unknown.

    Exists for exactly one caller — the alpha-overlay decoder selection in
    ``_inject_overlay_clips`` — and returns ``None`` (never raises) on ANY probe or
    parse failure, because an unprobeable overlay must degrade to "let ffmpeg pick
    the decoder", not take the shot down. See that function for why the answer
    matters.
    """
    try:
        out = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name",
                "-select_streams",
                "v:0",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — decoder selection is best-effort, never a gate
        log.warning("overlay_clip.codec_probe_failed", path=str(path), error=str(exc))
        return None
    return out.strip().splitlines()[0].strip() if out.strip() else None


class UnfetchableNarrationError(Exception):
    """A shot's ``audio_url`` (its narration) could not be fetched.

    Deliberately NOT a ``ValueError`` and NOT ``RetryableRenderError``, for two
    different reasons:

    1. It must stay OUT of ``_BAD_ASSET_ERRORS`` / ``_try_fetch``'s degrade net.
       Every OTHER shot asset (video, image, title-card clip, overlay, SFX)
       degrades one shot to its next-best visual and keeps rendering, because
       losing a picture still ships a watchable, narrated video. Narration is
       different: it is not decoration on top of the shot, it IS the
       deliverable's content. A run production evidence: a compile whose
       ken_burns clip 403'd (expired 1h presigned MinIO URL) delivered a
       ``completed`` master missing that shot's narration line and measuring
       68% digital silence — the exact shape ``shot_asset_unfetchable``'s
       silent fallback was never supposed to allow for audio.
    2. It must NOT map to ``RetryableRenderError`` (stamped ``interrupted`` /
       reaper re-runnable). The compiler's own reaper re-runs a reclaimed job
       from the SAME persisted ``job_options`` — the SAME expired presigned
       URL — so a reaper retry of an expired-URL failure cannot recover; it
       just burns the poison-attempt budget and eventually hard-fails anyway,
       hours later, having wasted every attempt. Left as a bare ``Exception``,
       this instead falls through to the generic ``except Exception`` branch
       in ``compile.py``'s ``_run``, which stamps the job terminal ``failed``
       (not reaper-retried). ``w2v_orchestrator``'s ``VideoCompileClient``
       never auto-resubmits a terminal ``failed`` job either (``stage_call``
       passes ``retries=0`` for a compile call — "a retry would duplicate the
       compile" — and only a *lost* (404) job is resubmitted once). So the run
       fails loudly and requires an explicit resume/regenerate — which reruns
       ``compile_stage.handle`` from scratch, including its presign phases —
       to actually recover with FRESH URLs. That is strictly better than the
       alternative this replaces: an unrecoverable expired URL degrading a
       shot forever while the run reports ``completed``.

    Video/image/title-card/overlay/SFX/music-bed loss is unaffected — see
    ``LocalFileInputError`` and ``compile_shots``'s music-bed handling for why
    those stay best-effort.
    """


class LocalFileInputError(Exception):
    """A ``file://`` / bare-path asset reference was refused at the point of USE.

    Deliberately NOT a ``ValueError``, so it stays OUT of ``_BAD_ASSET_ERRORS`` and
    propagates instead of degrading. The caller-facing gate lives at the API trust
    boundary (``_guard_local_file_inputs`` on /api/compile, the prefix check on
    /api/thumbnails), so by the time a local ref reaches ``_fetch`` it has already
    been admitted — a refusal HERE means a traversal token or a confinement escape,
    i.e. a bug in what staged the path or an attempt to walk past the boundary.
    Degrading that produced a *silently wrong* render: the asset vanished, the shot
    fell back to a text card, the delivered mp4 had no audio at all, and the only
    trace was a generic ``shot_asset_unfetchable`` line. A paid compile must fail
    loudly rather than ship a file that is quietly missing its content.
    """


# Percent-encoded separators/dots that must never survive into a local path. A
# caller that sends ``%2e%2e%2f`` (or the double-encoded ``%252e%252f``) is trying
# to get a decoder somewhere downstream to reconstitute ``../``. We do not decode
# — we refuse. ``%25`` is the encoding of ``%`` itself, so it covers EVERY
# double-encoded form in one check.
_ENCODED_PATH_MARKERS = ("%2e", "%2f", "%5c", "%25")


def _refuse_local_input(url: str, reason: str) -> LocalFileInputError:
    """Build the refusal AND log it at WARNING.

    Logged here, at the single refusal site, so a dropped asset can never again show
    up only as ``-inf LUFS`` in a harness: whoever reads the pod log sees the path
    and the reason, whether the caller re-raises, maps it to a 400, or (nobody does
    any more) degrades.
    """
    log.warning("local_file_input_refused", url=url[:200], reason=reason)
    return LocalFileInputError(f"local-file input {reason}: {url[:80]}")


def _resolve_local_input(url: str, *, local_root: Path | None) -> Path:
    """Resolve a ``file://`` / bare-path asset reference, refusing traversal.

    ``allow_local_file_inputs`` is NOT consulted here, and that is the point. It is
    a *submit-time* policy — its own config description says "Accept file:// and
    bare-path shot/clip inputs **at submit**" — enforced at the API trust boundary
    by ``_guard_local_file_inputs`` (/api/compile, /api/concat-audio) and the prefix
    check (/api/thumbnails). Re-evaluating a caller-input policy down here applied it
    to refs that never came from a caller (an in-process ``compile_shots`` call
    staging its own fixtures/produced files), which silently deleted their audio.

    What IS enforced here is trust-level-independent and therefore safe below the
    boundary:

    1. ``..`` and percent-encoded separators are refused outright, never decoded or
       normalised away. No legitimately-staged path contains them.
    2. If ``local_root`` is given, the resolved path must live INSIDE it. Symlinks
       are resolved FIRST (so a symlink planted in the scratch dir cannot point out),
       then containment is checked by path SEGMENTS via ``Path.parents`` — a string
       ``startswith`` would accept ``/tmp/video-compiler-evil/x`` against a
       ``/tmp/video-compiler`` root.

    ``local_root=None`` means the ref did not cross the API boundary (or was already
    admitted there) and no confinement root applies.
    """
    raw = url[len("file://") :] if url.startswith("file://") else url
    lowered = raw.lower()
    if ".." in raw or any(marker in lowered for marker in _ENCODED_PATH_MARKERS):
        raise _refuse_local_input(url, "rejected (traversal)")
    if local_root is None:
        return Path(raw)
    resolved = Path(raw).resolve()
    if local_root.resolve() not in resolved.parents:
        raise _refuse_local_input(url, "outside the permitted root")
    return resolved


async def _fetch(
    url: str, dest: Path, http: httpx.AsyncClient, *, local_root: Path | None = None
) -> Path:
    """Resolve a shot asset reference to a local file path.

    Accepts http(s) URLs (e.g. presigned MinIO), ``file://`` URIs, and bare
    local paths (used by tests). A remote URL is STREAMED to ``dest`` by
    :func:`w2v_common.net.safe_get_to_path` — same SSRF posture as ``safe_get``
    (per-hop redirect re-validation, IP pinning, ``allow_hosts``) but the body
    never exists in memory as a whole: one chunk at a time goes to a uuid7
    sibling temp file that is ``os.replace``d onto ``dest`` only once the
    transfer completes, so a partial fetch is never visible to the per-source
    dedupe and a failed one leaves ``dest`` untouched.

    That is the whole reason ``max_fetch_bytes`` can be 4 GiB. ``safe_get``
    buffers (chunk list + joined copy) and peaks at ~2x the file, i.e. ~8 GiB of
    pod RSS per concurrent fetch of a 4 GiB drone source — an OOM, not a
    slowdown. The cap (0 = unlimited) is still counted against DECODED bytes
    mid-stream, so an oversized or gzip-bombed asset aborts the transfer instead
    of being measured after the fact.

    ``file://`` and bare paths go through :func:`_resolve_local_input`. Pass
    ``local_root`` when the ref is CALLER-SUPPLIED and the caller has opted into
    local inputs: the path is then confined to that root, so the opt-in buys local
    files, not the pod's filesystem. Omit it for refs that never crossed the API
    boundary (an in-process ``compile_shots`` caller staging its own files) — those
    are still refused if they carry a traversal token, but there is no meaningful
    root to confine them to, and inventing one deletes legitimate assets.
    """
    if url.startswith("file://") or "://" not in url:
        return _resolve_local_input(url, local_root=local_root)
    # ``safe_get_to_path`` does NOT mkdir — it writes its temp file as a sibling of
    # ``dest``, so a missing parent is an OSError mid-fetch rather than a fetch.
    # Callers stage into per-job scratch dirs that normally exist already; making it
    # explicit here keeps a first-fetch-into-a-fresh-dir from degrading a good shot.
    dest.parent.mkdir(parents=True, exist_ok=True)
    # SSRF-safe: per-hop redirect re-validation + IP-pinning (DNS-rebind proof),
    # with our own MinIO hosts allow-listed so presigned internal shot/clip URLs
    # still fetch. `max_fetch_bytes=0` (unlimited) is passed through as-is.
    # Non-2xx raises ``httpx.HTTPStatusError`` from inside the helper, so there is
    # no response object to ``raise_for_status`` here; it is an ``httpx.HTTPError``
    # either way and stays inside ``_BAD_ASSET_ERRORS``.
    await safe_get_to_path(
        http,
        url,
        dest,
        allow_hosts=_minio_allow_hosts(),
        max_fetch_bytes=get_settings().max_fetch_bytes,
    )
    return dest


def _write_atomic(dest: Path, body: bytes) -> None:
    """Write ``body`` to ``dest`` via a temp file + ``os.replace``.

    ``dest.write_bytes`` is NOT atomic: it creates/truncates ``dest`` first and
    fills it afterwards, so between those two moments the path EXISTS and is
    short. That was survivable while every shot fetched into its own
    ``{shot.id}-clip.mp4``, but the per-source dedupe below makes one file the
    shared input of N shots — and the whole point of the dedupe is that a later
    shot uses a file some EARLIER shot fetched. A crash, a cancellation (the
    compile-level ``t.cancel()`` fan-out), a disk-full, or a torn-down work_root
    mid-write would then leave a truncated mp4 that the next shot happily hands
    to ffmpeg. That does not fail loudly: it decodes to a shorter clip, so the
    delivered film is silently missing footage — the exact class of defect
    ``LocalFileInputError`` exists to refuse.

    ``Path.replace`` (``os.replace``) is atomic within a filesystem, and the temp
    name is a sibling (same directory ⇒ same filesystem) so the rename never
    degrades to a copy. The temp name carries a uuid7 so two writers of the same
    ``dest`` cannot collide on the partial file; the loser's rename simply wins or
    loses whole, never half. A failed write unlinks its own partial and leaves
    ``dest`` untouched — an unfetchable asset degrades the shot rather than
    replacing a good earlier fetch with a bad one.
    """
    tmp = dest.with_name(f"{dest.name}.part-{uuid.uuid7().hex}")
    try:
        tmp.write_bytes(body)
        tmp.replace(dest)
    except BaseException:
        # Best-effort cleanup of the partial temp file. This must NOT throw: an
        # OSError from unlink (permission, races, a disk-full condition that also
        # makes the unlink fail) would replace the real failure above — including
        # asyncio.CancelledError, which must propagate unchanged for cooperative
        # cancellation to work — with a misleading cleanup error instead.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


# Everything a shot-asset fetch can fail with that must DEGRADE one shot rather than
# fail the whole paid compile. ``ValueError`` is load-bearing and not redundant with the
# two named subclasses: ``safe_get`` validates the URL BEFORE any I/O and raises a BARE
# ValueError for a malformed one — no hostname (``http://:8080/x.mp4``), an out-of-range
# port (``http://h:99999/x``), an unsupported scheme. ``httpx.InvalidURL`` is likewise
# NOT an ``httpx.HTTPError``. Neither was caught, and the submit-time guard admits such
# URLs (it only checks the http(s) prefix), so one malformed shot URL propagated through
# the shot ``gather`` and failed the entire render. SSRFError / ResponseTooLargeError
# stay named for documentation value — they are ValueError subclasses.
# ``LocalFileInputError`` is deliberately ABSENT (and deliberately not a ValueError):
# a refused local input is a policy/staging defect, not a flaky asset, and degrading
# it ships an mp4 that is silently missing content. See its docstring.
_BAD_ASSET_ERRORS = (
    httpx.HTTPError,
    httpx.InvalidURL,
    OSError,
    SSRFError,
    ResponseTooLargeError,
    ValueError,
)


# Bounded retry for a TRANSIENT shot-asset fetch failure (a blip, not a verdict).
# 2 retries ⇒ 3 attempts total; small linear backoff so a burst of shots hitting
# the same flaky hop don't hammer it in lockstep, jittered for the same reason
# ``compile.py``'s completion-write retry is (two replicas / shots retrying the
# same blip should not re-collide on every attempt). Kept small: this must not
# turn one bad asset into a multi-minute stall on a paid compile.
_TRANSIENT_FETCH_RETRIES = 2
_TRANSIENT_FETCH_BACKOFF_SECONDS = 0.5
# SystemRandom only to keep bandit's B311 quiet — this is retry jitter, not a secret.
_FETCH_JITTER = secrets.SystemRandom()

# HTTP statuses that are a VERDICT, not a blip: the object is gone (404/410), or the
# request is definitively unauthorized (401/403 — an expired or forged presigned MinIO
# signature returns exactly these). Retrying feeds the identical URL to the identical
# check and gets the identical answer, so these fail fast instead of burning the
# compile's time budget on attempts that cannot succeed.
_DEFINITIVE_FETCH_STATUSES = frozenset({401, 403, 404, 410})


def _is_transient_fetch_error(exc: BaseException) -> bool:
    """Whether a shot-asset fetch failure is worth a bounded retry.

    Timeouts, connection resets/refusals, and 5xx are transient — the same
    request against the same URL can plausibly succeed a moment later. A status
    in ``_DEFINITIVE_FETCH_STATUSES`` will not. Every non-HTTP member of
    ``_BAD_ASSET_ERRORS`` (malformed URL, SSRF block, oversized response,
    filesystem ``OSError``) is likewise definitive: retrying hands the exact
    same bytes/hostname to the exact same check.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in _DEFINITIVE_FETCH_STATUSES
    return isinstance(exc, httpx.HTTPError)


async def _try_fetch(
    url: str, dest: Path, http: httpx.AsyncClient, *, local_root: Path | None = None
) -> Path | None:
    """Best-effort ``_fetch``: return ``None`` (logging a warning) instead of
    raising when a shot asset can't be fetched — an expired presigned URL, a
    404/403, or a transport error.

    This lets ``render_shot`` degrade that one shot to its next-best visual (video
    → image → text card) rather than 500-ing the *whole* compile on a single bad
    asset. The orchestrator re-signs MinIO URLs right before compile so expiry
    should be rare, but a genuinely-deleted or transiently-unreachable asset must
    not take the entire video down with it.

    A TRANSIENT failure (timeout, connection reset, 5xx — see
    ``_is_transient_fetch_error``) is retried a bounded number of times with a
    small jittered backoff before this degrades the shot; a definitive one
    (401/403/404/410, or any non-HTTP ``_BAD_ASSET_ERRORS`` member) fails fast on
    the first attempt. This matters most for narration: ``_resolve_shot_assets``
    turns a ``None`` here into a compile-failing ``UnfetchableNarrationError``, so
    treating a passing network blip the same as a truly dead URL used to fail an
    otherwise-good compile outright instead of riding out the blip.

    A ``LocalFileInputError`` is NOT one of those conditions and is not caught here:
    it propagates so a refused/escaping local path fails the compile loudly instead
    of quietly dropping the asset (see ``LocalFileInputError``).
    """
    for attempt in range(_TRANSIENT_FETCH_RETRIES + 1):
        try:
            return await _fetch(url, dest, http, local_root=local_root)
        except _BAD_ASSET_ERRORS as exc:
            transient = _is_transient_fetch_error(exc)
            if transient and attempt < _TRANSIENT_FETCH_RETRIES:
                log.warning(
                    "shot_asset_fetch_retrying",
                    url=url,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                await asyncio.sleep(
                    _TRANSIENT_FETCH_BACKOFF_SECONDS
                    * (attempt + 1)
                    * _FETCH_JITTER.uniform(0.5, 1.5)
                )
                continue
            # Either a definitive failure, or a transient one that exhausted its
            # retries — a bad-asset condition like a 404 or an expired presigned
            # URL. Degrade that one shot, don't 500 the compile.
            log.warning("shot_asset_unfetchable", url=url, error=str(exc), transient=transient)
            return None
    raise AssertionError("unreachable")  # loop always returns on its final iteration


# ---------------------------------------------------------------------------
# Per-source fetch dedupe (one download per URL per compile job)
# ---------------------------------------------------------------------------
#: Directory INSIDE the job's ``work_root`` holding the deduped source downloads.
#: Kept in the job scratch (not a shared pod cache) on purpose — see
#: ``SourceFetchCache``'s docstring for why this must not be process-global.
_SOURCE_DIR_NAME = "sources"


class SourceFetchCache:
    """One download per remote URL per COMPILE JOB.

    WHY
    ---
    The music-cut product cuts many shots from ONE uploaded source: eight shots
    off a single 134 MB 4K file meant eight full downloads of the same bytes,
    because ``_fetch`` is keyed on the per-shot ``dest`` path and nothing above it
    ever asked "did another shot already pull this?". The rendered-shot cache
    (``_SHOT_CACHE_DIRNAME``) does not help — it is keyed on the shot's render
    params, and eight windows of one source are eight distinct keys by
    construction.

    WHY JOB-SCOPED, NOT PROCESS-GLOBAL
    ----------------------------------
    A module-level dict would outlive the job whose ``work_root`` holds the files.
    ``compile_shots`` ``rmtree``s that root in its ``finally``, so a later job
    hitting the same URL would be handed a path that no longer exists — and this
    service is MULTI-REPLICA behind a durable claim queue, so "it worked in my
    single process" proves nothing about the pod that picks up the next job. The
    cache is therefore constructed by ``compile_shots``, threaded down the render
    path, and dies with the job. Cross-job reuse is a different feature (a
    content-addressed disk cache with its own lifecycle) and is deliberately not
    smuggled in here.

    THE RACE
    --------
    Shots render under ``asyncio.gather``, so N coroutines can reach the same URL
    in the same event-loop tick. Two things must hold:

    1. **Exactly one download.** The map is consulted and populated under a single
       ``asyncio.Lock``, and what is stored is the in-flight ``Task``, not the
       result. A later arrival therefore always finds either the finished path or
       the running download — there is no window in which the map says "absent"
       while a download is already going.
    2. **No reader sees a half-written file.** Guaranteed one layer down by
       ``_write_atomic``: the shared path only ever appears complete
       (temp file + ``os.replace``). Without that, a second shot could ``-i`` a
       file the first shot was still filling and silently render short footage.

    Waiters ``await asyncio.shield(task)`` because the download Task is SHARED.
    Cancelling the coroutine that happens to be awaiting an unshielded Task
    cancels the Task itself — so one shot being cancelled (or timing out) would
    yank the download out from under every sibling waiting on the same URL. The
    shield makes cancellation local to the waiter; the job-level teardown
    (:meth:`aclose`) is what actually cancels the downloads.

    FAILURE
    -------
    A failed fetch resolves to ``None`` and that ``None`` is remembered FOR THE
    JOB. Degradation is therefore identical to today (each shot independently
    falls back video → image → text card, and each logs its own
    ``shot_asset_unfetchable``), but a dead URL is attempted once instead of once
    per shot — re-attempting a 404 eight times only burns the compile's time
    budget. It is not permanent: the memo dies with the job, so the next compile
    of the same shot list retries from scratch.

    A fault in the dedupe MACHINERY itself never propagates: :meth:`fetch` falls
    back to an undeduped ``_try_fetch`` into the caller's own ``dest``, which is
    exactly today's behaviour.
    """

    def __init__(self, root: Path) -> None:
        self._root = root / _SOURCE_DIR_NAME
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[Path | None]] = {}

    @staticmethod
    def _key(url: str) -> str:
        """Content-free identity of a source: a hash of the URL itself.

        Hashed rather than used raw because the key names a FILE in the job
        scratch — a presigned URL carries ``/``, ``?``, ``&`` and a signature far
        past any sane filename length. sha256 truncated to 32 hex chars is ample
        for a per-job namespace of at most a few dozen sources.

        The WHOLE url is hashed, query string included. A presigned MinIO URL's
        signature is part of its identity: two links to the same object with
        different expiries are different bytes on the wire only by accident, but
        stripping the query to "improve" hit rate would collide two genuinely
        different objects served by the same path under different buckets/hosts.
        Over-keying costs a duplicate download; under-keying renders the wrong
        footage.
        """
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

    async def fetch(
        self,
        url: str,
        dest: Path,
        http: httpx.AsyncClient,
        *,
        local_root: Path | None = None,
    ) -> Path | None:
        """Deduped ``_try_fetch``. Returns the SHARED local path, or ``None``.

        Only remote (http/https) refs are deduped. ``file://`` and bare paths are
        passed straight through: they are resolved, not downloaded — there is
        nothing to save — and their ``local_root`` confinement is a per-CALL trust
        decision that must not be memoised across callers with different roots.

        The returned path is shared between shots. Every consumer treats a fetched
        asset as read-only input to an ffmpeg ``-i``, so sharing is safe; nothing
        in the render path mutates or deletes a fetched source.
        """
        if url.startswith("file://") or "://" not in url:
            return await _try_fetch(url, dest, http, local_root=local_root)
        try:
            key = self._key(url)
            async with self._lock:
                task = self._tasks.get(key)
                if task is None:
                    self._root.mkdir(parents=True, exist_ok=True)
                    shared = self._root / f"src-{key}{dest.suffix}"
                    task = asyncio.ensure_future(
                        _try_fetch(url, shared, http, local_root=local_root)
                    )
                    self._tasks[key] = task
        except Exception:  # dedupe is an optimisation, never a gate
            log.warning("source_cache.unavailable", url=url, exc_info=True)
            return await _try_fetch(url, dest, http, local_root=local_root)
        # shield: the Task is shared, so an outer cancellation must not take the
        # download away from the siblings still waiting on it.
        return await asyncio.shield(task)

    async def aclose(self) -> None:
        """Cancel and drain any still-running download.

        Called from ``compile_shots``' teardown BEFORE the ``work_root`` is
        removed. Without it, a shielded download survives the render fan-out's
        ``t.cancel()`` and keeps writing into a directory that is being deleted —
        and its exception is never retrieved, so it surfaces as loop noise rather
        than as anything actionable.
        """
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _wrap_for_canvas(text: str, fontsize: int, canvas_width: int) -> str:
    """Pre-wrap ``text`` to fit ``canvas_width`` at ``fontsize`` (px).

    drawtext renders each `\n`-separated line at its own baseline but does NO
    width-based wrapping itself. Without this helper, a 50-character script
    line at fontsize=64 on a 1920px canvas computes to ~1760px text width and
    `x=(w-text_w)/2` then centers it so the line extends symmetrically off
    BOTH safe-area edges and gets cropped — exactly the "words flow off the
    screen" defect reported on the first promo runs (2026-05-31).

    Returns the input collapsed-and-wrapped, joined by literal `\n`s so the
    drawtext filter's textfile= path renders each chunk as a separate line.
    """
    safe_width = max(200, canvas_width - 2 * _safe_margin_px(canvas_width))
    max_chars = max(8, int(safe_width / (fontsize * _AVG_GLYPH_WIDTH_RATIO)))
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return ""
    # break_long_words=False keeps URLs / hyphenated tokens (e.g.
    # "w2v.nonagentic.ai" in the user's verbatim CTA) intact across wrap
    # boundaries even when they slightly exceed max_chars.
    wrapped = textwrap.wrap(collapsed, width=max_chars, break_long_words=False)
    return "\n".join(wrapped) if wrapped else collapsed


# drawtext's own ``line_spacing=`` option for the caption fragment — pulled out
# to a name so the text-card body's collision-avoidance math (below) uses the
# EXACT same gap the caption filter actually renders with, instead of a second
# hardcoded 10 that could silently drift from it.
_CAPTION_LINE_SPACING_PX = 10


def _text_block_height_px(text: str, fontsize: int, line_spacing: int) -> int:
    """Approximate rendered pixel height of a ``\\n``-joined drawtext block: N
    lines at ``fontsize`` px each plus N-1 gaps of ``line_spacing`` px between
    them (drawtext's own ``line_spacing=`` option semantics).

    This is Python-side geometry, not a real font metric (drawtext itself
    computes ``text_h`` from the actual glyph outlines, which routinely
    exceeds the nominal ``fontsize`` — see ``_DRAWTEXT_LINE_HEIGHT_FACTOR``)
    — it exists so two INDEPENDENT drawtext filters in the same chain (the
    text-card body + the shot's caption) can each reserve a non-overlapping
    vertical band, which neither filter can do on its own: drawtext has no
    concept of a sibling filter's footprint. ``0`` for empty text.
    """
    if not text:
        return 0
    lines = text.count("\n") + 1
    line_height = round(fontsize * _DRAWTEXT_LINE_HEIGHT_FACTOR)
    return lines * line_height + max(0, lines - 1) * line_spacing


def _caption_geometry(caption: str, settings: Settings) -> tuple[str, int, int]:
    """Wrapped caption text + its fontsize + bottom margin — the exact values
    ``_caption_filter_fragment`` renders with, factored out so a sibling filter
    (the text-card body) can reserve space above the caption without
    duplicating (and risking drift from) the font/margin derivation."""
    font_size = round(settings.render_height * settings.caption_drawtext_font_size_frac)
    margin_v = round(settings.render_height * settings.caption_drawtext_margin_v_frac)
    wrapped = _wrap_for_canvas(caption, fontsize=font_size, canvas_width=settings.render_width)
    return wrapped, font_size, margin_v


def _caption_top_px(caption: str | None, settings: Settings) -> int:
    """The y pixel ABOVE which the bottom-of-frame caption occupies the frame —
    ``render_height`` (i.e. "no reservation needed") when there is no caption.

    Used by the text-card branch to keep its body centered above the caption's
    footprint instead of across the whole frame, since two independent
    drawtext filters in one chain have no built-in awareness of each other's
    bounding box (see ``_text_block_height_px``)."""
    if not caption:
        return settings.render_height
    wrapped, font_size, margin_v = _caption_geometry(caption, settings)
    block_h = _text_block_height_px(wrapped, font_size, _CAPTION_LINE_SPACING_PX)
    return settings.render_height - block_h - margin_v


def _caption_filter_fragment(caption: str, settings: Settings, work_dir: Path, shot_id: str) -> str:
    """Build the ``drawtext`` fragment for the bottom-of-frame caption overlay.

    Caption uses a smaller fontsize than the text-card body and writes via
    ``textfile=`` (same reason as the text-card branch: textfile sidesteps
    drawtext's special-char escape minefield). y is computed from text_h so
    the multi-line caption sits above the lower safe margin.

    Font size and bottom margin scale with ``settings.render_height`` (same
    pattern the film-level ASS caption system uses in ``captions_ass.py`` —
    ``round(render_height * frac)``), via their own
    ``caption_drawtext_font_size_frac`` / ``caption_drawtext_margin_v_frac``
    settings. Those defaults (44/1080, 60/1080) make this a byte-identical
    no-op at the default 1080p output: 1080 * 44/1080 = 44, 1080 * 60/1080 = 60.
    """
    wrapped, font_size, margin_v = _caption_geometry(caption, settings)
    cap_path = work_dir / f"{shot_id}-caption.txt"
    cap_path.write_text(wrapped, encoding="utf-8")
    return (
        f"drawtext=fontfile={settings.render_font}:textfile={cap_path}"
        f":fontcolor={settings.render_text_color}:fontsize={font_size}"
        f":line_spacing={_CAPTION_LINE_SPACING_PX}"
        f":box=1:boxcolor=black@0.55:boxborderw=18"
        f":x=(w-text_w)/2:y=h-text_h-{margin_v}"
    )


def _text_card_body_y(
    caption: str | None, body_text: str, body_fontsize: int, settings: Settings
) -> str:
    """The text-card body drawtext's ``y=`` value.

    Absent a caption this is UNCHANGED — ffmpeg's own dynamic
    ``(h-text_h)/2``, centering across the whole frame — so the no-caption
    case stays byte-identical.

    With a caption, the body defaults to centering across the WHOLE frame
    while the caption is an INDEPENDENT drawtext filter anchored to the
    bottom safe margin with no awareness of the body's footprint (or vice
    versa) — a long wrapped body plus a multi-line caption can visually
    overlap. Reserve the caption's footprint (the exact geometry
    ``_caption_filter_fragment`` renders, via ``_caption_top_px``) and center
    the body inside the remaining band above it instead — a fixed numeric
    value (a Python-side estimate, see ``_text_block_height_px``) rather than
    ffmpeg's own dynamic ``text_h``, since the two filters can't reference
    each other's metrics inside the graph.

    ⚠️ When the body's own footprint EXCEEDS the reserved band
    (``body_block_h > caption_top``), centering still has nowhere to put the
    body without touching the caption band: ``max(0, ...)`` alone would clamp
    to the frame's top edge and render as a silent overlap, with no signal
    that collision-avoidance failed. Instead this pins the body flush to the
    frame top (y=0, the least-bad anchor — it maximises the gap to the
    caption band rather than centering into it) and logs a warning so the
    degradation is visible in the run's logs rather than only in the pixels.
    This does not shrink or truncate the body text; a true fix needs the
    caller to reduce ``body_fontsize`` or the source text, which this
    y-only helper cannot do.
    """
    if not caption:
        return "(h-text_h)/2"
    caption_top = _caption_top_px(caption, settings)
    body_block_h = _text_block_height_px(body_text, body_fontsize, _TEXT_CARD_BODY_LINE_SPACING_PX)
    if body_block_h > caption_top:
        log.warning(
            "text_card_body_exceeds_reserved_band",
            body_block_h=body_block_h,
            caption_top=caption_top,
        )
        return "0"
    return str((caption_top - body_block_h) // 2)


# ---------------------------------------------------------------------------
# Per-shot look: colour grade (eq) + burned-in text overlays (drawtext)
# ---------------------------------------------------------------------------
# The ffmpeg ``eq`` filter's per-component identity values. A component equal to
# its default contributes nothing, so we emit only the ones that actually move —
# and an all-default grade with no preset emits no ``eq`` at all (byte-identical
# no-op). ``gamma_r``/``gamma_b`` are per-channel gammas used by the warm/cool
# presets (their identity is also 1.0).
_EQ_DEFAULTS: dict[str, float] = {
    "brightness": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "gamma": 1.0,
    "gamma_r": 1.0,
    "gamma_b": 1.0,
}
# Deterministic emit order for the eq params.
_EQ_PARAM_ORDER: tuple[str, ...] = (
    "brightness",
    "contrast",
    "saturation",
    "gamma",
    "gamma_r",
    "gamma_b",
)
# Named presets → fixed eq component overrides (each value is non-default by
# construction). Explicit non-default ColorAdjust fields override these.
_COLOR_PRESETS: dict[str, dict[str, float]] = {
    "bw": {"saturation": 0.0},
    "vivid": {"saturation": 1.4, "contrast": 1.1},
    "warm": {"gamma_r": 1.06, "gamma_b": 0.94},
    "cool": {"gamma_b": 1.06, "gamma_r": 0.94},
    "cinematic": {"contrast": 1.1, "saturation": 0.9, "gamma": 0.95},
}

# Safe-area gutter for anchored overlays (matches the caption fragment's y=…-60).
_OVERLAY_MARGIN_PX = 60


def _eq_filter(adjust: ColorAdjust | None) -> str:
    """Build the ``eq=`` colour-grade fragment for a shot, or ``""`` for a no-op.

    Returns ``""`` when ``adjust`` is None, OR when every component equals its
    default AND no preset is set — an all-default eq is a real (identity) filter,
    not a no-op, so we emit NOTHING to keep the default render byte-identical.
    Otherwise a ``preset`` supplies a base grade and any explicitly-set
    (non-default) numeric field overrides the matching component; only the
    components that differ from the ffmpeg ``eq`` identity are emitted, ``:g``
    float-formatted (so ``1.4`` stays ``1.4`` and ``2.0`` collapses to ``2``).
    """
    if adjust is None:
        return ""
    explicit = {
        "brightness": adjust.brightness,
        "contrast": adjust.contrast,
        "saturation": adjust.saturation,
        "gamma": adjust.gamma,
    }
    if adjust.preset is None and all(explicit[k] == _EQ_DEFAULTS[k] for k in explicit):
        return ""
    values: dict[str, float] = {}
    if adjust.preset is not None:
        values.update(_COLOR_PRESETS.get(adjust.preset, {}))
    for k, v in explicit.items():
        if v != _EQ_DEFAULTS[k]:
            values[k] = v
    emitted = [
        f"{k}={values[k]:g}"
        for k in _EQ_PARAM_ORDER
        if k in values and values[k] != _EQ_DEFAULTS[k]
    ]
    return "eq=" + ":".join(emitted) if emitted else ""


def _overlay_position_xy(position: str) -> tuple[str, str]:
    """Map a nine-grid ``position`` to ffmpeg ``x``/``y`` drawtext expressions."""
    vert, horiz = position.split("-")
    if horiz == "left":
        x = str(_OVERLAY_MARGIN_PX)
    elif horiz == "right":
        x = f"w-text_w-{_OVERLAY_MARGIN_PX}"
    else:  # center
        x = "(w-text_w)/2"
    if vert == "top":
        y = str(_OVERLAY_MARGIN_PX)
    elif vert == "bottom":
        y = f"h-text_h-{_OVERLAY_MARGIN_PX}"
    else:  # middle
        y = "(h-text_h)/2"
    return x, y


def _overlay_enable_expr(start: float | None, end: float | None) -> str:
    """Build the ``:enable='…'`` timing clause for an overlay, or ``""``.

    Commas/colons inside the expression are backslash-escaped SEPARATELY from
    ``_escape_drawtext`` (they are filtergraph metacharacters, not text): a
    windowed overlay uses ``between(t\\,{start}\\,{end})``, a start-only overlay
    ``gte(t\\,{start})`` and an end-only overlay ``lte(t\\,{end})``.
    """
    if start is None and end is None:
        return ""
    if start is not None and end is not None:
        expr = f"between(t\\,{start:g}\\,{end:g})"
    elif start is not None:
        expr = f"gte(t\\,{start:g})"
    else:
        expr = f"lte(t\\,{end:g})"
    return f":enable='{expr}'"


# An ffmpeg colour token: a name (``white``), a ``#RRGGBB``/``0xRRGGBBAA`` hex,
# optionally with an ``@alpha`` suffix. ``fontcolor`` is inserted INLINE into the
# filtergraph (unlike the text, which goes through ``textfile=``), so a value
# carrying a ``:``/``;``/``,`` would break out of the drawtext options — anything
# not matching this shape falls back to the safe render default.
_SAFE_COLOR_RE = re.compile(r"\A#?[0-9A-Za-z]+(@[0-9.]+)?\Z")


def _safe_drawtext_color(color: str, settings: Settings) -> str:
    """Return ``color`` if it is a safe ffmpeg colour token, else the default."""
    return color if _SAFE_COLOR_RE.match(color) else settings.render_text_color


def _overlay_chain(
    overlays: list[TextOverlay], settings: Settings, work_dir: Path, shot_id: str
) -> str:
    """Build the comma-joined ``drawtext`` chain for a shot's text overlays.

    ``""`` for an empty list (no-op). Each overlay's text is written to a file and
    read back via ``textfile=`` — the SAME injection-proof path the caption
    fragment uses: drawtext does no escape interpretation on file contents, so no
    metacharacter in tenant-controlled text (``:``, ``;``, ``,``, quotes, ``\\``)
    can break out of the filtergraph (see the ``is_text`` note below on why the
    old ``text=`` + escape path was a losing cat-and-mouse). ``fontcolor`` is
    inline so it is validated by ``_safe_drawtext_color``; ``font_size``/``x``/``y``
    are ints and ``start``/``end`` are ``:g``-formatted floats, all injection-safe.
    Position maps to x/y expressions (explicit int ``x``/``y`` override the
    anchor); an optional start/end window becomes an ``enable=`` clause.
    """
    if not overlays:
        return ""
    frags: list[str] = []
    for i, ov in enumerate(overlays):
        text = re.sub(r"\s+", " ", ov.text).strip()
        ov_path = work_dir / f"{shot_id}-overlay-{i}.txt"
        ov_path.write_text(text, encoding="utf-8")
        anchor_x, anchor_y = _overlay_position_xy(ov.position)
        x = str(ov.x) if ov.x is not None else anchor_x
        y = str(ov.y) if ov.y is not None else anchor_y
        frags.append(
            f"drawtext=fontfile={settings.render_font}:textfile={ov_path}"
            f":fontcolor={_safe_drawtext_color(ov.color, settings)}:fontsize={ov.font_size}"
            f":x={x}:y={y}{_overlay_enable_expr(ov.start, ov.end)}"
        )
    return ",".join(frags)


def _apply_shot_look(video_vf: str, shot: Shot, settings: Settings, work_dir: Path) -> str:
    """Append a shot's colour grade (``eq``) then text overlays (``drawtext``) to
    its video filter body.

    Both sit AFTER scale/pad/format so the grade acts on the final framed picture
    and titles land in canvas coordinates. Returns ``video_vf`` UNCHANGED (no
    trailing comma) when the shot has neither a grade nor overlays — the
    byte-identical no-op invariant the default render depends on.
    """
    eq = _eq_filter(shot.adjust)
    if eq:
        video_vf = f"{video_vf},{eq}"
    overlays = _overlay_chain(shot.overlays, settings, work_dir, shot.id)
    if overlays:
        video_vf = f"{video_vf},{overlays}"
    return video_vf


# ---------------------------------------------------------------------------
# Directed footage (ORCH-3.2): synthetic cursor + auto-zoom-to-the-control.
# ---------------------------------------------------------------------------
# Both effects are opt-in per shot (``focus_track`` / ``cursor_track``) and
# pass-through when their track is falsy, so an absent track leaves the filter
# string byte-identical to before these fields existed. Every expression is
# single-quoted AND has its internal commas backslash-escaped (``\\,``) — the
# same belt-and-suspenders escaping the shipped ``_scroll_filter`` uses — so the
# string is valid whether it lands in a ``-vf`` or a ``-filter_complex`` seam.
#
# The auto-zoom is a ``zoompan`` (not ``crop``) because the zoom factor varies
# per frame (ease-in, hold, ease-out). The cursor is SYNTHESIZED — a brand-tinted
# filled disc built from a ``color``+``format=rgba``+``geq`` alpha-disc labeled
# source and overlaid; no binary PNG asset is embedded or required. Because the
# cursor needs labeled sub-graphs it emits a multi-chain graph string (contains
# ``;``); ``_video_audio_map_args`` detects that and routes the shot through the
# existing ``-filter_complex`` seam instead of the bare ``-vf`` text-card path.
_ZOOM_FILL = 0.60  # fraction of the frame the focus bbox should fill post-zoom
_CAPTURE_HEADROOM = 2.0  # max punch-in — the capture_scale headroom the producer allows
_FOCUS_EASE_SECONDS = 0.5  # ease-in / ease-out ramp of the punch-in
_DRIFT_ZOOM = (
    1.05  # gentle Ken-Burns zoom over a freeze-padded tail so it never dead-freezes (T7 / AC#5)
)
_CURSOR_DIAMETER_FRAC = 0.03  # cursor disc diameter as a fraction of frame height
_CURSOR_MIN_PX = 12
_CURSOR_COLOR = "0xF5F5F7"  # brand-tinted cursor fill (near-white)
_CURSOR_RING_COLOR = "0x3B82F6"  # click-pulse ring accent
_CLICK_RING_SCALE = 2.6  # click ring diameter vs the cursor disc
_CLICK_RING_THICKNESS = 0.16  # ring wall thickness as a fraction of its diameter
_CLICK_RING_ALPHA = 190  # ring opacity (0..255)
_CLICK_PULSE_LEAD = 0.05  # ring appears this many seconds before the click
_CLICK_PULSE_TAIL = 0.20  # ...and lingers this long after
# Footage callouts: a numbered badge disc (same synthesized-sprite trick as the
# cursor) glued to a clicked element's top-left corner, and a short lower-third
# label caption. Both ride the PRE-zoom canvas so a focus punch-in magnifies them
# with the element they annotate. Absent ``callouts`` is byte-identical.
_BADGE_DIAMETER_FRAC = 0.05  # badge disc diameter as a fraction of frame height
_BADGE_MIN_PX = 28
_BADGE_COLOR = "0x2563EB"  # brand accent — the numbered-badge disc fill
_BADGE_ALPHA = 235  # near-solid disc (a hair of tint over the element)
_BADGE_TEXT_COLOR = "white"  # badge number/short-label
_BADGE_FONT_FRAC = 0.5  # badge number fontsize as a fraction of disc diameter
_CALLOUT_LABEL_DWELL = 2.0  # max seconds a label caption dwells from its onset
# Callout-finish (AC3): a hard per-shot cap, a bbox-anchored label caption, plus
# two new sprite/box callout types (spotlight dim, arrow). All memory-safe on the
# 2Gi pod — no full-frame geq/soft-mask (the confirmed OOM class): spotlight is
# four bbox-complement drawbox bands, arrow is a tiny synthesized triangle sprite.
_CALLOUT_MAX_PER_SHOT = 4  # hard cap on callouts composited per shot
_CALLOUT_ANCHOR_GAP_PX = 24  # gap between the bbox edge and its anchored label caption
_SPOTLIGHT_DIM_ALPHA = 0.55  # black dim alpha over the bbox-complement bands
_ARROW_COLOR = "0x2563EB"  # arrow sprite fill (reuses the badge accent)
_ARROW_DIAMETER_FRAC = 0.06  # arrow sprite size as a fraction of frame height
_ARROW_GAP_PX = 12  # gap between the arrow apex and the bbox edge
_ARROW_MIN_PX = 24  # minimum arrow sprite size


def _smoothstep_expr(var: str, t0: float, t1: float) -> str:
    """A SMOOTHSTEP-eased 0→1 ramp of ``var`` over ``[t0, t1]`` as an ffmpeg
    expression string (Hermite ``p*p*(3-2p)`` on the clamped, normalized input).

    Returns the constant ``"1"`` for a degenerate/zero window. Commas are
    backslash-escaped so the fragment is safe inside a single-quoted filter
    option in either a ``-vf`` or ``-filter_complex`` graph.
    """
    if t1 <= t0:
        return "1"
    p = f"clip(({var}-{_ffnum(t0)})/{_ffnum(t1 - t0)}\\,0\\,1)"
    return f"({p})*({p})*(3-2*({p}))"


def _remap_tracks_to_canvas(
    focus_track: list[FocusKey] | None,
    cursor_track: list[CursorKey] | None,
    callouts: list[Callout] | None,
    footage_viewport: tuple[int, int] | None,
    settings: Settings,
    fit: str | None = None,
) -> tuple[list[FocusKey] | None, list[CursorKey] | None, list[Callout] | None]:
    """Remap focus/cursor/callout coordinates from recorded-source pixel space
    into the render-canvas pixel space, using the SAME fit transform
    ``_scale_pad_filter`` applies to the picture — so cursor/zoom/callouts keep
    tracking the picture at any tier.

    ``footage_viewport is None`` → every track is returned UNCHANGED (the exact
    same objects, no copy), so the legacy/identity path is provably byte-identical
    to the render before this field existed. Likewise when the recorded viewport
    already equals the render canvas: the transform is the identity, so the
    originals pass straight through.

    Otherwise the scale factor MATCHES ``_scale_pad_filter``'s branch for this
    ``fit``: ``fit="cover"`` fills the canvas (``force_original_aspect_ratio=
    increase`` then centered ``crop``) so ``s = max(W/vw, H/vh)``; every other fit
    (``contain``/``scroll``/default) letterboxes (``decrease`` + centered ``pad``)
    so ``s = min(W/vw, H/vh)``. Both share one centered-offset formula
    ``off = (canvas - v*s)/2`` (a positive pad for contain, a negative crop offset
    for cover), so ``x' = x*s + off_x`` / ``y' = y*s + off_y`` and focus/callout
    bbox dims scale ``w' = w*s`` / ``h' = h*s``. Coordinates stay floats.
    """
    if footage_viewport is None:
        return focus_track, cursor_track, callouts
    w, h = settings.render_width, settings.render_height
    vw, vh = footage_viewport
    if (vw, vh) == (w, h):
        return focus_track, cursor_track, callouts
    # cover fills the frame (scale up, crop) → s=max; contain/scroll letterbox
    # (scale down, pad) → s=min. The centered-offset formula is identical for both.
    s = max(w / vw, h / vh) if fit == "cover" else min(w / vw, h / vh)
    pad_x = (w - vw * s) / 2.0
    pad_y = (h - vh * s) / 2.0
    new_focus = (
        None
        if focus_track is None
        else [
            FocusKey(
                t=k.t,
                bbox=(
                    k.bbox[0] * s + pad_x,
                    k.bbox[1] * s + pad_y,
                    k.bbox[2] * s,
                    k.bbox[3] * s,
                ),
            )
            for k in focus_track
        ]
    )
    new_cursor = (
        None
        if cursor_track is None
        else [
            CursorKey(t=k.t, x=k.x * s + pad_x, y=k.y * s + pad_y, click=k.click)
            for k in cursor_track
        ]
    )
    new_callouts = (
        None
        if callouts is None
        else [
            Callout(
                at=c.at,
                type=c.type,
                bbox=(
                    c.bbox[0] * s + pad_x,
                    c.bbox[1] * s + pad_y,
                    c.bbox[2] * s,
                    c.bbox[3] * s,
                ),
                label=c.label,
            )
            for c in callouts
        ]
    )
    return new_focus, new_cursor, new_callouts


def _focus_zoom_target(bbox: tuple[float, float, float, float], w: int, h: int) -> float:
    """The punch-in zoom that makes ``bbox`` fill ``_ZOOM_FILL`` of the ``wxh``
    frame, clamped to ``[1.0, _CAPTURE_HEADROOM]``.

    ``z = clamp(sqrt(ZOOM_FILL * W*H / (bw*bh)), 1.0, CAPTURE_HEADROOM)`` — a
    small bbox needs a big zoom to fill the frame; a bbox already larger than the
    fill target clamps to 1.0 (no zoom). Degenerate (<=0) bbox dims floor to 1 px
    so the sqrt is always finite.
    """
    _, _, bw, bh = bbox
    bw = max(bw, 1.0)
    bh = max(bh, 1.0)
    z = math.sqrt(_ZOOM_FILL * w * h / (bw * bh))
    return max(1.0, min(_CAPTURE_HEADROOM, z))


def _focus_zoom_filter(video_vf: str, shot: Shot, settings: Settings, duration: float) -> str:
    """Append an eased ``zoompan`` punch-in centered on the shot's focus bbox.

    ``focus_track`` falsy → ``video_vf`` UNCHANGED (the no-op invariant). With a
    SINGLE keyframe its bbox sets the zoom target + centre and the punch eases in
    over ~``_FOCUS_EASE_SECONDS``, holds, then eases back out. With N>=2 keyframes
    the punch CENTRE and per-key zoom TARGET are smoothstep-interpolated across the
    keyframes over the ``on`` (output-frame) domain, so the focus follows every
    click. A track whose keys ALL already fill the frame (max target zoom == 1.0)
    is a pass-through. The crop window (``iw/zoom`` x ``ih/zoom``) tracks the bbox
    centre and is clamped inside the frame so the punch never reveals padding.

    Runs AFTER ``_cursor_filter`` in the chain. When a cursor sub-graph is already
    present (``video_vf`` contains ``;``) the zoompan chains on as a labeled
    ``filter_complex`` continuation (``…overlay[fz];[fz]zoompan=…``) so it
    magnifies the cursor along with the frame; otherwise it appends as a plain
    ``-vf`` step. The output carries no label so the caller's ``[0:v]…[v]`` seam
    still closes the graph.
    """
    track = shot.focus_track
    if not track:
        return video_vf
    w, h = settings.render_width, settings.render_height
    fps = settings.render_fps
    keys = sorted(track, key=lambda k: k.t)
    targets = [_focus_zoom_target(k.bbox, w, h) for k in keys]
    if max(targets) <= 1.0:
        return video_vf
    total = max(duration * fps, 1.0)
    ease = max(min(_FOCUS_EASE_SECONDS * fps, total / 2.0), 1.0)
    zin = _smoothstep_expr("on", 0.0, ease)
    zout = _smoothstep_expr("on", total - ease, total)
    if len(keys) == 1:
        # Single keyframe: byte-identical to the original whole-shot punch-in.
        key = keys[0]
        bx, by, bw, bh = key.bbox
        zt = targets[0]
        # Scaled into the PRE-SCALED coordinate space (see _ZOOMPAN_PRESCALE):
        # the bbox centre is in render-canvas pixels, and zoompan now receives a
        # frame _ZOOMPAN_PRESCALE times that size, so an unscaled centre would
        # punch at half the intended position.
        cx = (bx + bw / 2.0) * _ZOOMPAN_PRESCALE
        cy = (by + bh / 2.0) * _ZOOMPAN_PRESCALE
        z_expr = f"1+{_ffnum(zt - 1.0)}*min({zin}\\,1-{zout})"
        x_expr = f"max(0\\,min(iw-iw/zoom\\,{_ffnum(cx)}-iw/zoom/2))"
        y_expr = f"max(0\\,min(ih-ih/zoom\\,{_ffnum(cy)}-ih/zoom/2))"
    else:
        # N>=2: interpolate the zoom target AND punch centre across keyframes over
        # the zoompan ``on`` (output-frame) domain so the focus follows every click.
        ons = [k.t * fps for k in keys]
        # Same _ZOOMPAN_PRESCALE coordinate-space scaling as the single-key branch.
        cxs = [(k.bbox[0] + k.bbox[2] / 2.0) * _ZOOMPAN_PRESCALE for k in keys]
        cys = [(k.bbox[1] + k.bbox[3] / 2.0) * _ZOOMPAN_PRESCALE for k in keys]
        z_target = _focus_axis_expr(ons, targets)
        z_expr = f"1+({z_target}-1.0)*min({zin}\\,1-{zout})"
        x_expr = f"max(0\\,min(iw-iw/zoom\\,({_focus_axis_expr(ons, cxs)})-iw/zoom/2))"
        y_expr = f"max(0\\,min(ih-ih/zoom\\,({_focus_axis_expr(ons, cys)})-ih/zoom/2))"
    zoompan = (
        f"{_zoompan_prescale(w, h, settings)},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d=1:s={w}x{h}:fps={_ffnum(fps)}"
    )
    if ";" in video_vf:
        # A cursor sub-graph is already in play — chain the zoompan on as a
        # labeled filter_complex continuation so it magnifies the composited
        # cursor together with the frame.
        return f"{video_vf}[fz];[fz]{zoompan}"
    return f"{video_vf},{zoompan}"


def _cursor_axis_expr(track: list[CursorKey], axis: str, var: str = "t") -> str:
    """Nested-``if()`` smoothstep interpolation of one cursor axis over its
    keyframes, evaluated against the overlay per-frame time ``var`` (``t``).

    Before the first / after the last keyframe the value HOLDS (a smoothstep
    clamped to [0,1] returns the endpoint), so no explicit clamp branch is
    needed. A single keyframe degenerates to a constant. Commas are escaped for
    the single-quoted overlay option.
    """
    keys = sorted(track, key=lambda k: k.t)
    vals = [float(getattr(k, axis)) for k in keys]
    ts = [k.t for k in keys]
    expr = _ffnum(vals[-1])
    for i in range(len(keys) - 2, -1, -1):
        s = _smoothstep_expr(var, ts[i], ts[i + 1])
        seg = f"({_ffnum(vals[i])})+({_ffnum(vals[i + 1] - vals[i])})*({s})"
        expr = f"if(lt({var}\\,{_ffnum(ts[i + 1])})\\,{seg}\\,{expr})"
    return expr


def _focus_axis_expr(ons: list[float], vals: list[float]) -> str:
    """Nested-if() smoothstep interpolation of one focus quantity over its
    keyframes in the zoompan ``on`` (output-frame) domain. Mirrors
    ``_cursor_axis_expr`` but evaluates against ``on`` and takes precomputed
    parallel ``ons``/``vals`` lists (so it serves centre AND per-key zoom
    target). Before the first / after the last key the value HOLDS (smoothstep
    self-clamps). A single element degenerates to a constant."""
    expr = _ffnum(vals[-1])
    for i in range(len(vals) - 2, -1, -1):
        s = _smoothstep_expr("on", ons[i], ons[i + 1])
        seg = f"({_ffnum(vals[i])})+({_ffnum(vals[i + 1] - vals[i])})*({s})"
        expr = f"if(lt(on\\,{_ffnum(ons[i + 1])})\\,{seg}\\,{expr})"
    return expr


def _alpha_disc_source(
    diameter: int, color: str, alpha_expr: str, duration: float, fps: float
) -> str:
    """A labeled ``color``+``format=rgba``+``geq`` source: a solid ``color`` sprite
    whose alpha channel is ``alpha_expr`` (a disc or a ring). RGB pass through; the
    shape lives entirely in the alpha. No binary asset — the sprite is synthesized.
    """
    return (
        f"color=c={color}@1:s={diameter}x{diameter}:d={_ffnum(duration)}:r={_ffnum(fps)},"
        f"format=rgba,"
        f"geq=r='r(X\\,Y)':g='g(X\\,Y)':b='b(X\\,Y)':a='{alpha_expr}'"
    )


def _cursor_disc_source(diameter: int, duration: float, fps: float) -> str:
    """The filled cursor disc sprite sub-graph."""
    r = diameter / 2.0
    dist = f"(X-{_ffnum(r)})*(X-{_ffnum(r)})+(Y-{_ffnum(r)})*(Y-{_ffnum(r)})"
    alpha = f"if(lte({dist}\\,{_ffnum(r * r)})\\,255\\,0)"
    return _alpha_disc_source(diameter, _CURSOR_COLOR, alpha, duration, fps)


def _cursor_ring_source(diameter: int, duration: float, fps: float) -> str:
    """The hollow click-pulse ring sprite sub-graph (an annulus in the alpha)."""
    r = diameter / 2.0
    ri = max(1.0, r - max(2.0, diameter * _CLICK_RING_THICKNESS))
    dist = f"(X-{_ffnum(r)})*(X-{_ffnum(r)})+(Y-{_ffnum(r)})*(Y-{_ffnum(r)})"
    alpha = f"if(between({dist}\\,{_ffnum(ri * ri)}\\,{_ffnum(r * r)})\\,{_CLICK_RING_ALPHA}\\,0)"
    return _alpha_disc_source(diameter, _CURSOR_RING_COLOR, alpha, duration, fps)


def _badge_disc_source(diameter: int, duration: float, fps: float) -> str:
    """The tinted numbered-badge disc sprite sub-graph (a filled disc in the alpha
    at ``_BADGE_COLOR``/``_BADGE_ALPHA``) — the cursor-sprite trick reused for the
    callout badge; the number is drawn on top by ``_callout_filter``."""
    r = diameter / 2.0
    dist = f"(X-{_ffnum(r)})*(X-{_ffnum(r)})+(Y-{_ffnum(r)})*(Y-{_ffnum(r)})"
    alpha = f"if(lte({dist}\\,{_ffnum(r * r)})\\,{_BADGE_ALPHA}\\,0)"
    return _alpha_disc_source(diameter, _BADGE_COLOR, alpha, duration, fps)


def _arrow_sprite_source(size: int, direction: str, duration: float, fps: float) -> str:
    """A small filled-triangle arrow sprite synthesized in the alpha channel via
    the ``_alpha_disc_source`` trick — memory-safe because the sprite is
    ``size``-sized, NOT full-frame (no whole-frame geq).

    ``direction="right"`` points the apex to the RIGHT (base on the left edge), so
    it is placed to the LEFT of the bbox and points at it; ``"left"`` mirrors it
    (apex on the left, base on the right edge) for placement to the RIGHT of the
    bbox. Right-pointing alpha=255 where ``X <= size*(1 - |Y-size/2|/(size/2))`` —
    the fill width is ``size`` at mid-height and tapers to 0 at the top/bottom
    tips, so the apex lands on the pointing (right) side. Left mirrors X.
    """
    half = size / 2.0
    # available fill width at this row: full at mid-height, 0 at the top/bottom tips
    edge = f"({_ffnum(size)}*(1-abs(Y-{_ffnum(half)})/{_ffnum(half)}))"
    # right: base at X=0, apex at (size, size/2) → points +X; left: mirror X → apex at (0, size/2)
    inside = f"lte(X\\,{edge})" if direction == "right" else f"gte(X\\,{_ffnum(size)}-{edge})"
    alpha = f"if({inside}\\,255\\,0)"
    return _alpha_disc_source(size, _ARROW_COLOR, alpha, duration, fps)


def _cursor_filter(video_vf: str, shot: Shot, settings: Settings, duration: float) -> str:
    """Overlay a synthesized eased cursor (+ click-pulse rings) onto ``video_vf``.

    ``cursor_track`` falsy → ``video_vf`` UNCHANGED. Else returns a MULTI-CHAIN
    graph string (``base[cbase];disc[cursor];[cbase][cursor]overlay…``): the
    cursor disc rides the shot at the smoothstep-interpolated ``x(t)``/``y(t)``
    (offset by the disc radius so the tip lands on the keyframed point), and each
    ``click=True`` keyframe adds an expanding-ring layer gated by
    ``enable='between(t, tc-lead, tc+tail)'``. The final overlay carries NO output
    label so the caller's ``[0:v]{video_vf}[v]`` seam closes the graph; the ``;``
    it contains is what routes the shot onto the ``-filter_complex`` path.
    """
    track = shot.cursor_track
    if not track:
        return video_vf
    fps = settings.render_fps
    diameter = max(_CURSOR_MIN_PX, round(settings.render_height * _CURSOR_DIAMETER_FRAC))
    r = diameter / 2.0
    x_expr = f"({_cursor_axis_expr(track, 'x')})-{_ffnum(r)}"
    y_expr = f"({_cursor_axis_expr(track, 'y')})-{_ffnum(r)}"
    parts = [
        f"{video_vf}[cbase]",
        f"{_cursor_disc_source(diameter, duration, fps)}[cursor]",
        f"[cbase][cursor]overlay=x='{x_expr}':y='{y_expr}':eval=frame",
    ]
    clicks = [k for k in track if k.click]
    if not clicks:
        return ";".join(parts)
    parts[-1] += "[cov0]"
    ring_d = max(_CURSOR_MIN_PX, round(diameter * _CLICK_RING_SCALE))
    rr = ring_d / 2.0
    for i, k in enumerate(clicks):
        enable = (
            f"between(t\\,{_ffnum(k.t - _CLICK_PULSE_LEAD)}\\,{_ffnum(k.t + _CLICK_PULSE_TAIL)})"
        )
        out = "" if i == len(clicks) - 1 else f"[cov{i + 1}]"
        parts.append(f"{_cursor_ring_source(ring_d, duration, fps)}[ring{i}]")
        parts.append(
            f"[cov{i}][ring{i}]overlay=x='{_ffnum(k.x - rr)}':y='{_ffnum(k.y - rr)}'"
            f":eval=frame:enable='{enable}'{out}"
        )
    return ";".join(parts)


def _callout_filter(
    video_vf: str, shot: Shot, settings: Settings, duration: float, work_dir: Path
) -> str:
    """Overlay callouts (badge / label / spotlight / arrow) onto ``video_vf``.

    ``callouts`` falsy → ``video_vf`` UNCHANGED (the byte-identical no-op). At most
    ``_CALLOUT_MAX_PER_SHOT`` are composited (extras dropped). A ``badge``
    composites a tinted disc glued to the remapped element bbox top-left corner
    with its short number/label centered on the disc; a ``label`` draws a caption
    ANCHORED to the bbox (centered under it, flipped above if it would clip the
    frame bottom); a ``spotlight`` dims the four bbox-complement bands with
    ``drawbox`` (no full-frame geq → memory-safe); an ``arrow`` overlays a small
    synthesized triangle sprite pointing at the bbox. Every type is enabled over
    its ``at`` window. Geometry rides PRE-zoom canvas coords (this filter is
    inserted BEFORE ``_focus_zoom_filter``) so a punch-in magnifies it with the
    element.

    Runs BETWEEN ``_cursor_filter`` and ``_focus_zoom_filter``, so the incoming
    ``video_vf`` is either a simple single chain (no cursor) OR a cursor
    multi-chain whose final node is UNLABELED — in BOTH shapes the tail is
    label-less. This CLOSES that tail by labeling it ``[co0]`` and emits one
    ``;``-joined segment per callout; each reopens the previous label and the LAST
    segment is left UNLABELED so ``_focus_zoom_filter`` can append ``[fz]`` onto it
    exactly as it does after the cursor. The returned graph always contains ``;``,
    so ``_video_audio_map_args`` routes the shot onto ``-filter_complex`` and
    ``_focus_zoom_filter`` always takes its labeled ``;``-branch. Text goes through
    ``textfile=`` (like the caption/overlay fragments) so no tenant metacharacter
    can break out of the filtergraph.
    """
    callouts = shot.callouts
    if not callouts:
        return video_vf
    callouts = callouts[:_CALLOUT_MAX_PER_SHOT]
    w = settings.render_width
    fps = settings.render_fps
    diameter = max(_BADGE_MIN_PX, round(settings.render_height * _BADGE_DIAMETER_FRAC))
    font_px = max(10, round(diameter * _BADGE_FONT_FRAC))
    arrow_px = max(_ARROW_MIN_PX, round(settings.render_height * _ARROW_DIAMETER_FRAC))
    label_fontsize = max(1, round(settings.render_height * _CALLOUT_LABEL_FONTSIZE_FRAC))
    dwell = min(_CALLOUT_LABEL_DWELL, duration)
    m = _safe_margin_px(w)
    g = _CALLOUT_ANCHOR_GAP_PX
    segments: list[str] = []
    cur = "co0"
    last = len(callouts) - 1
    for i, c in enumerate(callouts):
        out = "" if i == last else f"co{i + 1}"
        tail = "" if not out else f"[{out}]"
        text_path = work_dir / f"{shot.id}-callout-{i}.txt"
        bx, by, bw, bh = c.bbox
        if c.type == "badge":
            # Badge: a disc pinned to the element's top-left corner + a centered
            # number. It appears at ``at`` and persists (gte(t,at)); the enable
            # rides BOTH the overlay (disc) and the number so they show together.
            enable = _overlay_enable_expr(c.at, None)
            text_path.write_text(re.sub(r"\s+", " ", c.label).strip(), encoding="utf-8")
            cx = bx + diameter / 2.0
            cy = by + diameter / 2.0
            number = (
                f"drawtext=fontfile={settings.render_font}:textfile={text_path}"
                f":fontcolor={_BADGE_TEXT_COLOR}:fontsize={font_px}"
                f":x={_ffnum(cx)}-text_w/2:y={_ffnum(cy)}-text_h/2{enable}"
            )
            segments.append(f"{_badge_disc_source(diameter, duration, fps)}[badge{i}]")
            segments.append(
                f"[{cur}][badge{i}]overlay=x={_ffnum(bx)}:y={_ffnum(by)}{enable},{number}{tail}"
            )
        elif c.type == "spotlight":
            # Spotlight: dim the COMPLEMENT of the bbox rect with four drawbox
            # bands (top / bottom / left / right) — no full-frame geq/soft-mask,
            # so it stays inside the 2Gi pod. Commas here are FILTER separators
            # (not escaped); the enable clause's own commas are single-quoted.
            en = _overlay_enable_expr(c.at, c.at + dwell)
            color = f"black@{_ffnum(_SPOTLIGHT_DIM_ALPHA)}"
            x1, y1 = _ffnum(bx), _ffnum(by)
            x2, y2 = _ffnum(bx + bw), _ffnum(by + bh)
            bh_s = _ffnum(bh)
            top = f"drawbox=x=0:y=0:w=iw:h={y1}:color={color}:t=fill{en}"
            bottom = f"drawbox=x=0:y={y2}:w=iw:h=ih-{y2}:color={color}:t=fill{en}"
            left = f"drawbox=x=0:y={y1}:w={x1}:h={bh_s}:color={color}:t=fill{en}"
            right = f"drawbox=x={x2}:y={y1}:w=iw-{x2}:h={bh_s}:color={color}:t=fill{en}"
            segments.append(f"[{cur}]{top},{bottom},{left},{right}{tail}")
        elif c.type == "arrow":
            # Arrow: a small triangle sprite pointing at the bbox. Default a
            # RIGHT-pointing sprite to the LEFT of the bbox; if it would clip the
            # safe left margin, flip to a LEFT-pointing sprite to the RIGHT.
            en = _overlay_enable_expr(c.at, c.at + dwell)
            if bx - _ARROW_GAP_PX - arrow_px < m:
                direction = "left"
                ax = bx + bw + _ARROW_GAP_PX
            else:
                direction = "right"
                ax = bx - _ARROW_GAP_PX - arrow_px
            ay = by + bh / 2.0 - arrow_px / 2.0
            segments.append(f"{_arrow_sprite_source(arrow_px, direction, duration, fps)}[arrow{i}]")
            segments.append(f"[{cur}][arrow{i}]overlay=x={_ffnum(ax)}:y={_ffnum(ay)}{en}{tail}")
        else:  # label caption anchored to the bbox
            wrapped = _wrap_for_canvas(c.label, fontsize=label_fontsize, canvas_width=w)
            text_path.write_text(wrapped, encoding="utf-8")
            enable = _overlay_enable_expr(c.at, c.at + dwell)
            # Centered under the bbox, clamped inside the safe margin; flipped
            # ABOVE the bbox if placing below would clip the frame bottom. The
            # commas inside the x=/y= option VALUES are ffmpeg expression argument
            # separators → escaped ``\,`` (a single backslash-comma in the graph).
            cx = bx + bw / 2.0
            below = _ffnum(by + bh + g)
            above = _ffnum(by - g)
            x_expr = f"clip({_ffnum(cx)}-text_w/2\\,{m}\\,w-text_w-{m})"
            y_expr = f"if(gt({below}+text_h\\,h-{m})\\,{above}-text_h\\,{below})"
            frag = (
                f"drawtext=fontfile={settings.render_font}:textfile={text_path}"
                f":fontcolor={_safe_drawtext_color(settings.render_text_color, settings)}"
                f":fontsize={label_fontsize}:line_spacing=10"
                f":box=1:boxcolor=black@0.55:boxborderw=18:x={x_expr}:y={y_expr}{enable}"
            )
            segments.append(f"[{cur}]{frag}{tail}")
        cur = out
    return f"{video_vf}[co0];" + ";".join(segments)


def _cadence_fragment(
    settings: Settings,
    source_fps: float | None = None,
    retime: str | None = None,
) -> str:
    """The ffmpeg retime fragment that snaps a shot onto the master cadence.

    ``settings.render_fps`` is already the resolved master cadence (``compile_shots``
    threads the chosen rate in via ``model_copy`` before the fan-out) and is always
    the TARGET. ``source_fps`` is the shot's DECLARED source rate
    (``Shot.target_fps`` — the same field ``_resolve_master_cadence`` feeds to the
    selector); it must NOT default to the target, or the ratio is always exactly 1.0
    and the interpolation branches are unreachable (the VC-3 retime was structurally
    dead this way). ``None`` (a still image, or a shot that declared no rate) keeps
    the plain ``fps={render_fps}`` — byte-identical to the pre-VC-3 filters.

    ``retime`` is the per-shot hint (``Shot.retime``): ``'motion'`` forces
    motion-compensated interpolation, ``'budget'`` the cheap blend, ``'clean'``
    forces the plain frame drop/dup even on a non-integer ratio. Unset/unknown
    falls back to the service-wide ``render_motion_blur``. Any of these only bites
    on a genuinely non-integer source→target ratio; ``cadence.retime_plan`` returns
    the clean fragment otherwise.

    The selection/verification logic lives in ``cadence`` / ``mastering``; this is
    the single wiring seam into the filter chain."""
    from src.services import cadence

    hint = (retime or "").strip().lower()
    if hint in {"clean", "motion", "budget"}:
        high_motion = hint == "motion"
        budget = hint == "budget"
    else:
        high_motion = settings.render_motion_blur
        budget = False
    return (
        cadence.retime_plan(
            source_fps,
            settings.render_fps,
            high_motion=high_motion,
            budget=budget,
        )
        + _STARTPTS_REBASE
    )


# The trailing timestamp rebase that makes a segment FRAME-EXACT.
#
# WHY: every footage segment is bounded by an OUTPUT-side ``-t`` (the
# ``trim_args`` built in ``render_shot``, rescaled for speed in
# ``_apply_shot_speed``). An output ``-t`` measures its window from the FIRST
# OUTPUT TIMESTAMP, not from zero — and the footage branch also emits an INPUT
# ``-ss`` (``_clip_seek_args``). When that seek lands mid-frame (the normal case:
# the cut planner's in-points are musical times, not source-frame multiples), the
# first decoded frame sits slightly AFTER the requested seek point, so the
# segment's video ``start_time`` is non-zero. Measured on a real 23.976 fps
# source (frame period 0.041708s) seeked to ``-ss 1.002``: ``start_time =
# 0.041016`` — very nearly a whole frame period. The ``-t`` window then runs
# ``[0.041016, 0.041016 + t)`` while the delivered frames still start at 0, so
# the LAST frame of the span falls outside the window and is dropped. Every such
# shot delivers exactly one frame less than its plan says.
#
# WHAT IT COST: a delivered ``action_reel`` carried 685 video frames against a
# plan of 690 — five two-frame PTS gaps, one at each of five shot seams. The
# accumulated shortfall walked the cuts up to 56.8 ms off the music beat, against
# a one-frame (41.7 ms) tolerance. Frame-aligned seeks are unaffected
# (``start_time`` 0.000000, count exact), which is why this only ever bit some
# shots — and why "it renders fine locally" was true for the plain no-seek path.
#
# WHY THIS FIX: ``setpts=PTS-STARTPTS`` subtracts the first frame's timestamp, so
# the segment's timeline genuinely begins at 0 and the output ``-t`` measures the
# window the planner asked for. It is a constant OFFSET, not a rate change, so it
# cannot alter frame spacing.
#
# Four candidates were rendered with real ffmpeg against a 23.976 fps source
# (non-integral shot expects 54 frames, integral shot 42, full reel 690):
#
#   * append ``fps=24`` alone .............. 53 ✗ | 42 | 685 ✗ (no change at all)
#   * ``-frames:v N`` instead of ``-t`` .... 54 ✓ | 42 ✓ | 690, but five segments'
#                                            CONTAINER duration inflated — the
#                                            concat is ``-c copy``, so a bad
#                                            container length ships as-is
#   * ``setpts=PTS-STARTPTS`` after ``fps=`` 54 ✓ | 42 ✓ | 690, all double-gaps
#     keeping ``-t`` unchanged (CHOSEN)     gone
#
# WHERE IT SITS, and why the position is load-bearing:
#
#   * AFTER the cadence ``fps=``/``minterpolate``/``tblend`` fragment. Those
#     filters RESAMPLE onto the master grid; rebasing before them would let the
#     resampler re-introduce an offset. Appending here is also what puts it
#     ahead of ``format=yuv420p`` in every chain (all three call sites splice the
#     fragment as ``…,setsar=1,{cadence},format=yuv420p``), i.e. still inside the
#     video body and nowhere near the encoder.
#   * AFTER — never instead of — the ``setpts=PTS/speed`` that
#     ``_apply_shot_speed`` PREPENDS to the whole filter body (see
#     ``_apply_speed_to_vf``). The resulting order is
#     ``setpts=PTS/speed , scale , pad , setsar , fps=N , setpts=PTS-STARTPTS ,
#     format``. The speed retime multiplies every timestamp and must run BEFORE
#     ``fps=`` so the frame COUNT actually changes; this rebase only subtracts a
#     constant from an already-retimed, already-resampled stream, so it neither
#     undoes nor double-applies the speed. ``_apply_shot_speed`` rescales the
#     ``-t`` to ``duration/speed`` and that bound now measures from a true zero,
#     so a sped-up seeked shot is frame-exact for the same reason a 1.0x one is.
#
# Do NOT "simplify" this into ``cadence.retime_plan``: that helper is the
# cadence-selection policy (unit-tested per ratio in ``tests/test_cadence.py``)
# and has no business knowing about the renderer's ``-ss``/``-t`` bounding.
_STARTPTS_REBASE = ",setpts=PTS-STARTPTS"


# ---------------------------------------------------------------------------
# Finishing (VC-5): brand letterbox colour
# ---------------------------------------------------------------------------
# The colour rides the PER-SHOT filter chain, not a finishing pass over the whole
# film: the final join is a stream copy (``concat_shots`` runs ``-c copy``), so
# there is no whole-film re-encode to hang a finishing filter on. It is threaded
# in on ``settings`` (``_settings_for_finish`` in the compile route
# ``model_copy``s it per job — the same seam ``_settings_for_format`` / the
# mastering tags use), so no renderer signature changes and an absent ``finish``
# profile leaves every filter string byte-identical.
#
# NOT SHIPPED — broadcast-safe legal-luma clamp. A pre-encode
# ``lutyuv=y=clip(val,LO,HI)`` CANNOT guarantee a decoded 16..235 master in this
# architecture, and the flag would have been a lie. Measured on real libx264:
# a white caption on a dark background at CRF 20 decodes at YMAX=244 unclamped
# and 242 with clip(17,234) — a ~2-codeword no-op. Landing <=235 at CRF 20 needs
# ~clip(25,225), which visibly crushes whites, and even that fails at CRF 28
# (YMIN=13 / YMAX=244). Root cause: the clamp runs PRE-encode, libx264 ringing
# around sharp edges (i.e. text) reintroduces out-of-range values, and the
# ringing grows with CRF. Since the concat is ``-c copy`` there is no later pass
# to clean up. A real fix needs a post-encode legalising pass over the delivered
# master (its own re-encode) — do not re-add a pre-encode lutyuv.
def _pad_color(settings: Settings) -> str:
    """The letterbox/pillarbox + canvas fill colour.

    ``render_letterbox_color`` (per-job brand colour, validated as ``#RRGGBB``
    by ``FinishProfile``) when set, else the service default ``render_bg_color``
    — so an unset finish profile keeps today's exact filter string.
    """
    return settings.render_letterbox_color or settings.render_bg_color


def _scale_pad_filter(
    settings: Settings,
    caption: str | None,
    work_dir: Path | None = None,
    shot_id: str | None = None,
    fit: str = "contain",
    source_fps: float | None = None,
    retime: str | None = None,
) -> str:
    """Build a video filter that fits the source onto the canvas + caption.

    ``fit="contain"`` (default) scales-to-fit and pads with the bg colour so
    nothing is cropped — correct for footage, slides and screenshots whose
    framing/text matters. ``fit="cover"`` scales-to-fill and center-crops so the
    canvas is fully covered (no bars) — for atmospheric AI-image backgrounds.

    ``source_fps``/``retime`` are the footage shot's declared source cadence and
    retime hint (VC-3); omitted (still images, undeclared shots) keeps the plain
    ``fps=`` resample.
    """
    w, h = settings.render_width, settings.render_height
    cadence = _cadence_fragment(settings, source_fps, retime)
    if fit == "cover":
        chain = (
            f"scale={w}:{h}:force_original_aspect_ratio=increase:{_sws_flags(settings)},"
            f"crop={w}:{h},"
            f"setsar=1,{cadence},format=yuv420p"
        )
    else:
        chain = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:{_sws_flags(settings)},"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={_pad_color(settings)},"
            f"setsar=1,{cadence},format=yuv420p"
        )
    if caption and work_dir is not None and shot_id is not None:
        chain += f",{_caption_filter_fragment(caption, settings, work_dir, shot_id)}"
    return chain


def _even_dim(n: int) -> int:
    """Floor to the nearest even integer — ``W//2``/``H//2`` of an even canvas dim
    can be odd (e.g. 1350//2 = 675), and yuv420p / hstack reject odd half dims."""
    return n - (n % 2)


def _split_filter(
    split: SplitScreen,
    idx: int,
    settings: Settings,
    work_dir: Path | None,
    shot_id: str | None,
    caption: str | None,
    duration: float,
    source_fps: float | None = None,
    retime: str | None = None,
) -> str:
    """Composite two sources into the shot canvas via hstack (h) / vstack (v).

    REBUILDS the base graph from scratch (the primary's incoming scale is
    discarded — each half is re-fitted into its half-canvas here), returning a
    ``;``-multi-chain string. When the caller wraps it as ``[0:v]{video_vf}[v]``
    the primary half consumes the prepended ``[0:v]`` and the final label-less
    stack chain receives ``[v]``.

    Each half is a FINITE ``tpad=clone`` sized off the caller's own ``duration``
    (see ``_cover_pad_stop_duration`` — this is the same shared math as the
    single-source cover pad; a split shot's speed is always 1.0, the retime seam
    is skipped for split shots). It holds its last frame so neither half starves
    the framesync while the caller's ``-t`` bounds the segment, but — UNLIKE the
    ``stop = -1`` this replaced — it can never livelock an encode that never
    arrives at a ``-t`` (every split shot forces one; see the call site). Each
    half is then scaled-to-fit + padded into an EVEN-floored half-canvas, then
    ``setsar=1`` + ``format=yuv420p`` so hstack/vstack see matching SAR + pixel
    format. The caption (if any) is re-applied on the composite, since the base
    scale — and with it the base caption — is discarded here.
    """
    w, h = settings.render_width, settings.render_height
    if split.layout == "h":
        hw, hh = _even_dim(w // 2), h
        stack = "hstack"
        stacked_w, stacked_h = hw * 2, hh
    else:
        hw, hh = w, _even_dim(h // 2)
        stack = "vstack"
        stacked_w, stacked_h = hw, hh * 2
    pad_color = _pad_color(settings)
    cadence = _cadence_fragment(settings, source_fps, retime)
    stop_duration = _ffnum(_cover_pad_stop_duration(duration, settings.render_fps, 1.0))
    half_body = (
        f"tpad=stop_mode=clone:stop_duration={stop_duration},"
        f"scale={hw}:{hh}:force_original_aspect_ratio=decrease:{_sws_flags(settings)},"
        f"pad={hw}:{hh}:(ow-iw)/2:(oh-ih)/2:color={pad_color},"
        f"setsar=1,format=yuv420p"
    )
    primary = f"{half_body}[sp0]"
    second = f"[{idx}:v]{half_body}[sp1]"
    # The EVEN-floor above is applied to the HALF, so the stack can land 1-2px
    # SHORT of the canvas (4x5 @1080p: 1350//2 -> 674, vstack -> 1080x1348).
    # Every non-split shot is exactly WxH and the final join is a stream copy
    # (``concat_shots`` runs ``-c copy``), which REJECTS a mismatched segment —
    # so re-pad (letterbox the odd 1-2px, never re-scale: a scale here would
    # resample every pixel of both halves for a 2px deficit). Emitted only when
    # the stack actually falls short, so an already-exact canvas stays
    # byte-identical to the pre-fix graph.
    restore = ""
    if (stacked_w, stacked_h) != (w, h):
        restore = f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color},"
    last = f"[sp0][sp1]{stack}=inputs=2,{restore}{cadence},format=yuv420p"
    if caption and work_dir is not None and shot_id is not None:
        last += f",{_caption_filter_fragment(caption, settings, work_dir, shot_id)}"
    return f"{primary};{second};{last}"


def _inject_split(
    cmd: list[str],
    shot: Shot,
    second_clip: Path | None,
    video_vf: str,
    next_idx: int,
    settings: Settings,
    work_dir: Path,
    duration: float,
) -> tuple[str, int]:
    """T8 split-screen: append the SECOND source ``-i`` and REBUILD ``video_vf`` as
    an hstack/vstack composite, returning the updated ``(video_vf, next_idx)``.

    The split ``-i`` is appended HERE — before ``_video_audio_map_args``, which
    appends SFX ``-i`` inputs starting at the ``next_idx`` it is passed — so the
    final input order is 0=primary, 1=audio, 2=split, 3+=SFX (split_idx=2). A
    no-op (returns the inputs unchanged, appends nothing) when the shot has no
    split or its second source was unfetchable → byte-identical to before.

    ``_inject_overlay_clips`` extends the SAME invariant one slot further: it also
    appends its ``-i`` inputs before ``_video_audio_map_args`` and threads
    ``next_idx`` on, so the full order is 0=primary, 1=audio, 2=split,
    3..=overlay clips, then SFX. Anything that appends an input must keep running
    BEFORE the SFX tail and must return the advanced index; appending after would
    silently renumber every SFX input in the audio graph.
    """
    if shot.split is None or second_clip is None:
        return video_vf, next_idx
    cmd += ["-i", str(second_clip)]
    split_idx = next_idx
    video_vf = _split_filter(
        shot.split,
        split_idx,
        settings,
        work_dir,
        shot.id,
        shot.caption,
        duration,
        source_fps=shot.target_fps,
        retime=shot.retime,
    )
    return video_vf, next_idx + 1


# ---------------------------------------------------------------------------
# Alpha overlay clips: animated graphics composited ON TOP of the shot picture
# ---------------------------------------------------------------------------
# Renderer-side mirror of ``schemas._MAX_OVERLAY_CLIPS``. The schema already 422s a
# third overlay at the API boundary; this bound exists so ``render_shot`` is safe
# on its own (the audiobook/thumbnail paths and the tests construct ``Shot``
# objects directly, and a future producer could grow the schema cap without
# anyone re-reading the per-frame cost of a full-canvas ``overlay`` pass).
_OVERLAY_CLIP_MAX_PER_SHOT = 2

#: The decoder that must be forced for a VP9 overlay. See ``_overlay_input_args``.
_VP9_ALPHA_DECODER = "libvpx-vp9"


def _overlay_input_args(path: Path, codec: str | None) -> list[str]:
    """Input args for one overlay clip: the alpha-safe decoder, then its ``-i``.

    THE DECODER TRAP (measured on this project's own VP9-alpha builds)
    -----------------------------------------------------------------
    The SAME VP9-alpha WebM decodes differently depending on which decoder ffmpeg
    picks, and the wrong one fails SILENTLY — no warning, no non-zero exit::

        decoder                        first pixel RGBA   alpha
        ffmpeg's native ``vp9``        fd 00 00 ff        255 — OPAQUE, alpha DISCARDED
        ``-c:v libvpx-vp9``            fd 00 00 00        0   — correct

    The native decoder is the DEFAULT. Left alone, the compositor lays an opaque
    black box over the drone footage and the render "succeeds". So the decoder is
    named explicitly, as an INPUT option, immediately before the ``-i`` it
    qualifies — position is load-bearing: ``-c:v`` is per-input and applies to the
    NEXT input only. (It cannot leak to the output: ``_video_codec_args`` sets the
    output ``-c:v`` explicitly, after every input.)

    Related trap, for whoever validates these assets: a CORRECT VP9-alpha file
    reports ``pix_fmt=yuv420p``, NOT ``yuva420p`` — the alpha plane rides in WebM
    ``BlockAdditional`` side data and is flagged by the container tag
    ``alpha_mode=1``. A "is the pix_fmt yuva420p?" check rejects every good build.

    WHY THIS IS PROBE-GATED rather than unconditional: ``libvpx-vp9`` can only
    decode VP9. Forcing it on a PNG/ProRes/H.264 overlay is a hard ffmpeg error,
    which would fail the whole shot — and the house rule is that a bad asset
    degrades, never 500s a compile. So the codec is probed (best-effort, ``None``
    on any failure) and the decoder is forced only when the file really is VP9.
    A non-VP9 overlay carries no alpha-in-side-data problem to begin with.
    """
    if codec == "vp9":
        return ["-c:v", _VP9_ALPHA_DECODER, "-i", str(path)]
    return ["-i", str(path)]


def _overlay_clip_chain(idx: int, clip: OverlayClip, settings: Settings) -> str:
    """The filter chain that turns input ``idx`` into a placed, timed alpha layer.

    ``scale='fit'`` fits the graphic INSIDE the canvas preserving aspect (no pad —
    the alpha is the framing), so one authored 1920x1080 asset lands correctly at
    every output tier; ``'native'`` leaves it at its authored size.

    ``format=yuva420p`` is what actually MATERIALISES the alpha plane for the
    filter graph. Combined with the forced decoder in ``_overlay_input_args`` it is
    the second half of the alpha contract — the decoder makes the plane available,
    this makes the graph carry it.

    ``trim`` (only when a duration is set) bounds the layer on its OWN timeline,
    before the shift, so the shift arithmetic stays independent of it.

    ``setpts=PTS-STARTPTS+start/TB`` re-bases the layer to zero and slides it to
    its onset. Before that onset the layer simply has no frames yet and
    ``overlay`` passes the base through untouched.

    The times on ``clip`` are already on the RENDERED timeline when this runs —
    ``_inject_overlay_clips`` divides them by the shot's ``speed`` first.
    """
    parts: list[str] = []
    if clip.scale == "fit":
        parts.append(
            f"scale={settings.render_width}:{settings.render_height}"
            f":force_original_aspect_ratio=decrease:{_sws_flags(settings)}"
        )
    else:
        # ``scale='native'`` skips the resize above, but the very next step
        # still performs an RGBA->YUV colour conversion (``format=yuva420p``,
        # below) — and that conversion runs through swscale exactly like a
        # real resize would, so it is silently subject to the same BT.601
        # default if no matrix node precedes it. A same-size ``scale=iw:ih``
        # is a geometric no-op but still routes the conversion through
        # swscale WITH the derived in/out_color_matrix, so a 'native' overlay
        # gets the same colour correctness as a 'fit' one.
        parts.append(f"scale=iw:ih:{_sws_flags(settings)}")
    parts.append("format=yuva420p")
    if clip.duration_seconds is not None:
        parts.append(f"trim=duration={_ffnum(clip.duration_seconds)}")
    parts.append(f"setpts=PTS-STARTPTS+{_ffnum(clip.start_seconds)}/TB")
    if clip.opacity < 1.0:
        # colorchannelmixer's ``aa`` is the alpha->alpha coefficient: it SCALES the
        # graphic's authored alpha rather than replacing it, so a soft-edged logo
        # stays soft-edged. aa=1 is the identity, which is why it is only emitted
        # below 1.0 — an opacity=1.0 overlay is byte-identical to no knob at all.
        parts.append(f"colorchannelmixer=aa={_ffnum(clip.opacity)}")
    return f"[{idx}:v]" + ",".join(parts)


def _inject_overlay_clips(
    cmd: list[str],
    overlays: list[tuple[Path, OverlayClip, str | None]],
    video_vf: str,
    next_idx: int,
    settings: Settings,
    speed: float,
) -> tuple[str, int]:
    """Composite alpha overlay clips on top of ``video_vf``; append their ``-i``s.

    Returns the updated ``(video_vf, next_idx)``. Empty ``overlays`` returns both
    unchanged and appends NOTHING to ``cmd`` — the byte-identical no-op that an
    absent ``overlay_clips`` depends on.

    INPUT ORDER (the invariant ``_inject_split`` documents): the overlay ``-i``
    inputs are appended HERE, before ``_video_audio_map_args`` appends the SFX
    inputs at the ``next_idx`` it is handed. Appending them after would renumber
    every SFX input out from under the audio graph's ``[N:a]`` labels.

    GRAPH SHAPE: each overlay closes the running chain into a named link, adds its
    own chain from input ``idx``, and re-joins them::

        <base chain>[ovb0];[3:v]<layer chain>[ovl0];[ovb0][ovl0]overlay=…

    The result is a MULTI-CHAIN graph (it contains ``;``), which is precisely the
    condition ``_video_audio_map_args`` uses to route a shot off the plain ``-vf``
    fast path onto ``-filter_complex``. That is not incidental — a bare ``-vf``
    cannot reference a second input at all, so an overlay MUST disable that fast
    path, and it does so by construction rather than by a flag that could be
    forgotten. The final chain is left label-less so the enclosing
    ``[0:v]{video_vf}[v]`` wrap still closes the graph correctly.

    ``eval=frame`` re-evaluates x/y per frame (so W/H/w/h/t expressions animate).

    ``repeatlast=0`` is load-bearing: ffmpeg's ``overlay`` DEFAULTS to holding the
    secondary's last frame forever once it EOFs. Without this a 3-second animated
    lower-third would freeze on its final frame for the rest of the shot instead
    of leaving — the "the graphic never goes away" bug — and a ``duration_seconds``
    would be inert.

    ``speed`` is the shot's playback factor. The overlay is a SEPARATE input and is
    NOT touched by ``_apply_shot_speed``'s ``setpts``, which only retimes the
    primary chain. So a cue authored against the SOURCE timeline
    (``retime_with_speed``, the default) has its onset/duration divided by
    ``speed`` here — the same 1/speed rescale ``_apply_directed_footage`` applies
    to cursor/callout/zoom keyframes, and for the same reason. ``retime_with_speed
    =False`` opts out and pins the overlay to rendered wall-clock time instead. At
    ``speed == 1.0`` the two are identical.
    """
    used = overlays[:_OVERLAY_CLIP_MAX_PER_SHOT]
    if not used:
        return video_vf, next_idx
    for n, (path, clip, codec) in enumerate(used):
        idx = next_idx + n
        cmd += _overlay_input_args(path, codec)
        timed = _retime_overlay_clip(clip, speed)
        base, layer = f"ovb{n}", f"ovl{n}"
        video_vf = (
            f"{video_vf}[{base}];"
            f"{_overlay_clip_chain(idx, timed, settings)}[{layer}];"
            f"[{base}][{layer}]overlay=x={timed.x}:y={timed.y}:eval=frame:repeatlast=0"
        )
    return video_vf, next_idx + len(used)


def _retime_overlay_clip(clip: OverlayClip, speed: float) -> OverlayClip:
    """``clip`` with its onset/duration moved onto the RENDERED timeline.

    Returns the clip UNCHANGED (the same object) at ``speed == 1.0`` or when
    ``retime_with_speed`` is off, so the default path allocates nothing and the
    emitted numbers are bit-for-bit the authored ones.
    """
    if speed == 1.0 or not clip.retime_with_speed:
        return clip
    return clip.model_copy(
        update={
            "start_seconds": clip.start_seconds / speed,
            "duration_seconds": (
                None if clip.duration_seconds is None else clip.duration_seconds / speed
            ),
        }
    )


def _scroll_filter(
    settings: Settings,
    duration: float,
    caption: str | None = None,
    work_dir: Path | None = None,
    shot_id: str | None = None,
) -> str:
    """Build a filter that fits a tall full-page screenshot to the canvas width
    and pans it top→bottom over the shot ``duration`` (a vertical scroll).

    A full-page website screenshot is far taller than 16:9, so ``contain`` would
    pillarbox it into an unreadable vertical sliver and ``cover`` would show only
    the top crop. Scrolling scales to the canvas width, pads the height up to at
    least the canvas height (so a short page still renders), then crops a
    full-width window whose y travels from the top to the bottom of the page
    across the shot — the page reads at full width and the whole thing is seen.
    Commas inside the crop/pad expressions are backslash-escaped for the
    filtergraph parser.
    """
    w, h = settings.render_width, settings.render_height
    cadence = _cadence_fragment(settings)
    d = max(duration, 0.1)
    chain = (
        f"scale={w}:-2:{_sws_flags(settings)},"
        f"pad={w}:max(ih\\,{h}):0:0:color={_pad_color(settings)},"
        f"crop={w}:{h}:0:'(ih-{h})*min(t/{d:.3f}\\,1)',"
        f"setsar=1,{cadence},format=yuv420p"
    )
    if caption and work_dir is not None and shot_id is not None:
        chain += f",{_caption_filter_fragment(caption, settings, work_dir, shot_id)}"
    return chain


def _image_vf(
    shot: Shot,
    settings: Settings,
    duration: float,
    work_dir: Path,
    *,
    defer_caption: bool = False,
) -> str:
    """Pick the still-image video filter: ``scroll`` pans a tall page over the
    shot; ``contain``/``cover`` use the scale/pad/crop fit.

    ``defer_caption`` omits the burned-in caption here so ``render_shot`` can draw
    it AFTER the focus punch-in — see the ``caption_after_zoom`` note there."""
    caption = None if defer_caption else shot.caption
    if shot.fit == "scroll":
        return _scroll_filter(settings, duration, caption, work_dir, shot.id)
    return _scale_pad_filter(settings, caption, work_dir, shot.id, shot.fit)


# The ms grid every emitted seek/span already lives on (``f"{x:.3f}"``).
_SEEK_MS = 1000.0
# Guard against float error turning an EXACTLY frame-aligned offset into the
# previous frame: 1e-6 of a frame period is ~0.04 microseconds at 24 fps, four
# orders of magnitude below the millisecond grid we then floor onto, so it can
# never move the emitted value.
_FRAME_EPS = 1e-6


async def _ffprobe_frame_rate(path: Path) -> float | None:
    """The source's ``r_frame_rate`` as a float, or ``None`` when unknowable.

    Only consulted by ``_clip_seek_args`` when the shot declared no
    ``target_fps`` — which is the COMMON case in production, because the cut
    planner states musical in/out times and leaves the rate to the file. Without
    this fallback the frame-grid snap below would be structurally inert on
    exactly the renders it was written for.

    Never raises: a missing/rejected/variable-rate file returns ``None`` and the
    caller emits the unsnapped args it emitted before this existed. A snap onto a
    GUESSED rate would be worse than the bug it fixes (it would move the in-point
    by an arbitrary amount), so "unknown" must mean "do nothing"."""
    try:
        out = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — advisory probe, never fail a render
        log.debug("clip_seek.frame_rate_probe_failed", path=str(path), error=str(exc))
        return None
    raw = out.strip()
    try:
        if "/" in raw:
            num, _, den = raw.partition("/")
            rate = float(num) / float(den)
        else:
            rate = float(raw)
    except ValueError, ZeroDivisionError:
        return None
    return rate if rate > 0.0 else None


def _snap_seek_to_source_grid(offset: float, source_fps: float | None) -> float:
    """Floor an input seek onto the SOURCE frame grid, then onto the ms grid.

    THE DEFECT (the input-side twin of the ``_STARTPTS_REBASE`` fix above).
    ``-ss`` is an ACCURATE seek: ffmpeg starts decoding at the first frame whose
    timestamp is >= the requested time. Ask for a time that lands mid-frame — the
    normal case, since the cut planner's in-points are musical times, not
    source-frame multiples — and up to one whole source frame is lost at the HEAD,
    while the ``-t`` derived from that same requested time still stops at the
    requested out-point. ``setpts=PTS-STARTPTS`` rebases the OUTPUT window onto
    that late first frame, which is correct and necessary, but it cannot conjure
    the lost frame back.

    Normally the output ``-t`` has enough tail slack to absorb the loss. It is
    only fatal when there is NO slack, i.e. the window ends at EOF. Measured on
    the live 19-shot reel, shot 12: ``out_seconds 10.010 == source duration
    10.010``, ``-ss 8.010`` on a 23.976 fps source lands 0.0397 s (0.95 frame
    periods) before the next real frame at 8.04972 (source frame 193), and the
    shot delivered 42 frames where the plan said 43. That was the LAST of the
    reel's frame deficit: the rebase took it from -5 to -1, this takes it to 0.

    THE FIX. ``floor(offset * fps) / fps`` moves the request back onto the frame
    boundary at or before it, so the frame that was being skipped is now the first
    one decoded (8.010 -> 8.008 in the case above). The standalone repro of the
    live shot went 42 frames -> 43 that way; the same zero-slack shape rebuilt
    here at 320x240 (2.0s window ending at a 10.010s EOF, 48-frame plan) goes
    47 -> 48, and is pinned by
    ``tests/render_harness/test_seek_frame_grid.py``.

    ⚠️ THE SEEK MOVES; THE WINDOW LENGTH DOES NOT. The span is still measured from
    the caller's REQUESTED in-point, so the window slides back by the recovered
    sub-frame delta rather than widening. Leaving ``end`` fixed and letting the
    span widen was tried first and MEASURED WRONG: rendering 19 pseudo-shots
    against a 23.976 fps source, the zero-slack shot went 47 -> 48 as intended but
    EIGHT of the other eighteen went one frame OVER plan (e.g. an 8-frame shot
    delivering 9), because the widened span becomes the segment's own output
    ``-t``. That is the #174 defect with the sign flipped — every seeked shot
    running long walks the cuts LATE off the beat. With the span pinned to the
    plan, the same sweep gives 47 -> 48 on the zero-slack shot and ZERO change on
    all eighteen others: the shot gains a head frame and drops a tail frame, a
    sub-frame CONTENT shift (<=41.7ms at 23.976 fps), never a count change.

    ORDERING — floor to the frame grid, then FLOOR (never round) to the ms grid.
    The emitted arg is ``f"{x:.3f}"``, which rounds to NEAREST millisecond; a
    frame boundary at, say, 8.00874 would round UP to 8.009, land after the frame
    again, and reinstate the whole bug in the half of cases that round up. Flooring
    to the ms below is always <= the true frame time, so the target frame is always
    included, and it can never reach the PREVIOUS frame as long as the frame period
    exceeds 1 ms (it does for every rate below 1000 fps).

    ``source_fps`` unknown, non-finite or non-positive => return exactly what this
    function returned before it existed (round to ms), byte-identical args. Guessing
    a rate would move the in-point by an arbitrary amount — strictly worse than the
    bug.

    ⚠️ ``math.isfinite`` is LOAD-BEARING, not defensive dressing. ``Shot.target_fps``
    is declared ``gt=0`` with NO upper bound, so a JSON ``1e400`` arrives as ``inf``
    — ``tests/test_cadence_nonfinite.py`` exists because exactly that once killed a
    whole job "with an opaque OverflowError before a single shot rendered", and
    ``cadence._coerce_fps`` was hardened for it. ``inf > 0.0`` is True, so a bare
    positivity guard lets it through and ``math.floor(offset * inf)`` raises
    ``OverflowError``. ``nan`` would fail the ``> 0.0`` test by luck; this makes both
    explicit rather than relying on it. Caught by an adversarial pass on THIS change,
    against real ``render_shot``: main rendered 47 frames, the unguarded branch
    raised."""
    if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0.0:
        return float(f"{offset:.3f}")
    frame = math.floor(offset * source_fps + _FRAME_EPS)
    snapped = frame / source_fps
    floored_ms = math.floor(snapped * _SEEK_MS + _FRAME_EPS) / _SEEK_MS
    return float(f"{floored_ms:.3f}")


async def _clip_seek_args(
    clip: Path,
    clip_seek_fraction: float,
    shot_duration: float,
    clip_anchor: str = "start",
    clip_lead_seconds: float = 0.0,
    clip_out_seconds: float | None = None,
    source_fps: float | None = None,
) -> tuple[list[str], float, float | None]:
    """Input-side args (``-ss`` and, with an out-point, ``-t`` — both BEFORE
    ``-i``) that choose WHICH span of a footage clip this shot shows.

    - ``clip_anchor='end'`` shows the SETTLED TAIL — the last ``shot_duration``
      seconds — so a slow app load / zoom-in at the head (white / loading frames)
      is skipped and only the final hero state appears. Adapts to however long the
      load actually took (the offset is derived from the real clip length).
    - ``clip_lead_seconds>0`` (anchor 'start') skips that many leading seconds (the
      white load-in) then plays forward — for clips whose hero is the MOTION near
      the head (typing, an in-progress detection).
    - else ``clip_seek_fraction`` (repeated-use panning) as before.

    Empty when no seek is warranted.

    ``clip_out_seconds`` (None = today's behaviour, byte-identical) names an
    EXPLICIT source-timeline out-point and switches this function into
    window mode, which is a different contract from the three cases above:

    * the in-point is taken VERBATIM from ``clip_lead_seconds`` — it is NOT
      clamped to ``seekable`` (``clip_len - shot_duration``). That clamp exists to
      stop a lead from eating the shot's own duration; a caller that states both
      ends has already decided how long the window is, and clamping the in-point
      would silently slide the window and show the wrong content. A window shorter
      than the narration freeze-pads downstream exactly as a short clip does.
    * ``clip_anchor='end'`` still means "the settled tail", but of the WINDOW: the
      in-point moves forward to ``out - shot_duration``, never before the stated
      lead.
    * the span is bounded by an INPUT-side ``-t``. This is the load-bearing half —
      an output ``-t`` (or a freeze-pad) would let ffmpeg keep decoding real frames
      past the out-point, so the shot would show content the caller excluded. As an
      input option it caps what is READ, and ``tpad`` then clones from the true
      last in-window frame.

    Returns ``(args, offset, span)``. ``offset`` is the seek APPLIED, in seconds,
    exactly as handed to ffmpeg (0.0 whenever no ``-ss`` is emitted). The caller
    needs it because this is an INPUT seek: output timestamps restart at the seek
    point, so the span that actually renders is only ``clip_dur - offset`` seconds
    long. The arithmetic lives here alone — no caller re-derives it. ``span`` is
    the in-window source length the ``-t`` bounds it to, or ``None`` in the
    (default) no-out-point case, where the readable length is the whole tail.

    ``source_fps`` (window mode only) is the shot's declared source rate. The
    emitted in-point is floored onto that frame grid, because ``-ss`` starts
    decoding at the first frame AT OR AFTER the requested time and a mid-frame
    request therefore loses up to one source frame at the head — see
    ``_snap_seek_to_source_grid``. ``None`` falls back to the clip's probed
    ``r_frame_rate``; when that is unknowable too, no snap happens at all and the
    args are byte-identical to before this parameter existed.

    The returned ``span`` (and the emitted ``-t``) is ALWAYS measured from the
    requested in-point, never from the snapped one: the snap recovers a head frame
    by sliding the window back, it must not lengthen the shot. ``offset`` is still
    the seek actually handed to ffmpeg, so ``_playable_clip_duration`` — which
    takes ``min(clip_len - offset, span)`` — keeps reading the segment's true
    playable length.

    Both are SOURCE seconds and deliberately unscaled by ``Shot.speed``: the
    decoder reads at native rate and the ``setpts``/``atempo`` retime happens
    downstream, so scaling here would move the window itself."""
    clip_len = await _ffprobe_duration(clip) or 0.0
    if clip_out_seconds is not None:
        # Window mode. Clamp the out-point to what the file actually holds — a
        # caller-stated out-point past EOF is not an error, it just means "to the
        # end" — then place the in-point per the anchor.
        end = min(clip_out_seconds, clip_len) if clip_len > 0.0 else clip_out_seconds
        if clip_anchor == "end":
            offset = max(end - shot_duration, clip_lead_seconds)
        else:
            offset = clip_lead_seconds
        args: list[str] = []
        # The in-point the SPAN is measured from: the caller's requested time on
        # the millisecond grid, exactly as before the frame-grid snap existed.
        # The snap below moves the SEEK earlier without moving the window LENGTH
        # — see the block comment there for why widening the span instead makes
        # every slack-bearing shot one frame LONGER than its plan.
        planned_offset = float(f"{offset:.3f}")
        # Below 0.1s a seek is noise (and matches the no-window branch's floor).
        if offset >= 0.1:
            # Snap the in-point back onto the SOURCE frame grid before emitting
            # it — see ``_snap_seek_to_source_grid`` for the measured case (shot
            # 12 of the live reel: 8.010 -> first real frame 8.04972, 42 frames
            # delivered against a plan of 43). ``shot.target_fps`` when the caller
            # declared one, else the probed ``r_frame_rate``; unknown => no snap
            # and byte-identical args.
            fps = source_fps if source_fps is not None and source_fps > 0.0 else None
            if fps is None:
                fps = await _ffprobe_frame_rate(clip)
            offset = _snap_seek_to_source_grid(planned_offset, fps)
            args += ["-ss", f"{offset:.3f}"]
        else:
            planned_offset = 0.0
            offset = 0.0
        # Re-derive the span from the REQUESTED (ms-rounded) in-point, not from
        # the snapped one: the emitted window is exactly as long as the caller
        # planned, shifted back by whatever sub-frame delta the snap recovered.
        # Measuring it from the snapped offset instead runs every seeked shot up
        # to one frame LONG (measured: 8 of 18 slack-bearing shots) — see
        # ``_snap_seek_to_source_grid``.
        span = max(end - planned_offset, 0.0)
        if span <= 0.0:
            # Degenerate (the schema validator rejects out <= lead, so this only
            # happens when the file is shorter than the stated in-point). Fall
            # back to "no bound" rather than emitting `-t 0`, which would produce
            # an empty input and fail the encode instead of degrading.
            return args, offset, None
        span = float(f"{span:.3f}")
        args += ["-t", f"{span:.3f}"]
        return args, offset, span
    seekable = max(clip_len - shot_duration, 0.0)
    if clip_anchor == "end":
        offset = max(seekable - 0.10, 0.0)
    elif clip_lead_seconds > 0.0:
        offset = min(clip_lead_seconds, seekable)
    elif clip_seek_fraction > 0.0:
        offset = clip_seek_fraction * seekable
    else:
        return [], 0.0, None
    if offset < 0.1:
        return [], 0.0, None
    # Report the rounded value we actually pass, so the caller's length arithmetic
    # matches what ffmpeg seeks to rather than the pre-format float.
    seek = f"{offset:.3f}"
    return ["-ss", seek], float(seek), None


# Safety margin appended to every FINITE cover-pad below so encoder/probe
# rounding can never leave a residual gap short of the caller's own ``-t``.
# Harmless: the segment's real length is set by that ``-t`` alone, never by
# this filter, so a clone that runs a little past it is simply truncated at
# mux time — the margin can only make the encoder do a LITTLE more cloning,
# never lengthen the segment or resurrect the livelock this replaces.
_COVER_PAD_MARGIN_SECONDS = 0.25

# Fallback freeze-pad floor for a nonsensical ``render_fps <= 0``, where "one output
# frame" is undefined. 1 ms is far below any real cadence's frame period (4 ms at
# 250 fps) so it can never be the binding floor on a sane setting.
_MIN_FREEZE_PAD_SECONDS = 0.001


def _bounded_cover_pad(duration: float, fps: float, speed: float) -> str:
    """The last line of defence for A/V coverage: clone the final frame for a
    FINITE duration. Emitted ONLY on a segment that carries an output-side
    ``-t`` (every call site below gates on ``bounded``) — the ``-t`` still
    sets the segment's real length, this filter only makes sure the picture
    does not stop before it.

    ⛔ HISTORY: this used to be a single constant, ``stop = -1`` — an
    UNCONDITIONAL, INFINITE last-frame clone that relied on the caller's
    ``-t`` alone to terminate it. Reproduced livelock: a cover-fit video input
    (1.933333s, 58 real frames) under a narration-driven segment target of
    1.95s (a ~17ms gap, well under one output frame) paired with a SECOND,
    finite audio input in the same graph. Under ``-vsync cfr`` the encoder
    keeps synthesising clones to fill the frame grid, every clone is a
    duplicate of a duplicate the muxer drops against the shorter audio
    stream, and ``-t`` never arrives because the video stream that would
    reach it never advances. ffmpeg froze at a fixed byte count with
    ``drop_frames`` climbing without bound and did not respond to SIGTERM —
    only SIGKILL stopped it. A single-input (no separate audio) or
    non-cfr case never observed the hang, which is why it shipped unnoticed:
    the trap needs BOTH a genuine, small under-run AND a second finite
    stream in the graph.

    THE FIX: never ask ``tpad`` for ``stop = -1`` — but also never size the
    clone from a COMPUTED gap (``duration - clip_dur``), because that
    arithmetic is exactly what a LYING container header defeats
    (video_compiler #212: a header can claim more real frames than the file
    decodes). A clip that PROBES as already covering the target
    (``clip_dur >= duration``) makes ``duration - clip_dur`` read as zero or
    negative — precisely when the real gap could be the size of the WHOLE
    segment (measured in production: a 12.0s-probing clip that decoded
    2.84s). ``tests/render_harness/test_picture_covers_narration_render.py``
    pins this shape end-to-end and went red under the gap-arithmetic version
    of this function during development.

    So this asks ``tpad`` to be ready to clone for up to the FULL segment
    ``duration`` (pre-retime seconds; scaled by ``1/speed`` into the
    post-retime timeline like every other emitted time in
    ``_freeze_pad_suffix``), floored at one output frame plus
    ``_COVER_PAD_MARGIN_SECONDS``. This costs nothing extra when the probe
    was right: ``tpad``'s ``stop_duration`` only bounds how long it is
    WILLING to keep padding once the real input frames run out — it clones
    nothing at all while real frames keep arriving, and the caller's own
    ``-t`` still truncates the output well before this ceiling in the normal
    case. ``duration`` is always a known, finite value (the segment's own
    ``-t``), so asking for the whole thing can never be unbounded the way
    ``stop = -1`` was, and it is correct however wrong ``clip_dur`` turns out
    to be — the probe is removed from this decision entirely."""
    return "," + _clone_pad(_cover_pad_stop_duration(duration, fps, speed), fps)


def _clone_pad(stop_duration: float, fps: float) -> str:
    """A last-frame clone that DECLARES its own frame rate first:
    ``fps=<render_fps>,tpad=stop_mode=clone:stop_duration=<n>``.

    WHY the ``fps=`` is load-bearing and not decoration: ``tpad`` turns
    ``stop_duration`` into a clone COUNT with the frame rate DECLARED on its
    input link, and every pad here sits downstream of the ``setpts=PTS-STARTPTS``
    rebase (``_STARTPTS_REBASE``). From ffmpeg 7.x ``setpts`` declares its output
    rate UNKNOWN (it may have rewritten every timestamp) and emits frames with
    ``duration=0``, so ``tpad`` computes zero clones and passes EOF straight
    through — silently, exit 0. Measured on production's 7.1.5 with an honest 4s
    clip, a 12s narration and ``-t 12``:

        [0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=12.25
            -> 100 frames, 3.96s of picture under 12.0s of audio
        [0:v]setpts=PTS-STARTPTS,fps=25,tpad=stop_mode=clone:stop_duration=12.25
            -> 300 frames, 12.0s

    ffmpeg 4.4 / 6.1.1 (local, CI) kept the rate across ``setpts``, which is why
    every pad test was green everywhere except the image that ships. The stream
    is ALREADY on the ``render_fps`` grid at this point (the cadence fragment
    resampled it), so the extra ``fps=`` passes every frame through untouched —
    it exists only to put a known rate back on the link. Applied to EVERY clone
    (including the terminal one after the drift ``zoompan``, which happens to
    declare a rate itself) so no pad ever depends on what its upstream filter
    chose to declare. A nonsensical ``fps <= 0`` has no rate to declare and keeps
    the bare ``tpad``."""
    pad = f"tpad=stop_mode=clone:stop_duration={_ffnum(stop_duration)}"
    return f"fps={_ffnum(fps)},{pad}" if fps > 0 else pad


def _cover_pad_stop_duration(duration: float, fps: float, speed: float) -> float:
    """Shared sizing for every FINITE last-frame clone in this module (the
    single-source cover pad above, and the split-screen half below): a
    ceiling of the whole segment ``duration`` (post-retime by ``1/speed``),
    floored at one output frame plus ``_COVER_PAD_MARGIN_SECONDS`` so the
    request is never sub-frame. See ``_bounded_cover_pad`` for why this is
    sized from the segment's own known-finite duration rather than a
    probed gap — the same reasoning applies to every caller, split-screen
    included, since a split half is just another cover pad that needs its
    output EOF to arrive from the caller's own ``-t``, never from a
    ``tpad`` clone that waits forever."""
    fps_floor = 1.0 / fps if fps > 0 else _MIN_FREEZE_PAD_SECONDS
    return (max(duration, fps_floor) + _COVER_PAD_MARGIN_SECONDS) / speed


def _freeze_pad_suffix(
    narration_driven: bool,
    clip_dur: float,
    duration: float,
    settings: Settings,
    pad_cap: float | None = None,
    speed: float = 1.0,
    bounded: bool = False,
) -> str:
    """Fill a per-line interaction clip that under-runs its narration line
    (``clip_dur < duration``). ``tpad`` clones the last frame to extend the
    stream to ``duration``; a gentle drift ``zoompan`` then eases the zoom over
    ONLY the padded tail so the cloned frames stop being identical (no dead
    freeze — T7/AC#5). Identity over the live region, so a clip that already
    fills its narration is byte-identical (empty return). Chained with ``,`` so
    it stays inside the base ``[0:v]`` chain.

    ``pad_cap`` (freeze-pad Step B, opt-in) bounds the END of the drift ramp: when
    set and ``clip_dur < pad_cap < duration``, the zoom eases over ``clip_dur →
    pad_cap`` (the clip length + freeze-pad ceiling) and then holds its settled
    value for the tail out to ``duration``. The ``tpad`` still clones to the FULL
    ``duration`` so the video always covers the narration — no black, no truncated
    audio — only the drift is capped at the ceiling. ``None`` (default) ramps over
    the whole tail (``pad_cap == duration``), byte-identical to before.

    ``speed`` is the shot's playback factor. This suffix is emitted UPSTREAM of the
    ``setpts=PTS/speed`` retime that ``_apply_shot_speed`` prepends, so its ``tpad``
    and drift ramp run on the POST-retime timeline (where the clip lasts
    ``clip_dur/speed`` and the segment ``duration/speed``). The trigger guard and
    the ramp-cap decision stay in REAL seconds (whether a clip under-runs its
    narration, and where the ceiling sits, are speed-independent); only the EMITTED
    filter times are scaled by ``1/speed`` so the frozen tail lands at the retimed
    length. At ``speed==1.0`` every scaled time is unchanged → byte-identical.

    ``bounded`` (the caller's ``trim``, i.e. "this segment carries an output-side
    ``-t``") appends a FINITE terminal clone (see ``_bounded_cover_pad``) sized
    to the gap between ``duration`` and whatever real picture is available,
    after the drift. THE PLANNED PAD ABOVE IS DECIDED FROM A PROBED NUMBER, and
    every consumer of that number inherits whatever the probe got wrong: a
    container whose header outlasts the frames it actually holds (a truncated
    download, a Chromium/Matroska capture, a fragmented writer) reports
    ``clip_dur >= duration``, reads as "already fills its narration", and emits
    only the FLOOR-sized clone below — while the ``-t`` keeps the segment at the
    planned length, so the delivered container duration AGREES with the plan
    and the picture simply stops. Measured on this shape: a 12.0s-probing clip
    that decodes 2.84s under a 12.0s narration delivered 2.84s of picture under
    12.0s of audio. ``_bounded_cover_pad`` fixes this too, DELIBERATELY: it does
    not size the clone from ``duration - clip_dur`` (a lying header can make
    that read as zero), it asks for a ceiling of the WHOLE ``duration`` instead
    — see its docstring — so the terminal clone alone is what makes coverage
    correct here, not just non-infinite.

    The terminal clone removes the probe from the coverage decision entirely: it
    is a concrete finite number (the segment's own ``duration``), so it can
    never diverge the way ``stop = -1`` did — the ``-t`` upstream still caps the
    segment exactly as before. It is NOT emitted on the unbounded own-audio clip
    branch (``trim=False``, no ``-t``), where any last-frame clone — finite or
    not — would serve no purpose (nothing needs the tail filled)."""
    fps = settings.render_fps
    if not (narration_driven and 0.0 < clip_dur < duration):
        return _bounded_cover_pad(duration, fps, speed) if bounded else ""
    # Bound the drift ramp's end at the ceiling when a valid cap is given; else it
    # ramps over the whole pad (today's behaviour). A cap outside (clip_dur, duration)
    # is meaningless and falls back to the uncapped ramp. Decided on RAW seconds —
    # the ceiling semantics do not move with speed.
    ramp_end = pad_cap if pad_cap is not None and clip_dur < pad_cap < duration else duration
    w, h = settings.render_width, settings.render_height
    # Scale the emitted times into the post-retime timeline (1/speed). At speed==1.0
    # cd/dur/re equal clip_dur/duration/ramp_end so the string is byte-identical.
    # (``re_end``, not ``re`` — the stdlib ``re`` module is imported at module scope.)
    cd, dur, re_end = clip_dur / speed, duration / speed, ramp_end / speed
    # ⛔ The guard above tests the CLIP's strict positivity (0 < clip_dur < duration),
    # never the PAD's magnitude — and the pad is ``duration - clip_dur``, a difference
    # of two PROBED numbers with no lower bound. ``clip_dur = duration - 4.2e-05``
    # sails through it and asks ffmpeg to clone 42 microseconds of frame.
    #
    # Below one OUTPUT FRAME that request is not merely useless, it is unsatisfiable:
    # ``tpad`` clones whole frames, so a ``stop_duration`` under ``1/fps`` cannot put a
    # single additional frame on the output. One frame is therefore the floor at which
    # the filter still has something to DO — derived from the mechanism rather than
    # picked as a round number, and it scales with the cadence instead of going stale
    # when ``render_fps`` moves. The drift ``zoompan`` is skipped with it: its whole job
    # is to keep the cloned tail from looking frozen, and there is no cloned tail.
    #
    # Nothing is given up by skipping the drift/zoompan below. On a BOUNDED segment
    # the output-side ``-t`` sets the length and the FINITE terminal clone
    # (``_bounded_cover_pad``, ceiling'd at the whole segment ``duration``) holds
    # the last frame out to it, so coverage is unchanged; on an UNBOUNDED one the
    # shortfall is a sub-frame residue the encoder rounds away. The hardened
    # ``_ffnum`` would now emit ``0.000042`` rather than crash, so this is the
    # second line of defence — a degenerate DRIFT filter is worse than no drift,
    # and it is better not to ask the zoompan ramp for something it cannot
    # deliver. This does NOT skip the terminal clone itself on a bounded segment:
    # that clone must still cover for the whole segment (see
    # ``_bounded_cover_pad``), it only skips the (pointless, sub-frame) drift
    # ramp on top of it.
    if dur - cd < (1.0 / fps if fps > 0 else _MIN_FREEZE_PAD_SECONDS):
        return _bounded_cover_pad(duration, fps, speed) if bounded else ""
    ramp = _smoothstep_expr("on", cd * fps, re_end * fps)
    z = f"1+{_ffnum(_DRIFT_ZOOM - 1.0)}*{ramp}"
    # The x/y expressions here are RELATIVE to iw/ih, so the _ZOOMPAN_PRESCALE
    # in front needs no coordinate fix-up (unlike the focus punch's bbox centres).
    zoompan = (
        f"{_zoompan_prescale(w, h, settings)},"
        f"zoompan=z='{z}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={w}x{h}:fps={_ffnum(fps)}"
    )
    drift = f",{_clone_pad(dur - cd, fps)},{zoompan}"
    # The drift's OWN clone pads to a number the probe produced (``dur - cd``):
    # if the clip decodes shorter than it probed, that lands short of the
    # ``-t`` too. A second, FINITE terminal clone — ceiling'd at the WHOLE
    # segment ``duration``, same as every other ``_bounded_cover_pad`` call —
    # closes that residue regardless of how wrong the probe was, chained
    # after the drift's tpad so it only ever activates if the drift's own
    # clone still leaves the picture short.
    return drift + _bounded_cover_pad(duration, fps, speed) if bounded else drift


# ---------------------------------------------------------------------------
# Per-clip playback speed (setpts video + atempo audio)
# ---------------------------------------------------------------------------
def _setpts_speed_fragment(speed: float) -> str:
    """The ``setpts`` video fragment that retimes a shot to ``speed``x.

    Empty (no filter) at ``speed==1.0`` so the default path stays byte-identical.
    ``:g`` formatting keeps whole factors integer (``2.0`` → ``setpts=PTS/2``).
    """
    if speed == 1.0:
        return ""
    return f"setpts=PTS/{speed:g}"


def _apply_speed_to_vf(video_vf: str, speed: float) -> str:
    """PREPEND the ``setpts`` retime to a shot's video filter body.

    Prepending (rather than appending) matters: the source is retimed BEFORE the
    master-cadence ``fps`` filter later in the chain, so that ``fps`` resamples
    the sped-up source to the output rate and the delivered frame COUNT actually
    changes. A trailing ``setpts`` would only re-stamp timestamps, leaving the
    frame count — and thus the visible motion smoothness — unchanged.
    """
    frag = _setpts_speed_fragment(speed)
    return f"{frag},{video_vf}" if frag else video_vf


def _atempo_chain(speed: float) -> str:
    """The ``atempo`` audio-filter chain that retimes audio to ``speed``x.

    A single ``atempo`` instance only accepts a factor in [0.5, 2.0], so a factor
    outside that range is decomposed into a product of in-range factors
    (4.0 → ``atempo=2,atempo=2``; 0.25 → ``atempo=0.5,atempo=0.5``). Empty at
    ``speed==1.0`` (no-op). The ``Shot.speed`` bounds [0.25, 4.0] need at most two
    stages, but the loop generalises to any factor.
    """
    if speed == 1.0:
        return ""
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:g}" for f in factors)


def _apply_shot_speed(
    video_vf: str, speed: float, duration: float, trim_args: list[str], retime_audio: bool = True
) -> tuple[str, str, float, list[str]]:
    """Fold per-clip ``speed`` into a shot's render args.

    Returns ``(video_vf, atempo, out_duration, trim_args)``: the video filter
    with the ``setpts`` retime prepended, the audio ``atempo`` chain, the retimed
    segment length (``duration/speed``), and the ``-t`` trim bound rescaled to it.
    ``speed==1.0`` returns every input unchanged (empty ``atempo``) so the default
    render path stays byte-identical.

    ``retime_audio=False`` keeps the VIDEO retime and suppresses ONLY the
    ``atempo`` chain. Its one caller is the synthesised-silence case: ffmpeg's
    ``atempo`` does NOT map an all-zero buffer to an all-zero buffer — fed
    ``anullsrc`` it emits a CONSTANT DC OFFSET (measured -0.011, ~-39 dBFS, on
    ffmpeg 4.4; reproduce with ``ffmpeg -f lavfi -i anullsrc=… -af atempo=1.25
    -t 3 out.wav``). Retiming silence is a semantic no-op anyway (``anullsrc`` is
    infinite and the output ``-t`` sets the length), so the right move is not to
    emit the chain. The DC is NOT cosmetic: it survives the stream-copy concat
    into the film bus, the ``loudnorm`` master then LIFTS it ~17 dB, and it rides
    ``[nar_key]`` — the sidechain key that drives the music ducker. On a
    music-only action reel (every shot muted, most shots retimed) it is the
    entire low end of the delivered file.
    """
    if speed == 1.0:
        return video_vf, "", duration, trim_args
    out_duration = duration / speed
    new_trim = ["-t", f"{out_duration}"] if trim_args else trim_args
    atempo = _atempo_chain(speed) if retime_audio else ""
    return _apply_speed_to_vf(video_vf, speed), atempo, out_duration, new_trim


# Unconditional per-shot edge fade. 8 ms ~= 384 samples at 48 kHz: long enough to
# reach true zero from any sample value, short enough to be inaudible as an
# envelope (a level ramp becomes perceptible as a "fade" around ~20 ms).
_SHOT_EDGE_FADE_S = 0.008


def _apply_shot_audio(audio: ShotAudio | None, duration: float) -> str:
    """Comma-joined ``afade`` fragment for a shot's PRIMARY audio, injected at its
    label in _audio_sfx_graph BEFORE the SFX amix/atempo.

    ONLY the fades are baked per-shot (pre-master): ``afade`` has no timeline
    ``enable=`` support, so it must ride with its segment. Per-clip GAIN/MUTE are
    deliberately NOT applied here — the default-ON dynamic loudness master
    (single-pass ``loudnorm`` over the whole bus) would flatten a pre-master gain
    back to target. They move to a POST-master, time-ranged pass keyed to each
    shot's absolute bus offset (see ``_postclip_filter`` / ``_finalize_audio``).

    Timing uses the shot's PRE-SPEED duration; atempo later rescales the baked
    envelope. Leading comma so it chains after _A48.

    EVERY shot gets an edge fade, author-specified or not. Until now this returned
    "" for ``audio is None`` — i.e. for MOST shots, since gain/mute/fades are all
    optional — so the overwhelming majority of segments began and ended on
    whatever sample the narration happened to be at. Joined by the sample-exact
    PCM concat, a boundary between two non-zero samples is a step discontinuity:
    a click, exactly where the old AAC pipeline used to put a 24 ms hole. The
    ``_SEGMENT_AUDIO_CODEC_ARGS`` change removes the hole; this removes the step.
    ``_SHOT_EDGE_FADE_S`` (8 ms ~= 384 samples at 48 kHz) is below the ~20 ms
    threshold at which a level ramp is audible as a fade, so it is inaudible as an
    envelope while being long enough to land the boundary on silence. An
    author-supplied ``fade_in_s``/``fade_out_s`` always wins — it is a deliberate
    creative choice and is never shortened to the edge fade.

    A shot too short to hold both edges (``duration <= 2 * _SHOT_EDGE_FADE_S``)
    gets no implicit fade: overlapping ramps would dip the whole shot rather than
    just its edges.
    """
    fade_in = _SHOT_EDGE_FADE_S
    fade_out = _SHOT_EDGE_FADE_S
    if duration <= 2 * _SHOT_EDGE_FADE_S:
        fade_in = fade_out = 0.0
    if audio is not None:
        if audio.fade_in_s > 0:
            fade_in = audio.fade_in_s
        if audio.fade_out_s > 0:
            fade_out = audio.fade_out_s
    # ⛔ CLAMP, never DROP. An author ``fade_out_s`` LONGER than the shot used to
    # take the `st = duration - fade_out; if st >= 0` branch and emit NOTHING —
    # and because the author value has already overwritten ``_SHOT_EDGE_FADE_S``
    # above, the shot then got NO fade at all, not even the implicit 8 ms one. The
    # segment boundary therefore landed on an arbitrary non-zero sample: exactly
    # the step-discontinuity click the edge fade exists to remove, reintroduced by
    # asking for MORE fade. Clamping to ``duration`` honours the intent (fade the
    # whole shot) and always lands the boundary on silence. ``fade_in`` is clamped
    # for the same reason — an over-long ``fade_in_s`` would otherwise ramp past
    # the end of the segment and leave the tail fading up from nothing.
    fade_in = min(fade_in, duration)
    fade_out = min(fade_out, duration)
    parts: list[str] = []
    # ``_ffdur``, not a bare float: ``ShotAudio.fade_in_s``/``fade_out_s`` are bounded
    # ``[0, 30]`` with NO exclusive lower bound, so an author may legitimately send
    # 4.2e-05 — it passes ``> 0``, and a bare f-string interpolates a float via ``repr``,
    # which turns to scientific notation below 1e-4 exactly as ``%g`` does. ``afade``'s
    # ``d``/``st`` are AV_OPT_TYPE_DURATION, so that is the crash shape, not a cosmetic
    # one. The ``min(…, duration)`` clamps above make it reachable a second way: a
    # sub-100µs shot drives BOTH fades under the threshold whatever the author asked for.
    if fade_in > 0:
        parts.append(f"afade=t=in:st=0:d={_ffdur(fade_in)}")
    if fade_out > 0:
        parts.append(f"afade=t=out:st={_ffdur(max(0.0, duration - fade_out))}:d={_ffdur(fade_out)}")
    return "," + ",".join(parts) if parts else ""


# A48 = 48 kHz stereo, the uniform per-segment audio format. Every SFX input is
# re-formatted to it before mixing so a mono whoosh / 44.1 kHz chime lines up
# with the AAC-48k narration (adelay needs the channel count to emit per-channel
# delays; amix needs matching layouts).
_A48 = "aformat=channel_layouts=stereo:sample_rates=48000"

# `_A48` names a rate but no RESAMPLER, so when the rate actually changes ffmpeg
# uses swr's default cubic interpolator. Cubic's stopband is shallow: on an
# UPSAMPLE it leaves the source's mirror image in the new top octave. This form
# pins soxr at precision=28, whose stopband is below the noise floor, and is the
# one to use wherever the rate can actually change.
#
# ⛔ THIS COMMENT USED TO READ "THIRD-PARTY ASSETS ONLY … the per-shot primary
# already arrives 48 kHz stereo, so `_A48` there is a genuine no-op". BOTH HALVES
# WERE FALSE and the claim is what stopped the previous investigation:
#   - `tts_backend` renders narration at **24000 Hz MONO**, hardcoded
#     (`src/services/tts_service.py` `_TORCH_SAMPLE_RATE`/`sr`/`_sample_rate`), and
#     the orchestrator's premaster pins its output back to the probed input rate,
#     so the per-shot primary arrives at 24 kHz — a REAL conversion, not a no-op.
#     (This module's own `_concat_audio_filtered` defaults agree: `24000, 1`.)
#   - Measured 2026-09-13, 24k mono -> 48k stereo, cubic vs soxr on a 2 s clip:
#     identical below 11.5 kHz (delta 0.00 dB), but cubic leaves the mirror image
#     at **-70.7 dBFS in 12-16 kHz against a -20.9 dBFS program (~-50 dBr)**,
#     where soxr leaves -180 dBFS — a 110 dB difference. soxr costs nothing in the
#     passband (it measures 0.3-0.9 dB HIGHER at 10.5-11.5 kHz; only the 11.5-12
#     kHz transition band, already ~60 dB down, drops 8 dB).
#
# ⚠️ It is a no-op TODAY at the master/postclip `[0:a]` sites, where the input is
# the concatenated segment stream that `_SEGMENT_AUDIO_CODEC_ARGS` already forced
# to 48 kHz stereo — `aresample=48000` on 48 kHz input is a byte-identical
# pass-through (verified). But "safe because nothing changes the rate" was true of
# the per-shot primary too, right up until it wasn't (see above). 4d (wave 8)
# swapped every bare `_A48` at those sites — `_master_narration_mix`'s
# `[nar_mix]`/`[nar_key]` split, both `master_audio` filter strings, and
# `apply_postclip_audio` — to `_A48_SOXR`: latent-proofing an invariant nothing
# in this module asserts, not a behaviour change (identical output, verified).
#
# ⚠️ The 192 kHz -> 48 kHz DECIMATION inside `_master_filter` (the `aformat` ahead
# of `alimiter`) was the intuitive suspect and was MEASURED AND LEFT ALONE: cubic
# vs soxr there is 0.00 dB across every band from 0 to 20 kHz, because loudnorm's
# 192 kHz bus carries nothing above 30 kHz (-121 dBFS) for cubic to alias down.
# Only 20-24 kHz moves (~0.5 dB), above audibility and discarded by the AAC
# delivery encode. Do not "fix" it without re-measuring.
_A48_SOXR = f"aresample=48000:resampler=soxr:precision=28,{_A48}"

# Historical alias: the SFX / ambient / music-bed paths. Library assets are
# overwhelmingly 44.1 kHz and 44.1 -> 48 is a NON-INTEGER ratio, so they were the
# first callers of the soxr form. Same string, kept as a separate name because the
# call sites read better for it.
_A48_ASSET = _A48_SOXR

# swscale flags for EVERY resize in this module. Unset, swscale runs its BICUBIC
# default. Shots routinely arrive under-sized and are scaled UP to the render
# canvas — a real delivered demo_video recorded 1920x1080 and mastered at
# 2560x1440 — and that upscale is where the scaler shows. Measured on a frame of
# that master's own source clip, variance-of-Laplacian (higher = more retained
# detail) for the 1920 -> 2560 upscale: bilinear 28.22, bicubic (today) 43.33,
# spline 45.83, **lanczos 47.57** — ~+9.8% over the shipped default for ~1-3% CPU.
# ``accurate_rnd`` + ``full_chroma_int`` measure neutral on luma (47.54) and are
# kept for chroma correctness, which that luma metric cannot see.
_SWS_FLAGS = "flags=lanczos+accurate_rnd+full_chroma_int"

# swscale's in/out colour matrix for the RGB->YUV conversion every ``scale=``
# node performs. Left unset, swscale defaults to BT.601 — but every encode is
# TAGGED bt709 (``mastering.colorspace_tag_args``, additive ``-colorspace``/
# ``-color_primaries``/``-color_trc`` at encode time), so an untagged scale
# node silently mismatches the container's own colr atom: RGB-origin content
# that actually PASSES THROUGH a ``scale=`` node (generated stills, slide
# PNGs, fitted overlays — ``_scale_pad_filter``/``_split_filter``/
# ``_scroll_filter``/``_overlay_clip_chain(scale="fit")``) gets converted with
# the WRONG matrix and then labelled with the RIGHT one.
#
# ⚠️ This does NOT cover drawtext/drawbox/color= (captions, badges, text-card
# backgrounds). Those filters draw straight onto an already-YUV frame via
# ffmpeg's ``drawutils``, which hardcodes BT.601 with no matrix option of its
# own, and the text-card branch in particular never traverses a ``scale=``
# node at all. Verified on ffmpeg 4.4:
# ``drawbox=color=red:t=fill,format=yuv420p`` -> Y=0x51 (BT.601 red luma)
# BOTH with and without a trailing ``scale=...:out_color_matrix=bt709`` — the
# matrix argument makes no measurable difference because the frame is never
# rescaled. Fixing that class requires a different mechanism (pre-converting
# the drawn color, or post-processing with ``colorspace=``), not this dict.
# Measured through the repo's verbatim chain,
# decoded back honouring the tag, vs the original RGB (mean per-pixel RGB
# distance): red patch 22.16 -> 3.17 (7.0x), teal 18.38 -> 2.48 (7.4x), orange
# 8.52 -> 2.12 (4.0x), navy 5.79 -> 1.26 (4.6x), red text (thin/AA) 6.81 ->
# 4.34 (1.6x). Control: the identical A/B on already-yuv420p footage input
# (screen capture, never RGB-origin) produced MD5-IDENTICAL output — a true
# no-op there, proven again in
# ``test_sws_color_matrix_noop_on_yuv420p_footage``.
#
# The matrix name is DERIVED from ``mastering.colorspace_tag_args`` — the same
# accessor that stamps the ``-colorspace`` tag — never hardcoded, so a job
# whose ``render_colorspace`` retargets away from bt709 keeps the scale and
# the tag in agreement. See ``_sws_color_matrix``.
_SWS_MATRIX_FOR_TAG: dict[str, str] = {
    # Keyed by the exact ``-colorspace`` value ``colorspace_tag_args`` emits
    # (``mastering._COLORSPACE_TRIPLETS``' third element) -> the name
    # swscale's ``in_color_matrix``/``out_color_matrix`` actually accepts
    # (``auto``/``bt601``/``bt470``/``bt709``/``fcc``/``smpte240m``/``bt2020``
    # — a DIFFERENT, narrower vocabulary than ``-colorspace``'s).
    "bt709": "bt709",  # bt709 tag, and srgb's tag (also "bt709")
    "smpte170m": "bt601",  # bt601 tag — smpte170m IS bt601's matrix, renamed
    "bt2020nc": "bt2020",  # bt2020 tag
}


def _sws_color_matrix(settings: Settings) -> str:
    """The swscale matrix name matching the colour tag this job's encode will
    carry. Falls back to ``bt709`` (the fleet default, and ffmpeg's own
    default output tag) only when ``render_colorspace`` is disabled/unknown —
    i.e. when ``colorspace_tag_args`` itself would emit no tag at all."""
    from src.services import mastering

    tag_args = mastering.colorspace_tag_args(settings)
    # tag_args is ["-color_primaries", P, "-color_trc", T, "-colorspace", M]
    # or [] when disabled/unknown; M (the last element) is the exact
    # -colorspace value the encoder will tag onto the delivered stream.
    matrix_tag = tag_args[-1] if tag_args else "bt709"
    return _SWS_MATRIX_FOR_TAG.get(matrix_tag, "bt709")


def _sws_flags(settings: Settings) -> str:
    """``_SWS_FLAGS`` plus the explicit in/out colour matrix (see
    ``_sws_color_matrix``) so every ``scale=`` node's RGB->YUV conversion
    agrees with the colr atom the encode carries."""
    matrix = _sws_color_matrix(settings)
    return f"{_SWS_FLAGS}:in_color_matrix={matrix}:out_color_matrix={matrix}"


# ``zoompan`` is an UNFLAGGED SCALER. It resamples internally on every output
# frame (that is what the zoom IS) and exposes no ``flags`` option at all, so
# swscale runs its BILINEAR default there — the worst of the four measured in
# ``_SWS_FLAGS`` above (28.22 vs lanczos' 47.57 variance-of-Laplacian). That
# constant only ever reaches explicit ``scale=`` nodes, so every ken-burns drift
# and every focus punch-in has been upscaling bilinear regardless of it.
#
# The standard fix is to hand zoompan a frame that is ALREADY larger than its
# output, with an explicit lanczos ``scale=``: zoompan's own resampler then only
# ever shrinks (where the kernel barely matters) instead of magnifying. 2x the
# target covers the zoom range both call sites can reach — ``_CAPTURE_HEADROOM``
# for the focus punch and ``_DRIFT_ZOOM`` for the drift — so no output frame is
# ever built from fewer source pixels than it has.
#
# ⚠️ It changes zoompan's INPUT COORDINATE SPACE. ``iw``/``ih`` double, so any
# x/y expression written in absolute source pixels (the focus punch's bbox
# centres) must be scaled by the SAME factor. Expressions written relative to
# ``iw``/``ih`` (the drift's ``iw/2-iw/zoom/2``) are unaffected.
_ZOOMPAN_PRESCALE = 2


def _zoompan_prescale(w: int, h: int, settings: Settings) -> str:
    """The explicit lanczos ``scale=`` that must precede a ``zoompan=…:s={w}x{h}``
    so zoompan's flagless internal resampler downsizes instead of upsizing."""
    return f"scale={w * _ZOOMPAN_PRESCALE}:{h * _ZOOMPAN_PRESCALE}:{_sws_flags(settings)}"


# The four loudnorm pass-1 JSON fields pass 2 needs to run in LINEAR mode.
_MASTER_MEASURE_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh")

# Pass-1 fields that are EVIDENCE ONLY — recorded into the compile report so a
# reader can reconstruct why loudnorm granted or refused `linear`, but never fed
# into pass 2's filter. Deliberately NOT in `_MASTER_MEASURE_KEYS`: that tuple is
# a hard requirement (a missing or non-finite member aborts the two-pass and
# fails open to the dynamic chain), and a missing `target_offset` must not be
# able to cost a render its linear master. Absent ⇒ simply unreported.
_MASTER_EVIDENCE_ONLY_KEYS = ("target_offset",)

# The pass-1 readings that DECIDE the linear/dynamic outcome, in report-field
# order. `input_thresh` is excluded on purpose: it is required by pass 2's filter
# but plays no part in either refusal condition, so it is not evidence.
_MASTER_EVIDENCE_KEYS = ("input_i", "input_tp", "input_lra", *_MASTER_EVIDENCE_ONLY_KEYS)

# dB of sample-peak headroom the master limiter aims BELOW `settings.master_tp`,
# so the delivered TRUE peak (inter-sample, which is what the mastering gate and
# every platform spec measure) lands at or under the configured ceiling rather
# than 0.07-0.13 dB above it. See `_master_filter` for the measurement.
_LIMITER_TP_HEADROOM_DB = 0.3


def _compressor_filter(settings: Settings) -> str:
    """The master bus' ``acompressor`` node, from Settings.

    Attack/release stay literal (20/250 ms — the glue envelope, not a level
    decision); threshold, ratio and makeup are the knobs that decide how hard
    the bus is squeezed and are configurable."""
    return (
        f"acompressor=threshold={_ffnum(settings.master_compressor_threshold_db)}dB"
        f":ratio={_ffnum(settings.master_compressor_ratio)}"
        ":attack=20:release=250"
        f":makeup={_ffnum(settings.master_compressor_makeup)}"
    )


def _loudnorm_filter(settings: Settings, measured: dict[str, str] | None = None) -> str:
    """The master bus' ``loudnorm`` node.

    With ``measured`` (pass-1 JSON) it runs in **linear** mode: one constant gain
    over the whole programme. Without it, it falls back to ffmpeg's single-pass
    DYNAMIC mode, which re-decides gain continuously and is what made a
    mostly-speech bus with pauses pump."""
    # `print_format=summary` is what makes the end-of-run `Normalization Type:`
    # line exist AT ALL. af_loudnorm defaults to `print_format=none`, so PASS 2
    # printed no summary, `_NORMALIZATION_TYPE_RE` matched nothing, and
    # `master_normalization_type` was `None` on every row ever recorded — an
    # observability hole, not a fallback to dynamic (`requested_linear: true`
    # already proves pass 1 succeeded). Measured locally, same command either way:
    #   without: no `Normalization Type` line in stderr at all
    #   with:    "Output Integrated: -32.8 LUFS / Output True Peak: -32.1 dBTP /
    #             Normalization Type:   Dynamic"
    # and the two output files are BYTE-IDENTICAL — the summary goes to stderr
    # only. It is applied to PASS 2 only; pass 1 keeps `print_format=json`
    # (`_master_measure_filter`), which is parsed as JSON.
    # ⛔⛔ READ THIS BEFORE "FIXING" THE INERT LINEAR REQUEST ⛔⛔
    #
    # `linear=true` below is a REQUEST that af_loudnorm is free to REFUSE, and at
    # this service's configured target pair it refuses it essentially always. The
    # refusal is silent: no warning, no non-zero exit, no change to the filter
    # string we asked for. The ONLY witness is pass 2's `Normalization Type:` line
    # (`_record_master_normalization_type`), and the first production master on
    # which that field has ever been observable — run
    # `01a0a582-2531-7145-ba7d-a1295e92b232` — reported `"dynamic"`.
    #
    # THE ARITHMETIC (af_loudnorm.c: linear is kept only while the ONE constant
    # gain it implies cannot breach the true-peak ceiling). Linear is REFUSED when
    #
    #     measured_TP + (target_I - measured_I) > target_TP
    #
    # i.e. when applying the whole integrated-loudness correction as a constant
    # offset would push the measured true peak above `TP`. Measured in isolation
    # against real ffmpeg with `I=-14:TP=-1:linear=true`:
    #
    #     measured_I=-22, measured_TP=-3  ->  -3 + 8 = +5 > -1   ->  Dynamic
    #     measured_I=-22, measured_TP=-9  ->  -9 + 8 = -1 <= -1  ->  Linear
    #
    # so the boundary is exact, not approximate. A SECOND, independent refusal
    # exists and is NOT covered by that inequality: `measured_LRA == 0.00` yields
    # Dynamic even with 23 dB of headroom. Both conditions look identical in the
    # report unless pass 1's own readings are recorded — which is why they are
    # (`_MASTER_EVIDENCE_KEYS` -> `master_input_i/_tp/_lra`, `master_target_offset`).
    #
    # ⇒ AT THE CURRENT TARGET PAIR (`master_lufs=-14`, `master_tp=-1.0`;
    # `config.py`) LINEAR IS STRUCTURALLY UNREACHABLE FOR A SPEECH BUS. This
    # service's bus arrives around -22 LUFS after the compressor, so linear needs
    # ~+8 dB of constant gain and therefore `measured_TP <= -9 dBTP` — a peak
    # headroom a compressed speech bus does not have (the real run measured
    # `input_tp` far above that). The request ships INERT while every
    # filter-string test in the suite stays green, because the string is what we
    # asked for and not what ran.
    #
    # ⛔ DO NOT "fix" this by guessing. The three tempting edits are all wrong or
    # not ours to make:
    #   - Loosening `master_tp` / raising `master_lufs` CHANGES DELIVERED AUDIO.
    #     The delivered master currently measures -14.2 LUFS / -1.2 dBTP with zero
    #     overs, so the dynamic fallback is plausibly PROTECTIVE here. Whether to
    #     move the target pair is an OWNER decision, not a code-hygiene one.
    #   - Dropping `linear=true` loses nothing today but removes the one thing
    #     that makes the refusal observable at all.
    #   - Asserting a harder `linear` (there is no such flag) or post-hoc
    #     re-normalising both re-introduce the gain riding this chain removed.
    #     (NOT the same thing as `_correct_master_loudness_miss`: that leaves the
    #     target, the mode and every master that LANDED untouched, and fires only
    #     on a master that MISSED the owner's target by more than the gate's
    #     tolerance — with ONE constant gain, not a re-normalise.)
    # The correct next step is to read `master_requested_linear` +
    # `master_input_*` off real rows and decide the target pair with evidence.
    base = f"loudnorm=I={settings.master_lufs}:TP={settings.master_tp}:LRA={settings.master_lra}"
    if not measured:
        return f"{base}:print_format=summary"
    return (
        f"{base}"
        f":measured_I={measured['input_i']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}"
        ":linear=true"
        ":print_format=summary"
    )


def _linear_refusal_predicted(settings: Settings, measured: dict[str, str] | None) -> bool:
    """Whether pass 1's readings already say loudnorm will REFUSE ``linear=true``.

    The two refusals documented (and measured against real ffmpeg) in
    :func:`_loudnorm_filter`, evaluated on the very ``measured`` strings that
    function threads into the filter and the very targets it writes:

    * ``measured_TP + (target_I - measured_I) > target_TP`` — the one constant gain
      linear implies would breach the true-peak ceiling;
    * ``measured_LRA == 0``.

    ``False`` for ``None`` (no measurements ⇒ linear is not requested, so there is
    nothing to predict) and for a reading that does not parse: an unpredicted
    downgrade keeps its WARNING, so every doubt resolves to the loud side.

    Observability only. It decides a LOG LEVEL — never the filter, the targets or
    a report field."""
    if not measured:
        return False
    try:
        measured_i = float(measured["input_i"])
        measured_tp = float(measured["input_tp"])
        measured_lra = float(measured["input_lra"])
    except KeyError, ValueError:
        return False
    return (
        measured_tp + (settings.master_lufs - measured_i) > settings.master_tp
        or measured_lra == 0.0
    )


# loudnorm's analysis frame. In DYNAMIC mode (single-pass, and every two-pass run
# ffmpeg silently demotes from ``linear=true``) it consumes its input in frames of
# exactly this length.
_LOUDNORM_FRAME_S = 0.1


def _loudnorm_frame_align(total_seconds: float | None) -> str:
    """Trailing-comma fragment placed directly BEFORE ``loudnorm`` that hands it a
    gapless bus a WHOLE number of 100 ms frames long, or ``""`` for ``None``.

    Two defects, both measured on production master 01a10e19 (ffmpeg 7.1.5) and
    reproduced on 7.1.1 — ffmpeg 4.4 shows NEITHER, its loudnorm restamps output:

    * A timestamp gap on the incoming bus is carried through to the encoder, which
      writes it into the MP4 as a hole in the audio track. ``asetpts=N/SR/TB``
      restamps by sample count, so whatever reaches loudnorm is contiguous.
    * Dynamic loudnorm DROPS the missing part of a partial last frame and jumps its
      output timestamps by the same amount — about 2.9 s before the END of the film
      (its lookahead), not at the end: a 37.610875 s bus is 376 frames + 10.875 ms,
      and 89.125 ms (= 100 - 10.875) went missing at 34.736 s, mid title card.
      Padding to the next 100 ms multiple leaves no partial frame.

    ``atrim`` before ``apad`` makes the length EXACT rather than "at least": a bus
    already longer than the target (``apad`` cannot shorten) would otherwise still
    end mid-frame. It removes nothing that ships — the target is never below the
    film's own length, and the caller trims the output back to that length (the bed
    pass with ``-t``, the no-bed pass with a trailing ``atrim``), so no film gets
    longer or shorter. The pad is digital silence past the end of the film.
    """
    if total_seconds is None:
        return ""
    frames = math.ceil(round(total_seconds / _LOUDNORM_FRAME_S, 6))
    target = f"{max(frames, 1) * _LOUDNORM_FRAME_S:.1f}"
    return f"asetpts=N/SR/TB,atrim=end={target},apad=whole_dur={target},"


def _master_measure_filter(settings: Settings, total_seconds: float | None = None) -> str:
    """Pass-1 chain: everything the measurement must see, ending in a
    ``print_format=json`` loudnorm that only PRINTS (its output is thrown away).

    The compressor is INCLUDED deliberately — it is upstream of loudnorm in the
    delivery chain, so measuring without it would hand pass 2 the loudness of a
    signal that never reaches it."""
    return (
        f"{_compressor_filter(settings)},"
        # Same alignment as pass 2 (``_master_filter``): pass 1 must measure the
        # bus pass 2 normalises.
        f"{_loudnorm_frame_align(total_seconds)}"
        f"loudnorm=I={settings.master_lufs}:TP={settings.master_tp}"
        f":LRA={settings.master_lra}:print_format=json"
    )


async def _measure_master_loudnorm(
    ffmpeg_args: list[str],
    filter_complex: str,
    out_label: str,
    tail_args: list[str] | None = None,
) -> dict[str, str] | None:
    """Pass 1 of the two-pass master: run ``filter_complex`` to ``-f null -`` and
    return loudnorm's measurement JSON fields, or ``None``.

    Mirrors ``w2v_orchestrator``'s ``audio_premaster`` two-pass shape (the
    per-clip premaster) rather than inventing a second one: measure, bail on a
    non-finite reading, then apply a constant gain.

    ⚠️ FAIL-OPEN BY DESIGN. Every failure path returns ``None`` and the caller
    falls back to the single-pass dynamic chain — the audio that ships today. A
    measurement hiccup must never fail a paid compile.

    ⚠️ The non-finite bail is NOT defensive padding. loudnorm prints
    ``input_i: "-inf"`` for a digitally silent bus; left to run, pass 2 dies with
    "Result too large" and takes the whole render with it. ``audio_premaster``
    documents the same trap (``UNMEASURABLE_LEVEL``)."""
    from src.services.loudness import _extract_json_block

    cmd = [
        "ffmpeg",
        "-hide_banner",
        *ffmpeg_args,
        "-filter_complex",
        filter_complex,
        "-map",
        out_label,
        *(tail_args or []),
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = await _run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — fail open to the dynamic chain
        _record_master_measure_failure("ffmpeg_failed", error=str(exc))
        return None
    data = _extract_json_block(stderr) or {}
    measured: dict[str, str] = {}
    for key in _MASTER_MEASURE_KEYS:
        raw = data.get(key)
        if raw is None:
            _record_master_measure_failure("incomplete", missing=key)
            return None
        try:
            value = float(raw)
        except TypeError, ValueError:
            _record_master_measure_failure("unparseable", key=key, raw=str(raw))
            return None
        if not math.isfinite(value):
            _record_master_measure_failure("unmeasurable", key=key, raw=str(raw))
            return None
        measured[key] = str(raw)
    # Record the readings that DECIDE linear-vs-dynamic into the per-render
    # evidence channel. Done only on the fully-successful path: every bail above
    # fails open to the dynamic chain, where `requested_linear` is False and there
    # is no refusal to explain — and the unmeasurable bail's own values are
    # `±inf`, which is not JSON-serialisable and would break the response model.
    _record_master_loudnorm_evidence(data)
    log.info("master.measured", **measured)
    return measured


# ffmpeg's loudnorm prints its end-of-run summary to STDERR, and the line that
# says which mode it actually ran in is `Normalization Type:   Linear|Dynamic`
# (af_loudnorm.c). Case-insensitive + tolerant of the run of spaces ffmpeg emits.
_NORMALIZATION_TYPE_RE = re.compile(r"Normalization Type:\s*([A-Za-z]+)")

# The normalization mode the LAST master pass on this task actually ran in, or
# None when no master pass ran (mastering off / no audio / a bed-less concat that
# never reached one).
#
# ⚠️ WHY THIS EXISTS. `linear=true` is a REQUEST, not a guarantee: ffmpeg silently
# falls back to its dynamic single-pass mode when the measured values cannot be hit
# with one constant gain (loudnorm checks the measured offset against its own
# range and drops `linear` without failing, warning, or changing its exit code).
# So the headline fix of this chain — one constant gain instead of audible gain
# riding — can be INERT in production while every test that asserts the filter
# STRING stays green, because the string is what we asked for and not what ran.
# Reading it back off pass-2's stderr is the only thing that can tell those apart.
#
# A ContextVar rather than a sixth element on `compile_shots`' return tuple: the
# master runs several call sites deep (`master_audio`, `overlay_audio_bed`,
# `_finalize_audio`), the value is per-render, and a plain `await` chain shares one
# context — so the route handler that assembles the mastering report reads exactly
# what its own render wrote, with no threading through four signatures.
_master_normalization_type: ContextVar[str | None] = ContextVar(
    "master_normalization_type", default=None
)

# Whether the last master pass ASKED for `linear=true` (i.e. pass 1 produced a
# usable measurement set). Rides its own ContextVar for the same reason as the
# mode above, and exists because `master_normalization_type` alone cannot be
# read: `"dynamic"` means "loudnorm REFUSED our linear request" when this is
# True, and merely "we never asked, pass 1 failed open" when it is False. The
# value was already computed at both call sites and passed into
# `_record_master_normalization_type` — but only ever as a LOG FIELD, so no run
# row has ever carried it and the two cases were indistinguishable on the record.
_master_requested_linear: ContextVar[bool | None] = ContextVar(
    "master_requested_linear", default=None
)

# Pass 1's own readings (`_MASTER_EVIDENCE_KEYS`), as floats. These are the
# INPUTS to loudnorm's refusal decision, so without them a `"dynamic"` row cannot
# be attributed: the true-peak-headroom refusal
# (`measured_TP + (target_I - measured_I) > target_TP`) and the `measured_LRA == 0`
# refusal produce byte-identical reports. See `_loudnorm_filter` for the
# arithmetic and the measurements it was derived from.
_master_loudnorm_evidence: ContextVar[dict[str, float] | None] = ContextVar(
    "master_loudnorm_evidence", default=None
)

# Whether the master's MEASURE pass (pass 1) actually FAILED, and why.
#
# ⚠️ This is a DEGRADATION, not a mode. `_measure_master_loudnorm` fails open on
# five distinct paths — ffmpeg raised, a key was missing, a key was unparseable, a
# key was non-finite, and (in `_acx_program_master`) the whole measure command
# raised — and every one of them logged at WARNING and shipped the film. So a
# paid deliverable could be mastered by the fallback chain with the run record
# carrying no trace of it at all.
#
# ⛔ It is NOT the same signal as `master_requested_linear=False`. That flag says
# "linear was not requested", which is ALSO what a `master_enabled` render with no
# audio, or an older-shaped render, reports — absence of a request, from any
# cause. This says "we tried to measure and could not", and names which of the
# five ways. Reading a degradation off the absence of a field is exactly the
# absence-of-evidence-as-evidence confusion `degraded_report` documents.
#
# ⚠️ NO PUBLIC GETTER FOR THE REASON, deliberately. ``get_master_loudnorm_evidence``
# publishes the BOOLEAN (it is already read by the compile route), and the reason
# string rides the ERROR log, which is the channel that works today. A second
# public getter nothing calls would read to the next session as a shipped report
# field when the report has no field for it — see the note on that getter.
#
# ⚠️ Do NOT read this as "the loudness target was missed". Linear loudnorm being
# refused at -14 LUFS / -1.0 dBTP is MEASURED and EXPECTED on this material, and
# "dynamic" is ffmpeg's own word for the fallback, not a defect — see
# `_master_filter`. This flag fires only when the MEASUREMENT itself did not
# complete, which is the case nothing else can observe.
_master_measure_failed: ContextVar[str | None] = ContextVar("master_measure_failed", default=None)


def reset_master_normalization_type() -> None:
    """Clear the recorded normalization type — and the linear request + pass-1
    evidence that explain it — at the start of a render.

    Without this a render whose master pass never ran (or whose stderr carried no
    summary) would report the PREVIOUS render's mode if it somehow shared a
    context — a stale value is worse than an absent one here, because the whole
    point of the channel is to be trusted. The same argument applies to the
    evidence fields, and MORE strongly: a stale `master_input_tp` would be read as
    the reason THIS render was refused linear. All three reset together or none
    of them should."""
    _master_normalization_type.set(None)
    _master_requested_linear.set(None)
    _master_loudnorm_evidence.set(None)
    _master_measure_failed.set(None)
    _master_loudness_correction.set(None)


def _record_master_measure_failure(reason: str, **fields: object) -> None:
    """Record — and LOG AS AN ERROR — that the master's measure pass could not
    complete, so the fallback chain mastered a paid deliverable.

    ⛔ ERROR, not WARNING, and that is the point of this function existing. All
    five bail paths warned and shipped; a warning on a service that logs warnings
    for routine facts (an unfetchable overlay, a missing loudnorm summary) is not a
    signal anyone acts on. The reason is recorded so the run row can say WHICH bail
    fired, not merely that one did.

    Never raises: this is an observability channel on a paid render and the
    fail-open behaviour it describes is deliberate (see
    ``_measure_master_loudnorm``). It makes the degradation visible; it does not
    change what ships."""
    _master_measure_failed.set(reason)
    log.error("master.measure_failed", reason=reason, degraded=True, **fields)


def get_master_normalization_type() -> str | None:
    """The normalization mode the last master pass ran in (``"linear"`` /
    ``"dynamic"``), or ``None`` if no master pass ran or ffmpeg printed no
    summary. Read by the compile route into the mastering report."""
    return _master_normalization_type.get()


def _record_master_loudnorm_evidence(data: dict[str, object]) -> dict[str, float]:
    """Record pass 1's refusal-deciding readings from its ``print_format=json``
    block. Returns what was recorded (possibly empty).

    Keys that are absent or non-finite are SKIPPED rather than failing: this is an
    observability channel on a paid render, and a non-finite reading cannot cross
    the JSON response boundary anyway."""
    evidence: dict[str, float] = {}
    for key in _MASTER_EVIDENCE_KEYS:
        raw = data.get(key)
        # loudnorm's JSON block types every reading as a STRING; the isinstance
        # narrowing is what makes `float()` well-typed, and it also drops a
        # structurally wrong value (list/dict/None) without an except clause.
        if not isinstance(raw, str | int | float):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            evidence[key] = value
    _master_loudnorm_evidence.set(evidence)
    return evidence


def get_master_loudnorm_evidence() -> dict[str, float | bool]:
    """Everything the record needs to answer "was linear requested, and why was it
    refused?" — ``requested_linear`` plus pass 1's ``input_i`` / ``input_tp`` /
    ``input_lra`` / ``target_offset``.

    Only keys that were actually observed are present, so a mastering-off job (and
    any older-shaped render) contributes nothing and the report keeps the shape it
    has today. Read by the compile route into the mastering report.

    ``measure_failed`` (and ``measure_failed_reason``) appear ONLY when pass 1
    actually failed — the degradation ``_record_master_measure_failure`` exists to
    surface. A clean render, and a render that never mastered, contribute neither
    key, so absence keeps meaning "nothing to report" rather than "fine"."""
    out: dict[str, float | bool] = {}
    requested = _master_requested_linear.get()
    if requested is not None:
        out["requested_linear"] = requested
    out.update(_master_loudnorm_evidence.get() or {})
    failure = _master_measure_failed.get()
    if failure is not None:
        out["measure_failed"] = True
    # The post-master miss correction (`_correct_master_loudness_miss`), present
    # ONLY when it ran — absence means the master landed within tolerance (or
    # mastering was off), never "corrected by 0".
    out.update(_master_loudness_correction.get() or {})
    return out


def _record_master_normalization_type(
    stderr: str,
    *,
    requested_linear: bool,
    master_enabled: bool = True,
    refusal_predicted: bool = False,
) -> str | None:
    """Parse pass-2's stderr for loudnorm's `Normalization Type`, record it, and
    WARN when we asked for linear and got dynamic WITHOUT having seen it coming.

    ``refusal_predicted`` is :func:`_linear_refusal_predicted` on pass 1's
    readings. A downgrade those readings already implied is the documented,
    expected outcome on a speech bus at this target pair (see
    :func:`_loudnorm_filter`) and is logged at INFO with ``predicted=True`` — as a
    warning it fired on nearly every master and so warned of nothing. WARNING is
    kept for the case that still means something: the arithmetic said linear was
    reachable and ffmpeg ran dynamic anyway. The recorded mode and every report
    field are identical on both branches.

    Returns the lowercased mode, or None when the summary is absent (an ffmpeg
    build that does not print it, a truncated stderr). Never raises — an
    observability channel must not be able to fail a paid render."""
    if not master_enabled:
        # No loudnorm on that branch, so there is no summary and nothing to observe.
        # Guarded HERE rather than at the call site so the bed path stays one
        # straight-line statement (it is already at the C901 ceiling).
        return None
    # PERSIST the request, not just log it. Recorded BEFORE the summary match on
    # purpose: "we asked for linear and could not observe the outcome" is a
    # different, and reportable, state from "we never asked" — and until now both
    # rendered as an all-None row. This is the only write of the flag; every
    # earlier use was a log field that no run row carried.
    _master_requested_linear.set(requested_linear)
    match = _NORMALIZATION_TYPE_RE.search(stderr or "")
    if match is None:
        # Not a warning: the bed path and the audio-only path both reach here, and
        # a missing summary means "could not observe", not "downgraded".
        log.info("master.normalization_type_unknown", requested_linear=requested_linear)
        return None
    mode = match.group(1).strip().lower()
    _master_normalization_type.set(mode)
    if requested_linear and mode != "linear":
        # We threaded measurements in, the filter string said `linear=true`, and
        # ffmpeg ran dynamic anyway — so the film shipped with the gain riding the
        # fix was supposed to remove. Same event either way; the LEVEL says whether
        # pass 1 had already told us it would happen.
        emit = log.info if refusal_predicted else log.warning
        emit(
            "master.normalization_downgraded",
            normalization_type=mode,
            predicted=refusal_predicted,
        )
    else:
        log.info("master.normalization_type", normalization_type=mode)
    return mode


def _master_filter(
    settings: Settings,
    measured: dict[str, str] | None = None,
    total_seconds: float | None = None,
) -> str:
    """Compressor + loudnorm + limiter master chain for the final audio bus, or
    ``""`` when disabled (``master_enabled=False``).

    ⚠️ **``measured`` is the whole point of this chain's second revision.** With
    pass-1 measurements threaded in, ``loudnorm`` runs ``linear=true`` — ONE
    constant gain across the film. Without them it runs ffmpeg's single-pass
    dynamic mode, which re-derives gain continuously; over a mostly-speech bus
    with pauses that is audible gain riding, and it is what the owner heard as
    "digital". Narration already arrives normalised to -18 LUFS per clip from
    ``tts_backend``, so there is nothing on this bus that NEEDS levelling — the
    dynamic mode was solving a problem that had already been solved upstream.
    ``None`` is the fail-open path and reproduces today's audio exactly.

    Density (compression) is the real perceived-loudness lever — a raw mix sits
    near -22 LUFS and reads quiet vs typical platform content; the compressor
    tightens dynamics, loudnorm lifts the whole thing to ``master_lufs``, and the
    limiter guards the true-peak ceiling. The compressor params are fixed (the
    proven chain); the loudnorm target and the ceiling are configurable.

    ⚠️ **The limiter runs LAST, and that ordering is load-bearing.** It used to sit
    between the compressor and loudnorm, where it could not bound the delivered
    peak at all: loudnorm then added ~5.5 dB on the way to ``master_lufs``, and its
    own true-peak limiter is only advisory in the single-pass dynamic mode this
    chain runs in. Measured on a real delivered master
    (run 01a08f66-7e66-712d-a80e-74ad92f62991): the shipped order landed
    **-0.15 dBTP against a configured -1.0 ceiling** — 0.85 dB of overshoot on
    every film. With the limiter last the same input lands **-0.90 dBTP**, at an
    unchanged -12.55 LUFS and an unchanged 2.70 LU range. The fix costs nothing
    and changes neither loudness nor dynamics; it only stops the ceiling being a
    suggestion.

    ⚠️ **``level=disabled`` is equally load-bearing.** ``alimiter``'s ``level``
    option defaults to TRUE, which auto-normalizes the output back up to full
    scale — so ``limit=`` does NOT bound the output while it is on. Measured: a
    +12 dB signal through ``alimiter=limit=0.95`` emerges at peak **1.0000**, and
    at **0.9500** only once ``level`` is disabled. The old chain therefore carried
    a limit that never bound anything.

    ``aformat`` before the limiter is NOT decoration: ``loudnorm`` emits **192 kHz**
    regardless of its input rate (the same gotcha ``audio_premaster`` documents), so
    without it the limiter would do its true-peak work at 192 kHz and the
    downstream resample back to 48 kHz could reintroduce inter-sample overshoot
    after the only thing guarding against it.

    The ceiling is derived from ``master_tp`` rather than hardcoded, so the
    configured dBTP is the value actually enforced.
    """
    if not settings.master_enabled:
        return ""
    # dBTP -> linear amplitude for alimiter's `limit`, which takes 0..1, not dB.
    #
    # ⚠️ MINUS THE HEADROOM, and that subtraction is the whole point. `alimiter`'s
    # `limit` is a SAMPLE-peak bound; the mastering gate (and every platform spec)
    # measures TRUE peak, which is the reconstructed inter-sample peak and is
    # always >= the sample peak. Feeding `master_tp` straight in therefore asks
    # for -1.0 dB of sample peak and DELIVERS a true peak slightly above it: the
    # last 33 masters measured -0.93 and -0.87 dBTP against a gate that tests
    # -1.0, i.e. every single one "failed" a check it missed by 0.07 dB.
    #
    # That is a CALIBRATION mismatch, not clipping — the limiter is doing its job
    # (fleet max was +7.38 dBTP before it existed). The fix is to aim the sample
    # ceiling low enough that the inter-sample peak still lands under
    # `master_tp`. 0.3 dB covers the measured 0.07-0.13 dB of overshoot with
    # margin and costs 0.3 dB of peak headroom nobody can hear (loudness is set
    # by loudnorm's integrated target upstream, not by the ceiling).
    limit = _limiter_linear_limit(settings)
    return (
        f"{_compressor_filter(settings)},"
        # ``total_seconds`` (the film's length) arms :func:`_loudnorm_frame_align`.
        # ``None`` — a caller that only asks "is mastering on?" — adds nothing.
        f"{_loudnorm_frame_align(total_seconds)}"
        f"{_loudnorm_filter(settings, measured)},"
        "aformat=sample_rates=48000:channel_layouts=stereo,"
        f"alimiter=limit={_ffnum(limit)}"
        f":attack={_ffnum(settings.master_limiter_attack_ms)}"
        f":release={_ffnum(settings.master_limiter_release_ms)}"
        ":level=disabled"
    )


def _limiter_linear_limit(settings: Settings) -> float:
    """``master_tp`` (dBTP) minus ``_LIMITER_TP_HEADROOM_DB``, as the linear
    amplitude ``alimiter``'s ``limit=`` wants (0..1, not dB).

    Shared by the master bus limiter (``_master_filter``) and every per-shot
    safety limiter (``_audio_sfx_graph``) so a single configured ceiling governs
    both, instead of the bus deriving its ceiling from ``master_tp`` while a
    per-shot mix independently hardcodes its own (0.95, i.e. -0.45 dBFS,
    unrelated to any setting). Bench (4a, wave 8): swapping the per-shot
    ``alimiter`` for an ``acompressor`` was measured to NOT bound output to full
    scale — a 0.9+0.9 amplitude two-tone sum (true peak +5.10 dBFS unprocessed)
    still left the compressor at +3.59 dBFS even at its maximum ratio (20:1) with
    peak detection, because a ratio-based compressor has no hard ceiling. Only a
    real limiter (`alimiter`, uncapped ratio) reliably held the same input to
    -0.45 dBFS. So 4a keeps `alimiter` per shot (the ONLY filter in this codebase
    proven to bound sample peak) and fixes the actual defect named in the audit
    — the ceiling being disconnected from `master_tp` — by deriving it here
    instead of replacing the filter type.
    """
    return min(max(10.0 ** ((settings.master_tp - _LIMITER_TP_HEADROOM_DB) / 20.0), 0.0625), 1.0)


# ── Post-master loudness MISS correction ──────────────────────────────────────
#
# ⛔ THE DEFECT. At the configured target pair loudnorm refuses `linear` on
# essentially every film (see `_loudnorm_filter`) and runs its DYNAMIC mode. That
# mode decides its starting gain from the first ~3 s window: when the film OPENS
# on silence, that window is below the gate, the gain starts at 0 dB and only
# creeps up while speech plays. A film that opens on a long silent stretch
# therefore ships several LU under target. Measured on real ffmpeg 7.1.5 (the
# prod major) through `master_audio` itself, synthetic speech-like bus at a pass-1
# `input_i` of -23.05 (prod: ~-22.85), dynamic mode in every case:
#
#     15 s, first 65% silent           -> -21.8 LUFS  (7.8 LU under)
#     15 s, first 10% silent (control)  -> -14.9 LUFS
#     15 s, 65% silent at the TAIL      -> -15.1 LUFS
#     15 s, 65% silent as 3 s gaps      -> -15.1 LUFS
#     60 s, first 65% silent           -> -17.1 LUFS  (more speech = more ramp)
#
# i.e. the silence RATIO is a proxy; the cause is silence at the START. Prod:
# the three quiet masters (-20.73, -20.73, -21.69) are the only ones over 0.6
# silence; every other master lands -14.03…-14.52.
#
# THE FIX DOES NOT TOUCH THE TARGET, THE MODE OR ANY MASTER THAT LANDED. It
# measures the delivered master and, ONLY when it missed `master_lufs` by more
# than the mastering gate's own tolerance, re-renders pass 2 ONCE with one
# CONSTANT gain (`volume`, no riding) plus the same true-peak limiter the chain
# already ends in. Never loops. Recorded in the mastering report
# (`master_loudness_correction_db`), never silent.

# Beyond this the "miss" is not an under-corrected speech bus but something
# pathological (a near-silent hum, a broken mix) that a blind constant gain would
# only make worse. Logged and left alone; the mastering gate still reports it.
_MAX_LOUDNESS_CORRECTION_DB = 12.0

# The correction the last master pass applied, or None when none ran. Same
# ContextVar channel and reasons as `_master_loudnorm_evidence`.
_master_loudness_correction: ContextVar[dict[str, float] | None] = ContextVar(
    "master_loudness_correction", default=None
)


def _loudness_correction_db(measured_lufs: float | None, settings: Settings) -> float | None:
    """The constant gain (dB) that brings a master that MISSED its target back to
    ``master_lufs``, or ``None`` when no correction may run.

    ``None`` when: mastering is off, the master is unmeasurable, it is already
    within ``mastering.LOUDNESS_GATE_TOLERANCE_LU`` (the gate's own tolerance, so
    the correction fires exactly on the masters the gate would convict), or the
    miss exceeds ``_MAX_LOUDNESS_CORRECTION_DB``."""
    from src.services import mastering

    if not settings.master_enabled or measured_lufs is None or not math.isfinite(measured_lufs):
        return None
    gain = settings.master_lufs - measured_lufs
    if abs(gain) <= mastering.LOUDNESS_GATE_TOLERANCE_LU:
        return None
    if abs(gain) > _MAX_LOUDNESS_CORRECTION_DB:
        log.warning(
            "master.loudness_correction_refused",
            measured_lufs=measured_lufs,
            target_lufs=settings.master_lufs,
            gain_db=gain,
        )
        return None
    return gain


def _loudness_correction_chain(gain_db: float, settings: Settings) -> str:
    """``volume`` (ONE constant gain) + the master's own true-peak limiter, appended
    after ``_master_filter``. Same ceiling and same ``level=disabled`` as the bus
    limiter, for the same reasons (see ``_master_filter``)."""
    return (
        f"volume={gain_db:.2f}dB,"
        f"alimiter=limit={_ffnum(_limiter_linear_limit(settings))}"
        f":attack={_ffnum(settings.master_limiter_attack_ms)}"
        f":release={_ffnum(settings.master_limiter_release_ms)}"
        ":level=disabled"
    )


# The read-only measure of a DELIVERED master the miss correction decides on.
# loudnorm's `input_i` is the same EBU R128 integrated reading the mastering gate
# takes (`loudness.final_mux_loudness`). A command of its OWN, not that function:
# `final_mux_loudness` is also the narration measure of the VO-relative ducker,
# and sharing it would make "did a narration measurement run?" unanswerable.
_DELIVERED_LOUDNESS_FILTER = "loudnorm=print_format=json"


async def _measure_delivered_lufs(path: Path) -> float | None:
    """Integrated loudness (LUFS) of ``path``'s first audio stream, or ``None`` on
    any failure or a non-finite reading (digital silence). Never raises."""
    from src.services.loudness import _extract_json_block

    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0"]
    cmd += ["-af", _DELIVERED_LOUDNESS_FILTER, "-f", "null", "-"]
    try:
        stderr = await _run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — measurement is best-effort
        log.warning("master.delivered_measure_failed", path=str(path), error=str(exc))
        return None
    raw = (_extract_json_block(stderr) or {}).get("input_i")
    try:
        value = float(raw) if isinstance(raw, str | int | float) else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


async def _correct_master_loudness_miss(
    out_path: Path,
    settings: Settings,
    build_cmd: Callable[[str, Path], list[str]],
) -> None:
    """Measure the delivered master; on a miss, re-render pass 2 ONCE with the
    correction chain appended and re-measure.

    ``build_cmd(extra_chain, dest)`` must return the exact pass-2 command with
    ``,{extra_chain}`` appended to the master chain, writing to ``dest``. The
    re-render goes to a sibling file that replaces ``out_path`` only on success,
    so a failed correction ships the uncorrected master rather than a broken one.

    Fail-open like every other master step: a measurement or render failure is
    logged and the uncorrected master ships (the mastering gate still sees it)."""
    if not settings.master_enabled:
        return
    pre_lufs = await _measure_delivered_lufs(out_path)
    gain = _loudness_correction_db(pre_lufs, settings)
    if gain is None or pre_lufs is None:
        return
    tmp_path = out_path.with_name(f"{out_path.stem}.loudfix{out_path.suffix}")
    try:
        await _run(build_cmd(_loudness_correction_chain(gain, settings), tmp_path))
        tmp_path.replace(out_path)
    except Exception as exc:  # noqa: BLE001 — fail open: ship the uncorrected master
        tmp_path.unlink(missing_ok=True)
        log.error("master.loudness_correction_failed", gain_db=gain, error=str(exc))
        return
    post_lufs = await _measure_delivered_lufs(out_path)
    record: dict[str, float] = {
        "loudness_correction_db": round(gain, 2),
        "loudness_pre_correction_lufs": pre_lufs,
    }
    if post_lufs is not None:
        record["loudness_post_correction_lufs"] = post_lufs
    _master_loudness_correction.set(record)
    log.warning("master.loudness_corrected", **record, target_lufs=settings.master_lufs)


def _fixed_point(value: float) -> str:
    """``value`` as a plain decimal string that can NEVER carry an exponent.

    Nanosecond resolution (9 dp) — three orders finer than ffmpeg's own
    ``AV_TIME_BASE`` microsecond granularity, so nothing ffmpeg can represent is
    lost — with the dead trailing zeros stripped. Anything that underflows 1 ns
    (and the ``-0`` it can strip down to) normalises to ``"0"``. Large magnitudes
    come out in full digits rather than ``1e+20``.
    """
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "0", "-0") else text


def _ffnum(value: float) -> str:
    """Render a float as a plain ffmpeg filter argument.

    ffmpeg wants bare numbers in filter args; a Python float would emit
    ``attack=30.0`` / ``f=150.0``. ``%g`` drops the dead ``.0`` (30.0 → ``30``)
    while keeping real fractions (0.03 → ``0.03``).

    ⛔ THIS DOCSTRING USED TO CLAIM "the schema's ge/le bounds keep every value
    well clear of ``%g``'s scientific-notation range". THAT WAS FALSE, and the
    false confidence is what shipped the crash. ``%g`` switches to scientific
    notation below ``1e-4`` (and at/above ``1e+6``), and the bounds argument only
    covers values read STRAIGHT off a schema field. It says nothing about a
    computed DIFFERENCE of two of them, which has no lower bound at all: a
    42-microsecond freeze pad (``duration - clip_dur``, both probed) rendered as
    ``4.2e-05`` and killed a paid render outright —

        [Parsed_tpad_6] Unable to parse option value "4.2e-05" as duration
        Error applying option 'stop_duration' to filter 'tpad': Invalid argument

    — because ffmpeg's DURATION parser (``av_parse_time``, behind every
    ``AV_OPT_TYPE_DURATION`` option: ``tpad:stop_duration``, ``trim:duration``,
    ``afade:d``, ``color:d``, and the CLI ``-t``) rejects an exponent outright.
    Expression contexts (``zoompan``'s ``z``/``x``/``y``, ``geq``'s ``a``) go
    through ``av_strtod`` and WOULD have tolerated it — which is exactly why this
    sat latent: most call sites here are expressions.

    So the guarantee is now structural rather than argued: out-of-``%g``-range
    magnitudes fall back to :func:`_fixed_point`, and no caller — present or
    future, schema-bounded or computed — can emit an exponent into a filter
    argument. Every value ``%g`` already rendered plainly is untouched, so this
    is byte-identical on every path that was not already broken.

    For a value whose FULL precision matters (an output-side duration), prefer
    :func:`_ffdur`, which is exponent-free without ``%g``'s 6-significant-digit
    truncation.
    """
    text = f"{value:g}"
    return text if "e" not in text else _fixed_point(value)


def _ffdur(value: float) -> str:
    """Render a duration as a plain ffmpeg argument, at FULL float precision.

    The duration twin of :func:`_ffnum`, for the ``AV_OPT_TYPE_DURATION`` options
    and the CLI ``-t`` — the arguments whose parser rejects an exponent (see
    :func:`_ffnum`) and where ``%g``'s 6 significant digits would be a real loss:
    ``%g`` renders a 123.456789 s segment as ``123.457``, moving the bound that
    terminates it by 0.2 ms. ``repr`` keeps every digit, so this is byte-identical
    to the bare ``f"{value}"`` these sites used to interpolate — for every value
    except the one shape that crashed the render.
    """
    text = repr(float(value))
    return text if "e" not in text else _fixed_point(value)


def _median(values: list[float]) -> float:
    """The median of ``values`` — the T9 "one camera" luma target across shots.

    The median (not the mean) so a single blown-out or crushed outlier shot can't
    drag the target it is supposed to be corrected toward. Odd count → the middle
    element; even → the mean of the two middle; empty → ``0.0`` (the caller guards
    on the shot count before it matters). PURE, never raises."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ---------------------------------------------------------------------------
# Whole-film colour grade (VC-5): one post-concat re-encode over the delivery
# master. Pure, side-effect-free builder + report — mirrors _master_filter.
# ---------------------------------------------------------------------------
# Grade tuning constants. Chosen so a full-strength (grade_intensity=1) knob is a
# strong-but-sane finishing move; every one is scaled by grade_intensity so 0
# collapses the whole grade to identity. All resolve to CORE ffmpeg filters
# (eq/colorbalance/vignette/noise/unsharp/gblur/blend/colorlevels) — no new deps.
_TEMP_GAMMA_SCALE = 0.12  # temperature ±1 → per-channel gamma ±0.12 (warm=+R/-B)
_SPLIT_SCALE = 0.5  # tint chroma-cast → colorbalance shadow/highlight shift
_VIGNETTE_MAX_ANGLE = math.pi / 4  # vignette=1 → widest (darkest) corner falloff
_GRAIN_SCALE = 40.0  # grain=1 → noise alls=40 (0..100)
_SHARPEN_SCALE = 1.5  # sharpen=1 → unsharp luma_amount=1.5
_GLOW_SIGMA = 8.0  # bloom blur radius (px) for the screen-blended glow
# match_sources: gentle bias of the eq midpoints toward the film's own measured
# signalstats (YAVG/SATAVG are 0..255 averages). Bounded so a mismeasure can't
# blow the grade out.
_MATCH_Y_TARGET = 120.0
_MATCH_GAMMA_EXP = 0.3
_MATCH_GAMMA_MIN = 0.85
_MATCH_GAMMA_MAX = 1.2
_MATCH_SAT_TARGET = 90.0
_MATCH_SAT_EXP = 0.3
_MATCH_SAT_MIN = 0.9
_MATCH_SAT_MAX = 1.15

# T9 source-matching (per-shot exposure normalisation): a post-render pass nudges
# each outlier shot's exposure toward the film's MEDIAN luma (the "one camera"
# target) so mixed sources (browser recording / AI image / slide / motion card)
# stop drifting shot-to-shot. Distinct from the whole-film ``match_sources`` grade
# bias above: this equalises SHOTS AGAINST EACH OTHER, not the film toward a fixed
# target. A single gamma pass moves an outlier onto the median. Gamma pins 0 and 1,
# so no legal-range clamp is needed. The correction gamma is the CLOSED FORM that
# exactly maps a flat field's measured luma onto the target (see
# ``_source_match_correction``) — no empirical exponent, no systematic overshoot —
# still clamped so an extreme dark-vs-bright shot is protected (converges as far as
# the clamp allows) rather than blown out.
_SRC_MATCH_GAMMA_MIN = 0.6
_SRC_MATCH_GAMMA_MAX = 1.7
# Saturation clamp is WIDER than gamma: a near-grey outlier (tiny SATAVG) yields a
# large target/satavg ratio, and chroma scaling is linear (no exponent) rather than
# gamma's log-domain compression, so it needs more headroom to converge while still
# refusing a runaway over-saturation.
_SRC_MATCH_SAT_MIN = 0.5
_SRC_MATCH_SAT_MAX = 2.0


def _source_match_correction(
    stats: dict[str, float], target_yavg: float, target_satavg: float, settings: Settings
) -> str:
    """The T9 per-shot source-match ``eq=`` fragment, or ``""`` for a no-op.

    Nudges ONE shot toward the film's median luma (``target_yavg``) AND median
    saturation (``target_satavg``) in a SINGLE combined ``eq=`` pass (one re-encode
    per outlier). Two independent terms:

    * LUMA — a single gamma bias: a darker-than-median shot (low ``YAVG``) is lifted
      (gamma > 1), a brighter one is pulled down (gamma < 1).
    * SAT — a linear chroma scale ``saturation = target/satavg``: an
      under-saturated shot is boosted, an over-saturated one is pulled in.

    Each term is emitted ONLY when its metric is valid AND out of tolerance AND its
    clamped bias is not identity, so an in-tolerance shot on BOTH terms returns
    ``""`` (byte-identical, no re-encode) and a shot that trips only one term emits
    only that term (a luma-only shot stays byte-identical to the gamma-only era).
    Both biases are bounded so a mismeasure can't blow the shot out. PURE (no
    ffmpeg), deterministic, never raises."""
    params: list[str] = []
    # LUMA term. Gamma is undefined at the 0/255 endpoints (log domain), so both the
    # shot and the target must sit strictly inside the open (0, 255) luma range.
    y = stats.get("YAVG")
    if (
        y is not None
        and 0.0 < y < 255.0
        and 0.0 < target_yavg < 255.0
        and abs(target_yavg - y) > settings.source_match_tolerance
    ):
        # Exact single-pass gamma: eq maps a flat field v -> 255*(v/255)**(1/g), so
        # the gamma landing this shot's mean luma ``y`` on the median ``target_yavg``
        # solves (y/255)**(1/g) = target/255  =>  g = ln(y/255) / ln(target/255).
        # Bounded so an extreme shot converges only as far as the clamp allows.
        raw = math.log(y / 255.0) / math.log(target_yavg / 255.0)
        gbias = min(_SRC_MATCH_GAMMA_MAX, max(_SRC_MATCH_GAMMA_MIN, raw))
        if gbias != 1.0:
            params.append(f"gamma={_ffnum(gbias)}")
    # SAT term. eq's ``saturation`` scales chroma LINEARLY, so the exact single-pass
    # scale landing ``sv`` on the median is simply target/satavg (no exponent).
    sv = stats.get("SATAVG")
    if (
        sv is not None
        and sv > 0.0
        and target_satavg > 0.0
        and abs(target_satavg - sv) > settings.source_match_tolerance
    ):
        sbias = min(_SRC_MATCH_SAT_MAX, max(_SRC_MATCH_SAT_MIN, target_satavg / sv))
        if sbias != 1.0:
            params.append(f"saturation={_ffnum(sbias)}")
    return ("eq=" + ":".join(params)) if params else ""


def _hex_rgb(value: str) -> tuple[float, float, float]:
    """Parse a permissive hex colour → (r, g, b) in 0..1. Tolerant of a leading
    ``#`` and 3- or 6-digit forms; unparseable input → neutral ``(0, 0, 0)`` (a
    no-op split-tone shift), never raises."""
    s = (value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return 0.0, 0.0, 0.0
    try:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except ValueError:
        return 0.0, 0.0, 0.0


def _clamp_unit(x: float) -> float:
    """Clamp to colorbalance's [-1, 1] shift range."""
    return max(-1.0, min(1.0, x))


def _split_tone_frag(shadow_tint: str, highlight_tint: str, intensity: float) -> str:
    """``colorbalance`` split-tone from the two tints, or ``""`` for no cast.

    Only the tint's CHROMA CAST (deviation from its own grey) drives the shift, so
    a neutral tint (black shadows / white highlights — the defaults, or any grey)
    contributes nothing regardless of its brightness. Shadow tint drives ``*s``,
    highlight tint drives ``*h``; each shift is scaled by ``intensity`` and
    clamped to colorbalance's [-1, 1]."""
    sr, sg, sb = _hex_rgb(shadow_tint)
    s_avg = (sr + sg + sb) / 3.0
    hr, hg, hb = _hex_rgb(highlight_tint)
    h_avg = (hr + hg + hb) / 3.0
    rs = _clamp_unit((sr - s_avg) * _SPLIT_SCALE * intensity)
    gs = _clamp_unit((sg - s_avg) * _SPLIT_SCALE * intensity)
    bs = _clamp_unit((sb - s_avg) * _SPLIT_SCALE * intensity)
    rh = _clamp_unit((hr - h_avg) * _SPLIT_SCALE * intensity)
    gh = _clamp_unit((hg - h_avg) * _SPLIT_SCALE * intensity)
    bh = _clamp_unit((hb - h_avg) * _SPLIT_SCALE * intensity)
    if all(x == 0.0 for x in (rs, gs, bs, rh, gh, bh)):
        return ""
    return (
        f"colorbalance=rs={_ffnum(rs)}:gs={_ffnum(gs)}:bs={_ffnum(bs)}"
        f":rh={_ffnum(rh)}:gh={_ffnum(gh)}:bh={_ffnum(bh)}"
    )


def _apply_match_bias(base: dict[str, float], stats: dict[str, float]) -> None:
    """Bias the eq gamma/saturation toward the film's measured midpoints, in place.

    A darker-than-target film (low ``YAVG``) is lifted (gamma up), an
    under-saturated one (low ``SATAVG``) is nudged up — both gently and bounded.
    Missing/zero metrics are skipped (best-effort; never raises)."""
    yavg = stats.get("YAVG")
    if yavg and yavg > 0:
        raw = (_MATCH_Y_TARGET / yavg) ** _MATCH_GAMMA_EXP
        bias = min(_MATCH_GAMMA_MAX, max(_MATCH_GAMMA_MIN, raw))
        base["gamma"] = base.get("gamma", 1.0) * bias
    satavg = stats.get("SATAVG")
    if satavg and satavg > 0:
        raw = (_MATCH_SAT_TARGET / satavg) ** _MATCH_SAT_EXP
        bias = min(_MATCH_SAT_MAX, max(_MATCH_SAT_MIN, raw))
        base["saturation"] = base.get("saturation", 1.0) * bias


# Legal-range (16..235) output clamp — the broadcast_safe tail filter. Compresses
# the full-range signal INTO legal levels via colorlevels' OUTPUT range so the
# delivered luma sits inside 16..235 (best-effort; x264 ringing can still nudge a
# few codes out — the colour analog of the true-peak limiter).
_BROADCAST_SAFE_CLAMP = (
    "colorlevels=romin=16/255:romax=235/255:gomin=16/255:gomax=235/255:bomin=16/255:bomax=235/255"
)


def _grade_eq_fragment(grade: GradeSpec, stats: dict[str, float] | None, intensity: float) -> str:
    """The ``eq=`` fragment: the ``look`` base preset + explicit
    contrast/saturation/gamma knobs + a ``temperature`` per-channel gamma bias,
    optionally biased toward the film's measured midpoints (``match_sources``),
    then each component scaled toward its identity default by ``intensity``.
    Returns ``""`` when nothing deviates from the eq identity."""
    base: dict[str, float] = dict(_COLOR_PRESETS.get(grade.look, {}))
    if grade.contrast != _EQ_DEFAULTS["contrast"]:
        base["contrast"] = grade.contrast
    if grade.saturation != _EQ_DEFAULTS["saturation"]:
        base["saturation"] = grade.saturation
    if grade.gamma != _EQ_DEFAULTS["gamma"]:
        base["gamma"] = grade.gamma
    if grade.temperature != 0.0:
        t = grade.temperature * _TEMP_GAMMA_SCALE
        base["gamma_r"] = base.get("gamma_r", 1.0) + t
        base["gamma_b"] = base.get("gamma_b", 1.0) - t
    if stats and grade.match_sources:
        _apply_match_bias(base, stats)
    eq_parts: list[str] = []
    for k in _EQ_PARAM_ORDER:
        if k in base:
            scaled = _EQ_DEFAULTS[k] + (base[k] - _EQ_DEFAULTS[k]) * intensity
            if scaled != _EQ_DEFAULTS[k]:
                eq_parts.append(f"{k}={_ffnum(scaled)}")
    if not eq_parts:
        return ""
    return "eq=" + ":".join(eq_parts)


def _grade_texture_fragments(grade: GradeSpec, intensity: float) -> list[str]:
    """The intensity-scaled vignette / film-grain / output-sharpen fragments, in
    chain order — each emitted only when its knob stays positive after scaling
    (an identity knob contributes nothing)."""
    frags: list[str] = []
    vignette = grade.vignette * intensity
    if vignette > 0:
        frags.append(f"vignette=a={_ffnum(vignette * _VIGNETTE_MAX_ANGLE)}")
    grain = grade.grain * intensity
    if grain > 0:
        frags.append(f"noise=alls={_ffnum(grain * _GRAIN_SCALE)}:allf=t+u")
    sharpen = grade.sharpen * intensity
    if sharpen > 0:
        frags.append(
            f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={_ffnum(sharpen * _SHARPEN_SCALE)}"
        )
    return frags


def _grade_compose(linear: list[str], glow: float, tail: str) -> str:
    """Assemble the linear fragments (+ optional legal-range ``tail``) into the
    final filtergraph. When ``glow`` (already intensity-scaled) is positive the
    linear chain is wrapped in the screen-blended gaussian bloom sub-graph, so
    the result may be a full split/blend filtergraph rather than a bare comma
    chain. The ``tail`` clamp, when present, is always applied LAST."""
    if glow <= 0:
        parts = [*linear, tail] if tail else linear
        return ",".join(parts)
    opacity = min(1.0, glow)
    prefix = ",".join(linear)
    head = f"{prefix},split[gbase][gblur]" if prefix else "split[gbase][gblur]"
    graph = (
        f"{head};[gblur]gblur=sigma={_ffnum(_GLOW_SIGMA)}[gb];"
        f"[gbase][gb]blend=all_mode=screen:all_opacity={_ffnum(opacity)}"
    )
    return f"{graph},{tail}" if tail else graph


def _grade_vf(grade: GradeSpec, stats: dict[str, float] | None = None) -> str:
    """Build the whole-film grade filter chain, or ``""`` for a no-op grade.

    Returns ``""`` when the grade is disabled OR every knob sits at its identity
    default (or ``grade_intensity`` scales the whole grade back to identity) — the
    byte-identity gate the ``-c copy`` fast path depends on. Otherwise emits a
    single filtergraph, in order:

    1. ``eq`` — the ``look`` base preset + contrast/saturation/gamma + a
       ``temperature`` per-channel gamma bias (reuses the per-shot
       ``_COLOR_PRESETS`` / ``_EQ_*`` machinery), each component scaled toward its
       identity by ``grade_intensity``;
    2. ``colorbalance`` — shadow/highlight split-tone from the two tints;
    3. ``vignette`` — corner falloff;
    4. ``noise`` — film grain (``alls``/``allf=t+u``);
    5. ``unsharp`` — output sharpening;
    6. a ``gblur`` + ``blend=screen`` bloom for ``glow`` (a self-contained
       split/blend sub-graph, so the return may be a full filtergraph, not a bare
       comma chain);
    7. ``colorlevels`` legal-range clamp LAST — only when ``broadcast_safe`` (this
       one is NOT scaled by intensity: it is a delivery guard, not a look knob).

    ``stats`` (when ``match_sources``) biases the eq midpoints toward the film's
    own measured luma/chroma before the intensity scale. Every filter name is core
    ffmpeg — no new system dependency. ``lut_url`` is carried on the model but not
    applied here.
    """
    if not grade.enabled:
        return ""
    intensity = grade.grade_intensity
    linear: list[str] = []
    # 1. eq base preset + knobs + temperature; 2. split-tone; 3-5. texture.
    eq = _grade_eq_fragment(grade, stats, intensity)
    if eq:
        linear.append(eq)
    split = _split_tone_frag(grade.shadow_tint, grade.highlight_tint, intensity)
    if split:
        linear.append(split)
    linear.extend(_grade_texture_fragments(grade, intensity))
    tail = _BROADCAST_SAFE_CLAMP if grade.broadcast_safe else ""
    # 6. glow bloom (may return a full split/blend filtergraph); 7. legal clamp.
    return _grade_compose(linear, grade.glow * intensity, tail)


def grade_report(grade: GradeSpec | None) -> dict[str, object] | None:
    """Resolved grade knobs + whether a whole-film re-encode engages (VC-5).

    Route-level observability: reports the REQUESTED knobs (never a fabricated
    "measured grade"). ``reencoded`` is True whenever the whole-film re-encode
    actually engages: either a static knob makes ``_grade_vf`` non-empty, OR
    ``match_sources`` is enabled — its adaptive bias re-encodes on any footage
    that isn't perfectly neutral (i.e. in practice, always), so folding it in
    keeps ``reencoded`` honest for cost/QA callers rather than reporting a false
    "no re-encode". ``None`` when the request carried no grade."""
    if grade is None:
        return None
    report: dict[str, object] = {
        "enabled": grade.enabled,
        "look": grade.look,
        "contrast": grade.contrast,
        "saturation": grade.saturation,
        "gamma": grade.gamma,
        "temperature": grade.temperature,
        "shadow_tint": grade.shadow_tint,
        "highlight_tint": grade.highlight_tint,
        "vignette": grade.vignette,
        "grain": grade.grain,
        "glow": grade.glow,
        "sharpen": grade.sharpen,
        "grade_intensity": grade.grade_intensity,
        "match_sources": grade.match_sources,
        "broadcast_safe": grade.broadcast_safe,
        "lut_url": grade.lut_url,
        "reencoded": bool(_grade_vf(grade)) or (grade.enabled and grade.match_sources),
    }
    return report


# The duck's DEPTH anchor: the ratio `_sidechain_ratio` yields for the schema
# default ``duck_db=-12``.
#
# ⚠️ WAS 9, AND 9 PUMPED. A 9:1 slam that recovers inside every inter-word gap is
# textbook breathing/pumping — the bed audibly heaves once per syllable, which is
# one of the "digital artifacts" reported on delivered films. 4:1 with a longer
# release rides UNDER the narration instead of chasing it. The ratio is the depth
# knob and the release is the speed knob; they must move together or a shallower
# ratio just pumps more often.
_DUCK_RATIO_AT_DEFAULT_DB = 4

# The ambient branch carries no ``duck_db`` knob (tight surface), so its fixed
# ratio is DERIVED from the same anchor as music's default rather than written as
# a second literal — the two branches cannot disagree by construction.
_AMBIENT_DUCK_DB = -12.0

# Ramp on an accent SFX's tail as it reaches the shot boundary (see
# `_audio_sfx_graph`): without it a cue whose tail outlives the shot is chopped
# at a non-zero sample by the segment's `-t` bound.
_ACCENT_TAIL_FADE_S = 0.15


def _sidechain_ratio(duck_db: float) -> int:
    """Map a music bed's advisory ``duck_db`` to the sidechain compression ratio:
    a deeper (more negative) requested duck → a higher ratio so the bed sits
    further under the narration. Anchored so the schema default ``duck_db=-12``
    yields ``_DUCK_RATIO_AT_DEFAULT_DB`` (4), clamped to [2, 20].

    ``duck_db >= 0`` ("no extra attenuation requested") maps to the floor ratio 2 (a
    light touch), NOT the heaviest ratio — the formula already yields 2 there (0 →
    clamped up), so it must NOT be short-circuited (a prior ``if duck_db >= 0:
    return 9`` slammed a bed the caller asked to barely duck)."""
    return max(2, min(20, round(_DUCK_RATIO_AT_DEFAULT_DB * (duck_db / -12.0))))


def _offset_gain_db(
    offset_lu: float | None,
    fallback_gain_db: float,
    vo_lufs: float | None,
    asset_lufs: float | None,
) -> tuple[float, bool]:
    """VO-relative leveling (T1): the gain that sits a bed/cue ``offset_lu``
    LU below the measured narration, or the literal fallback when that isn't
    possible.

    ``offset_lu`` unset, or either measurement missing/non-finite (the
    best-effort ``final_mux_loudness`` failed), falls back to
    ``(fallback_gain_db, False)`` — the exact literal path this feature is
    additive over. Otherwise the target level is ``vo_lufs - offset_lu`` and
    the gain is that target minus the asset's own measured level, clamped to
    ``[-40.0, 12.0]`` so a wildly mismeasured asset can't produce a silent or
    a blown-out bed. Returns ``(applied_gain_db, was_measured)``."""
    if (
        offset_lu is None
        or vo_lufs is None
        or asset_lufs is None
        or not math.isfinite(vo_lufs)
        or not math.isfinite(asset_lufs)
    ):
        return fallback_gain_db, False
    target = vo_lufs - offset_lu
    gain = target - asset_lufs
    return max(-40.0, min(12.0, gain)), True


def _vo_relative_threshold(
    open_lu_below_vo: float | None, vo_lufs: float | None, fallback: float
) -> float:
    """VO-relative sidechain open threshold (T1): converts ``vo_lufs -
    open_lu_below_vo`` (dBFS) to the LINEAR key level ``sidechaincompress``
    wants, replacing the absolute ``duck_threshold`` literal.

    ``open_lu_below_vo`` unset, or ``vo_lufs`` missing/non-finite (measurement
    unset or the best-effort measure failed), returns ``fallback`` unchanged —
    the exact literal path this feature is additive over. Clamped to the
    schema's ``duck_threshold`` bounds ``[0.001, 1.0]``."""
    if open_lu_below_vo is None or vo_lufs is None or not math.isfinite(vo_lufs):
        return fallback
    dbfs = vo_lufs - open_lu_below_vo
    linear = 10 ** (dbfs / 20)
    return max(0.001, min(1.0, linear))


def _audio_sfx_graph(
    primary_label: str,
    transition: tuple[int, SoundCue] | None,
    accents: list[tuple[int, AccentCue]],
    settings: Settings,
    *,
    out_label: str = "aout",
    audio_frag: str = "",
    shot_seconds: float = 0.0,
) -> str:
    """Build the ``-filter_complex`` audio fragment mixing the shot's primary
    audio (narration / clip audio / synthesized silence at ``primary_label``,
    e.g. ``"1:a"``) with a transition SFX at t=0 and accent SFX at offsets.

    Returns the fragment producing ``[out_label]`` (default ``[aout]``).
    ``out_label`` lets a caller redirect the mix into a downstream stage (e.g. a
    per-clip ``atempo`` retime) instead of terminating at the mapped output.
    ``normalize=0`` keeps the primary at full level (default amix divides by
    input count, which would bury the narration); ``duration=first`` anchors the
    mix to the primary so a long accent tail can't extend the segment past its
    ``-t`` bound.

    ``shot_seconds`` (the shot's own pre-speed length; the accent ``adelay``
    offsets are in that same timeline because the retime is appended AFTER this
    mix) arms two guards that ``duration=first`` alone does not give:

    * an accent whose tail runs past the shot end was CHOPPED at whatever sample
      the ``-t`` bound landed on — a click at every such cut. Each accent now
      ramps out into the shot boundary. ``0.0`` (a caller that has no length to
      offer) omits the ramp rather than computing a negative ``st``.
    * ``normalize=0`` sums the primary and every accent at unity, so a shot with
      two accents over a hot line can exceed full scale. Every other audio bus in
      this service ends in a limiter; this one was the last unbounded branch, and
      the per-shot segment is encoded (and stream-copied into the film) BEFORE the
      post-concat master ever sees it, so the master cannot undo the clipping.
      ``level=disabled`` is load-bearing — ``alimiter`` otherwise re-normalises
      its output back to 1.0 and ``limit=`` bounds nothing.
    """
    # `_A48_SOXR` on the primary for the same reason as the no-SFX branch in
    # `_clip_segment_filter_args`: the primary IS the narration, 24 kHz mono out of
    # tts_backend, so this is a real 24k -> 48k conversion and bare `_A48` did it on
    # swr's cubic. The SFX inputs below already used the soxr form; the narration
    # they are mixed against did not.
    parts: list[str] = [f"[{primary_label}]{_A48_SOXR}{audio_frag}[a_pri]"]
    labels: list[str] = ["[a_pri]"]
    if transition is not None:
        idx, cue = transition
        parts.append(f"[{idx}:a]{_A48_ASSET},volume={cue.gain_db}dB[a_tr]")
        labels.append("[a_tr]")
    for n, (idx, cue) in enumerate(accents):
        ms = round(cue.at_seconds * 1000)
        # Room between this accent's onset and the shot end. <= 0 means the cue is
        # at/after the end (amix's duration=first drops it anyway) → no fade.
        room = shot_seconds - cue.at_seconds
        tail = ""
        if room > 0.0:
            fade = min(_ACCENT_TAIL_FADE_S, room)
            tail = f",afade=t=out:st={max(0.0, shot_seconds - fade):.3f}:d={fade:.3f}"
        parts.append(
            f"[{idx}:a]{_A48_ASSET},adelay={ms}|{ms},volume={cue.gain_db}dB{tail}[a_ac{n}]"
        )
        labels.append(f"[a_ac{n}]")
    limit = _limiter_linear_limit(settings)
    parts.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:normalize=0,"
        + f"alimiter=limit={_ffnum(limit)}:level=disabled[{out_label}]"
    )
    return ";".join(parts)


async def _fetch_asset(
    url: str,
    dest: Path,
    http: httpx.AsyncClient,
    source_cache: SourceFetchCache | None,
) -> Path | None:
    """``_try_fetch``, deduped per-source when the job supplied a cache.

    ``source_cache is None`` is the undeduped path and stays exactly what every
    caller did before the cache existed — that is what keeps the thumbnail route,
    the audiobook concat path and every direct ``render_shot`` caller (tests
    included) unchanged.
    """
    if source_cache is None:
        return await _try_fetch(url, dest, http)
    return await source_cache.fetch(url, dest, http)


async def _fetch_shot_sfx(
    shot: Shot,
    work_dir: Path,
    http: httpx.AsyncClient,
    source_cache: SourceFetchCache | None = None,
) -> tuple[tuple[Path, SoundCue] | None, list[tuple[Path, AccentCue]]]:
    """Best-effort fetch of a shot's transition + accent SFX. A cue whose asset
    is unfetchable (expired URL / 404) is dropped — one bad SFX never fails the
    shot, mirroring ``_resolve_shot_assets`` for the visual/audio degradation.

    SFX go through the per-job dedupe too: a single whoosh/chime is typically
    referenced by EVERY shot in a compile, so this is the highest-hit-rate source
    in the whole request even though each asset is small."""
    transition: tuple[Path, SoundCue] | None = None
    if shot.transition_sfx:
        path = await _fetch_asset(
            shot.transition_sfx.url, work_dir / f"{shot.id}-trans", http, source_cache
        )
        if path is not None:
            transition = (path, shot.transition_sfx)
    accents: list[tuple[Path, AccentCue]] = []
    for j, cue in enumerate(shot.sfx):
        path = await _fetch_asset(cue.url, work_dir / f"{shot.id}-sfx{j}", http, source_cache)
        if path is not None:
            accents.append((path, cue))
    return transition, accents


async def _fetch_overlay_clips(
    shot: Shot,
    work_dir: Path,
    http: httpx.AsyncClient,
    source_cache: SourceFetchCache | None = None,
) -> list[tuple[Path, OverlayClip, str | None]]:
    """Best-effort fetch + codec-probe of a shot's alpha overlay clips.

    Returns ``(local_path, clip, codec_name)`` per SURVIVING overlay. An overlay
    whose asset is unfetchable (expired presigned URL, 404, transport error) is
    DROPPED with a log — mirroring ``_fetch_shot_sfx`` / ``_resolve_shot_assets``:
    one bad decoration must never fail the shot, let alone the compile. The shot
    then renders exactly as it would without that overlay.

    Goes through ``_fetch_asset`` like every other asset, so the per-job
    ``SourceFetchCache`` applies. This is the highest-value dedupe target in the
    whole feature: a logo bug is by definition the SAME url on every shot in the
    film, so twenty shots cost one download.

    The ``.webm`` dest suffix is deliberate — ``SourceFetchCache`` names the shared
    file ``src-<hash><dest.suffix>``, and the VP9 alpha assets this feature exists
    for are WebM. The suffix is cosmetic to ffmpeg (which sniffs the container),
    but it keeps the job scratch legible when a render is being debugged.

    ``codec_name`` is probed here rather than at graph-build time because the probe
    is async and the graph builder is pure. ``None`` = unknown → no forced decoder;
    see ``_overlay_input_args`` for why that gate exists.
    """
    if not shot.overlay_clips:
        return []
    fetched: list[tuple[Path, OverlayClip, str | None]] = []
    for i, clip in enumerate(shot.overlay_clips[:_OVERLAY_CLIP_MAX_PER_SHOT]):
        path = await _fetch_asset(clip.url, work_dir / f"{shot.id}-ovl{i}.webm", http, source_cache)
        if path is None:
            log.warning("overlay_clip.dropped", shot_id=shot.id, url=clip.url)
            continue
        fetched.append((path, clip, await _probe_video_codec(path)))
    return fetched


async def _resolve_shot_assets(
    shot: Shot,
    work_dir: Path,
    http: httpx.AsyncClient,
    source_cache: SourceFetchCache | None = None,
) -> tuple[Path | None, Path | None, Path | None, float, Path | None]:
    """Best-effort fetch of a shot's audio + visual up front, returning
    ``(audio_path, clip, image, duration, second_clip)``.

    Each VISUAL asset is fetched via ``_try_fetch`` so a single unfetchable one
    degrades this shot — video → image → (in ``render_shot``) the text card —
    instead of aborting the whole compile. Video wins over image, so the image
    is only fetched when the clip is absent.

    ``audio_url`` is different: it is not fetched best-effort. A shot that
    declares one is a NARRATED shot, and a narrated shot whose audio cannot be
    fetched must not silently render as a mute one — see
    ``UnfetchableNarrationError``. ``shot.audio_url`` absent (no narration
    requested for this shot) is unaffected and still resolves to
    ``audio_path=None`` / the shot's own ``seconds``.

    ``second_clip`` (T8 split-screen) is the SECOND source composited beside the
    primary; a dangling/unfetchable second source returns None so the split is
    simply skipped (graceful degradation), never failing the shot.

    ``source_cache`` (job-scoped, ``None`` = undeduped as before) collapses N
    shots cut from ONE source file into a single download. This is where it pays:
    the music-cut product sends many shots whose ``video_url`` is the same
    uploaded 4K file, distinguished only by their in/out points.
    """
    audio_path: Path | None = None
    if shot.audio_url:
        audio_path = await _fetch_asset(
            shot.audio_url, work_dir / f"{shot.id}-audio", http, source_cache
        )
        if audio_path is None:
            raise UnfetchableNarrationError(
                f"shot {shot.id}: narration audio unfetchable: {shot.audio_url[:200]}"
            )
    duration = (await _ffprobe_duration(audio_path) or shot.seconds) if audio_path else shot.seconds
    clip: Path | None = None
    if shot.video_url:
        clip = await _fetch_asset(
            shot.video_url, work_dir / f"{shot.id}-clip.mp4", http, source_cache
        )
    image: Path | None = None
    if clip is None and shot.image_url:
        image = await _fetch_asset(shot.image_url, work_dir / f"{shot.id}-img", http, source_cache)
    second_clip: Path | None = None
    if shot.split is not None:
        second_clip = await _fetch_asset(
            shot.split.second_source_url, work_dir / f"{shot.id}-split.mp4", http, source_cache
        )
    return audio_path, clip, image, duration, second_clip


def _narrated_picture_floor(
    duration: float, floor_seconds: float | None, *, frame_seconds: float
) -> float:
    """A narrated shot's segment length under its ``Shot.seconds`` floor.

    ``duration`` is what :func:`_resolve_shot_assets` resolved — the PROBED narration
    on a narrated shot. That used to be final, so ``shot.seconds`` was discarded
    whenever narration existed. The orchestrator clamps ``seconds`` into
    reconcile_timeline's ``[min, max]`` window and pins a footage beat's ``min`` to its
    measured clip, so the plan that should have kept the footage never reached the
    picture: run ``01a0ab23`` recorded 10.96 / 11.92 / 8.40s of footage and rendered
    10.200 / 7.367 / 5.100s (the narration lengths) — a 30.0s order delivered 22.67s.

    A FLOOR, never a cap: the result is ``max(narration, seconds)`` so narration is
    never cut. It is a no-op on a shot without narration BY ARITHMETIC, not by a flag:
    there ``duration`` already IS ``shot.seconds`` (no gap to floor), and the own-audio
    clip branch reassigns ``duration`` to the clip length afterwards, so it keeps its
    natural length (both pinned by the render harness). A floor that exceeds the narration by less than ``frame_seconds``
    (one output frame) is ignored: that is metadata-vs-probe rounding, not a planned
    hold, and it cannot change the picture — so those shots stay byte-identical.

    ⛔ ONLY A RECONCILED ``seconds`` IS A FLOOR. The caller passes ``None`` unless the
    shot carries ``max_seconds``, which the orchestrator stamps ONLY on a shot
    reconcile_timeline bounded (``shot_assembly._freeze_pad_ceiling_patch``). Without
    bounds, ``seconds`` on a narrated shot can be the STORYBOARD guess: when TTS cannot
    measure a clip, narrate stores ``seconds=None`` and ``_reconcile_seconds`` treats
    the shot as un-narrated while still sending its ``audio_url``. lesson / app_review /
    audit_video / website_review send no bounds, so flooring there would freeze the
    picture over silence. ``None`` ⇒ narration-length, byte-identical to before.
    """
    if floor_seconds is None or floor_seconds - duration < frame_seconds:
        return duration
    return float(floor_seconds)


# Half a sample at 48 kHz. A duration this close to a frame boundary IS on the grid:
# it is a probe's 6-decimal rounding of N/fps (12.666667 for 380/30), not a planned
# sub-frame length, and treating it as off-grid would add a whole frame to the shot.
_FRAME_GRID_TOLERANCE_S = 1e-5


def _snap_to_frame_grid(seconds: float, fps: int, *, speed: float = 1.0, up: bool = True) -> float:
    """``seconds`` moved onto the OUTPUT frame grid: the PRE-speed duration whose
    rendered length (``seconds / speed``) is exactly ``N / fps``.

    WHY. A segment is cut with one ``-t`` for both streams, but the two streams
    quantise it differently: audio to the sample (1/48000 s), video to the FRAME —
    and ffmpeg rounds the frame count to nearest. So an off-grid ``-t`` always
    leaves the segment's audio and video different lengths, by up to half a frame
    either way. ``concat_shots`` stream-copies, and the concat demuxer starts every
    file at the previous file's LONGEST stream:

    * audio SHORTER than video -> the next segment's audio starts late: a hole in
      the audio timestamps at the cut (production run 01a10e19: 12.661312 s of
      audio under 380 frames = 12.666667 s -> 258 samples, 5.4 ms, missing).
    * audio LONGER than video -> the next segment's video starts late: the last
      frame before the cut is stretched (+9.6 ms and +6.7 ms on that same run,
      which is the whole of its ``avg_frame_rate`` 29.987 against 30).

    On the grid both streams end on the same instant, so neither happens.

    ``up=True`` (narrated shots, image and text cards): never shorter than asked, so
    narration is never cut — the shot gains < 1 frame. ``up=False`` (a silent clip,
    whose picture is FINITE): never longer than the source can cover — the shot
    loses < 1 frame. Either way the drift is under one output frame per shot.

    A value already on the grid (within ``_FRAME_GRID_TOLERANCE_S``) returns the
    exact ``N / fps``, which for every representable grid point (6.0, 10.5, ...) is
    the same float that came in — those segments' commands stay byte-identical.
    Never returns less than one frame.
    """
    frames_exact = seconds / speed * fps
    slack = _FRAME_GRID_TOLERANCE_S * fps
    frames = math.ceil(frames_exact - slack) if up else math.floor(frames_exact + slack)
    return max(frames, 1) / fps * speed


def _nearest_frame(seconds: float, fps: int) -> float:
    """``seconds`` moved to the NEAREST output frame boundary (never below one frame).

    For a length nothing bounds from either side — a crossfade overlap, a join
    offset: rounding to nearest keeps the error under half a frame and unbiased,
    where :func:`_snap_to_frame_grid`'s one-sided snap would shift every join the
    same way. A probe-rounded grid value (12.666667) returns the exact ``N / fps``.
    """
    return max(math.floor(seconds * fps + 0.5), 1) / fps


def _grid_speed(shot: Shot) -> float:
    """The speed that maps a shot's ``duration`` to its RENDERED length. A split
    shot skips the speed retime entirely, so its output length is ``duration``."""
    return shot.speed if shot.split is None else 1.0


def _snap_decided_lengths(
    duration: float,
    narration_seconds: float,
    *,
    narrated: bool,
    has_clip: bool,
    fps: int,
    speed: float,
) -> tuple[float, float]:
    """``(duration, narration_seconds)`` with ``duration`` snapped UP onto the frame
    grid for every shot whose length is already final when ``render_shot`` resolves
    its assets: a narrated shot of any picture kind, and an image or text card.

    A clip WITHOUT narration is returned untouched — its ``duration`` is reassigned
    to the clip's own length inside the clip branch (and snapped there, or left at
    its natural length on the own-audio path).

    On an un-narrated card there is no narration to outlast, so
    ``narration_seconds`` follows the snap: the card simply IS that long, and
    ``narration_padded`` stays False for it exactly as before.
    """
    if has_clip and not narrated:
        return duration, narration_seconds
    snapped = _snap_to_frame_grid(duration, fps, speed=speed)
    return snapped, (narration_seconds if narrated else snapped)


def _own_audio_bound(
    clip_dur: float, fallback_seconds: float, fps: int, speed: float
) -> tuple[float, bool, str]:
    """``(duration, bounded, audio_pad)`` for a clip that plays its OWN audio.

    Probed (``clip_dur > 0``): the clip's length snapped DOWN onto the frame grid,
    ``bounded=True`` (the segment gets an output-side ``-t``) and the ``apad``
    fragment that extends the clip's audio to that same length. Unprobeable:
    ``fallback_seconds`` is only what gets REPORTED — unbounded, no pad, exactly
    the pre-snap shape. See the own-audio branch of ``render_shot`` for why.
    """
    if clip_dur <= 0.0:
        return fallback_seconds, False, ""
    duration = _snap_to_frame_grid(clip_dur, fps, speed=speed, up=False)
    return duration, True, _pad_audio_to(duration)


def _pad_audio_to(seconds: float) -> str:
    """Leading-comma ``apad`` fragment that extends the primary audio to ``seconds``.

    Chained after :func:`_apply_shot_audio` on a shot whose picture floor outlasts its
    narration, so the segment's audio stream spans its picture instead of ending early
    (a short audio leg is exactly what a container duration — the max over streams —
    hides). ``whole_dur`` cannot shorten, so it is inert on audio already that long;
    ``_ffdur`` for the same parse-safety reason as the segment ``-t``.
    """
    return f",apad=whole_dur={_ffdur(seconds)}"


def _primary_audio_frag(
    audio: ShotAudio | None,
    duration: float,
    narration_seconds: float,
    *,
    narration_padded: bool,
) -> str:
    """The primary-audio fragment for a segment of ``duration`` seconds.

    Unpadded (every shot whose picture floor did not raise it): exactly
    ``_apply_shot_audio(audio, duration)``, byte-identical to before. Padded: fade the
    narration at ITS end, then :func:`_pad_audio_to` the segment length so the audio
    stream spans the picture. ``narration_padded`` is latched by the caller BEFORE the
    own-audio clip branch reassigns ``duration`` — re-deriving it here from
    ``duration > narration_seconds`` would pad an own-audio clip sent with a placeholder
    ``seconds``.
    """
    if not narration_padded:
        return _apply_shot_audio(audio, duration)
    return _apply_shot_audio(audio, narration_seconds) + _pad_audio_to(duration)


def _defers_caption(shot: Shot) -> bool:
    """Whether this shot's narration caption must be drawn AFTER the punch-in.

    The caption used to be baked into the fit/scale branch, i.e. UPSTREAM of
    :func:`_apply_directed_footage`'s zoompan — so on any shot carrying a focus
    track the caption text was magnified and resampled by the punch-in and went
    soft, while shot titles stayed crisp because :func:`_apply_shot_look` already
    runs downstream of the zoom.

    True only where it actually matters: a zoompan is composed only when a
    ``focus_track`` is set, and ``_apply_directed_footage`` is skipped outright on
    a split shot. Every other shot keeps the caption where it was, so its filter
    graph stays byte-identical.

    ⚠️ This is NOT the callouts. Badge/label captions ride the PRE-zoom canvas on
    purpose so the punch-in magnifies them together with the element they
    annotate. Only the bottom-of-frame narration caption moves.
    """
    return bool(shot.caption) and shot.split is None and bool(shot.focus_track)


def _deferred_caption_suffix(shot: Shot, settings: Settings, work_dir: Path, deferred: bool) -> str:
    """``,<caption fragment>`` when the caption was deferred past the zoom, else ``""``."""
    if not deferred or not shot.caption:
        return ""
    return "," + _caption_filter_fragment(shot.caption, settings, work_dir, shot.id)


def _shift_tracks_to_seek(shot: Shot, track_shift: float, duration: float) -> Shot:
    """Move a shot's cursor/focus/callout times onto the window the decoder plays.

    Track times are measured from the shot's DECLARED in-point,
    ``clip_lead_seconds`` (0 when unset): the caller rebases recorder times by the
    lead it asked for. The decoder's real in-point is ``seek_offset`` from
    :func:`_clip_seek_args`, which also covers the seeks this compiler chooses on
    its own: the repeated-use pan (``clip_seek_fraction``), the ``'end'`` settled
    tail, and a lead clamped to ``clip_len - duration``. ``track_shift`` is the
    difference, so every key is moved by exactly the seek the caller could not see.

    Before this, a clip shared by two shots (showcase spike 01a0fd4d-9948, clip-2
    under l2 and l3) panned the second shot ~15s into the recording while its
    cursor ring and click badge still fired at the recording's raw 3.07s, over the
    "Sales" nav of a page that was no longer on screen.

    A key that lands before the window or after the shot is dropped, not clamped:
    it points at a frame this shot never shows. An emptied track becomes ``None``
    (the byte-identical no-op). ``track_shift == 0`` returns ``shot`` unchanged.
    """
    if not track_shift:
        return shot

    def _in_shot(t: float) -> bool:
        return 0.0 <= t <= duration

    cursor = None
    if shot.cursor_track:
        cursor = [
            CursorKey(t=k.t - track_shift, x=k.x, y=k.y, click=k.click)
            for k in shot.cursor_track
            if _in_shot(k.t - track_shift)
        ] or None
    focus = None
    if shot.focus_track:
        focus = [
            FocusKey(t=k.t - track_shift, bbox=k.bbox)
            for k in shot.focus_track
            if _in_shot(k.t - track_shift)
        ] or None
    callouts = None
    if shot.callouts:
        callouts = [
            Callout(at=c.at - track_shift, type=c.type, bbox=c.bbox, label=c.label)
            for c in shot.callouts
            if _in_shot(c.at - track_shift)
        ] or None
    return shot.model_copy(
        update={"cursor_track": cursor, "focus_track": focus, "callouts": callouts}
    )


def _apply_directed_footage(
    video_vf: str, shot: Shot, settings: Settings, duration: float, speed: float, work_dir: Path
) -> str:
    """Compose the ORCH-3.2 directed-footage filters onto ``video_vf``: synthetic
    cursor THEN badge/label callouts THEN auto-zoom punch-in.

    ``speed`` != 1.0 retimes the segment via a leading ``setpts=PTS/speed`` added
    downstream, which compresses BOTH the overlay ``t`` axis and the zoompan ``on``
    range by 1/speed — so every authored keyframe/callout time is divided by
    ``speed`` here and the filters run against the retimed ``duration/speed``.

    Applied AFTER the fit/scale (+caption) branch and BEFORE the look, so all sit
    in framed-canvas coordinates and the grade/titles land on top. The cursor +
    callouts are composited BEFORE the zoompan on purpose: cursor keyframes, focus
    bboxes and callout bboxes share the same pre-zoom render-frame pixel space, so
    the punch-in magnifies them together with the frame instead of drifting them
    off-target during the hold. Each is a pass-through when its field is None →
    byte-identical to before these fields existed.

    WS-A: when the shot declares the recorded source resolution
    (``footage_viewport``) and it differs from the render canvas, the coords are
    remapped through the same decrease-scale+centered-pad transform the picture is
    fitted with, so cursor/zoom land on-target at non-default tiers.
    ``footage_viewport is None`` (every legacy caller) returns the ORIGINAL track
    objects, so ``_directed is shot`` and the exact same values flow through.

    SKIPPED entirely on a split shot: the primary now occupies only half the
    canvas, so cursor/callout/zoom coords (captured in the primary's full-canvas
    px) would land off-target — returns ``video_vf`` unchanged.
    """
    if shot.split is not None:
        return video_vf
    _focus_track, _cursor_track, _callouts = _remap_tracks_to_canvas(
        shot.focus_track,
        shot.cursor_track,
        shot.callouts,
        shot.footage_viewport,
        settings,
        shot.fit,
    )
    # A downstream setpts=PTS/speed compresses the overlay `t` and zoompan `on`
    # ranges by 1/speed, so scale every authored keyframe/callout time by 1/speed
    # and run the filters against the retimed duration. speed==1.0 is untouched.
    eff_dur = duration / speed if speed != 1.0 else duration
    if speed != 1.0:
        _focus_track = (
            None
            if _focus_track is None
            else [FocusKey(t=k.t / speed, bbox=k.bbox) for k in _focus_track]
        )
        _cursor_track = (
            None
            if _cursor_track is None
            else [CursorKey(t=k.t / speed, x=k.x, y=k.y, click=k.click) for k in _cursor_track]
        )
        _callouts = (
            None
            if _callouts is None
            else [
                Callout(at=c.at / speed, type=c.type, bbox=c.bbox, label=c.label) for c in _callouts
            ]
        )
    _directed = shot
    if (
        speed != 1.0
        or _focus_track is not shot.focus_track
        or _cursor_track is not shot.cursor_track
        or _callouts is not shot.callouts
    ):
        _directed = shot.model_copy(
            update={
                "focus_track": _focus_track,
                "cursor_track": _cursor_track,
                "callouts": _callouts,
            }
        )
    video_vf = _cursor_filter(video_vf, _directed, settings, eff_dur)
    video_vf = _callout_filter(video_vf, _directed, settings, eff_dur, work_dir)
    return _focus_zoom_filter(video_vf, _directed, settings, eff_dur)


async def _playable_clip_duration(clip: Path, seek_offset: float, clip_span: float | None) -> float:
    """How many seconds of ``clip`` will actually PLAY, after seek and out-point.

    THE PLAYABLE length, not the file length, and the single place it is derived.
    ``-ss`` is an INPUT seek, so the decoder starts at ``seek_offset`` and only the
    tail renders — subtracting once, here, is what makes every consumer correct for
    free: both duration branches AND the freeze-pad guard that decides whether the
    clip under-runs its narration. Subtracting per-branch instead (the earlier
    shape) left the silent branch and the freeze pad reading the unseeked length.

    An explicit out-point bounds the READ (the input-side ``-t``), so the playable
    length is the window, not the whole post-seek tail — folded in here for the
    same reason, rather than at each consumer, which is exactly how the seek offset
    came to be wrong before.
    """
    clip_dur = max((await _ffprobe_duration(clip) or 0.0) - seek_offset, 0.0)
    if clip_span is not None:
        return min(clip_dur, clip_span)
    return clip_dur


def _all_shots_muted(shots: list[Shot]) -> bool:
    """True when every shot of the request is muted AND can actually be silenced at
    its SOURCE — the music-only reel shape, and the ONE property that decides WHERE
    a mute is realised.

    There are two correct implementations of ``audio.mute`` and which one is
    correct depends on the whole request, not on the shot:

    * **Every shot muted** (the music-cut product: the orchestrator sets ``mute``
      on all of them and the bed is the entire soundtrack) → mute AT SOURCE, by
      never mapping ``0:a`` in ``render_shot``. Every blend is then
      silence-to-silence, so no neighbour can be harmed by it, the source never
      reaches ``[nar_key]`` (the ducker's sidechain key), and NOTHING needs the
      post-master pass — which matters, because that pass runs on the FINISHED
      bus: re-muting an already-silent shot there zeroes the BED under it, and
      with the solo windows tiling the whole reel the delivered file measured
      -91 dBFS.

    * **A mixed request** (the narration products: some shots muted, some not) →
      mute POST-MASTER, via the time-ranged ``volume=0`` keyed to each shot's
      SOLO window (``_solo_windows`` / ``_postclip_filter``). Source-level
      silence cannot be used here: an ``acrossfade`` between an unmuted
      neighbour and a silenced shot ramps the neighbour's audio against silence,
      damaging audio the user never edited. The solo window deliberately
      EXCLUDES the blend regions, which is exactly what spares the neighbour.

    ⚠️ "Muted" is NOT sufficient on its own: the shot must also be SILENCEABLE at
    source, and a shot carrying its own ``audio_url`` is not. All three branches of
    ``render_shot`` map that narration track (``1:a``) unconditionally — the mute is
    never consulted there, and for the image/text branches there is no source audio
    to drop in the first place. So an "all muted" request that contains a narrated
    shot would silence NOTHING at source while ``_postclip_filter`` skipped it as
    already-silent, and the narration was delivered at full level. Requiring
    ``not audio_url`` hands any such request back to the post-master path, where the
    ``volume=0`` clause is the only mechanism that can actually mute narration.

    An empty request is False — there is no reel to be music-only.
    """
    return bool(shots) and all(
        s.audio is not None and s.audio.mute and not s.audio_url for s in shots
    )


def _record_narration_audio(sink: list[Path | None] | None, audio_path: Path | None) -> None:
    """Append ``audio_path`` to ``sink`` when the caller supplied one — a plain
    guarded append, split out so ``render_shot``/``compile_shots`` don't each pay
    a branch of cyclomatic complexity for an optional out-parameter."""
    if sink is not None:
        sink.append(audio_path)


async def render_shot(
    shot: Shot,
    out_path: Path,
    settings: Settings,
    work_dir: Path,
    http: httpx.AsyncClient,
    clip_seek_fraction: float = 0.0,
    *,
    deterministic: bool = False,
    freeze_pad_cap: bool = False,
    all_shots_muted: bool = False,
    source_cache: SourceFetchCache | None = None,
    narration_audio_sink: list[Path | None] | None = None,
) -> float:
    """Render a single shot to ``out_path`` (uniform H.264 + AAC). Returns its duration.

    ``narration_audio_sink``, when given, is an out-parameter: this function
    appends the shot's resolved local narration audio path (``None`` for a shot
    with no ``audio_url``) to it. It exists so ``_render_shot_cached`` can learn
    where this shot's SOURCE narration clip landed on disk without widening this
    function's return type — dozens of tests call ``render_shot`` directly for
    its plain ``float`` duration. See ``mastering.narration_presence_report`` for
    the consumer: the post-mux presence check reuses this same local copy instead
    of re-fetching an expired presigned URL after the final mux.

    ``clip_seek_fraction`` (0..1) seeks into a footage clip so repeated uses of
    the same recording pan through different parts of the page instead of all
    replaying its opening hero. Ignored for image/text shots.

    ``shot.clip_lead_seconds`` / ``shot.clip_out_seconds`` name an EXPLICIT source
    window instead, which is how several shots are cut from one long recording.

    ``source_cache`` (job-scoped, ``None`` = undeduped) makes those several shots
    download that one recording once — see ``SourceFetchCache``.

    ``all_shots_muted`` is a WHOLE-REQUEST property (``_all_shots_muted``, computed
    once in ``compile_shots``) and decides where this shot's ``audio.mute`` is
    realised: at SOURCE here (never map ``0:a``) only when the entire request is
    muted, otherwise post-master in ``_postclip_filter``. Read that helper for why
    a mixed request must NOT silence at source. Default off = the post-master
    behaviour, so a direct caller is unchanged.

    A shot whose VISUAL asset is unfetchable degrades (video → image → text
    card) rather than failing the compile — see ``_resolve_shot_assets``. A shot
    whose NARRATION audio (``audio_url``) is unfetchable does NOT degrade to
    silence: it raises ``UnfetchableNarrationError``, failing the whole compile
    loudly rather than shipping a ``completed`` master missing that line. The
    same graceful-degrade holds for ``shot.overlay_clips``: an unfetchable alpha
    overlay is dropped and the shot renders without it.
    """
    audio_path, clip, image, duration, second_clip = await _resolve_shot_assets(
        shot, work_dir, http, source_cache
    )
    _record_narration_audio(narration_audio_sink, audio_path)
    # The narration's own length, kept so its edge fades land on ITS end even when the
    # plan's picture floor below outlasts it. Every branch that maps ``audio_path``
    # leaves ``duration`` alone after this point, so the floor reaches each ``-t``.
    #
    # ⛔ LATCHED HERE, not re-derived at the audio fragment: the own-audio clip branch
    # below REASSIGNS ``duration`` to the clip length, so ``duration > narration_seconds``
    # read later would be true for a placeholder ``seconds=1.0`` and anchor that clip's
    # fade-out at 1s. Only a floor that actually raised a NARRATED shot pads audio.
    narration_seconds = duration
    duration = _narrated_picture_floor(
        duration,
        shot.seconds if shot.max_seconds is not None else None,
        frame_seconds=1.0 / settings.render_fps,
    )
    # ⛔ END ON THE FRAME GRID — see :func:`_snap_to_frame_grid`. Every branch whose
    # length is decided HERE (a narrated shot of any picture kind; an image or text
    # card) is snapped UP now, before any consumer below reads ``duration`` (the
    # freeze pad, ``color=d=``, the cursor/zoom tracks, ``-t``), so they all agree.
    # A clip WITHOUT narration reassigns ``duration`` to its own length inside the
    # clip branch and is handled there. A split shot skips the speed retime, so its
    # output length is ``duration`` itself.
    grid_speed = _grid_speed(shot)
    duration, narration_seconds = _snap_decided_lengths(
        duration,
        narration_seconds,
        narrated=audio_path is not None,
        has_clip=clip is not None,
        fps=settings.render_fps,
        speed=grid_speed,
    )
    # True on a planned picture floor AND on the sub-frame snap above: in both the
    # segment outlasts its narration, and the audio must be padded to span it —
    # an audio stream shorter than its video is exactly the hole described there.
    narration_padded = duration > narration_seconds
    transition_sfx, accent_sfx = await _fetch_shot_sfx(shot, work_dir, http, source_cache)
    has_sfx = transition_sfx is not None or bool(accent_sfx)
    # Alpha overlay clips are fetched (and codec-probed) up front with every other
    # asset; an unfetchable one is simply absent from this list and the shot renders
    # without it. Empty list = the byte-identical no-op path.
    overlay_clips = await _fetch_overlay_clips(shot, work_dir, http, source_cache)

    cmd: list[str] = ["ffmpeg", "-y", *_bitexact_input_args(deterministic)]
    # Each branch sets: the video filter body applied to ``[0:v]`` (``video_vf``),
    # the bare primary-audio map label (``audio_map``), the ``-t`` trim args, the
    # next free input index (``next_idx``, where SFX inputs are appended), and
    # whether this is the text-card branch (which keeps a plain ``-vf`` when no
    # SFX are present).
    is_text = False
    # True when the primary audio map is a SYNTHESISED ``anullsrc`` track rather
    # than real samples (a muted shot, or any image/text card with no narration).
    # ⭐ CAPTION vs PUNCH-IN ORDER — see :func:`_defers_caption` for why and when.
    caption_after_zoom = _defers_caption(shot)

    # Consumed by the speed block below to keep ``atempo`` off an all-zero buffer.
    silent_primary = False
    # Only the clip branch measures a real source length; default 0.0 keeps the
    # split-screen freeze contract well-defined for the image/text branches.
    clip_dur = 0.0
    # Non-empty only for a clip playing its OWN audio whose length was probed: the
    # one segment whose primary audio is a finite stream that nothing else pads to
    # the ``-t`` (narration is padded by ``narration_padded``; silence is infinite).
    own_audio_pad = ""
    # Source seconds between the caller's declared in-point (``clip_lead_seconds``,
    # which its track times are measured from) and where the decoder really starts.
    # 0.0 for every non-clip shot. See :func:`_shift_tracks_to_seek`.
    track_shift = 0.0

    if clip is not None:
        seek_args, seek_offset, clip_span = await _clip_seek_args(
            clip,
            clip_seek_fraction,
            duration,
            shot.clip_anchor,
            shot.clip_lead_seconds,
            shot.clip_out_seconds,
            # The DECLARED source rate, so the window's in-point can be floored
            # onto the source frame grid. None here is not fatal: _clip_seek_args
            # falls back to probing the clip's own r_frame_rate.
            shot.target_fps,
        )
        track_shift = seek_offset - shot.clip_lead_seconds
        # ``seek_args`` carries the input-side ``-ss`` and (out-point only) ``-t``,
        # so it MUST stay before the ``-i`` it qualifies.
        cmd += seek_args
        cmd += ["-i", str(clip)]
        clip_has_audio = await _has_audio_stream(clip)
        # THE PLAYABLE length, not the file length. ``-ss`` above is an INPUT
        # seek, so the decoder starts at ``seek_offset`` and only the tail
        # renders. Subtracting once, here, is what makes every consumer below
        # correct for free — the two duration branches AND the freeze-pad guard.
        # Subtracting per-branch instead (the earlier shape) left the silent
        # branch and the freeze pad reading the unseeked file length.
        clip_dur = await _playable_clip_duration(clip, seek_offset, clip_span)
        # ON AN ALL-MUTED REQUEST ONLY, a muted shot never maps its own audio: it
        # takes the silence branch below instead. Dropping the map (rather than
        # relying only on the post-master ``volume=0`` pass) is what makes the
        # mute EXACT — that pass is time-ranged to the shot's SOLO window, so a
        # shot fully consumed by crossfades gets no mute at all and the source
        # stays audible inside every blend. It also keeps the source out of
        # ``[nar_key]``, the sidechain key that drives the music ducker in
        # ``overlay_audio_bed`` — otherwise drone/wind noise on a music-only
        # reel ducks the bed by the full ``duck_db`` for its whole length. The
        # post-master pass then skips these shots entirely (``_postclip_filter``),
        # since re-muting silence on the finished bus would zero the music bed.
        #
        # ⚠️ It is gated on the WHOLE request because silencing at source is only
        # harmless when EVERY blend is silence-to-silence. In a MIXED request an
        # ``acrossfade`` between an unmuted neighbour and a source-silenced shot
        # ramps that neighbour's audio against silence — damaging audio nobody
        # edited. Those requests keep the post-master path. See ``_all_shots_muted``.
        muted = all_shots_muted and shot.audio is not None and shot.audio.mute
        # A clip with its own audio plays at its natural remaining length (on the
        # frame grid when that length could be probed, see below); otherwise we
        # bound the segment with -t (to the narration, or to the remaining clip
        # length when a silent track is synthesised).
        trim = True
        if audio_path is not None:
            cmd += ["-i", str(audio_path)]
            audio_map = "1:a"
            next_idx = 2
        elif clip_has_audio and not muted:
            audio_map = "0:a"
            # Natural remaining length — never a length taken from ``shot.seconds``.
            #
            # ``shot.seconds`` is only the fallback for an unprobeable file; it can
            # NOT be read as a cut length, because a "play this recording" shot
            # routinely carries a placeholder (a 12s lesson clip sent as
            # ``seconds=1.0``) and truncating to it would cut such a product
            # mid-sentence. A shot that IS one cut out of a longer recording says so
            # with an OUT-point, and that is already bounded upstream:
            # ``_clip_seek_args`` emitted the INPUT-side ``-t`` for the window and
            # ``_playable_clip_duration`` folded it in, so ``clip_dur`` here IS the
            # window. (An earlier revision of this branch ALSO bounded a shot with an
            # IN-point but no OUT-point to ``shot.seconds``. That broke the contract
            # pinned by ``test_clip_out_point.py::
            # test_no_out_point_still_delivers_the_whole_tail`` — omitting the
            # out-point must still deliver the whole post-seek tail — and bought
            # nothing, since the cut planner always sends an out-point.)
            #
            # ``clip_dur`` is POST-seek: reporting the probed file length would
            # over-state the segment by the whole offset (an anchor='end' shot
            # claiming 20s for a 6s segment).
            #
            # ⛔ A PROBED length is bounded ON THE FRAME GRID — see
            # :func:`_snap_to_frame_grid`. "Natural length" is two lengths: the
            # clip's audio ends on a sample and its picture on a frame of ITS OWN
            # rate, neither on the output grid, and the two streams of a recording
            # rarely agree with each other to begin with. Left unbounded the segment
            # goes to the stream-copy concat with audio and video different lengths:
            # a hole in the audio timestamps (audio short) or a stretched frame
            # (audio long) at the next cut. So the CLIP's length (never
            # ``shot.seconds``) is snapped DOWN — the recording is finite, the
            # segment may not ask for a frame it cannot supply, and under one output
            # frame of tail is given up — an output-side ``-t`` cuts both streams
            # there, and the audio is padded to it below (``own_audio_pad``) so an
            # audio stream that ends before the picture still spans it.
            #
            # An UNPROBEABLE clip keeps the old shape exactly: there is no length to
            # snap, ``shot.seconds`` is only what gets REPORTED, and bounding the
            # encode to a placeholder would cut the recording — no ``-t``, no pad.
            duration, trim, own_audio_pad = _own_audio_bound(
                clip_dur, shot.seconds, settings.render_fps, grid_speed
            )
            next_idx = 1
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            audio_map = "1:a"
            silent_primary = True
            # anullsrc is infinite, so -t always drives the container length here
            # (``trim`` stays True). Same post-seek length as the own-audio branch
            # above — using the unseeked file length made the AUDIO longer than the
            # video by exactly the offset, an A/V mismatch the container duration
            # hides, which is why the test asserts per-STREAM lengths.
            #
            # Snapped DOWN onto the frame grid (:func:`_snap_to_frame_grid`): the
            # picture is finite here (no freeze pad on an un-narrated clip), so the
            # segment may not ask for a frame the source cannot supply.
            duration = _snap_to_frame_grid(
                clip_dur or shot.seconds, settings.render_fps, speed=grid_speed, up=False
            )
            next_idx = 2
        vf = _scale_pad_filter(
            settings,
            None if caption_after_zoom else shot.caption,
            work_dir,
            shot.id,
            shot.fit,
            source_fps=shot.target_fps,
            retime=shot.retime,
        )
        # When the clip is shorter than its narration line, freeze the last
        # frame for the remainder so the video fills the audio; -t below still
        # bounds the segment exactly to the narration length. Step B (opt-in): a
        # ``freeze_pad_cap`` job caps the drift at the shot's ``max_seconds`` ceiling.
        pad_cap = shot.max_seconds if freeze_pad_cap else None
        # Thread the shot's speed so the freeze tail is sized in the POST-retime
        # timeline (the tpad runs downstream of _apply_shot_speed's setpts). A split
        # shot SKIPS the speed retime (see `if shot.split is None` below) and
        # _inject_split rebuilds video_vf anyway, so pass 1.0 there.
        vf += _freeze_pad_suffix(
            audio_path is not None,
            clip_dur,
            duration,
            settings,
            pad_cap,
            speed=shot.speed if shot.split is None else 1.0,
            # ``trim`` IS "this segment gets an output-side -t" (see trim_args
            # below), which is the only thing that terminates the terminal clone.
            # An UNPROBEABLE own-audio clip sets trim=False and must never receive
            # it. A probed one is bounded like every other segment, and the clone is
            # what holds its picture to the ``-t`` its padded audio already reaches
            # (a recording whose video stream ends before its container does).
            bounded=trim,
        )
        video_vf = vf
        # ``_ffdur``, not ``f"{duration}"``: the CLI ``-t`` goes through the same
        # ``av_parse_time`` that rejected ``4.2e-05``, and ``duration`` here is
        # ``_ffprobe_duration(audio_path) or shot.seconds`` — a PROBED number falling
        # back to a field bounded only by ``exclusiveMinimum: 0``. Neither has a floor
        # above 1e-4. ``_ffdur`` (repr, not ``%g``) because this is the bound that
        # TERMINATES the segment: 6 significant digits would move it by up to 0.2 ms.
        trim_args = ["-t", _ffdur(duration)] if trim else []
    elif image is not None:
        cmd += ["-loop", "1", "-i", str(image)]
        if audio_path is not None:
            cmd += ["-i", str(audio_path)]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            silent_primary = True
        audio_map = "1:a"
        next_idx = 2
        video_vf = _image_vf(shot, settings, duration, work_dir, defer_caption=caption_after_zoom)
        trim_args = ["-t", f"{duration}"]
    else:
        is_text = True
        # textfile= reads the raw bytes of the file — drawtext does NO
        # escape interpretation on its contents (no \:, \,, \', \[, \]
        # gymnastics needed). The previous text= path with
        # _escape_drawtext was a constant cat-and-mouse: every new
        # special character we missed surfaced as ffmpeg exit=8 / 234
        # on the next LLM output. Investigated 2026-05-28 on website_review
        # run 47fde1d9 — even with `\,` `\:` `\'` correctly applied to a
        # 1.7 KiB critique, ffmpeg still rejected the filter spec with
        # exit=234 (EINVAL). textfile= sidesteps the entire class.
        #
        # Whitespace collapse (one line, no newlines) is preserved here —
        # drawtext's text rendering still can't word-wrap and the
        # single-line layout matches the prior visual.
        # Use the smaller body fontsize for description; the previous 64 was
        # tuned for a single short title and overflowed every full-sentence
        # script line (a 50-char line at fs=64 needs ~1760 px — past the
        # 1920-px canvas safe area). Wrapping via _wrap_for_canvas computes
        # the right line length for the font and canvas, and drawtext renders
        # each `\n`-separated chunk on its own baseline.
        #
        # Scaled off render_height (identity at 1080p = 56px) — the same
        # pattern the caption already follows (VC-6) — so this stays 5.19% of
        # canvas height at 1440p instead of shrinking to 3.89% (a flat 56px
        # was -25% relative there, packing ~76 chars/line via _wrap_for_canvas
        # instead of the intended ~55).
        body_fontsize = max(1, round(settings.render_height * _TEXT_CARD_BODY_FONTSIZE_FRAC))
        body_text = _wrap_for_canvas(
            shot.description or shot.id,
            fontsize=body_fontsize,
            canvas_width=settings.render_width,
        )
        text_path = work_dir / f"{shot.id}-text.txt"
        text_path.write_text(body_text, encoding="utf-8")
        # If the shot ALSO has a caption (the verbatim script line for this
        # beat), draw it bottom-of-frame on top of the description card so the
        # spoken narration is visible. Without this, the no-image branch was
        # silently dropping the caption that the orchestrator forwarded —
        # only image/video branches above ever rendered shot.caption.
        cap_fragment = ""
        if shot.caption:
            cap_fragment = "," + _caption_filter_fragment(shot.caption, settings, work_dir, shot.id)
        color = (
            f"color=c={_pad_color(settings)}:s={settings.render_width}x{settings.render_height}"
            # ``color``'s ``d`` is AV_OPT_TYPE_DURATION and ``duration`` reaches here by
            # the same probe-or-``shot.seconds`` route as the ``-t`` above — same crash
            # shape, and it must stay digit-for-digit equal to that ``-t``.
            f":d={_ffdur(duration)}:r={settings.render_fps}"
        )
        body_y = _text_card_body_y(shot.caption, body_text, body_fontsize, settings)
        # Path itself can contain no `:` / `\` problematic to drawtext because
        # the work_dir is a controlled tmp path with simple ASCII.
        drawtext = (
            f"drawtext=fontfile={settings.render_font}:textfile={text_path}"
            f":fontcolor={settings.render_text_color}:fontsize={body_fontsize}"
            f":line_spacing={_TEXT_CARD_BODY_LINE_SPACING_PX}"
            f":x=(w-text_w)/2:y={body_y}{cap_fragment}"
        )
        cmd += ["-f", "lavfi", "-i", color]
        if audio_path is not None:
            cmd += ["-i", str(audio_path)]
        else:
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
            silent_primary = True
        audio_map = "1:a"
        next_idx = 2
        video_vf = drawtext
        trim_args = ["-t", f"{duration}"]

    # T8 split-screen: composite a SECOND source beside the primary via hstack/vstack.
    # REBUILDS video_vf (half-canvas re-fit + stack) and injects the second ``-i``; a
    # no-op when the shot has no split or the source was unfetchable → byte-identical.
    # The directed-footage + speed seams below are then skipped on a split shot because
    # their coords/timeline assume the primary occupies the whole canvas.
    video_vf, next_idx = _inject_split(
        cmd,
        shot,
        second_clip,
        video_vf,
        next_idx,
        settings,
        work_dir,
        duration,
    )
    # A split composites two FINITE last-frame-clone halves (see
    # ``_cover_pad_stop_duration``), each ceiling'd at this same ``duration`` — but
    # that ceiling is only a WILLING-TO-CLONE bound, not the segment's real length;
    # the output still only EOFs when a `-t` bounds it. The own-audio clip branch
    # sets `trim=False` ⇒ `trim_args=[]`, which for a split would run the encode
    # past the clone ceiling with nothing to truncate it. Force the primary's
    # `duration` as the bound on any split shot that lacks one — the second half
    # just clones to fill.
    if shot.split is not None and not trim_args:
        trim_args = ["-t", f"{duration}"]

    # Directed footage (ORCH-3.2): synthetic cursor → callouts → auto-zoom, applied
    # after the fit/scale (+caption) branch and before the look. A pass-through when
    # no track is set, and SKIPPED on a split shot (its coords assume the full canvas).
    video_vf = _apply_directed_footage(
        video_vf,
        _shift_tracks_to_seek(shot, track_shift, duration),
        settings,
        duration,
        shot.speed,
        work_dir,
    )

    # The narration caption, drawn AFTER the punch-in (see :func:`_defers_caption`).
    # Before ``_apply_shot_look`` so shot titles still land on top of it, which is
    # where they sat when the caption was baked into the fit/scale branch.
    video_vf += _deferred_caption_suffix(shot, settings, work_dir, caption_after_zoom)

    # Per-shot look: colour grade (eq) + burned-in text overlays (drawtext),
    # appended AFTER scale/pad/format so the grade acts on the framed picture and
    # titles land in canvas coordinates. Both empty (no adjust / no overlays) is a
    # byte-identical no-op. This must sit BEFORE the speed retime below so the
    # setpts still leads the chain.
    video_vf = _apply_shot_look(video_vf, shot, settings, work_dir)

    # Per-clip primary-audio gain/mute/fade. Computed HERE — after every branch
    # above has resolved the segment's final PRE-SPEED ``duration`` (an own-audio
    # clip reassigns it to the clip length with trim off) — so the fade-out
    # timestamp (duration - fade_out_s) anchors to the real segment length, not
    # the ``shot.seconds`` fallback. Baked at the primary label below, before the
    # atempo retime rescales it. "" (audio is None / all-default) is the
    # byte-identical no-op, so where it's computed can't affect an unedited shot.
    #
    # ``own_audio_pad`` goes AFTER the edge fades (they belong to the clip's own last
    # samples) and is PRE-speed like every other time in this fragment. The ``-t``
    # trims an audio stream that outlasts the bound; the pad covers one that ends
    # before it.
    audio_frag = (
        _primary_audio_frag(
            shot.audio, duration, narration_seconds, narration_padded=narration_padded
        )
        + own_audio_pad
    )

    # Per-clip playback speed: retime the video (setpts, prepended so the
    # cadence fps filter downstream resamples the frame count) and the audio
    # (atempo). speed=1.0 is a no-op — both fragments are empty and the args are
    # byte-identical to before. The rendered segment now lasts duration/speed, so
    # the -t bound (when present) and the returned duration scale accordingly.
    # A split shot disables the speed retime: _apply_shot_speed prepends setpts to
    # only the FIRST chain, which would retime the primary half but not the second
    # → the two halves desync. Keep video_vf/trim_args as-is (atempo empty).
    # ``retime_audio`` is False only for a SYNTHESISED-SILENCE primary with no SFX
    # riding the same bus — see _apply_shot_speed for why atempo must not touch an
    # all-zero buffer. With SFX the atempo runs on the MIX (_video_audio_map_args),
    # which carries real samples and must stay locked to the picture.
    if shot.split is None:
        video_vf, atempo, out_duration, trim_args = _apply_shot_speed(
            video_vf,
            shot.speed,
            duration,
            trim_args,
            retime_audio=not silent_primary or has_sfx,
        )
    else:
        atempo, out_duration = "", duration

    # Alpha overlay clips (lower-thirds / logo bugs) composite LAST, on top of the
    # graded, titled, retimed picture — they are the finishing layer, not part of
    # the shot's own image. Appended here, still BEFORE _video_audio_map_args, so
    # the overlay ``-i`` inputs land ahead of the SFX inputs and every ``[N:a]``
    # label in the audio graph keeps its meaning (the invariant _inject_split
    # documents). Running AFTER the speed block is deliberate: _apply_speed_to_vf
    # prepends its ``setpts`` to the FIRST chain only, and by now that chain is
    # closed, so the overlay layers can never be retimed by it — which is exactly
    # why their own onsets are rescaled by ``speed`` instead. A split shot skips
    # the speed retime entirely, so its effective factor is 1.0.
    video_vf, next_idx = _inject_overlay_clips(
        cmd,
        overlay_clips,
        video_vf,
        next_idx,
        settings,
        shot.speed if shot.split is None else 1.0,
    )

    cmd += _video_audio_map_args(
        has_sfx=has_sfx,
        is_text=is_text,
        video_vf=video_vf,
        audio_map=audio_map,
        next_idx=next_idx,
        transition_sfx=transition_sfx,
        accent_sfx=accent_sfx,
        trim_args=trim_args,
        settings=settings,
        atempo=atempo,
        audio_frag=audio_frag,
        # PRE-speed length: accent `adelay` offsets are in the same timeline, and
        # the atempo retime is appended AFTER the SFX mix, so the accent tail fade
        # must be computed against `duration`, not the retimed `out_duration`.
        shot_seconds=duration,
    )
    cmd += [
        *_video_codec_args(settings, deterministic),
        # Lossless intermediate: this segment is stream-copied into the film by
        # ``concat_shots``, so it must carry NO encoder delay/padding. See
        # ``_SEGMENT_AUDIO_CODEC_ARGS``.
        *_segment_audio_codec_args(deterministic),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    await _run(cmd)
    return out_duration


def _video_audio_map_args(
    *,
    has_sfx: bool,
    is_text: bool,
    video_vf: str,
    audio_map: str,
    next_idx: int,
    transition_sfx: tuple[Path, SoundCue] | None,
    accent_sfx: list[tuple[Path, AccentCue]],
    trim_args: list[str],
    settings: Settings,
    atempo: str = "",
    audio_frag: str = "",
    shot_seconds: float = 0.0,
) -> list[str]:
    """Tail args for ``render_shot``: the SFX ``-i`` inputs (if any) plus the
    video/audio ``-filter_complex``/``-vf`` + ``-map`` + ``-t`` args.

    ``atempo`` (empty at speed=1.0) is the per-clip audio-speed retime folded
    into the audio graph. When it is present the audio can no longer be a bare
    ``-map`` — even the no-SFX paths route through ``-filter_complex`` to apply
    it — so the argument shape depends on BOTH ``has_sfx`` and ``atempo``.

    Shapes: (1) with SFX, video moves into the same ``-filter_complex`` as the
    audio mix (a plain ``-vf`` can't coexist with ``-filter_complex``); the SFX
    mix is retimed by appending ``atempo`` after the amix when requested.
    (2) a text card with no SFX and no atempo keeps the plain ``-vf`` drawtext
    path mapped from ``0:v`` — byte-identical to before — UNLESS ``video_vf`` is a
    multi-chain graph (a ``;`` from a directed-footage cursor sub-graph, or an
    alpha overlay clip's layer chain), which a plain ``-vf`` can't express, so it
    falls through to the filter_complex path. For an overlay clip that fall-through
    is mandatory, not merely nicer: a bare ``-vf`` cannot reference a second input.
    (3) any remaining case (clip/image, a cursor graph, or a text card that needs
    atempo) uses ``-filter_complex [0:v]…[v]`` with the audio either bare-mapped
    (no atempo) or retimed through ``[audio_map]atempo…[aout]``. The
    ``[0:v]{video_vf}[v]`` wrap closes a cursor graph correctly: its base chain
    takes ``[0:v]`` and its final label-less overlay receives ``[v]``.
    """
    if has_sfx:
        inputs: list[str] = []
        transition_spec: tuple[int, SoundCue] | None = None
        accent_specs: list[tuple[int, AccentCue]] = []
        idx = next_idx
        if transition_sfx is not None:
            path, cue = transition_sfx
            inputs += ["-i", str(path)]
            transition_spec = (idx, cue)
            idx += 1
        for path, accent in accent_sfx:
            inputs += ["-i", str(path)]
            accent_specs.append((idx, accent))
            idx += 1
        if atempo:
            audio_fc = _audio_sfx_graph(
                audio_map,
                transition_spec,
                accent_specs,
                settings,
                out_label="a_mix",
                audio_frag=audio_frag,
                shot_seconds=shot_seconds,
            )
            audio_fc += f";[a_mix]{atempo}[aout]"
        else:
            audio_fc = _audio_sfx_graph(
                audio_map,
                transition_spec,
                accent_specs,
                settings,
                audio_frag=audio_frag,
                shot_seconds=shot_seconds,
            )
        return [
            *inputs,
            "-filter_complex",
            f"[0:v]{video_vf}[v];{audio_fc}",
            "-map",
            "[v]",
            "-map",
            "[aout]",
            *trim_args,
        ]
    if is_text and not atempo and not audio_frag and ";" not in video_vf:
        return ["-vf", video_vf, "-map", "0:v", "-map", audio_map, *trim_args]
    if atempo or audio_frag:
        # audio_frag already leads with a comma and is designed to follow _A48.
        # The normalisation is UNCONDITIONAL: the speed-only path (audio_frag == "")
        # used to skip it and feed raw `[idx:a]` straight into atempo, making this
        # the one audio node in the service whose inputs were not provably 48
        # kHz/stereo — a 44.1 kHz or mono clip then reached the segment encoder
        # unnormalised and the concat joined mismatched streams.
        #
        # `_A48_SOXR`, not bare `_A48`: this is the PER-SHOT PRIMARY, i.e. the
        # narration clip, which arrives 24 kHz MONO from tts_backend. Bare `_A48`
        # resampled it 24k -> 48k on swr's default cubic, which leaves the source's
        # mirror image ~50 dB under the program in 12-16 kHz; soxr leaves none.
        # See `_A48_SOXR` for the measurement. On a shot whose audio is already 48
        # kHz (a recorded clip) `aresample=48000` is a byte-identical pass-through,
        # so this costs those shots nothing.
        pre = f"{_A48_SOXR}{audio_frag}"
        atempo_seg = ("," + atempo) if atempo else ""
        return [
            "-filter_complex",
            f"[0:v]{video_vf}[v];[{audio_map}]{pre}{atempo_seg}[aout]",
            "-map",
            "[v]",
            "-map",
            "[aout]",
            *trim_args,
        ]
    return [
        "-filter_complex",
        f"[0:v]{video_vf}[v]",
        "-map",
        "[v]",
        "-map",
        audio_map,
        *trim_args,
    ]


#: The audio-stream fields that must be IDENTICAL across every segment a
#: stream-copy concat joins. The concat demuxer exposes ONE codec context (the
#: first file's), so any difference here is decoded with the wrong parameters —
#: for ALAC a bit-depth mismatch decodes as broadband noise, not an error.
_CONCAT_AUDIO_FIELDS = (
    "codec_name",
    "sample_fmt",
    "bits_per_raw_sample",
    "sample_rate",
    "channels",
)


def _segment_audio_signature(probe_json: str) -> tuple[str, ...] | None:
    """PURE: the ``_CONCAT_AUDIO_FIELDS`` of the first audio stream in an ffprobe
    JSON payload, as strings; ``None`` when the payload carries no audio stream.

    Raises ``FfmpegError`` on an unparseable payload — an unreadable segment is not
    evidence the join is safe."""
    try:
        payload = json.loads(probe_json)
    except ValueError as exc:
        raise FfmpegError(f"segment audio probe returned unparseable output: {exc}") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        return None
    stream = streams[0]
    return tuple(str(stream.get(field, "")) for field in _CONCAT_AUDIO_FIELDS)


async def _assert_uniform_segment_audio(segment_paths: Sequence[Path]) -> None:
    """Refuse a stream-copy concat whose segments' audio parameters differ.

    ``-c copy`` cannot reconcile them and does not fail: it muxes a stream whose
    single codec header describes only the FIRST segment, and the decoder turns
    every other segment into noise (see ``_SEGMENT_AUDIO_CODEC_ARGS``). Every
    segment producer pins the same args, so a mismatch here is a pipeline defect
    (a new producer, a stale cache entry) — surfaced LOUDLY as a deterministic
    ``FfmpegError`` rather than delivered as a master with no narration."""
    if len(segment_paths) < 2:
        return
    signatures: list[tuple[str, ...] | None] = []
    for path in segment_paths:
        out = await _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=" + ",".join(_CONCAT_AUDIO_FIELDS),
                "-of",
                "json",
                str(path),
            ]
        )
        signatures.append(_segment_audio_signature(out))
    if len(set(signatures)) == 1:
        return
    detail = "; ".join(
        f"{p.name}: "
        + (
            "no audio stream"
            if sig is None
            else ",".join(f"{k}={v}" for k, v in zip(_CONCAT_AUDIO_FIELDS, sig, strict=True))
        )
        for p, sig in zip(segment_paths, signatures, strict=True)
    )
    log.error("renderer.concat.segment_audio_mismatch", segments=detail)
    raise FfmpegError(
        "refusing stream-copy concat: segment audio parameters differ "
        f"(would decode as noise): {detail}"
    )


async def concat_shots(
    segment_paths: list[Path], out_path: Path, deterministic: bool = False
) -> None:
    """Concat uniformly-encoded segments via the ffmpeg concat demuxer.

    Stays a stream-copy join (``-c copy``); ``deterministic`` only adds the global
    ``-fflags +bitexact`` so the concatenated container carries no wall-clock
    ``creation_time``/writing-app tag. No re-encode tokens are added.

    "Uniformly-encoded" is VERIFIED, not assumed: ``_assert_uniform_segment_audio``
    probes every segment and raises before joining if their audio differs."""
    await _assert_uniform_segment_audio(segment_paths)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file {shlex.quote(str(p.resolve()))}" for p in segment_paths) + "\n"
    )
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        await _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


_MIX_SR = 48000


def _timeline_crossfade_mix(
    pieces: Sequence[tuple[str, float, float, float, float]],
    out_label: str,
    *,
    curve: str = "tri",
) -> list[str]:
    """PURE: crossfade audio pieces on a timeline WITHOUT ``acrossfade``.

    ``pieces`` is ``(source_head, length, fade_in, fade_out, start)`` per piece, in
    timeline order. ``source_head`` is a filter prefix that yields the piece's raw
    audio and ends ready for another filter (``"[3:a]"`` or ``"[3:a]<norm>,"``).
    Each piece is conformed to exactly ``length`` (``atrim``+``apad``), ramped in and
    out (``afade``, ``curve=`` below), placed at
    ``start`` with a SAMPLE-exact ``adelay``, and all pieces are summed by
    ``amix=normalize=0`` into ``[out_label]``.

    ``curve`` picks the crossfade LAW, and the choice depends on whether the two
    sides of a seam are correlated:

    * ``tri`` (default, equal-GAIN — also ``acrossfade``'s default) is right for
      a seam between two DIFFERENT segments, where the caller is ramping one
      programme out and an unrelated one in.
    * ``qsin`` (equal-POWER) is right for a seam that joins ONE asset to ITSELF
      at a different offset — a music-bed loop point. Those two sides are
      decorrelated, so their sum is ``sqrt(a^2 + b^2)``, and an equal-gain pair
      (``a + b = 1``) lands at ``0.707`` = **-3 dB** at the seam centre: a
      periodic level dip every ``asset_seconds`` under the music.

    MEASURED (pink noise, two decorrelated 4.0 s reads, ``xfade=0.25``, RMS over
    the 0.15 s straddling the seam centre at 3.875 s, versus the steady single-
    source level either side)::

        curve   steady_pre   seam_centre   steady_post
        tri     -23.61 dBFS  -26.38 dBFS   -22.66 dBFS    <- -2.77 dB DIP
        qsin    -24.02 dBFS  -24.32 dBFS   -22.66 dBFS    <- -0.29 dB, flat

    The default stays ``tri`` so the segment-join caller (``concat``/xfade) is
    byte-identical; only the bed loop opts into ``qsin``.

    ⛔ WHY NOT ``acrossfade``: its output depends on the arrival ORDER of its two
    inputs' frames/EOF, i.e. on thread timing. Measured with docker ``--cpus=1``
    (2 x 1.0 s pieces, d=0.5, correct = 72000 samples): ffmpeg 7.1.5 (prod image)
    emitted 0 / 576 / 12864 / 16960 / 21056-sample stubs, audio-less files, or
    "Could not open encoder before EOF"; ffmpeg 6.1.1 (PR CI) emitted 10/10 group
    files with NO audio stream. This construction: 60/60 (2 pieces) and 60/60
    (3 pieces) sample-exact on both versions at 1 CPU, 180/180 muxed with video.
    """
    parts: list[str] = []
    labels: list[str] = []
    for k, (head, length, fade_in, fade_out, start) in enumerate(pieces):
        sep = "" if head.endswith((",", "]")) else ","
        chain = f"{head}{sep}atrim=end={length:.6f},apad=whole_dur={length:.6f}"
        if fade_in > 0.0:
            chain += f",afade=t=in:curve={curve}:st=0:d={fade_in:.6f}"
        if fade_out > 0.0:
            chain += (
                f",afade=t=out:curve={curve}:st={max(0.0, length - fade_out):.6f}:d={fade_out:.6f}"
            )
        delay = round(start * _MIX_SR)
        if delay > 0:
            chain += f",adelay=delays={delay}S:all=1"
        label = f"[{out_label}_p{k}]"
        parts.append(chain + label)
        labels.append(label)
    parts.append(
        "".join(labels) + f"amix=inputs={len(labels)}:duration=longest:normalize=0[{out_label}]"
    )
    return parts


# ---------------------------------------------------------------------------
# Per-clip visual transitions (xfade video + acrossfade audio)
# ---------------------------------------------------------------------------
# Map a Transition.type to an ffmpeg ``xfade`` transition name. ``dissolve`` is
# an alias for the plain cross-dissolve ``fade``; wipe/slide are native names.
_XFADE_TRANSITIONS: dict[str, str] = {
    "fade": "fade",
    "dissolve": "fade",
    "wipeleft": "wipeleft",
    "slideright": "slideright",
}


def _xfade_groups(transitions: Sequence[Transition | None]) -> list[list[int]]:
    """Segment indices partitioned at hard cuts: a ``None`` transition starts a new
    group, a real one extends the current group (``transitions[0]`` is ignored).

    The ONE partition, shared by ``xfade_concat_shots`` (what is rendered) and
    ``_bus_offsets_xfade`` (where the audio edits are told each shot sits)."""
    groups: list[list[int]] = [[0]] if transitions else []
    for i in range(1, len(transitions)):
        if transitions[i] is None:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _xfade_group_joins(
    lengths: Sequence[float], transitions: Sequence[Transition | None], fps: int
) -> list[tuple[float, float]]:
    """``(overlap, offset)`` of every segment in ONE crossfaded group, group-relative:
    the crossfade INTO segment ``i`` lasts ``overlap`` and begins at ``offset``,
    which is also where segment ``i`` starts. Entry 0 is ``(0.0, 0.0)``. The group's
    composite length is ``joins[-1][1] + lengths[-1]``.

    The ONE place this arithmetic lives. ``_render_xfade_group`` feeds these exact
    numbers to ``xfade`` and to the audio placement; ``_bus_offsets_xfade`` returns
    them as the per-shot bus windows. They used to be computed twice — the renderer
    on the frame grid, the bus offsets with an unsnapped ``min(requested, prev,
    cur)`` — and disagreed by up to half a frame per off-grid join, compounding.

    ⛔ ON THE FRAME GRID, both numbers — see :func:`_snap_to_frame_grid`. ``xfade``
    passes the left side through while ``pts < offset`` and then consumes the right
    side frame for frame, so the picture is ``ceil(offset * fps)`` frames plus the
    right side WHATEVER the overlap is; the audio piece is placed at ``offset`` to
    the SAMPLE. An off-grid offset therefore leaves the group's picture up to a
    frame longer than its audio, and a group file is stream-copied into the film by
    ``concat_shots`` like any other segment: a hole in the audio timestamps at the
    hard cut after it. So the overlap goes to the NEAREST frame, bounded by the
    shorter neighbour's WHOLE frames (a crossfade longer than either neighbour
    would run past that clip's own length — a negative offset / a black hole in the
    timeline), and the offset follows from it on the grid."""
    joins: list[tuple[float, float]] = [(0.0, 0.0)]
    running = lengths[0]
    for i in range(1, len(lengths)):
        tr = transitions[i] or Transition()
        d = min(
            _nearest_frame(tr.duration, fps),
            _snap_to_frame_grid(min(lengths[i - 1], lengths[i]), fps, up=False),
        )
        offset = _nearest_frame(running - d, fps) if running - d > 0.0 else 0.0
        joins.append((d, offset))
        running = offset + lengths[i]
    return joins


def _bus_offsets_plain(lengths: list[float]) -> list[tuple[float, float]]:
    """Absolute ``[start, end)`` of each shot on a plain stream-copy concat bus.

    No overlap: ``start_k = sum(lengths[:k])``, ``end_k = start_k + lengths[k]``.
    Lengths are the PROBED segment durations (the bytes the concat demuxer
    stitches), never ``render_shot``'s nominal ``out_duration`` — a deband
    re-encode can drift the nominal value sub-frame, which would smear the
    time-ranged post-clip window off the edited shot."""
    offsets: list[tuple[float, float]] = []
    start = 0.0
    for length in lengths:
        offsets.append((start, start + length))
        start += length
    return offsets


def _bus_offsets_xfade(
    lengths: list[float], transitions: list[Transition | None], fps: int
) -> list[tuple[float, float]]:
    """Absolute ``[start, end)`` of each shot on the crossfaded bus.

    Not a mirror of the renderer's math but the SAME math: the hard-cut partition
    is ``_xfade_groups`` (as in ``xfade_concat_shots``) and every in-group start is
    the ``offset`` ``_xfade_group_joins`` hands ``_render_xfade_group`` for that
    very join, from the same probed ``lengths`` and the same ``fps``. A group
    starts flush against the previous group's composite end (no overlap), which is
    the duration ``_render_xfade_group`` returns for it.

    So ``start_k`` is on the output frame grid, NOT ``start_{k-1} + L_{k-1} - d_k``
    to the sample: with off-grid lengths the two differ by up to half a frame.
    Worked example on the grid (durations [4,6,7], xfade d=1 between all):
    [(0,4),(3,9),(8,15)]."""
    offsets: list[tuple[float, float]] = []
    base = 0.0
    for group in _xfade_groups(transitions):
        group_lengths = [lengths[i] for i in group]
        joins = _xfade_group_joins(group_lengths, [transitions[i] for i in group], fps)
        for (_overlap, offset), length in zip(joins, group_lengths, strict=True):
            offsets.append((base + offset, base + offset + length))
        base += joins[-1][1] + group_lengths[-1]
    return offsets


async def _render_xfade_group(
    segment_paths: list[Path],
    lengths: list[float],
    transitions: list[Transition | None],
    out_path: Path,
    settings: Settings,
    deterministic: bool = False,
) -> float:
    """xfade + ramped-amix crossfade a run of >=2 uniform segments into one re-encoded file.

    ``transitions[i]`` (for i>=1) is the crossfade INTO segment ``i`` from the
    accumulated left side; ``transitions[0]`` is ignored (no preceding segment).
    Each pair is folded with ``xfade`` (video) + a ramped audio crossfade; the
    ``xfade`` offset is the running composite length minus that pair's overlap,
    so overlaps compound correctly down the chain. Returns the composite duration
    (sum of segment lengths minus the sum of overlaps).

    ⚠️ **4c: every ``[i:a]`` is conformed to ``lengths[i]`` (``atrim``+``apad``)
    BEFORE the crossfade.** (Historical measurement below is for ``acrossfade``,
    since replaced — see the comment in the body.) ``acrossfade`` takes no offset — it always
    crossfades the literal tail of its left input against the literal head of its
    right input, so its overlap point is wherever the AUDIO STREAM happens to end.
    ``xfade``'s ``offset=`` is computed from ``lengths[i]`` (the probed VIDEO
    stream duration, see ``_ffprobe_duration``). Any segment whose audio stream
    length differs from its video length — an ALAC frame boundary, an encoder
    priming/remainder sample, anything short of exact equality — puts the two
    joins at different points in the timeline, and the gap COMPOUNDS across every
    transition in the group. Forcing each audio input to exactly ``lengths[i]``
    (``atrim=end=`` cannot lengthen, ``apad=whole_dur=`` cannot shorten, so
    together they land on exactly that value regardless of which side the real
    stream erred on) makes ``acrossfade``'s overlap point identical to the offset
    ``xfade`` was given, by construction. Measured (wave 8, synthetic 4.08s/5.95s
    audio against declared 4.0s/6.0s video lengths, d=1.0): unconformed output
    duration 9.030s (0.03s off, and the crossfade lands 0.08s into the first
    segment's audio where the video crossfade lands at 3.0s exactly); conformed
    output lands at exactly 9.000s, matching ``running``'s math below.
    """
    cmd: list[str] = ["ffmpeg", "-y", *_bitexact_input_args(deterministic)]
    for path in segment_paths:
        cmd += ["-i", str(path)]
    fps = settings.render_fps
    # ⛔ Audio is NOT joined with ``acrossfade`` — it races (see
    # :func:`_timeline_crossfade_mix` for the measurements). Each segment is ramped
    # over its transition windows and placed at its xfade offset instead.
    parts: list[str] = []
    overlaps: list[float] = [0.0] * (len(segment_paths) + 1)
    v_prev = "[0:v]"
    # The overlap and offset of every join come from the ONE helper that
    # ``_bus_offsets_xfade`` also reads (frame-grid reasoning lives there), and the
    # audio below uses the very same two numbers.
    joins = _xfade_group_joins(lengths, transitions, fps)
    starts: list[float] = [offset for _d, offset in joins]
    for i in range(1, len(segment_paths)):
        tr = transitions[i] or Transition()
        d, offset = joins[i]
        xname = _XFADE_TRANSITIONS.get(tr.type, "fade")
        v_out = f"[vx{i}]"
        # Six decimals, not three: ``xfade`` rounds ``offset`` onto the stream
        # timebase, and 365/30 printed as 12.167 is past frame 365, not on it.
        parts.append(
            f"{v_prev}[{i}:v]xfade=transition={xname}:duration={d:.6f}:offset={offset:.6f}{v_out}"
        )
        v_prev = v_out
        overlaps[i] = d
    running = starts[-1] + lengths[-1]
    # `atrim=end=` alone would only ever shorten; `apad=whole_dur=` alone would only
    # ever lengthen. Both together (inside the helper) land every input on exactly
    # `lengths[i]` regardless of which side the real audio stream erred on.
    parts += _timeline_crossfade_mix(
        [
            (f"[{i}:a]", length, overlaps[i], overlaps[i + 1], starts[i])
            for i, length in enumerate(lengths)
        ],
        "ax",
    )
    a_prev = "[ax]"
    cmd += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        v_prev,
        "-map",
        a_prev,
        # ⛔ INTERMEDIATE CELL, and it is not optional. The `xfade` filter_complex
        # above spans the ENTIRE pair — not just the transition window — so EVERY
        # frame of EVERY shot in a crossfaded group is decoded and re-encoded here.
        # At the delivery cell that is a second full x264 generation re-quantising
        # material generation 1 already quantised at that exact cell, which is
        # precisely what `render_intermediate_crf_delta` exists to prevent, and it
        # was happening on the deliverable rather than on a temp file. Sibling
        # segment re-encodes (`_deband_segment` / `_regrade_segment` /
        # `_polish_segment`) already pass this; the group encode was the one that
        # did not. Still concat-stream-copy compatible: `_video_codec_args` changes
        # only the CRF cell and drops VBV — the colour stamp, the CFR pin, the GOP
        # and the pixel format are identical, which is what `concat_shots` checks.
        *_video_codec_args(settings, deterministic, intermediate=True),
        # A crossfaded group file is itself fed to the stream-copy
        # ``concat_shots`` below, so it must stay delay-free lossless too —
        # re-encoding it to AAC here would reintroduce exactly the join gap this
        # pipeline removes (and a second lossy generation).
        *_segment_audio_codec_args(deterministic),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    await _run(cmd)
    return running


async def xfade_concat_shots(
    segment_paths: list[Path],
    transitions: list[Transition | None],
    out_path: Path,
    settings: Settings,
    lengths: list[float] | None = None,
    deterministic: bool = False,
) -> float:
    """Join per-shot segments with crossfades, preserving hard cuts.

    ``transitions`` is index-aligned with ``segment_paths``: ``transitions[i]``
    is the transition INTO segment ``i`` (``transitions[0]`` is ignored). A
    ``None`` entry is a HARD CUT and starts a new group; consecutive segments
    linked by a transition form one crossfaded group. Each group of >=2 is
    ``xfade``/``acrossfade``-joined into a re-encoded (but still uniform:
    same codec/fps/pixfmt) file, singleton groups pass through untouched, and the
    group files are finally stitched with the fast stream-copy ``concat_shots``.
    A single crossfaded run renders straight to ``out_path`` (no extra concat).

    Sibling of ``concat_shots`` so the no-transition fast path stays untouched.
    Returns the delivered duration (segment lengths minus the summed overlaps).

    ``lengths`` may be supplied by the caller (``_join_segments`` probes each
    segment ONCE and shares the result with the offset math) so the join and the
    post-clip bus offsets are computed from byte-identical durations; when None
    they are probed here (the standalone / test path)."""
    if lengths is None:
        lengths = [await _ffprobe_duration(p) or 0.0 for p in segment_paths]
    # One transition per segment: a shorter list would silently drop the trailing
    # segments from the partition below.
    if len(transitions) < len(segment_paths):
        raise ValueError(
            f"xfade join needs one transition per segment: {len(transitions)} for {len(segment_paths)}"
        )
    # Partition at hard-cut (None) boundaries.
    groups = _xfade_groups(transitions[: len(segment_paths)])
    # No actual crossfade boundary (every group is a singleton) → plain concat.
    if all(len(g) == 1 for g in groups):
        await concat_shots(segment_paths, out_path, deterministic)
        return float(sum(lengths))
    # A single crossfaded run covers the whole timeline → render to out_path.
    if len(groups) == 1:
        return await _render_xfade_group(
            segment_paths, lengths, transitions, out_path, settings, deterministic
        )
    # Mixed cuts + crossfades: render each >=2 group, pass singletons through,
    # then stream-copy concat the (uniform) group files.
    group_files: list[Path] = []
    total = 0.0
    for gi, g in enumerate(groups):
        if len(g) == 1:
            group_files.append(segment_paths[g[0]])
            total += lengths[g[0]]
        else:
            gf = out_path.with_name(f"{out_path.stem}-xf{gi:03d}{_SEGMENT_EXT}")
            total += await _render_xfade_group(
                [segment_paths[i] for i in g],
                [lengths[i] for i in g],
                [transitions[i] for i in g],
                gf,
                settings,
                deterministic,
            )
            group_files.append(gf)
    await concat_shots(group_files, out_path, deterministic)
    return total


def _build_shot_timeline(
    shots: Sequence[Shot], shot_offsets: Sequence[tuple[float, float]]
) -> list[dict[str, object]]:
    """Zip each shot to its probed [start, end) bus offset → authoritative per-shot
    timeline keyed by line_id. Index-aligned (offsets come from the same ordered shot
    list). A shot with no line_id emits line_id=None."""
    return [
        {"line_id": s.line_id, "start": float(start), "end": float(end)}
        for s, (start, end) in zip(shots, shot_offsets, strict=False)
    ]


def _has_transitions(shots: list[Shot]) -> bool:
    """True when any shot PAST THE FIRST carries a transition. The first shot's
    transition has no preceding shot to dissolve from and is ignored, so a
    single-shot compile (``shots[1:]`` empty) is always False."""
    return any(shot.transition is not None for shot in shots[1:])


async def _join_segments(
    shots: list[Shot],
    segment_paths: list[Path],
    total_seconds: float,
    out_path: Path,
    settings: Settings,
    need_offsets: bool = False,
    deterministic: bool = False,
) -> tuple[float, list[tuple[float, float]]]:
    """Join the rendered per-shot segments into ``out_path``; return the delivered
    total duration AND each shot's absolute ``[start, end)`` on the final bus.

    Fast stream-copy concat (byte-identical to before WS-2) unless a shot carries
    a transition, in which case the re-encoding xfade/acrossfade chain is used —
    which OVERLAPS shots, so the returned total shrinks by the summed overlaps.

    The bus offsets feed the POST-master per-clip gain/mute pass. They are
    computed from the PROBED segment lengths — the same durations the join
    stitches — so a time-ranged ``volume`` node lands exactly on its shot. Probing
    (an extra N ffprobe reads on the plain path) only happens when
    ``need_offsets`` is set (a shot actually requests gain/mute); otherwise the
    offsets are empty and the plain path stays as cheap as before.
    """
    if _has_transitions(shots):
        transitions: list[Transition | None] = [shot.transition for shot in shots]
        lengths = [await _ffprobe_duration(p) or 0.0 for p in segment_paths]
        total = await xfade_concat_shots(
            segment_paths,
            transitions,
            out_path,
            settings,
            lengths=lengths,
            deterministic=deterministic,
        )
        offsets = (
            _bus_offsets_xfade(lengths, transitions, settings.render_fps) if need_offsets else []
        )
        return total, offsets
    await concat_shots(segment_paths, out_path, deterministic)
    if need_offsets:
        lengths = [await _ffprobe_duration(p) or 0.0 for p in segment_paths]
        return total_seconds, _bus_offsets_plain(lengths)
    return total_seconds, []


def _grade_video_args(grade_vf: str, settings: Settings, deterministic: bool = False) -> list[str]:
    """The video-map ffmpeg args for a re-encoding audio pass: fold the whole-film
    grade (VC-5) into the video map when ``grade_vf`` is non-empty — the shared
    encode ladder (``_video_codec_args``), which itself emits ``-r render_fps``
    to re-assert CFR (VC-3; ``_video_codec_args`` is the single source of that
    flag now — do not append a second ``-r`` here). With no grade the video
    stays a plain ``-c:v copy`` stream so the no-grade path is byte-identical.
    ``deterministic`` threads the bit-reproducible encode flags into the graded
    video map (a ``-c:v copy`` stream needs none)."""
    from src.services import mastering

    if grade_vf:
        return [
            "-map",
            "0:v",
            "-vf",
            grade_vf,
            *_video_codec_args(settings, deterministic),
        ]
    # ⛔ RE-ASSERT THE COLOUR TRIPLET ON THE COPY BRANCH. Measured on three
    # delivered masters (2026-09-13): each carried `color_space=bt709` with
    # `color_primaries` and `color_transfer` BOTH `unknown` — which is the entire
    # reason `colorspace_ok` reads false on 16/16 masters. The check is correct;
    # the tags really are missing.
    #
    # They are not missing at the encode. `_video_codec_args` emits all three, a
    # bench with its verbatim args tags correctly, and a `-f concat -c copy` join
    # of two such segments PRESERVES all three. The loss is at THIS pass — the
    # last one to touch video before the master ships — where a `-c:v copy` video
    # map rides in the same invocation as an active audio `-filter_complex` and so
    # depends entirely on the muxer propagating the input's codecpar. Production
    # runs ffmpeg 7.1.5; the behaviour reproduces as intact on 4.4, so the
    # propagation is version-dependent and must not be relied on.
    #
    # These flags are legal alongside `-c:v copy` (they rewrite codecpar without
    # re-encoding), so this costs nothing and makes the delivered master's tagging
    # independent of muxer behaviour instead of incidental to it.
    #
    # CORRECTION (measured on 7.1.5, 2026-10-06): the diagnosis above was wrong
    # about WHERE. The tags were lost at the ENCODE (see
    # `mastering.colorspace_bsf_args`), and these flags do NOT repair a stream
    # that arrives without them — `-c:v copy` + all three flags over a
    # half-tagged input still probes `bt709/unknown/unknown`, because ffprobe
    # reads the H.264 SPS VUI and a stream copy never rewrites it. A correctly
    # stamped input keeps all three through this pass with or without the flags.
    # They are harmless and stay; the `h264_metadata` stamp in
    # `_video_codec_args` is what tags the master.
    return ["-map", "0:v", "-c:v", "copy", *mastering.colorspace_tag_args(settings)]


# Seam-crossfaded bed looping.
#
# ⚠️ WHY NOT `-stream_loop`: it BUTT-JOINS the asset to itself. The last sample of
# repetition N and the first of N+1 are at whatever level the asset happens to
# start/end on, so a loop of a 45 s track under a 4-minute film emits a step
# discontinuity — an audible CLICK — every 45 s, forever. `_music_bed_filter`
# fades only the FILM's head and tail, so it never touched a single interior seam.
# The fix reads the asset N times and crossfades consecutive repetitions
# (:func:`_timeline_crossfade_mix`), so every seam is a continuous 0.25 s linear
# ramp instead of a step.
_BED_LOOP_XFADE_S = 0.25
# In/out ramp on every ambient WINDOW edge. Not a schema knob: `AmbientCue` has no
# fade fields and the fix is a de-clicking requirement, not a creative choice.
_AMBIENT_EDGE_FADE_S = 0.15
# Above this, the crossfade mix (and the input list) stops being worth it —
# a pathologically short asset under a long film. Such a bed falls back to the
# old butt-joined `-stream_loop -1` rather than minting hundreds of decoders.
_BED_LOOP_MAX_COPIES = 60


def _bed_loop_plan(asset_seconds: float, needed: float) -> tuple[int, float]:
    """How many reads of a bed asset to crossfade together to cover ``needed``.

    Returns ``(copies, xfade_seconds)``:

    * ``(1, 0.0)`` — the asset already covers ``needed``; do NOT loop at all
      (today's behaviour: one ``-i``, ``atrim`` does the rest).
    * ``(n>=2, d)`` — read the asset ``n`` times and crossfade the seams.
      ``n`` repetitions crossfaded by ``d`` cover ``n*L - (n-1)*d``.
    * ``(0, 0.0)`` — NO plan: the probe failed (``_ffprobe_duration`` returns
      0.0) or the asset is too short to cover ``needed`` within
      ``_BED_LOOP_MAX_COPIES``. The caller must keep ``-stream_loop -1``; a
      clicking bed is worse than nothing but far better than no bed.
    """
    if not math.isfinite(asset_seconds) or asset_seconds <= 0.0:
        return 0, 0.0
    if asset_seconds >= needed:
        return 1, 0.0
    xfade = min(_BED_LOOP_XFADE_S, asset_seconds / 4.0)
    if xfade <= 0.0 or asset_seconds <= xfade:
        return 0, 0.0
    copies = math.ceil((needed - xfade) / (asset_seconds - xfade))
    if copies < 2 or copies > _BED_LOOP_MAX_COPIES:
        # ⛔ BOTH arms fall back to NO PLAN. `copies < 2` is unreachable given
        # asset < needed, but a float-edge 1 would silently produce a SHORT bed.
        # `copies > _BED_LOOP_MAX_COPIES` used to RETURN THE OVERSIZED COUNT — the
        # cap named in this function's own docstring did nothing, and the caller
        # opened `copies` separate `-i` decoders in a 2 Gi pod (a 20 s bed under a
        # 30 min film = 90 inputs). The docstring's contract is `(0, 0.0)` => the
        # caller keeps `-stream_loop -1`: a butt-joined, occasionally clicking bed,
        # which is far better than an OOM-killed render.
        return 0, 0.0
    return copies, xfade


def _bed_loop_filter(
    first_idx: int, copies: int, xfade: float, out_label: str, asset_seconds: float
) -> str:
    """Crossfade ``copies`` independent reads of one bed asset into ``[out_label]``.

    Separate ``-i`` inputs rather than one ``asplit``: an ``asplit`` branch would
    have to buffer the whole decoded asset in RAM per pending repetition (tens of
    MB each, in a 2Gi pod). Independent inputs each decode lazily.

    Every copy is normalised (``_A48_ASSET``) first, conformed to the probed
    ``asset_seconds``, and placed at ``k * (asset_seconds - xfade)`` with its seams
    ramped — :func:`_timeline_crossfade_mix`, NOT a chain of ``acrossfade``s, which
    truncated or dropped audio depending on thread timing.
    """
    step = asset_seconds - xfade
    pieces = [
        (
            f"[{first_idx + k}:a]{_A48_ASSET},",
            asset_seconds,
            xfade if k > 0 else 0.0,
            xfade if k < copies - 1 else 0.0,
            k * step,
        )
        for k in range(copies)
    ]
    # ``qsin`` (equal-POWER), NOT the default ``tri``: both sides of a loop seam
    # are the SAME asset at different offsets, i.e. decorrelated, so an equal-gain
    # sum dips -3 dB at every seam centre — a periodic duck under the music once
    # per ``asset_seconds``. Measured -2.77 dB with ``tri`` vs -0.29 dB with
    # ``qsin`` (see :func:`_timeline_crossfade_mix`).
    return ";".join(_timeline_crossfade_mix(pieces, out_label, curve="qsin"))


def _ambient_window_filter(
    idx: int,
    window: float,
    gain_db: float,
    delay: str,
    out_label: str,
    *,
    src_label: str = "",
) -> str:
    """Filtergraph for one ambient cue's window → ``[out_label]``.

    ⚠️ THE EDGE FADES ARE THE POINT. ``atrim``+``volume`` alone start and stop the
    bed at FULL LEVEL on a non-zero sample — a click into and out of every ambient
    section — and ``AmbientCue`` carries no fade fields to ask for anything else.
    The fades sit BEFORE ``adelay`` so ``st`` is in the trimmed window's own
    (``asetpts``-rebased) timeline, and the ramp is halved for a window too short
    to hold both (``window`` floors at 0.1 s upstream): a negative ``st`` is not a
    quiet degradation, it aborts the render.

    ``src_label`` consumes the already-normalised seam-crossfaded loop label
    instead of reading ``[idx:a]`` directly (see :func:`_append_bed_input`).
    """
    fade = min(_AMBIENT_EDGE_FADE_S, window / 2.0)
    fades = ""
    if fade > 0:
        fades = (
            f",afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, window - fade):.3f}:d={fade:.3f}"
        )
    head = f"[{src_label}]" if src_label else f"[{idx}:a]{_A48_ASSET},"
    return (
        f"{head}atrim=duration={window:.3f},asetpts=PTS-STARTPTS,"
        f"volume={gain_db}dB{fades}{delay}[{out_label}]"
    )


async def _append_bed_input(
    cmd: list[str], parts: list[str], path: Path, first_idx: int, needed: float, label: str
) -> tuple[int, str]:
    """Append a bed asset's ``-i`` input(s) (and its loop filter, if any) in place.

    Returns ``(inputs_consumed, src_label)``. ``src_label`` is the crossfaded-loop
    output label when a loop was minted, and ``""`` when the caller's own
    ``[idx:a]{_A48_ASSET}`` head applies (asset already long enough, or unprobeable
    and falling back to ``-stream_loop -1``).
    """
    asset_seconds = await _ffprobe_duration(path)
    copies, xfade = _bed_loop_plan(asset_seconds, needed)
    if copies >= 2:
        cmd += ["-i", str(path)] * copies
        parts.append(_bed_loop_filter(first_idx, copies, xfade, label, asset_seconds))
        return copies, label
    if copies == 1:
        cmd += ["-i", str(path)]
        return 1, ""
    cmd += ["-stream_loop", "-1", "-i", str(path)]
    return 1, ""


def _music_bed_filter(
    idx: int, total: float, music_gain_db: float, bed: MusicBed, *, src_label: str = ""
) -> str:
    """Filtergraph for the full-length music bed → ``[bed_music]``.

    Guards the fades: a 0-duration ``afade`` (``fade_in_s``/``fade_out_s`` allow ge=0) is
    degenerate and can abort the pass. Omitting it IS "no fade" — semantically identical.
    Mirrors the per-shot ``_apply_shot_audio`` guard.

    ``src_label`` (e.g. ``"bed_src_music"``) consumes an ALREADY-normalised
    upstream label — the seam-crossfaded loop from :func:`_bed_loop_filter` — and
    skips the ``[idx:a]`` read + ``_A48_ASSET`` (the loop applied it per copy).
    Empty (the default) is the unlooped path and stays byte-identical.
    """
    fade_out_start = max(0.0, total - bed.fade_out_s)
    head = f"[{src_label}]" if src_label else f"[{idx}:a]{_A48_ASSET},"
    chain = [
        f"atrim=duration={total:.3f}",
        "asetpts=PTS-STARTPTS",
        f"volume={music_gain_db}dB",
    ]
    # ``_ffdur`` for the same reason as the shot-edge fades: ``MusicBed.fade_in_s`` /
    # ``fade_out_s`` are operator-supplied with ``minimum: 0`` and NO upper bound and no
    # exclusive lower one, so a sub-100µs fade is a legal request that ``> 0`` accepts
    # and a bare f-string would render as ``5e-05`` into a DURATION option.
    if bed.fade_in_s > 0:
        chain.append(f"afade=t=in:d={_ffdur(bed.fade_in_s)}")
    if bed.fade_out_s > 0:
        chain.append(f"afade=t=out:st={fade_out_start:.3f}:d={_ffdur(bed.fade_out_s)}")
    return head + ",".join(chain) + "[bed_music]"


def _bed_pass2_cmd(
    extra_chain: str,
    dest: Path,
    *,
    input_cmd: list[str],
    parts: list[str],
    master_chain: str,
    tail: list[str],
) -> list[str]:
    """``overlay_audio_bed``'s pass-2 command, optionally with ``extra_chain``
    appended to the master node (the LAST part on the master-on branch — the only
    branch ``_correct_master_loudness_miss`` ever reaches), writing to ``dest``."""
    graph = parts
    if extra_chain:
        graph = [*parts[:-1], f"[premaster]{master_chain},{extra_chain}[aout]"]
    return [*input_cmd, "-filter_complex", ";".join(graph), *tail, str(dest)]


async def overlay_audio_bed(
    video_path: Path,
    out_path: Path,
    total_seconds: float,
    music: tuple[Path, MusicBed] | None,
    ambient: list[tuple[Path, AmbientCue]],
    settings: Settings,
    grade_vf: str = "",
    deterministic: bool = False,
    *,
    premaster_mute: str = "",
    audio_args: list[str] | None = None,
) -> tuple[Path, dict[str, float | bool | None] | None]:
    """Lay a continuous music + ambient bed over the concatenated video.

    Runs ONCE after ``concat_shots``, re-encoding only the audio (``-c:v copy``
    keeps the per-shot stream-copy fast path intact). The concat output's own
    audio (``[0:a]`` — the baked per-shot narration / silence) is the sidechain
    key, so the bed ducks under whatever narration is already in the video
    without re-deriving it. Music + every ambient cue are summed into one bus and
    ducked by a single ``sidechaincompress`` (a filtergraph label is consumable
    once, so the narration key is ``asplit``-duplicated for the audible mix and
    the key).

    A bed shorter than its window is looped by crossfading (``_timeline_crossfade_mix``) N independent
    reads of the asset (``_bed_loop_plan`` / ``_bed_loop_filter``) so every loop
    seam is a ramp, not a click; one already long enough is NOT looped, and an
    unprobeable one falls back to the old butt-joined ``-stream_loop -1``. Every
    bed is then ``atrim``-ed to its window. Caller guarantees at least one of
    ``music``/``ambient`` is present (else this pass is skipped and the concat
    output returned as-is).

    VO-relative leveling (T1): a ``MusicBed``/``AmbientCue`` whose
    ``*_offset_lu``/``duck_open_lu_below_vo`` is set trades its blind literal
    ``gain_db``/``duck_threshold`` for one MEASURED against the narration bus
    (``final_mux_loudness`` on ``video_path``) and the asset's own measured
    loudness — see ``_offset_gain_db``/``_vo_relative_threshold``. Measurement
    is LAZY: with no offset field set anywhere this is zero extra ffmpeg
    passes, byte-identical to before T1. Measurement is best-effort (never
    fails the job) — a failed pass silently falls back to the literal.

    Returns ``(out_path, music_bed_report)``. ``music_bed_report`` is an
    auditable dict of the music bed's mix (``vo_lufs``, ``music_lufs``,
    ``music_gain_db``, ``music_was_measured``, ``duck_db``,
    ``sidechain_ratio``, ``duck_threshold``) — ``None`` when there is no music
    bed (ambient-only compiles report nothing; the fields above are all
    music-bed-specific). It does not report a "measured duck depth": after
    ``sidechaincompress`` sums bed+narration the bed is no longer separately
    addressable, so only the REQUESTED duck (``duck_db``) + derived ratio are
    reported.

    ``premaster_mute`` (``_premaster_mute_filter``) is spliced onto ``[0:a]``
    BEFORE the ``asplit``, so a per-clip mute reaches BOTH the audible narration
    AND the sidechain key: a muted shot must not duck the bed under itself. It
    also lands before the bed mix, so it can only silence the narration and never
    the bed. Empty (the default and every compile with no mixed mute) leaves the
    graph byte-identical.

    ``audio_args`` overrides the output audio codec args. The caller passes
    LOSSLESS (``_segment_audio_codec_args``) when a per-clip gain pass will follow
    this one, so the film pays for exactly ONE AAC generation instead of being
    encoded here and decoded + re-encoded there. ``None`` → the delivery AAC spec,
    i.e. today's behaviour for every compile without that pass.
    """
    total = max(total_seconds, 0.1)
    cmd: list[str] = ["ffmpeg", "-y", *_bitexact_input_args(deterministic), "-i", str(video_path)]
    mute_frag = f",{premaster_mute}" if premaster_mute else ""
    # ⛔ #235 — the narration bus is padded to ``total`` BEFORE the split, so BOTH
    # the audible branch and the sidechain KEY last the whole film.
    #
    # ``sidechaincompress`` ends ``[ducked]`` when its KEY input reaches EOF and does
    # not drain the bed frames still queued behind it; ``amix=duration=first`` then
    # carries on with the narration alone. So the music bed simply STOPPED before the
    # end of the film and the outro title card played over digital silence. Measured
    # on production master 01a0b00d (ffmpeg 7.1.5): bed present at -28 dB to 13.75s,
    # then -inf for the last ~1.05s of a 15.05s film. Re-running that exact graph on
    # the production assets gave a ``[ducked]`` of 14.06 / 14.14 / 14.21 / 14.57s
    # against a 15.050s bed — a scheduling race, not a length the caller chose — and
    # a key that is merely SHORTER than ``total`` (a title card whose AAC leg ends
    # early) truncates the bed deterministically. With the key padded here ``[ducked]``
    # came back 15.050s on every run. ``whole_dur`` cannot shorten, so audio already
    # that long is untouched; the pad is digital silence, which neither ducks the bed
    # nor adds anything audible to the narration branch.
    parts: list[str] = [
        "[0:a]" + _A48_SOXR + mute_frag + _pad_audio_to(total) + ",asplit=2[nar_mix][nar_key]"
    ]
    bed_labels: list[str] = []
    idx = 1

    # Measure the narration bus ONCE, only if something actually needs it —
    # every offset/threshold field is None on the default (pre-T1) path, so
    # `wants_measurement` is False and this stays a zero-cost no-op.
    wants_measurement = (
        music is not None
        and (music[1].music_offset_lu is not None or music[1].duck_open_lu_below_vo is not None)
    ) or any(
        cue.ambient_offset_lu is not None or cue.duck_open_lu_below_vo is not None
        for _, cue in ambient
    )
    vo_lufs: float | None = None
    if wants_measurement:
        from src.services.loudness import final_mux_loudness

        vo_report = await final_mux_loudness(str(video_path))
        vo_lufs = vo_report.get("integrated_lufs") if vo_report else None
        if vo_lufs is None:
            log.warning("overlay_audio_bed.vo_measure_failed", path=str(video_path))
    # VC-W14/V6 — THE DISCRIMINATOR, published rather than logged.
    #
    # ``vo_lufs is None`` is AMBIGUOUS on its own: it is the normal, correct state
    # for every job that requested no VO-relative knobs at all (``wants_measurement``
    # False ⇒ no measurement is even attempted). Only the CONJUNCTION says something
    # went wrong — calibration was asked for and could not be computed — and that
    # conjunction existed nowhere outside this function, which is why
    # ``vo_measure_failed`` was a log line no consumer could act on.
    #
    # With it None, BOTH ``_offset_gain_db`` and ``_vo_relative_threshold`` take
    # their literal fallbacks, so the bed sits at an arbitrary level under the
    # narration and the ducker's open threshold is an absolute linear constant the
    # bus may never cross. That is an audible mix defect shipped on a warning.
    vo_measure_failed = wants_measurement and vo_lufs is None

    music_bed_report: dict[str, float | bool | None] | None = None
    music_lufs: float | None = None
    music_gain_db = 0.0
    music_was_measured = False

    if music is not None:
        path, bed = music
        # Seam-crossfaded loop (see _bed_loop_plan): N independent reads whose
        # joins are 0.25 s ramps, instead of `-stream_loop`'s butt-joined click
        # every asset-length seconds.
        music_inputs, music_src_label = await _append_bed_input(
            cmd, parts, path, idx, total, "bed_src_music"
        )
        music_gain_db = bed.gain_db
        if bed.music_offset_lu is not None:
            from src.services.loudness import final_mux_loudness

            asset_report = await final_mux_loudness(str(path))
            music_lufs = asset_report.get("integrated_lufs") if asset_report else None
            music_gain_db, music_was_measured = _offset_gain_db(
                bed.music_offset_lu, bed.gain_db, vo_lufs, music_lufs
            )
        parts.append(_music_bed_filter(idx, total, music_gain_db, bed, src_label=music_src_label))
        bed_labels.append("[bed_music]")
        idx += music_inputs
    for n, (path, cue) in enumerate(ambient):
        end = cue.end_seconds if cue.end_seconds is not None else total
        window = max(0.1, min(end, total) - cue.start_seconds)
        start_ms = round(cue.start_seconds * 1000)
        delay = f",adelay={start_ms}|{start_ms}" if start_ms > 0 else ""
        amb_inputs, amb_src_label = await _append_bed_input(
            cmd, parts, path, idx, window, f"bed_amb_src{n}"
        )
        ambient_gain_db = cue.gain_db
        if cue.ambient_offset_lu is not None:
            from src.services.loudness import final_mux_loudness

            asset_report = await final_mux_loudness(str(path))
            asset_lufs = asset_report.get("integrated_lufs") if asset_report else None
            ambient_gain_db, _ = _offset_gain_db(
                cue.ambient_offset_lu, cue.gain_db, vo_lufs, asset_lufs
            )
        parts.append(
            _ambient_window_filter(
                idx,
                window,
                ambient_gain_db,
                delay,
                f"bed_amb{n}",
                src_label=amb_src_label,
            )
        )
        bed_labels.append(f"[bed_amb{n}]")
        idx += amb_inputs
    if len(bed_labels) == 1:
        bed_out = bed_labels[0]
    else:
        # duration=longest, NOT first: each bed is atrim'd to its OWN window, so mixing to
        # the FIRST input's length truncates the whole bed to bed[0]'s window — an ambient
        # cue that ends before a later/longer cue would cut ALL ambient at its end.
        # normalize=0 keeps summed levels independent of input count. (Music-present path is
        # unaffected: bed_music is appended first AND atrim'd to full `total`.)
        parts.append(
            "".join(bed_labels) + f"amix=inputs={len(bed_labels)}:duration=longest:normalize=0[bed]"
        )
        bed_out = "[bed]"
    # Duck the whole bed under the narration, then sum the (full-level) narration
    # back over the ducked bed. duck_db still sets the DEPTH (via the ratio); the
    # bed's timing/sensitivity knobs set when the compressor opens and how fast it
    # lets go. The key is high-passed first so the duck triggers on speech presence
    # rather than rumble/plosives.
    if music is not None:
        bed = music[1]
        duck_threshold = _vo_relative_threshold(
            bed.duck_open_lu_below_vo, vo_lufs, bed.duck_threshold
        )
        ratio = _sidechain_ratio(bed.duck_db)
        parts.append(f"[nar_key]highpass=f={_ffnum(bed.sidechain_hpf_hz)}[nar_hpf]")
        parts.append(
            f"{bed_out}[nar_hpf]sidechaincompress="
            f"threshold={_ffnum(duck_threshold)}:"
            f"ratio={ratio}:"
            f"attack={_ffnum(bed.duck_attack_ms)}:"
            f"release={_ffnum(bed.duck_release_ms)}[ducked]"
        )
        music_bed_report = {
            # VC-W14/V6. See ``vo_measure_failed`` above: the only unambiguous
            # statement that the ducker/bed gain are UNCALIBRATED rather than
            # deliberately literal. ``mastering.bed_calibration_violation`` is the
            # consumer.
            "vo_measure_failed": vo_measure_failed,
            "vo_lufs": vo_lufs,
            "music_lufs": music_lufs,
            "music_gain_db": music_gain_db,
            "music_was_measured": music_was_measured,
            "duck_db": bed.duck_db,
            "sidechain_ratio": ratio,
            "duck_threshold": duck_threshold,
        }
    else:
        # Ambient-only bed (no music): the SAME breathing duck as the music branch,
        # driven by the ambient cue's own knobs. The prior legacy filter
        # (threshold=0.125:release=1000) was dead — 0.125 is an absolute linear key
        # level the -18 LUFS narration bus never crosses (compressor never opened)
        # and a 1000 ms release pinned the bed flat through every VO gap. The duck
        # applies to the SUMMED ambient bus (`bed_out`), which is one node, so it
        # reads ONE cue's knobs: ambient[0] — the cue that owns the (first/only)
        # bed. Cues are per-section but the compressor is a single global node, so
        # the first cue is the least-surprising source and matches the single
        # full-span cue the orchestrator emits today. The key is high-passed into a
        # DISTINCT label ([nar_hpf_amb]) — the music branch's [nar_hpf] is only
        # minted when music is present, so reusing it here would dangle.
        amb_bed = ambient[0][1]
        amb_duck_threshold = _vo_relative_threshold(
            amb_bed.duck_open_lu_below_vo, vo_lufs, amb_bed.duck_threshold
        )
        parts.append(f"[nar_key]highpass=f={_ffnum(amb_bed.sidechain_hpf_hz)}[nar_hpf_amb]")
        parts.append(
            f"{bed_out}[nar_hpf_amb]sidechaincompress="
            f"threshold={_ffnum(amb_duck_threshold)}:"
            # MusicBed's effective ratio for its default duck_db, DERIVED (not a
            # second literal — a hardcoded 9 here survived the music branch's
            # recalibration and the two branches disagreed). Ambient carries no
            # duck_db knob (tight surface).
            f"ratio={_sidechain_ratio(_AMBIENT_DUCK_DB)}:"
            f"attack={_ffnum(amb_bed.duck_attack_ms)}:"
            f"release={_ffnum(amb_bed.duck_release_ms)}[ducked]"
        )
    # Fold the final master (compressor+limiter+loudnorm) into this same audio
    # re-encode so a bed video pays for exactly one audio pass, not two.
    measured: dict[str, str] | None = None
    master_chain = ""
    if settings.master_enabled:
        parts.append("[nar_mix][ducked]amix=inputs=2:duration=first:normalize=0[premaster]")
        # The length the output ``-t`` below trims to, digit for digit — loudnorm is
        # padded up from THIS, so the pad can never fall short of what ships.
        align_to = float(f"{total:.3f}")
        # Two-pass loudnorm: the bus that pass 2 normalises does not exist as a
        # file — it is minted by THIS filtergraph (narration + ducked bed) — so
        # pass 1 re-runs the whole graph to `-f null -` and measures the same
        # `[premaster]` label. `-t` bounds it: the bed inputs are `-stream_loop -1`.
        # ``cmd[1:]`` is exactly the input side (`-y`, bitexact flags, every `-i`);
        # no output args have been appended yet.
        measured = await _measure_master_loudnorm(
            cmd[1:],
            ";".join(
                [*parts, f"[premaster]{_master_measure_filter(settings, align_to)}[ameasure]"]
            ),
            "[ameasure]",
            ["-t", f"{total:.3f}"],
        )
        master_chain = _master_filter(settings, measured, align_to)
        parts.append(f"[premaster]{master_chain}[aout]")
    else:
        # MASTER OFF. `normalize=0` is correct — it keeps the narration at full
        # level instead of amix's default divide-by-input-count — but it also means
        # bed + narration SUM, and with the master chain disabled there is no
        # compressor, no loudnorm and (until now) NOTHING between that sum and the
        # AAC encoder. A loud bed under a loud line clips. The master-on branch
        # above has always ended in `alimiter`; this branch gets the same guard and
        # nothing else, so a `master_enabled=False` render keeps its levels and its
        # dynamics and only loses the ability to go past full scale.
        # `level=disabled` is load-bearing for the same reason it is above:
        # alimiter's `level` defaults to TRUE and re-normalises the output back up
        # to 1.0, which makes `limit=` bound nothing at all.
        # The ceiling is DERIVED from `master_tp` (`_limiter_linear_limit`), the
        # same single configured ceiling the master bus and every per-shot safety
        # limiter use — wave 8 fixed `_master_filter` and `_audio_sfx_graph` and
        # left this branch on a hardcoded 0.95 (-0.45 dBFS), unrelated to any
        # setting. This is the LAST limiter a `master_enabled=False` render sees,
        # so its ceiling is the delivered one.
        limit = _limiter_linear_limit(settings)
        parts.append(
            "[nar_mix][ducked]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit={_ffnum(limit)}:level=disabled[aout]"
        )
    # Whole-film grade (VC-5): when a non-empty ``grade_vf`` is threaded in, fold
    # the grade into THIS pass' video map — the bed pass is already re-encoding
    # audio, so the graded video rides the same pass instead of paying for a
    # second one. ``-r render_fps`` re-asserts CFR (VC-3). With no grade the video
    # stays a plain ``-c:v copy`` stream — byte-identical to before VC-5.
    video_args = _grade_video_args(grade_vf, settings, deterministic)
    pass2_cmd = functools.partial(
        _bed_pass2_cmd,
        input_cmd=list(cmd),
        parts=parts,
        master_chain=master_chain,
        tail=[
            *video_args,
            "-map",
            "[aout]",
            *(audio_args if audio_args is not None else _audio_codec_args(deterministic)),
            "-t",
            f"{total:.3f}",
            "-max_muxing_queue_size",
            "4096",
            "-movflags",
            "+faststart",
        ],
    )

    # Same read-back as `master_audio`: pass 2's stderr is the only witness to
    # whether loudnorm actually ran `linear`. Skipped when mastering is off — there
    # is no loudnorm on that branch and no summary to parse.
    _record_master_normalization_type(
        await _run(pass2_cmd("", out_path), capture_stderr=True),
        requested_linear=measured is not None,
        master_enabled=settings.master_enabled,
        refusal_predicted=_linear_refusal_predicted(settings, measured),
    )
    # Same miss correction as `master_audio`; a no-op with mastering off.
    await _correct_master_loudness_miss(out_path, settings, pass2_cmd)
    return out_path, music_bed_report


async def master_audio(
    video_path: Path,
    out_path: Path,
    settings: Settings,
    deterministic: bool = False,
    *,
    premaster_mute: str = "",
    audio_args: list[str] | None = None,
) -> Path:
    """Apply the final loudness master to a video that has NO music/ambient bed.

    The bed path folds the identical ``_master_filter`` chain into
    ``overlay_audio_bed``; this is the dedicated audio-only pass for bed-less
    videos so every compiled film gets the same platform-competitive loudness.
    Re-encodes only the audio (``-c:v copy``) — the stream-copied video is
    untouched. Caller invokes this only when ``_master_filter`` is non-empty.

    Guard: a bed-less video with NO audio stream (a fully audio-less concat)
    can't be mastered — the ``[0:a]`` map below would fail the render. Such a
    file skips the loudnorm master entirely and is stream-copied through
    unchanged (``-c copy``), so its output stays byte-identical. Real audio
    (every normal narrated compile) is untouched by this guard and still gets
    the #71 master.

    ``premaster_mute`` / ``audio_args``: the bed path's parameters, same contract
    (see ``overlay_audio_bed``). The mute is spliced into BOTH passes — pass 1 must
    measure the SAME bus pass 2 normalises, or ``loudnorm``'s constant gain is
    derived from audio the deliverable does not contain.
    """
    if not await _has_audio_stream(video_path):
        log.info("master_audio.no_audio_skip", path=str(video_path))
        await _run(
            [
                "ffmpeg",
                "-y",
                *_bitexact_input_args(deterministic),
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        return out_path
    # Two-pass loudnorm (pass 1): measure the same `[0:a]` bus through the same
    # compressor pass 2 will see, so pass 2 can apply ONE constant gain instead
    # of ffmpeg's dynamic gain riding. Fails open to the single-pass chain.
    measure_args = ["-y", *_bitexact_input_args(deterministic), "-i", str(video_path)]
    # The per-clip mute rides the HEAD of the chain, shared by both passes.
    head = f"[0:a]{_A48_SOXR}" + (f",{premaster_mute}" if premaster_mute else "")
    # The bus' own length, so loudnorm can be handed whole 100 ms frames
    # (:func:`_loudnorm_frame_align`) and the output trimmed back to exactly this.
    # ``None`` (unprobeable) fails open to the unaligned chain — today's audio.
    audio_seconds = await _audio_stream_seconds(video_path)
    measured = await _measure_master_loudnorm(
        measure_args,
        f"{head},{_master_measure_filter(settings, audio_seconds)}[ameasure]",
        "[ameasure]",
    )
    master = _master_filter(settings, measured, audio_seconds)
    # In-graph, not an output ``-t``: ``-t`` would also bound the stream-copied
    # VIDEO, and a film whose picture outlasts its audio by a frame must keep it.
    # LAST, after any miss-correction chain, so every pass-2 shape ends the same.
    trim_back = f",atrim=end={_ffdur(audio_seconds)}" if audio_seconds is not None else ""

    def _pass2_cmd(extra_chain: str = "", dest: Path = out_path) -> list[str]:
        chain = (f"{master},{extra_chain}" if extra_chain else master) + trim_back
        return [
            "ffmpeg",
            "-y",
            *_bitexact_input_args(deterministic),
            "-i",
            str(video_path),
            "-filter_complex",
            f"{head},{chain}[aout]",
            # VC-W14/V1: re-assert the colour triplet on the `-c:v copy` video map.
            # Same defect, same shape as the bed branch (`_grade_video_args`): a video
            # stream-copied alongside an ACTIVE audio `-filter_complex` depends on the
            # muxer propagating the input's codecpar, which ffmpeg 7.1.5 does not do
            # reliably. `overlay_audio_bed` already re-stamped here; this is the
            # NO-BED master, which did not — so a film without a music/ambient bed
            # shipped untagged. That audio branch, not any asset class, is the
            # `colorspace_ok` failure population.
            *_grade_video_args("", settings, deterministic),
            "-map",
            "[aout]",
            *(audio_args if audio_args is not None else _audio_codec_args(deterministic)),
            "-max_muxing_queue_size",
            "4096",
            "-movflags",
            "+faststart",
            str(dest),
        ]

    # `capture_stderr=True` so loudnorm's end-of-run summary can be read back: it
    # is the ONLY place ffmpeg says whether it honoured `linear=true` or silently
    # fell back to dynamic. `_run` already captures stderr either way — the flag
    # only decides whether it is RETURNED — so this costs nothing.
    _record_master_normalization_type(
        await _run(_pass2_cmd(), capture_stderr=True),
        requested_linear=measured is not None,
        refusal_predicted=_linear_refusal_predicted(settings, measured),
    )
    # A master that MISSED `master_lufs` (dynamic loudnorm on a film that opens on
    # silence) is re-rendered once with a constant gain. No-op when it landed.
    await _correct_master_loudness_miss(out_path, settings, _pass2_cmd)
    return out_path


async def apply_finish_grade(
    video_path: Path,
    out_path: Path,
    grade_vf: str,
    settings: Settings,
    deterministic: bool = False,
) -> Path:
    """Whole-film colour grade (VC-5) as its OWN video-only re-encode pass.

    The bed path folds the grade into ``overlay_audio_bed``'s single re-encode;
    this is the dedicated pass for a bed-less film (mirrors ``master_audio`` /
    ``_deband_segment``). Re-encodes only the video through ``grade_vf`` with the
    shared encode ladder (``_video_codec_args`` — BT.709 tag + CRF, never a second
    hardcoded crf) and stream-copies the already-mastered audio (``-c:a copy``, so
    the VC-1 loudness master is untouched). ``_video_codec_args`` itself emits
    ``-r render_fps`` to re-assert CFR so the VC-3 cadence holds (single source —
    do not append a second ``-r`` here). Caller invokes this only when
    ``grade_vf`` is non-empty."""
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(video_path),
        "-vf",
        grade_vf,
        *_video_codec_args(settings, deterministic),
        "-c:a",
        "copy",
        "-max_muxing_queue_size",
        "4096",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    await _run(cmd)
    return out_path


# A solo window shorter than this is not worth a filter node (and a zero/inverted
# window is an invalid `enable=`), so such a shot emits no clause at all.
_MIN_SOLO_WINDOW_S = 0.05


def _solo_windows(
    shots: list[Shot], offsets: list[tuple[float, float]]
) -> list[tuple[float, float] | None]:
    """Narrow each shot's raw bus ``[start, end)`` to its SOLO region — the span
    where the bus carries ONLY that shot.

    During a crossfade the bus samples in the overlap are a crossfade BLEND
    of two shots. A per-clip ``volume`` keyed to the RAW window would therefore
    reach into the neighbour: e.g. A(4 s, no override) + B(6 s, mute) with a 1 s
    xfade gives raw offsets [(0,4), (3,9)], so B's ``enable`` would silence
    [3,4) — which is A's fade-out tail. A was never edited by the user. Worse, if
    BOTH neighbours carry an override, the two ``volume`` clauses both fire in the
    overlap and COMPOUND (double gain; a mute zeroing the other's boost).

    So each shot is trimmed by the overlap it shares with a transitioned
    neighbour:
        lead_d  = overlap of the xfade INTO  this shot (0 on a hard cut / shot 0)
        trail_d = overlap of the xfade OUT OF this shot (0 on a hard cut / last)
        solo = [start + lead_d, end - trail_d)
    Each overlap is read straight back off the offsets ``_bus_offsets_xfade``
    produced (``lead_i = end_{i-1} - start_i``): the span in which the PREVIOUS
    shot's audio is still on the bus. Those offsets are the very numbers
    ``_xfade_group_joins`` gave the renderer, so the window edges sit on the
    rendered join. ``lead_i`` is the join's overlap to within half a frame, not
    identically: the offset is snapped to the frame grid, so with off-grid segment
    lengths the previous shot's audio ends up to half a frame either side of
    ``start_i + overlap`` — and it is where that audio ENDS that bounds the solo
    region. ``None`` = the shot is fully consumed by its overlaps (no solo region
    worth touching) and must emit no clause.

    Plain concat has no transitions → lead/trail are 0 → the windows are the raw
    offsets, unchanged.

    ACCEPTED RESIDUAL: within a crossfade, the EDITED clip's blended onset/tail
    keeps its ORIGINAL level (the gain/mute starts only once the clip is solo).
    That is inherent to editing a post-mix bus — the overlap samples are one
    blended signal and cannot be attributed to a single clip after the fact. We
    prefer leaving the edited clip's own blend untouched over DAMAGING a
    neighbour the user never edited.

    Shared by BOTH per-clip edit positions — the pre-master mute
    (``_premaster_mute_filter``) and the post-master gain (``_postclip_filter``) —
    so the two can never disagree about which samples belong to which shot.
    """
    n = len(shots)
    windows: list[tuple[float, float] | None] = []
    for i, (start, end) in enumerate(offsets):
        lead = 0.0
        if i > 0 and shots[i].transition is not None:
            lead = max(0.0, offsets[i - 1][1] - start)
        trail = 0.0
        if i + 1 < n and shots[i + 1].transition is not None:
            trail = max(0.0, end - offsets[i + 1][0])
        solo_start = start + lead
        solo_end = end - trail
        windows.append(
            (solo_start, solo_end) if solo_end - solo_start > _MIN_SOLO_WINDOW_S else None
        )
    return windows


# Edge ramp on a per-clip MUTE. ``volume``'s ``enable=`` toggles INSTANTLY: the
# bus steps from a non-zero sample straight to 0 at the window's start and back at
# its end, which is a step discontinuity — the exact click ``_SHOT_EDGE_FADE_S``
# exists to remove at every segment join, reintroduced in the middle of the film
# by editing it. 10 ms (~480 samples at 48 kHz) sits below the ~20 ms at which a
# level ramp becomes audible AS a fade (the same threshold argued in
# ``_apply_shot_audio``), so the boundary lands on silence without the mute
# reading as a fade-out. Clamped to a quarter of the solo window so the two ramps
# of a short window can never overlap and dip the middle.
_POSTCLIP_MUTE_RAMP_S = 0.010


def _ramped_mute_fragment(start: float, end: float) -> str:
    """One ``aeval`` node that ramps the bus DOWN to silence across the first
    ``_POSTCLIP_MUTE_RAMP_S`` of ``[start, end]``, holds it at zero, and ramps it
    back UP across the last — or ``""`` for a window too short to ramp at all.

    The envelope is the complement of a trapezoidal gate::

        1 - clip(min((t-a)/r, (b-t)/r), 0, 1)

    1 outside ``[a, b]``, 0 across ``[a+r, b-r]``, linear between. Verified at the
    four corners: ``t<a`` and ``t>b`` make one term negative so ``clip`` floors to
    0 (gain 1); ``t==a+r`` and ``t==b-r`` make ``min`` exactly 1 (gain 0).

    ⛔ **``aeval``, NOT ``volume``, and the difference is the entire fix.**
    ``volume``'s expression support is ``eval=frame`` — the gain is recomputed
    ONCE PER AUDIO FRAME, not per sample. Measured on ffmpeg 4.4 with a 48 kHz
    stereo bus and this exact expression: a 10 ms ramp came out as a TWO-STEP
    staircase, ``1.000 -> 0.738 -> 0`` over ~21 ms, because one decoder frame is
    ~1024 samples and the ramp is only 480. A 0.738-to-0 step is still a step
    discontinuity, i.e. still the click this exists to remove — just a quieter one.
    ``aeval`` evaluates per SAMPLE: the same expression measured as a clean
    monotonic ramp (``1.000, 0.983, 0.885, 0.783, … 0.083, 0.000`` in 1 ms
    samples), exact digital silence across the hold, and audio bit-identical to the
    input on both sides of the window.

    ⚠️ One expression PER CHANNEL, and stereo is a real invariant here, not an
    assumption: every splice point for this fragment sits immediately after
    ``_A48_SOXR``, whose trailing ``aformat=channel_layouts=stereo`` guarantees
    exactly two channels. ``c=same`` keeps the layout rather than deriving it from
    the expression count.
    """
    ramp = min(_POSTCLIP_MUTE_RAMP_S, (end - start) / 4.0)
    if ramp <= 0.0:
        return ""
    envelope = f"1-clip(min((t-{start:.3f})/{ramp:.4f},({end:.3f}-t)/{ramp:.4f}),0,1)"
    return f"aeval=exprs='val(0)*({envelope})'|'val(1)*({envelope})':c=same"


def _premaster_mute_filter(shots: list[Shot], offsets: list[tuple[float, float]]) -> str:
    """Comma-joined, edge-ramped ``volume`` chain muting each MIXED-request shot on
    the NARRATION bus, or ``""`` (byte-identical no-op) when no shot needs it.

    ⚠️ **Position is the whole point.** This chain is spliced onto ``[0:a]`` —
    BEFORE the bed mix, BEFORE the ``asplit`` that mints the ducker's sidechain
    key, and BEFORE ``loudnorm``. Two defects live in the alternative (the
    finished-bus position this used to share with ``_postclip_filter``):

    * the finished bus carries the MUSIC BED, so a ``volume=0`` window there zeroes
      the bed under the muted shot, not just the shot — a hole in the soundtrack;
    * the sidechain key was taken from the narration bus BEFORE the mute, so a
      muted shot still ducked the bed under it — audible pumping under silence.

    Splicing ahead of both fixes both, and it is safe HERE (unlike a pre-master
    GAIN, which ``loudnorm`` flattens back to target — see ``_postclip_filter``)
    because zero survives any gain: neither the dynamic nor the linear mode can
    resurrect a silenced span. The measured ``loudnorm``-corrupts-the-timeline trap
    documented on ``apply_postclip_audio`` does not apply either — it is about a
    timeline filter placed DOWNSTREAM of ``loudnorm``; this one reads the input
    file's own clean timeline.

    Each clause is still keyed to the shot's SOLO window (``_solo_windows``), which
    is what spares a crossfaded neighbour's audio, and each edge is ramped
    (``_ramped_mute_fragment``) rather than switched.

    Emitted only for a MIXED request: when ``_all_shots_muted`` holds,
    ``render_shot`` already rendered every segment silent at source and the solo
    windows tile the whole reel, so a chain here would be a redundant second mute.
    """
    if not offsets or len(offsets) != len(shots):
        return ""
    # One whole-request decision, read once (see ``_all_shots_muted``): it is the
    # same property ``compile_shots`` threaded into ``render_shot``, so the two
    # halves of the mute cannot disagree about which one owns it.
    if _all_shots_muted(shots):
        return ""
    parts: list[str] = []
    for shot, window in zip(shots, _solo_windows(shots, offsets), strict=True):
        audio = shot.audio
        if audio is None or window is None or not audio.mute:
            continue
        fragment = _ramped_mute_fragment(*window)
        if fragment:
            parts.append(fragment)
    return ",".join(parts)


def _postclip_filter(shots: list[Shot], offsets: list[tuple[float, float]]) -> str:
    """Comma-joined, time-ranged ``volume`` chain applying each shot's per-clip
    GAIN on the FINAL mastered bus, or ``""`` (byte-identical no-op) when no shot
    requests one.

    Runs AFTER the loudness master (see ``apply_postclip_audio``) because that is
    the only position where a gain SURVIVES: a PRE-master gain is flattened back to
    target by ``loudnorm``. Each clause is keyed to that shot's SOLO window on the
    bus (``_solo_windows`` — the raw offset narrowed out of any crossfade blend it
    shares with a neighbour) via ``volume``'s timeline ``enable=``. ``afade`` has
    no timeline support, which is why per-clip FADES stay pre-master.

    ⚠️ MUTE is NOT emitted here any more. It has no reason to be post-master (zero
    survives ``loudnorm``) and two reasons not to be — it would zero the music bed
    and it would arrive too late for the sidechain key — so it moved to
    ``_premaster_mute_filter``. ``mute`` still wins over ``gain_db``: a muted shot
    is skipped here because the mute has already been realised, at source or on the
    narration bus, and gain over silence is silence either way.

    ⚠️ A positive gain on this bus lands DOWNSTREAM of the master's ``alimiter``.
    The caller re-asserts the ceiling after the chain (``_postclip_limiter``, gated
    on ``_postclip_raises_level``); without it a boost clipped in the AAC encoder,
    and ``true_peak_dbtp`` is advisory-only so it shipped unflagged.
    """
    if not offsets or len(offsets) != len(shots):
        return ""
    parts: list[str] = []
    for shot, window in zip(shots, _solo_windows(shots, offsets), strict=True):
        audio = shot.audio
        if audio is None or window is None or audio.mute or audio.gain_db == 0.0:
            continue
        start, end = window
        parts.append(f"volume={audio.gain_db}dB:enable='between(t,{start:.3f},{end:.3f})'")
    return ",".join(parts)


def _postclip_raises_level(shots: list[Shot]) -> bool:
    """Whether any shot asks the post-master pass to BOOST the bus, i.e. whether
    that pass can breach the delivered true-peak ceiling and therefore needs
    ``_postclip_limiter`` after it.

    A cut (``gain_db < 0``) and a mute can only lower the bus, so a compile that
    only attenuates keeps a command byte-identical to today's."""
    return any(
        shot.audio is not None and not shot.audio.mute and shot.audio.gain_db > 0.0
        for shot in shots
    )


def _postclip_limiter(settings: Settings) -> str:
    """The delivered peak ceiling, re-asserted after a post-master per-clip GAIN.

    ``_master_filter`` ends in ``alimiter`` precisely because that filter must be
    the LAST thing the bus sees; the per-clip gain pass then ran after it with
    NOTHING downstream, so a ``+6 dB`` clause handed the AAC encoder samples above
    full scale. The gate's ``true_peak_dbtp`` is advisory-only (no pass/fail check
    is applied to it), so every such master shipped unflagged.

    Same ceiling, same knobs, same ``level=disabled`` as the master bus — derived
    from ``master_tp`` via ``_limiter_linear_limit`` so one configured value
    governs every limiter in the module. ``level`` defaults to TRUE in ffmpeg and
    re-normalises the output back up to 1.0, which makes ``limit=`` bound nothing
    at all; it is as load-bearing here as it is in ``_master_filter``."""
    return (
        f"alimiter=limit={_ffnum(_limiter_linear_limit(settings))}"
        f":attack={_ffnum(settings.master_limiter_attack_ms)}"
        f":release={_ffnum(settings.master_limiter_release_ms)}"
        ":level=disabled"
    )


async def apply_postclip_audio(
    video_path: Path,
    out_path: Path,
    postclip: str,
    settings: Settings,
    deterministic: bool = False,
) -> Path:
    """Apply the per-clip gain chain to an ALREADY-MASTERED bus, as its OWN
    ffmpeg pass. Audio-only re-encode (``-c:v copy``); the video is untouched.

    ⚠️ The caller (``_finalize_audio``) composes ``postclip`` and owns what rides
    it: the ``volume`` gain clauses, a trailing ``_postclip_limiter`` whenever one
    of them can BOOST (this pass used to be the last thing the bus saw, so a
    positive gain clipped in the AAC encoder below), and — only on the branch with
    no bed and no master, where this pass IS the narration bus — the per-clip mute.
    This function does not inspect the chain.

    ⚠️ It is also the ONLY AAC generation: when this pass runs, the bed/master
    pass upstream was told to emit lossless ALAC instead of AAC, so the film is not
    encoded twice.

    ⚠️ This MUST be a separate pass — it cannot be folded into the master's
    filtergraph. ``loudnorm`` (the last node of ``_master_filter``, and dynamic /
    single-pass by design) hands its downstream filters a CORRUPTED timeline: on
    ffmpeg 4.4 a 6.01 s stream emerges from ``loudnorm`` with frame ``pts_time``
    advancing only to ~3.11, so a downstream ``volume=…:enable='between(t,a,b)'``
    fires on the wrong samples and never switches back off. Verified: the same
    ``enable`` window is EXACT after ``acompressor`` and after ``alimiter``, and
    broken only after ``loudnorm``; an ``asetpts=N/SR/TB`` reset in-graph does NOT
    repair it. Re-reading the muxed, mastered file gives the ``volume`` filter a
    clean, sample-accurate timeline, and the window then lands exactly (measured:
    -20 dB inside the window, unchanged either side).

    The cost is one extra audio-only pass, and ONLY for a compile that actually
    requests a per-clip gain/mute — ``_finalize_audio`` skips this entirely
    otherwise, so the default render is untouched.

    Guard: a bus with NO audio stream is stream-copied through unchanged (the
    ``[0:a]`` map would otherwise fail the render) — mirrors ``master_audio``.
    """
    if not await _has_audio_stream(video_path):
        log.info("apply_postclip_audio.no_audio_skip", path=str(video_path))
        await _run(
            [
                "ffmpeg",
                "-y",
                *_bitexact_input_args(deterministic),
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        return out_path
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(video_path),
        "-filter_complex",
        f"[0:a]{_A48_SOXR},{postclip}[aout]",
        # VC-W14/V1: re-assert the colour triplet on the `-c:v copy` video map —
        # this pass is the LAST thing to touch the master on any compile that
        # requests a per-clip gain/mute, so an untagged copy here is what ships.
        *_grade_video_args("", settings, deterministic),
        "-map",
        "[aout]",
        *_audio_codec_args(deterministic),
        "-max_muxing_queue_size",
        "4096",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    await _run(cmd)
    return out_path


async def _encode_delivery_audio(
    video_path: Path, out_path: Path, settings: Settings, deterministic: bool = False
) -> Path:
    """Transcode the intermediate's lossless PCM bus to the DELIVERY audio spec.

    Segments carry lossless ALAC (see ``_SEGMENT_AUDIO_CODEC_ARGS``) so the
    stream-copy join is sample-exact; every normal compile then converts that to
    AAC in a pass it was already running (``overlay_audio_bed`` /
    ``master_audio`` / ``apply_postclip_audio``). This is the fallback for the
    two configurations where NO such pass exists — mastering disabled with no
    bed, with or without a whole-film grade — so a deliverable can never leave
    this service as ALAC. Audio-only (``-c:v copy``), so it adds no video
    generation and exactly one (the first and only) AAC generation.

    Guard mirrors ``master_audio``/``apply_postclip_audio``: a bus with no audio
    stream is stream-copied through unchanged.
    """
    if not await _has_audio_stream(video_path):
        log.info("encode_delivery_audio.no_audio_remux", path=str(video_path))
        await _run(
            [
                "ffmpeg",
                "-y",
                *_bitexact_input_args(deterministic),
                "-i",
                str(video_path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        return out_path
    await _run(
        [
            "ffmpeg",
            "-y",
            *_bitexact_input_args(deterministic),
            "-i",
            str(video_path),
            # VC-W14/V1: re-assert the colour triplet on the `-c:v copy` video
            # map. This is the terminal pass on the two mastering-disabled
            # configurations, so it is the last writer of the delivered master.
            *_grade_video_args("", settings, deterministic),
            "-map",
            "0:a",
            *_audio_codec_args(deterministic),
            "-max_muxing_queue_size",
            "4096",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return out_path


async def _finalize_audio(
    concat_path: Path,
    work_root: Path,
    total_seconds: float,
    music_asset: tuple[Path, MusicBed] | None,
    ambient_assets: list[tuple[Path, AmbientCue]],
    settings: Settings,
    shots: list[Shot] | None = None,
    shot_offsets: list[tuple[float, float]] | None = None,
    grade_vf: str = "",
    deterministic: bool = False,
) -> tuple[Path, dict[str, float | bool | None] | None]:
    """Produce the deliverable's final audio over the stream-copied concat.

    Stage 1 — the loudness bus: a music/ambient bed present → ``overlay_audio_bed``
    (which folds the master into its single audio re-encode); no bed but mastering
    enabled → a dedicated ``master_audio`` pass; neither → the concat carries
    through. A per-clip MUTE (``_premaster_mute_filter``) is spliced into the HEAD
    of whichever of those runs — ahead of the bed mix, the sidechain key and
    ``loudnorm`` — because on the finished bus it zeroed the music bed and arrived
    too late to stop a muted shot ducking that bed.

    Stage 2 — per-clip GAIN: when any shot requests one, the time-ranged ``volume``
    chain is applied to the FINISHED bus in its OWN pass
    (``apply_postclip_audio``), because a pre-master gain is flattened back to
    target by ``loudnorm`` AND because ``loudnorm`` corrupts the timeline of any
    filter downstream of it in the same graph (see that function). A gain that can
    BOOST is followed by ``_postclip_limiter`` — it would otherwise be the only
    thing after the master's own limiter. Skipped entirely when no shot requests
    gain/mute, so a default compile runs exactly the passes it ran before — and
    with neither bed, master nor tail the concat is returned UNCHANGED (the
    byte-identical no-op).

    Exactly ONE lossy generation in every configuration: whichever pass is LAST
    owns the AAC encode, and the earlier ones stay lossless when a later one
    exists (``lossless_bus`` below).

    Returns ``(path, music_bed_report)`` — ``music_bed_report`` (T1) is
    ``overlay_audio_bed``'s auditable VO-relative-leveling report, or ``None``
    when there is no music bed (mastering-only / postclip-only / unchanged
    paths never produce one).
    """
    shot_list = shots or []
    premaster_mute = _premaster_mute_filter(shot_list, shot_offsets or [])
    postclip = _postclip_filter(shot_list, shot_offsets or [])
    if postclip and _postclip_raises_level(shot_list):
        # C4: the gain lands downstream of the master's limiter with nothing after
        # it. Re-assert the SAME ceiling so a boost cannot clip the AAC encoder.
        postclip = f"{postclip},{_postclip_limiter(settings)}"
    music_bed_report: dict[str, float | bool | None] | None = None
    # ONE lossy generation, always. The bus arriving here is lossless ALAC (see
    # ``_SEGMENT_AUDIO_CODEC_ARGS``) and the deliverable must be AAC
    # 192k/48 kHz/stereo. When a per-clip GAIN pass will follow, the bed/master
    # pass is told to stay LOSSLESS and that final pass owns the only AAC encode —
    # otherwise the film was encoded to AAC here and then decoded and re-encoded
    # to AAC again, two lossy generations on any compile requesting a gain.
    lossless_bus = bool(postclip)
    bus_audio_args = _segment_audio_codec_args(deterministic) if lossless_bus else None
    # Does the chosen branch already encode the delivery audio? Bed and master
    # passes re-encode audio and therefore do the conversion for free (unless told
    # to stay lossless above); a grade-only pass (``-c:a copy``) and the
    # do-nothing branch do not.
    delivery_encoded = False
    if music_asset is not None or ambient_assets:
        # A bed already re-encodes audio → the grade rides that SAME pass' video
        # map (grade_vf folded into overlay_audio_bed), not a second pass.
        bus_path = work_root / "final.mp4"
        _, music_bed_report = await overlay_audio_bed(
            concat_path,
            bus_path,
            total_seconds,
            music_asset,
            ambient_assets,
            settings,
            grade_vf,
            deterministic,
            premaster_mute=premaster_mute,
            audio_args=bus_audio_args,
        )
        delivery_encoded = not lossless_bus
        premaster_mute = ""
    elif _master_filter(settings):
        bus_path = work_root / "mastered.mp4"
        await master_audio(
            concat_path,
            bus_path,
            settings,
            deterministic,
            premaster_mute=premaster_mute,
            audio_args=bus_audio_args,
        )
        delivery_encoded = not lossless_bus
        premaster_mute = ""
        # No bed to fold into → grade the mastered video in its own pass
        # (audio stream-copied through, so the loudness master is preserved).
        if grade_vf:
            graded = work_root / "graded.mp4"
            await apply_finish_grade(bus_path, graded, grade_vf, settings, deterministic)
            bus_path = graded
    elif grade_vf:
        # No bed, no audio master → a dedicated video-only grade pass. It
        # stream-copies audio (``-c:a copy``), so the bus is still lossless
        # afterwards and ``_encode_delivery_audio`` below owns the one AAC
        # generation.
        bus_path = work_root / "graded.mp4"
        await apply_finish_grade(concat_path, bus_path, grade_vf, settings, deterministic)
    else:
        bus_path = concat_path
    # ``premaster_mute`` is cleared by the two branches that actually HAVE a
    # pre-master position to splice it into. If it survives to here, no bed and no
    # loudnorm ran — so this pass IS the narration bus, the mute has nowhere
    # earlier to go, and neither of the defects that moved it off the finished bus
    # (zeroing the bed, arriving after the sidechain key) can exist on this branch.
    # It leads the chain so a gain on the same bus applies after it.
    tail = ",".join(fragment for fragment in (premaster_mute, postclip) if fragment)
    if tail:
        # This pass re-encodes audio to the delivery spec itself, and — because the
        # branches above stayed lossless for it — it is the ONLY AAC generation.
        postclip_path = work_root / "postclip.mp4"
        await apply_postclip_audio(bus_path, postclip_path, tail, settings, deterministic)
        return postclip_path, music_bed_report
    if not delivery_encoded:
        delivery_path = work_root / "delivery.mp4"
        await _encode_delivery_audio(bus_path, delivery_path, settings, deterministic)
        return delivery_path, music_bed_report
    return bus_path, music_bed_report


def _clip_seek_fractions(shots: list[Shot]) -> list[float]:
    """Per-shot seek fraction (0..1) that spreads repeated uses of a footage clip
    across its timeline. A clip used by N shots yields 0, 1/N, 2/N, … (N-1)/N so
    the promo pans through the page; 0 for images, text cards, and single-use
    clips (which keep the natural start). Order-preserving, one entry per shot."""
    counts: dict[str, int] = {}
    for shot in shots:
        if shot.video_url:
            counts[shot.video_url] = counts.get(shot.video_url, 0) + 1
    seen: dict[str, int] = {}
    fractions: list[float] = []
    for shot in shots:
        fraction = 0.0
        if shot.video_url and counts[shot.video_url] > 1:
            k = seen.get(shot.video_url, 0)
            fraction = k / counts[shot.video_url]
            seen[shot.video_url] = k + 1
        fractions.append(fraction)
    return fractions


async def _deband_segment(seg_path: Path, settings: Settings, deterministic: bool = False) -> None:
    """Re-encode one segment through the ffmpeg ``deband`` filter, in place.

    A second pass applied ONLY to a segment whose measured banding risk tripped
    (``compile_shots`` decides). Keeps the same video-codec args (so the BT.709
    colr tag, the CFR pin and the GOP are preserved) at the INTERMEDIATE CRF
    cell, and stream-copies the already-uniform audio. It used to encode at the
    DELIVERY cell — a second x264 generation re-quantising material x264 had
    already quantised at that exact cell, for nothing. See
    ``mastering.encode_ladder``'s ``intermediate`` argument. Never raises — a deband failure leaves the clean single-pass segment
    untouched (best-effort picture polish, never fails a compile)."""
    from src.services import mastering

    tmp_path = seg_path.with_name(seg_path.stem + "-deband" + _SEGMENT_EXT)
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(seg_path),
        "-vf",
        mastering.deband_filter(),
        *_video_codec_args(settings, deterministic, intermediate=True),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        await _run(cmd)
    except Exception as exc:  # noqa: BLE001 — best-effort polish, keep the clean seg
        log.warning("deband_segment.failed", path=str(seg_path), error=str(exc))
        # Best-effort cleanup of the failed deband output. This must NOT itself
        # raise: an OSError from unlink (permission, a race on tmp_path) would
        # break the "never raises" contract this function promises its caller
        # (``_apply_conditional_deband`` gathers these with
        # ``return_exceptions=True`` precisely so a deband failure degrades
        # instead of killing the compile).
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        return
    tmp_path.replace(seg_path)


async def _deband_decide(segment_paths: list[Path], settings: Settings) -> list[Path]:
    """Measure each rendered segment and return the ones that trip the banding
    risk. Pure decision + metrics/logging — no encode. Shared by the standalone
    ``_apply_conditional_deband`` (decide, then encode alone) and the merged
    ``_apply_picture_master`` (decide banding + exposure concurrently, then one
    combined encode). Skipped entirely when ``render_deband`` is off. Never
    raises — measurement is best-effort."""
    from src.services import mastering, measure

    if not mastering.deband_enabled(settings):
        return []
    # MEASURE PASS — concurrent. Each measure is a read-only ffprobe/ffmpeg over a
    # DISTINCT segment file, and every one of them still queues behind the global
    # ``_run`` semaphore (``max_concurrent_ffmpeg_subprocesses``), so this raises
    # utilisation from 1 slot to the configured bound without adding a knob. The
    # loop ran strictly serially while all other slots sat idle: this pass is
    # ~34% of a job.
    # PER-FRAME, matching the gate. An averaged measurement here made the gate
    # unsatisfiable — see mastering.banding_risk_frames.
    #
    # ⚠️ STILL THE WHOLE-FRAME (pixel-free) score here, deliberately, even
    # after the spatial term (``mastering.spatial_banding_factor``) landed —
    # see that function's module comment for what it measures and why. The
    # spatial re-measure is wired into ``mastering.master_verify`` /
    # ``bucket_banding`` (the REPORTED/gated ``banding_advisories`` and
    # per-shot ``needs_deband`` a caller reads), not into this internal
    # encode-time TRIGGER. Two reasons: (1) this trigger's own false-positive
    # cost is a no-op re-encode, already measured inert on the false-positive
    # material this gate exists to catch (SSIM 0.9998 — see ``banding_score``
    # note 3's "score went UP on 13 of 13 segments"), not a wrong verdict a
    # caller acts on; (2) doing it here would mean seeking the source segment
    # back to a specific frame's pts for a SECOND decode per segment on the
    # measure pass that is already ~34% of a job, multiplying that cost across
    # every segment instead of the bounded, already-flagged set the report-time
    # gate re-measures. Re-derive this call site's cost/benefit before wiring
    # the spatial term in here too.
    # ``return_exceptions=True`` is load-bearing: deband is best-effort picture
    # polish that must NEVER fail a compile, and a bare gather would propagate the
    # first segment's failure and kill the job. gather preserves INPUT ORDER, so
    # zip() below maps every result back to its own segment.
    measured = await asyncio.gather(
        *(measure.measure_shot_frames(seg) for seg in segment_paths),
        return_exceptions=True,
    )
    # DECISION PASS — pure classification, no encoding. Kept separate so the
    # encodes below can overlap; the counters increment here exactly as they did
    # when the encode was inline, so applied/skipped/failed totals are unchanged.
    to_deband: list[Path] = []
    for seg, frames in zip(segment_paths, measured, strict=True):
        if isinstance(frames, BaseException):
            # Re-raise anything the serial loop's `except Exception` would NOT have
            # caught. CancelledError is a BaseException, not an Exception, so this
            # covers it: swallowing a cancel would make the job un-cancellable.
            # Enumerating CancelledError alone would silently widen the catch to
            # every other BaseException, which the code this replaced never did.
            if not isinstance(frames, Exception):
                raise frames
            # NOT logger.exception: a return_exceptions result is a VALUE, not an
            # active exception, so there is no traceback in sys.exc_info() here.
            log.warning(
                "conditional_deband.measure_failed",
                path=str(seg),
                error=str(frames),
                exc_info=frames,
            )
            metrics.compile_deband_segments_total.labels(result="failed").inc()
            continue
        if mastering.banding_risk_frames(frames):
            log.info(
                "conditional_deband.applying",
                path=str(seg),
                score=max((mastering.banding_score(f) for f in frames), default=0.0),
            )
            metrics.compile_deband_segments_total.labels(result="applied").inc()
            to_deband.append(seg)
        else:
            # The skip path logged NOTHING before, which is why "does deband ever
            # actually fire?" was unanswerable.
            metrics.compile_deband_segments_total.labels(result="skipped").inc()

    return to_deband


async def _apply_conditional_deband(
    segment_paths: list[Path], settings: Settings, deterministic: bool = False
) -> None:
    """Standalone deband pass: decide, then encode the tripped segments alone.

    Kept for callers that want ONLY deband (the A/B script, direct tests) —
    the production pipeline instead runs ``_apply_picture_master``, which
    decides banding + exposure concurrently and folds a segment's deband and
    exposure correction into a SINGLE re-encode. Never raises — encode is
    best-effort."""
    to_deband = await _deband_decide(segment_paths, settings)
    if not to_deband:
        return
    # ENCODE PASS — concurrent, for exactly the reason the measure pass above is.
    # It used to run strictly serially, justified in-code by "the re-encode is
    # believed rare — the measure pass is the dominant cost". PRODUCTION DISAGREED:
    # 129 applied / 2 skipped = 98.5% of segments re-encoded. ``banding_risk_frames``
    # takes a MAX over frames, so one smooth frame trips a whole segment, which is
    # why the "rare" branch was the common one and held ONE ffmpeg slot while the
    # rest of the configured bound sat idle. Concurrency is still the right shape.
    #
    # ⚠️ BUT THAT 98.5% IS STALE — DO NOT PLAN AGAINST IT (corrected 2026-09-15).
    # It was measured 2026-08-28, and the SPATIAL TERM landed in ``banding_score``
    # on 2026-09-02, five days later. On the five real-master segments measured at
    # that change (see ``mastering.banding_score`` note 3), where the OLD score
    # fired on 99.4-99.6% of frames, FOUR now score 0.000 and one 0.184 — i.e. the
    # trigger rate collapsed by roughly 5x and this branch is plausibly rare again.
    # Nobody has re-counted ``compile_deband_segments_total{result=...}`` on the
    # current formula, so the live rate is UNMEASURED, not 98.5% and not "rare".
    #
    # The same staleness applies to the SSIM-0.9998 / "score went UP on 13 of 13
    # segments" evidence in ``mastering.banding_score``, measured 2026-09-01: it
    # shows the deband filter was inert on segments the OLD stillness-detector
    # score selected — segments with no spatial banding at all, where a spatial
    # filter having no effect is the expected result. It does NOT show the filter
    # is inert on the segments the CURRENT score selects. That is why this pass is
    # still UNGATED beyond ``render_deband``: gating it on a score threshold would
    # be a constant derived from a measurement of the superseded formula, exactly
    # the mistake ``BANDING_GATE_CEILING``'s history documents. Re-measure first.
    #
    # Safe to overlap: each call re-encodes a DISTINCT segment into its OWN temp
    # (``seg-NNNN-deband.mp4``, provably distinct) and ``.replace()``s only its own
    # path, so no two tasks ever touch a shared file. Every one still queues behind
    # the same ``_run`` semaphore, so the per-pod ffmpeg ceiling — and with it the
    # ~1GiB-per-encode memory bound — is UNCHANGED; this raises utilisation within
    # the existing cap rather than lifting it. Completion order now varies, which
    # nothing downstream reads: ``join`` runs after this whole stage returns.
    #
    # ``return_exceptions=True`` mirrors the measure pass and is load-bearing.
    # ``_deband_segment`` swallows its own ffmpeg failure, but ``.replace()`` can
    # still raise OSError, and a bare gather would propagate it and kill a compile
    # over best-effort picture polish. Metric semantics are deliberately unchanged:
    # ``applied`` counts the DECISION, as it did when the encode was inline.
    outcomes = await asyncio.gather(
        *(_deband_segment(seg, settings, deterministic) for seg in to_deband),
        return_exceptions=True,
    )
    for seg, outcome in zip(to_deband, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            # Same BaseException split as the measure pass: CancelledError is not an
            # Exception, and swallowing a cancel would make the job un-cancellable.
            if not isinstance(outcome, Exception):
                raise outcome
            # NOT logger.exception: a return_exceptions result is a VALUE, so there
            # is no active traceback in sys.exc_info() here.
            log.warning(
                "conditional_deband.encode_failed",
                path=str(seg),
                error=str(outcome),
                exc_info=outcome,
            )


async def _regrade_segment(
    seg_path: Path, filter_str: str, settings: Settings, deterministic: bool = False
) -> None:
    """Re-encode one segment through an arbitrary ``-vf`` chain, in place (T9).

    The exposure-match sibling of ``_deband_segment``: same video-codec args at
    the same INTERMEDIATE CRF cell (so the BT.709 colr tag survives and the
    re-encode does not re-quantise at the delivery cell) and stream-copied
    audio, differing only in the caller-supplied filter (``_source_match_correction``'s gamma
    and/or saturation).
    Never raises — a regrade failure leaves the clean single-pass segment
    untouched (best-effort exposure polish, never fails a compile)."""
    tmp_path = seg_path.with_name(seg_path.stem + "-match" + _SEGMENT_EXT)
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(seg_path),
        "-vf",
        filter_str,
        *_video_codec_args(settings, deterministic, intermediate=True),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        await _run(cmd)
    except Exception as exc:  # noqa: BLE001 — best-effort polish, keep the clean seg
        log.warning("regrade_segment.failed", path=str(seg_path), error=str(exc))
        tmp_path.unlink(missing_ok=True)
        return
    tmp_path.replace(seg_path)


async def _source_match_decide(
    segment_paths: list[Path],
    settings: Settings,
    finish: FinishProfile | None,
) -> list[tuple[Path, str]]:
    """Measure every rendered segment's average luma/saturation, take the MEDIAN
    across shots as a "one camera" target, and return the ``(segment, eq_filter)``
    pairs for outlier shots that need a correction. Pure decision + logging — no
    encode. Shared by the standalone ``_apply_source_match`` (decide, then
    regrade alone) and the merged ``_apply_picture_master``. Gated on
    ``render_source_match`` AND the per-request ``finish.grade.match_sources``.
    Needs at least 3 valid shots — a median over one or two is meaningless. Never
    raises — measurement is best-effort."""
    from src.services import mastering, measure

    if not mastering.source_match_enabled(settings):
        return []
    grade = finish.grade if finish is not None else None
    # Mirror the whole-film match path (``_resolve_grade_vf`` requires ``enabled``):
    # a request that sets ``match_sources`` with the grade disabled was a no-op
    # before, so per-shot matching must not silently start firing for it either.
    if grade is None or not grade.enabled or not grade.match_sources:
        return []
    # MEASURE PASS — concurrent, bounded by the SAME global ``_run`` semaphore (see
    # _deband_decide). ``return_exceptions=True`` keeps per-segment
    # failures isolated (best-effort exposure polish, never fails a compile), and
    # gather's INPUT ORDER guarantee is what lets zip() map each stats dict back to
    # its own segment.
    results = await asyncio.gather(
        *(measure.measure_shot_stats(seg) for seg in segment_paths),
        return_exceptions=True,
    )
    measured: list[tuple[Path, dict[str, float]]] = []
    for seg, stats in zip(segment_paths, results, strict=True):
        if isinstance(stats, BaseException):
            # See _deband_decide: mirror the old `except Exception`
            # width exactly rather than special-casing CancelledError.
            if not isinstance(stats, Exception):
                raise stats
            # NOT logger.exception — a gathered result is a value, not an active
            # exception; pass the object as exc_info instead.
            log.warning(
                "source_match.measure_failed", path=str(seg), error=str(stats), exc_info=stats
            )
            continue
        measured.append((seg, stats))
    valid = [(seg, s) for seg, s in measured if s.get("YAVG", 0.0) > 0]
    if len(valid) < 3:
        return []
    target = _median([s["YAVG"] for _, s in valid])
    # SAT median drops 0/missing SATAVG shots (a valid grey shot legitimately
    # measures ~0 chroma and must NOT drag the target down); the YAVG set above is
    # never thinned. Empty → 0.0, which makes every SAT term a no-op.
    sat_vals = [s["SATAVG"] for _, s in valid if s.get("SATAVG", 0.0) > 0]
    target_sat = _median(sat_vals) if sat_vals else 0.0
    pairs: list[tuple[Path, str]] = []
    for seg, s in valid:
        f = _source_match_correction(s, target, target_sat, settings)
        if f:
            log.info(
                "source_match.applying",
                path=str(seg),
                target=target,
                yavg=s["YAVG"],
                target_sat=target_sat,
                satavg=s.get("SATAVG"),
            )
            pairs.append((seg, f))
    return pairs


async def _apply_source_match(
    segment_paths: list[Path],
    settings: Settings,
    finish: FinishProfile | None,
    deterministic: bool = False,
) -> None:
    """Standalone source-match pass: decide, then regrade the outlier segments
    alone. Kept for callers that want ONLY exposure correction — the production
    pipeline instead runs ``_apply_picture_master``. Never raises — regrade is
    best-effort."""
    pairs = await _source_match_decide(segment_paths, settings, finish)
    for seg, f in pairs:
        await _regrade_segment(seg, f, settings, deterministic)


async def _polish_segment(
    seg_path: Path, filter_str: str, settings: Settings, deterministic: bool = False
) -> None:
    """Re-encode one segment through a COMBINED ``-vf`` chain (deband and/or
    source-match), in place. Single-generation sibling of ``_deband_segment`` /
    ``_regrade_segment`` — same video-codec args at the same INTERMEDIATE CRF
    cell and stream-copied audio,
    differing only in that ``filter_str`` may already be a comma-joined chain of
    both fragments. Never raises — best-effort picture polish, never fails a
    compile."""
    tmp_path = seg_path.with_name(seg_path.stem + "-master" + _SEGMENT_EXT)
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(seg_path),
        "-vf",
        filter_str,
        *_video_codec_args(settings, deterministic, intermediate=True),
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        await _run(cmd)
    except Exception as exc:  # noqa: BLE001 — best-effort polish, keep the clean seg
        log.warning("picture_master.polish_failed", path=str(seg_path), error=str(exc))
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        return
    tmp_path.replace(seg_path)


async def _apply_picture_master(
    segment_paths: list[Path],
    settings: Settings,
    finish: FinishProfile | None,
    deterministic: bool = False,
) -> None:
    """Merged deband + source-match pass (3a): decides BOTH conditions
    concurrently, then does AT MOST ONE re-encode per segment instead of the two
    sequential decode/encode passes ``_apply_conditional_deband`` +
    ``_apply_source_match`` used to run. A segment tripping only one condition
    gets only that filter; a segment tripping neither is untouched (stays a
    single-pass, byte-identical encode — never a needless generation); a segment
    tripping both gets ONE encode with a comma-joined ``<deband>,<eq>`` filter
    chain (deband first, exposure correction second — the same order the two
    passes ran in before, since source-match ran AFTER the deband pass).

    This is the call the production pipeline (``compile_shots``) uses; the two
    single-purpose functions above stay available standalone (A/B script,
    focused tests). Never raises — decode/encode here are best-effort."""
    to_deband, match_pairs = await asyncio.gather(
        _deband_decide(segment_paths, settings),
        _source_match_decide(segment_paths, settings, finish),
    )
    filters: dict[Path, list[str]] = {}
    if to_deband:
        from src.services import mastering

        deband_vf = mastering.deband_filter()
        for seg in to_deband:
            filters.setdefault(seg, []).append(deband_vf)
    for seg, f in match_pairs:
        filters.setdefault(seg, []).append(f)
    if not filters:
        return
    # ENCODE PASS — concurrent, mirroring _apply_conditional_deband: each call
    # re-encodes a DISTINCT segment into its own temp and .replace()s only its
    # own path, so no two tasks touch a shared file, and every one still queues
    # behind the same ``_run`` semaphore (per-pod ffmpeg ceiling unchanged).
    # ``return_exceptions=True`` is load-bearing for the same reason as the two
    # passes it replaces: this is best-effort picture polish that must never
    # fail a compile.
    items = list(filters.items())
    outcomes = await asyncio.gather(
        *(_polish_segment(seg, ",".join(fs), settings, deterministic) for seg, fs in items),
        return_exceptions=True,
    )
    for (seg, _fs), outcome in zip(items, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            log.warning(
                "picture_master.encode_failed",
                path=str(seg),
                error=str(outcome),
                exc_info=outcome,
            )


def _cadence_candidates(settings: Settings) -> frozenset[int]:
    """Parse the configured candidate cadences into a set of legal rates."""
    from src.services import cadence

    out: set[int] = set()
    for part in (settings.render_cadence_candidates or "").split(","):
        part = part.strip()
        if part:
            out.add(cadence.clamp_cadence(part))
    return frozenset(out) or frozenset({24, 25, 30, 50, 60})


def _resolve_master_cadence(shots: list[Shot], settings: Settings) -> int:
    """Pick the single master output cadence for this compile (VC-3).

    ``render_cadence_mode`` in {'auto','on'} runs the retime-cost selector over
    the shots' declared source rates (``Shot.target_fps``); with none declared —
    the default compile — it returns ``settings.render_fps`` unchanged, so the
    filter chain stays byte-identical. Any other mode (incl. an explicit request
    ``cadence`` which the route pins as 'fixed') keeps ``render_fps`` as-is.
    Selection is pure + never-raises, so this needs no error handling."""
    from src.services import cadence

    mode = (settings.render_cadence_mode or "auto").strip().lower()
    if mode not in {"auto", "on"}:
        return settings.render_fps
    declared = [shot.target_fps for shot in shots if shot.target_fps]
    if not declared:
        return settings.render_fps
    profile = cadence.source_rate_profile(declared)
    return cadence.master_cadence([profile], candidates=_cadence_candidates(settings))


async def _resolve_grade_vf(finish: FinishProfile | None, joined_path: Path) -> str:
    """Resolve the whole-film grade filter chain (VC-5) over the JOINED film, or
    ``""`` for an absent / disabled / all-identity grade (keeps the ``-c copy``
    fast path byte-identical).

    A ``match_sources`` grade measures the joined bus signalstats (best-effort —
    never fails the job) so the grade can bias toward the film's own luma/chroma
    before ``_grade_vf`` builds the chain."""
    grade = finish.grade if finish is not None else None
    if grade is None or not grade.enabled:
        return ""
    stats: dict[str, float] | None = None
    if grade.match_sources:
        from src.services import measure

        try:
            stats = await measure.measure_shot_stats(joined_path)
        except Exception as exc:  # noqa: BLE001 — never fail a job on measurement
            log.warning("compile_shots.match_sources_measure_failed", error=str(exc))
            stats = None
    return _grade_vf(grade, stats)


def _resolve_captions_vf(
    captions: list[ShotCaptions] | None,
    shots: list[Shot],
    shot_offsets: list[tuple[float, float]] | None,
    settings: Settings,
    work_root: Path,
    captions_render: str,
    caption_mood: str | None = None,
    captions_karaoke: bool = False,
    caption_look: CaptionLook | None = None,
) -> str:
    """Build the ``ass=`` filter fragment for burned-in captions, or "" when none.

    Assembles the ASS document from shot-local cues mapped onto the film bus,
    writes it into ``work_root``, and returns the ``ass=`` filtergraph fragment.
    Returns "" when there are no captions, no surviving cues, or ``captions_render``
    is not ``"burned_in"`` (``"overlay"`` muxes a soft track instead of burning),
    so the combined ``-vf`` (and the ``-c copy`` fast path) stays byte-identical.

    ``caption_mood`` selects a per-video caption look (``None``/unknown ⇒ the
    default style, byte-identical). ``captions_karaoke`` renders per-word ``{\\k}``
    highlight (burned_in only). ``caption_look`` carries the request's caption
    typography (font family, text colour, one box per cue); ``None`` ⇒ the same
    ASS document and the same ``ass=`` fragment as before it existed.
    """
    if not captions or captions_render != "burned_in":
        return ""
    from src.services.captions_ass import (
        build_film_ass,
        captions_vf_fragment,
        resolve_caption_style,
    )

    speeds = [shot.speed for shot in shots]
    style = resolve_caption_style(caption_mood)
    ass_text = build_film_ass(
        captions,
        shot_offsets or [],
        speeds,
        settings,
        style=style,
        karaoke=captions_karaoke,
        look=caption_look,
    )
    if not ass_text:
        return ""
    ass_path = work_root / "captions.ass"
    ass_path.write_text(ass_text, encoding="utf-8")
    return captions_vf_fragment(
        ass_path, caption_look.fontsdir if caption_look is not None else None
    )


async def _mux_soft_captions(
    final_path: Path,
    captions: list[ShotCaptions] | None,
    shots: list[Shot],
    shot_offsets: list[tuple[float, float]] | None,
    settings: Settings,
    work_root: Path,
    captions_render: str,
    deterministic: bool = False,
) -> Path:
    """Mux word-timed cues into the delivered MP4 as a selectable mov_text soft
    subtitle track — the ``"overlay"`` counterpart to ``_resolve_captions_vf``'s
    burn. A follow-on `-c copy` remux — no video re-encode, so nothing about the
    picture changes. Returns the original `final_path` UNCHANGED (byte-identical)
    unless ``captions_render == "overlay"`` with surviving cues, and also on any
    mux failure (captions must NEVER fail a render)."""
    if captions_render != "overlay" or not captions:
        return final_path
    from src.services.captions_ass import build_film_srt

    speeds = [shot.speed for shot in shots]
    srt_text = build_film_srt(captions, shot_offsets or [], speeds, settings)
    if not srt_text:
        return final_path
    srt_path = work_root / "captions.srt"
    srt_path.write_text(srt_text, encoding="utf-8")
    out_path = work_root / "subbed.mp4"
    try:
        await _run(
            [
                "ffmpeg",
                "-y",
                *_bitexact_input_args(deterministic),
                "-i",
                str(final_path),
                "-i",
                str(srt_path),
                "-map",
                "0",
                "-map",
                "1",
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                "-movflags",
                "+faststart",
                "-metadata:s:s:0",
                "language=eng",
                str(out_path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — captions must never fail the render
        log.warning("compile_shots.soft_caption_mux_failed", error=str(exc))
        return final_path
    return out_path


# ---------------------------------------------------------------------------
# Content-addressed per-shot render cache (C10/AC10)
#
# The orchestrator's evaluator-optimizer loop re-submits a shot list where only
# a few shots changed between calls; the spec forbids re-rendering the unchanged
# shots. Each shot renders to a self-contained ``seg-*.mp4`` BEFORE the join, so
# an unchanged shot's segment is reusable byte-for-byte. We key that segment by a
# content hash of exactly the inputs ``render_shot`` reads into its pixels/audio.
#
# Correctness rule: a hit must NEVER serve stale pixels, so the key over-includes
# rather than under-includes; a MISS is byte-identical to a normal render.
# ---------------------------------------------------------------------------
_SHOT_CACHE_DIRNAME = "shot-render-cache"
# Bump to invalidate every existing entry after a change to the hashing logic or
# to what render_shot draws into a segment.
#
# 3: the seeked-clip duration fix. Before it, a clip rendered with an input seek
#    (``-ss``) reported its PRE-seek probed length, so the value written into the
#    ``{key}.json`` sidecar was overstated by the seek offset. The fix corrected
#    the arithmetic but did NOT change the key, so every entry cached before it
#    still hashes the same and ``_shot_cache_get`` happily replays the wrong
#    duration — the bad number outlives the bug. The segment PIXELS are fine;
#    it is the persisted metadata that is stale, and that metadata flows into
#    ``total_seconds`` and from there into the music-bed ``atrim``/``afade``
#    timing, so a stale hit mis-times the fade-out on an otherwise correct video.
# 4: the VC-3 retime wiring fix. ``_cadence_fragment`` used to pass
#    ``render_fps`` as BOTH source and target, so the ratio was always exactly 1.0
#    and every shot got the plain ``fps=N`` drop/dup — the minterpolate/tblend
#    branches were unreachable. Now the shot's declared ``target_fps`` (source
#    rate) and ``retime`` hint drive the fragment. Both inputs are already hashed
#    (``shot.model_dump`` / ``render_motion_blur``), but an entry cached BEFORE
#    the fix hashes the same key and would replay the old drop/dup PIXELS for a
#    shot that should now be interpolated — so the version is bumped to retire it.
# 5: the atempo carve-out on a synthesised silent track. ``_apply_shot_speed`` now
#    passes ``retime_audio=not silent_primary or has_sfx``, so a silent image/text
#    (or source-silenced) shot at a non-unit ``speed`` no longer runs ``atempo`` over
#    its all-zero ``anullsrc`` buffer. Every input to the key (``shot.model_dump``,
#    the settings view) is unchanged by that, so an entry cached BEFORE it hashes
#    identically and would replay the old SAMPLES for a shot whose audio track is now
#    generated differently — the same class of stale hit as 3 and 4.
# 6: the ``setpts=PTS-STARTPTS`` frame-exactness rebase (``_STARTPTS_REBASE``,
#    #174). ``_cadence_fragment`` now appends a filter, so a shot rendered with a
#    mid-frame input seek gains the final frame the old chain dropped — the segment
#    has one MORE frame than an entry cached before the fix, for the same shot and
#    the same settings. Neither input to the key changed (``shot.model_dump``, the
#    settings view), so a stale entry hashes identically and would replay the SHORT
#    segment: exactly the class of stale hit as 3, 4 and 5.
#
#    ⚠️ Measured, so nobody over-reads this: today the cache dir is an ``emptyDir``
#    (`/tmp/video-compiler`, `sizeLimit: 5Gi`), so a rollout starts every pod with an
#    empty cache and #174's fix could not have been masked in production — 87 MB of
#    pre-fix entries were live on the old pods and died with them. This bump is
#    therefore NOT a live-defect fix; it is the module's own stated rule
#    ("bump ... after a change to ... what render_shot draws into a segment") being
#    honoured, so that the day the cache becomes a PVC or survives a restart, the
#    dropped frame does not quietly come back with it.
# 6 -> 7: the -t-bounded clip branch now carries a terminal
#    ``tpad=stop_mode=clone:stop = -1``, so a segment whose source hands over less
#    picture than its header claims is filled instead of ending early. That
#    changes what render_shot DRAWS into such a segment, and a pre-bump entry for
#    exactly that shot is a segment with the short picture baked in — the module's
#    rule below applies verbatim.
# 7 -> 8: the segment's AUDIO CODEC changed (AAC -> lossless ALAC, see
#    ``_SEGMENT_AUDIO_CODEC_ARGS``). This is the one bump that MUST NOT be
#    skipped, and it is the first entry here that is about SAMPLES rather than
#    pixels. A stale AAC entry served for one shot next to a freshly rendered
#    ALAC segment would be concat-COPIED together by ``concat_shots``: a
#    heterogeneous join, and one that reintroduces the exact 24 ms-per-cut hole
#    this change removes — intermittently, on whichever shots happened to hit the
#    cache. The bump is the whole barrier: every key computed after it differs, so
#    no pre-bump entry is addressable. (The container is deliberately unchanged,
#    so the stale entries stay ordinary ``<key>.mp4`` files that the evictor
#    already counts and reclaims — nothing is orphaned or pinned.)
#
# v9 (1c419ed): caption_drawtext_font_size_frac / caption_drawtext_margin_v_frac
# joined the hashed view. They size/place the per-shot `drawtext` caption
# fragment baked into the cached segment (_caption_filter_fragment), so a
# resolution-driven change to either would otherwise replay a stale segment
# captioned at the wrong size.
# v10: TWO changes, one bump.
#   (a) The narration caption moved to AFTER the focus punch-in on shots that carry
#       a focus_track (_defers_caption), so those segments' pixels differ.
#   (b) ⛔ A MISSED BUMP, corrected here. #218 made `-vsync cfr` unconditional and
#       #219 added explicit `-g`/`-keyint_min`; both change the bytes `render_shot`
#       bakes into a cached segment and NEITHER bumped this. The version exists
#       precisely so an encode change cannot replay a stale segment — leaving it at
#       9 meant a warm cache could serve pre-#218 segments that never got the CFR
#       pin. (In practice a new image starts a new pod with a cold cache, so this
#       was latent rather than live; it is still the invariant, so restore it.)
# v11: segment audio pinned to ``-sample_fmt s32p`` (24-bit ALAC). A v10 entry for a
#   narration shot is 16-bit ALAC; replayed next to a fresh 24-bit segment it is a
#   mixed-depth copy-concat — exactly the no-narration/noise defect the pin removes
#   (``concat_shots`` would now refuse it and fail the job instead).
# v12 (#232): ``_freeze_pad_suffix`` / ``_bounded_cover_pad`` stopped emitting the
#   unconditional, infinite ``tpad=stop_mode=clone:stop = -1`` on every bounded
#   (narration/cover) segment and replaced it with a FINITE ``stop_duration``
#   ceiling'd at the segment's own ``duration``. The emitted filtergraph string
#   for every bounded shot is different bytes, so a warm v11 entry would be
#   replayed under a DIFFERENT command than the one that produced it. More than
#   cosmetic: a v11 entry for the lying-header shape (video probes as covering
#   the narration but decodes far less — video_compiler #212) was cached from a
#   run that shipped a SHORT picture under the full narration; that cache entry
#   must not survive to be served as "already correct" once the fix can actually
#   repair it. And the sub-one-output-frame shape (the ~17ms gap that triggered
#   the livelock itself) could never have produced a cached v11 entry at all — a
#   livelocked render is killed by the subprocess timeout, never completes, and
#   is retried, so there is nothing stale to worry about there specifically; the
#   lying-header case is the one that actually needs the bump.
# v13 (#233, merged independently as its own v12 on main): a narrated shot's
#   ``seconds`` is now a picture FLOOR (``_narrated_picture_floor``). A v12 entry
#   for a footage beat whose ``seconds`` outlasts its narration is the SHORT
#   narration-length segment; the shot's key inputs are unchanged, so without a
#   further bump a warm cache would replay exactly the truncated segment this
#   change removes. Bumped once more (not left at the colliding v12) because
#   BOTH branches changed the bytes ``render_shot`` bakes into a cached segment
#   for overlapping shot shapes (any bounded, narrated segment) — v12-from-#232
#   and v12-from-#233 are two different command lines claiming the same key.
# v14: cursor/focus/callout times are shifted by the seek the compiler chose
#   (``_shift_tracks_to_seek``), so a shared-clip pan or an 'end'-anchored shot now
#   draws its cursor, click rings and badges at different times than a v13 entry.
# v15: every bounded segment now ends on the output frame grid
#   (``_snap_to_frame_grid``) with its audio padded to exactly its picture. A v14
#   entry for an off-grid shot is the segment whose audio and video differ in
#   length — replaying it would put the timestamp hole / stretched frame straight
#   back into the film.
# v16: every last-frame clone declares its frame rate first (``_clone_pad``). On the
#   deployed ffmpeg 7.1.5 the bare ``tpad`` after ``setpts`` cloned NOTHING, so a
#   v15 entry for any clip shorter than its narration is the segment whose picture
#   stops at the clip's own length under the full audio — the defect itself, which
#   must not be replayed as "already rendered".
_SHOT_CACHE_KEY_VERSION = 16


def _shot_cache_dir(settings: Settings) -> Path:
    """Stable per-pod dir for cached shot segments.

    Lives under ``settings.work_dir`` but is a SIBLING of the throwaway
    ``job-*`` scratch dirs, so ``_cleanup_job_dir`` (which only ever removes a
    strict ``job-*`` subdirectory of ``work_dir``) never deletes it. It therefore
    survives across separate ``POST /api/compile`` calls within one pod — exactly
    what the re-compile loop needs. Cross-pod persistence is best-effort only:
    ``work_dir`` is ephemeral container scratch, so a re-compile that lands on a
    different / restarted pod hits a cold cache and simply renders in full (==
    today's behaviour). A config change forces a new pod (Recreate strategy)
    whose cold cache guarantees pod-constant settings can never go stale.
    """
    return Path(settings.work_dir) / _SHOT_CACHE_DIRNAME


def _shot_cache_settings_view(settings: Settings) -> dict[str, object]:
    """Render settings that change a per-shot SEGMENT's bytes.

    Only the knobs ``render_shot`` bakes into the segment belong here. Deband
    (applied to the segments AFTER render, pre-concat), the final-audio master,
    and whole-film caption/grade settings are excluded on purpose — they are
    applied to the materialized segment / joined film downstream, so they never
    alter the CACHED bytes and must not shift the key. Pod-constant encode knobs
    are included anyway (harmless: equal across a pod's compiles); the per-job
    overridable ones (dimensions / fps / crf / letterbox) are the load-bearing
    entries the loop actually varies by tier/format.
    """
    return {
        "render_width": settings.render_width,
        "render_height": settings.render_height,
        "render_fps": settings.render_fps,
        "render_crf": settings.render_crf,
        "render_preset": settings.render_preset,
        "render_encoder": settings.render_encoder,
        "render_hw_bitrate": settings.render_hw_bitrate,
        "render_colorspace": settings.render_colorspace,
        "render_motion_blur": settings.render_motion_blur,
        "render_bitrate_mode": settings.render_bitrate_mode,
        "render_vbv_maxrate": settings.render_vbv_maxrate,
        "render_vbv_bufsize": settings.render_vbv_bufsize,
        "render_letterbox_color": settings.render_letterbox_color,
        "render_bg_color": settings.render_bg_color,
        "render_text_color": settings.render_text_color,
        "render_font": settings.render_font,
        # Per-shot drawtext caption size/margin (_caption_filter_fragment) — baked
        # into the cached segment's pixels, unlike the whole-film ASS caption
        # knobs below, which are excluded (applied downstream of the segment).
        "caption_drawtext_font_size_frac": settings.caption_drawtext_font_size_frac,
        "caption_drawtext_margin_v_frac": settings.caption_drawtext_margin_v_frac,
    }


# Every Settings field that is deliberately NOT in _shot_cache_settings_view,
# because it either (a) resolves EARLIER than the per-shot fan-out and is already
# folded into a hashed field — render_cadence_mode / render_cadence_candidates
# collapse into the single render_fps before render_shot runs — or (b) applies
# DOWNSTREAM of the cached segment (source-match / deband / final-audio master /
# whole-film captions / grade) or is pure infra (minio / metering / concurrency /
# work_dir / timeouts / logging) that never touches a segment's pixels.
#
# Invariant (enforced by tests/test_shot_render_cache.py): every Settings field
# is in EXACTLY one of {hashed view, this excluded set}. A new segment-affecting
# field must go into _shot_cache_settings_view AND bump _SHOT_CACHE_KEY_VERSION;
# any other new field goes here.
_SHOT_CACHE_EXCLUDED_SETTINGS: frozenset[str] = frozenset(
    {
        "port",
        "log_level",
        "minio_endpoint",
        "minio_access_key",
        "minio_secret_key",
        "minio_secure",
        "minio_bucket",
        "minio_audio_bucket",
        "minio_url_expiry_seconds",
        # Job-row retention / idempotency dedup window — job-store housekeeping,
        # never read by segment rendering, so it cannot change a segment's pixels.
        "compile_job_retention_seconds",
        "render_cadence_mode",
        "render_cadence_candidates",
        "render_vfr_epsilon",
        "render_dead_frame_luma_spread_floor",
        # Same classification as the floor beside it: both are read by the
        # post-encode mastering GATE, never by segment rendering, so neither
        # changes a segment's pixels and neither belongs in the cache key.
        "render_dead_frame_min_ratio",
        # Whether a CLEARED post-mux gate fails the render or merely annotates it.
        # It decides what happens AFTER the mux, off the delivered file — it cannot
        # change a single segment's pixels, so it must not enter the shot-cache key.
        "mastering_gate_mode",
        # The six per-gate kill switches. Same ground as ``mastering_gate_mode``
        # above: each decides whether a CLEARED post-mux gate fails the render,
        # read only by ``mastering.gate_armed`` and the route's verify branch —
        # never by segment rendering, so none can change a segment's pixels.
        # Proven by read-site audit rather than analogy: zero reads in
        # renderer.py / measure.py.
        "mastering_gate_av_coverage_enabled",
        "mastering_gate_dead_frame_enabled",
        "mastering_gate_banding_measurement_enabled",
        "mastering_gate_verify_enabled",
        "mastering_gate_colorspace_enabled",
        "mastering_gate_cfr_enabled",
        "mastering_gate_loudness_enabled",
        # VC-W14: same classification as their five siblings above — all six are
        # read only by ``mastering.gate_armed`` over the FINAL delivered mux,
        # downstream of every cached segment, so none can change a segment's
        # samples.
        "mastering_gate_true_peak_enabled",
        "mastering_gate_measure_enabled",
        "mastering_gate_bed_calibration_enabled",
        # Same classification as its siblings above: read only by
        # ``mastering.gate_armed`` from inside ``post_mux_verify``, which runs
        # AFTER the mux and only records a ``MasteringGateViolation`` — it never
        # touches segment rendering, so it cannot change a segment's pixels.
        "mastering_gate_speech_presence_enabled",
        # Same classification as its siblings above: read only by
        # ``mastering.gate_armed`` from inside ``post_mux_verify``'s
        # ``narration_presence_report`` call, which runs AFTER the mux over the
        # delivered file and never touches segment rendering.
        "mastering_gate_narration_presence_enabled",
        # The INTERMEDIATE encode cell (#wave11). Excluded on the same ground as
        # ``render_deband`` beside it, and provably so rather than by analogy:
        # ``mastering.encode_ladder`` reads this field at EXACTLY ONE line, inside
        # ``if intermediate:`` — so a caller that does not pass
        # ``intermediate=True`` cannot observe it at all. Exactly FOUR call sites
        # pass it (``_deband_segment`` / ``_regrade_segment`` / ``_polish_segment``
        # / ``_render_xfade_group``, pinned by
        # ``tests/unit/test_av_quality_wave11.py::_INTERMEDIATE_ENCODERS``), and
        # this comment said THREE until VC-W14/V8 — ``_render_xfade_group`` was
        # added in wave 13 and the pin test was updated while the prose was not.
        # The first three run from ``_apply_picture_master``, which is invoked in
        # ``compile_shots``
        # AFTER the render gather — i.e. after ``_shot_cache_put`` has already
        # published the segment, and against the caller's own COPY of it (a cache
        # hit copies bytes out, never symlinks). ``render_shot``'s own encode calls
        # ``_video_codec_args`` WITHOUT ``intermediate``, so the cached bytes are
        # delta-independent by construction.
        #
        # Consequence, which is the whole point of the classification: a segment
        # cached under delta=0 and replayed under delta=6 is still correct, because
        # the polish re-encode runs fresh on the materialized copy with the CURRENT
        # delta every time. Hashing it would force a full cache invalidation that
        # buys nothing. Pinned by
        # tests/test_shot_render_cache.py::test_intermediate_crf_delta_is_not_segment_affecting.
        "render_intermediate_crf_delta",
        "render_deband",
        "render_source_match",
        "source_match_tolerance",
        "render_output_platform",
        "master_enabled",
        "master_lufs",
        "master_tp",
        "master_lra",
        # Master bus compressor/limiter shape (#325b871). Same classification as
        # the loudness targets beside them: read only by ``_master_filter`` on the
        # FINAL audio bus, downstream of every cached segment, so they cannot
        # change a segment's samples and must not enter the shot-cache key.
        "master_compressor_threshold_db",
        "master_compressor_ratio",
        "master_compressor_makeup",
        "master_limiter_attack_ms",
        "master_limiter_release_ms",
        # Audiobook ACX program-master knobs — concat-audio path only, never
        # touch per-shot video rendering, so excluded from the shot-cache view.
        "render_audiobook_lufs_target",
        "render_audiobook_lufs_min",
        "render_audiobook_lufs_max",
        "render_audiobook_tp",
        "render_audiobook_lra",
        "render_audiobook_noise_floor_db",
        "caption_font_family",
        "caption_font_size_frac",
        "caption_margin_v_frac",
        "caption_outline_px",
        "caption_primary",
        "caption_outline_colour",
        "caption_secondary",
        "caption_max_line_chars",
        "caption_box_enabled",
        "caption_box_colour",
        "caption_box_padding_frac",
        "work_dir",
        "shot_cache_max_bytes",
        "max_concurrent_ffmpeg_subprocesses",
        "max_concurrent_compile_jobs",
        "job_work_timeout_seconds",
        "ffmpeg_subprocess_timeout_seconds",
        "max_fetch_bytes",
        # Submit-time input-source policy (reject file://); gates acceptance, never a shot's pixels.
        "allow_local_file_inputs",
        "metering_redis_url",
        # Crash-resumption reaper knobs — dispatch/lifecycle policy, never a shot's pixels.
        "reaper_enabled",
        "reaper_interval_seconds",
        "job_lease_seconds",
        "reclaim_max_attempts",
        "reclaim_max_lifetime_seconds",
        "prune_interval_seconds",
        # Terminal-row retention = the Idempotency-Key dedup horizon. Store-lifecycle
        # policy (and a billing guard), never a shot's pixels.
        "job_retention_seconds",
        # Durable job-store connection — infra config, never touches a shot's pixels.
        "database",
    }
)


def _shot_cache_key(
    shot: Shot,
    settings: Settings,
    clip_seek_fraction: float,
    deterministic: bool = False,
    freeze_pad_cap: bool = False,
    all_shots_muted: bool = False,
) -> str:
    """SHA-256 over EXACTLY the inputs that determine a shot's rendered segment.

    A hit replays a prior ``seg-*.mp4``, so this must fold in every input
    ``render_shot`` reads into the segment: all of the shot's render fields (only
    ``line_id`` is excluded — a pure identifier ``render_shot`` never reads;
    ``id`` IS included because the text-card fallback draws it onto the frame),
    the per-shot ``clip_seek_fraction`` (footage in-point), and the
    segment-affecting render settings. Fields whose effect lands only at the
    join (``transition``, ``audio`` gain/mute) stay in the payload too — over-
    hashing merely costs a re-render, whereas under-hashing serves stale pixels.

    ``deterministic`` changes the segment BYTES (``-threads 1``/``+bitexact``), so
    a deterministic render must never collide with a normal one in the cache. It
    is folded into the key ONLY when set, so the default (prod) key is UNCHANGED
    — the existing pod cache stays valid and a deterministic job simply occupies
    its own namespace. ``freeze_pad_cap`` likewise changes the drift pixels of an
    over-narrated footage shot, so it is folded in the same way (only when set) —
    the shot's ``max_seconds`` is already in the payload via ``model_dump``, but the
    job-level ENFORCEMENT flag is not, so it must be keyed too.

    ``all_shots_muted`` is a third such job-level flag, but it changes the SAMPLES
    of a muted shot only (source silence instead of ``0:a``) and nothing at all for
    an unmuted one. So it is folded in only for a shot that is ACTUALLY muted on an
    all-muted request — the music-only reel gets its own namespace while every other
    job's key, muted shots included, stays exactly what it is today.
    """
    payload: dict[str, object] = {
        "v": _SHOT_CACHE_KEY_VERSION,
        "shot": shot.model_dump(mode="json", exclude={"line_id"}),
        "clip_seek_fraction": clip_seek_fraction,
        "settings": _shot_cache_settings_view(settings),
    }
    if deterministic:
        payload["deterministic"] = True
    if freeze_pad_cap:
        payload["freeze_pad_cap"] = True
    if all_shots_muted and shot.audio is not None and shot.audio.mute:
        payload["mute_at_source"] = True
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _place_cache_entry(src: Path, seg: Path, meta: Path, duration: float) -> None:
    """Atomically publish ``src`` + its duration as the cache entry for a key.

    Writes to temp names in the cache dir (same filesystem → atomic ``replace``)
    then swaps the mp4 in FIRST and the sidecar SECOND. A reader requires BOTH to
    exist, so the only visible intermediate state (segment present, sidecar not)
    reads as a miss — never a half-entry served as a hit.
    """
    tmp_seg = seg.with_name(f"{seg.name}.{uuid.uuid7().hex}.tmp")
    tmp_meta = meta.with_name(f"{meta.name}.{uuid.uuid7().hex}.tmp")
    shutil.copyfile(src, tmp_seg)
    tmp_meta.write_text(json.dumps({"duration": duration}), encoding="utf-8")
    tmp_seg.replace(seg)
    tmp_meta.replace(meta)


def _evict_shot_cache(cache_dir: Path, max_bytes: int) -> None:
    """Evict oldest cache entries (by mtime) until the dir is within ``max_bytes``.

    Best-effort and synchronous (run under ``to_thread`` before a put). An entry
    is a ``key<_SEGMENT_EXT>`` + ``key.json`` pair, removed together, oldest-first,
    so the cache that deliberately survives per-job cleanup can't grow without
    bound and fill ``work_dir`` on the long-lived single pod. Racing a concurrent
    get is safe: the get is guarded and simply falls back to a render on any error.

    Entries left behind by a ``_SHOT_CACHE_KEY_VERSION`` bump are no longer
    addressable but are still ORDINARY entries here (same extension), so they are
    counted and reclaimed oldest-first like any other — a version bump costs cache
    warmth, never resident bytes.
    """
    entries: list[tuple[float, int, Path, Path]] = []
    total = 0
    for seg in cache_dir.glob(f"*{_SEGMENT_EXT}"):
        try:
            st = seg.stat()
        except OSError:
            continue
        meta = seg.with_suffix(".json")
        try:
            meta_size = meta.stat().st_size if meta.is_file() else 0
        except OSError:
            meta_size = 0
        size = st.st_size + meta_size
        total += size
        entries.append((st.st_mtime, size, seg, meta))
    if total <= max_bytes:
        return
    entries.sort(key=lambda e: e[0])  # oldest first
    for _mtime, size, seg, meta in entries:
        if total <= max_bytes:
            break
        for path in (seg, meta):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        total -= size


async def _shot_cache_get(key: str, dest: Path, settings: Settings) -> float | None:
    """Materialize the cached segment for ``key`` into ``dest``; return its duration.

    Returns ``None`` on a miss OR any error — the caller then renders normally,
    so the cache can never fail a compile. Copies (never symlinks) so the later
    in-place deband and the per-job dir cleanup can't mutate/delete the shared
    cache entry. A non-positive ``shot_cache_max_bytes`` disables the cache.
    """
    if settings.shot_cache_max_bytes <= 0:
        return None
    try:
        cache_dir = _shot_cache_dir(settings)
        seg = cache_dir / f"{key}{_SEGMENT_EXT}"
        meta = cache_dir / f"{key}.json"
        if not (seg.is_file() and meta.is_file()):
            return None
        duration = float(json.loads(meta.read_text(encoding="utf-8"))["duration"])
        await asyncio.to_thread(shutil.copyfile, seg, dest)
    except Exception:  # cache is best-effort, never fail the render
        log.warning("shot_cache.get_failed", key=key, exc_info=True)
        return None
    log.debug("shot_cache.hit", key=key)
    return duration


async def _shot_cache_put(key: str, src: Path, duration: float, settings: Settings) -> None:
    """Publish a freshly rendered ``src`` segment into the cache under ``key``.

    Fully best-effort — any error is swallowed. A no-op when the entry already
    exists (a concurrent renderer won the race), which is safe because identical
    keys imply identical bytes. A non-positive ``shot_cache_max_bytes`` disables
    the cache; otherwise oldest entries are evicted to stay under that cap.
    """
    if settings.shot_cache_max_bytes <= 0:
        return
    try:
        cache_dir = _shot_cache_dir(settings)
        seg = cache_dir / f"{key}{_SEGMENT_EXT}"
        meta = cache_dir / f"{key}.json"
        if seg.is_file() and meta.is_file():
            return
        await asyncio.to_thread(cache_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_place_cache_entry, src, seg, meta, duration)
        # Evict AFTER publishing so the just-added (newest-mtime) entry is counted
        # and protected — only older entries are pruned back under the cap.
        await asyncio.to_thread(_evict_shot_cache, cache_dir, settings.shot_cache_max_bytes)
    except Exception:  # cache is best-effort, never fail the render
        log.warning("shot_cache.put_failed", key=key, exc_info=True)


async def _render_shot_cached(
    shot: Shot,
    dest: Path,
    settings: Settings,
    work_root: Path,
    http: httpx.AsyncClient,
    *,
    clip_seek_fraction: float,
    gate: asyncio.Semaphore,
    deterministic: bool = False,
    freeze_pad_cap: bool = False,
    all_shots_muted: bool = False,
    source_cache: SourceFetchCache | None = None,
) -> tuple[float, Path | None]:
    """``render_shot`` with the content-addressed segment cache in front.

    A hit replays a prior segment (no ffmpeg, no ``gate`` budget consumed); a miss
    renders normally under the subprocess gate and then publishes the result. Key
    computation and every cache IO are guarded so ANY cache fault falls back to a
    plain render — the cache can never fail a compile, and a miss is byte-identical
    to today's behaviour. The segment is cached as produced by ``render_shot`` —
    i.e. BEFORE the post-render deband pass — so a later hit is debanded once (per
    this job's settings), never doubly.

    Returns ``(duration, narration_audio_path)``. The second element is this
    shot's local narration source clip, for ``mastering.narration_presence_report``
    to reuse post-mux instead of re-fetching a possibly-expired presigned URL —
    ``None`` for a shot with no ``audio_url``, and ALSO ``None`` on a cache HIT:
    a replayed segment never calls ``render_shot``/``_resolve_shot_assets``, so no
    raw narration source was fetched THIS compile and there is nothing local to
    hand back. That shot is reported UNMEASURED (never "missing") downstream.
    """
    key: str | None = None
    try:
        key = _shot_cache_key(
            shot,
            settings,
            clip_seek_fraction,
            deterministic,
            freeze_pad_cap,
            all_shots_muted,
        )
    except Exception:  # cache is best-effort, never fail the render
        log.warning("shot_cache.key_failed", shot_id=shot.id, exc_info=True)
    if key is not None:
        hit = await _shot_cache_get(key, dest, settings)
        if hit is not None:
            return hit, None
    audio_sink: list[Path | None] = []
    async with gate:
        duration = await render_shot(
            shot,
            dest,
            settings,
            work_root,
            http,
            clip_seek_fraction=clip_seek_fraction,
            deterministic=deterministic,
            freeze_pad_cap=freeze_pad_cap,
            all_shots_muted=all_shots_muted,
            source_cache=source_cache,
            narration_audio_sink=audio_sink,
        )
    if key is not None:
        await _shot_cache_put(key, dest, duration, settings)
    audio_path = audio_sink[0] if audio_sink else None
    return duration, audio_path


def _flag_unapplied_music_bed(
    report: dict[str, float | bool | None] | None,
    *,
    requested: bool,
    unfetchable: bool,
    url: str | None = None,
    ambient_requested: int = 0,
    ambient_lost: int = 0,
) -> dict[str, float | bool | None] | None:
    """Guarantee a requested-but-not-applied bed (music OR ambient) is NEVER
    reported as ``None``.

    Three ways a requested bed can fail to land, all flagged the same channel:

    1. ``unfetchable`` — ``_fetch_asset`` returned ``None`` for ``music.url``. This
       was the only case originally flagged (see git history), by a caller-side
       check right after the fetch.
    2. Every OTHER way a MUSIC bed goes missing: the fetch reported success (or
       the fetch step itself never ran on the code path that produced this
       report) yet ``overlay_audio_bed`` still did not come back with a
       populated report. ``overlay_audio_bed`` unconditionally builds a
       ``music_bed_report`` dict whenever it is entered with ``music is not
       None`` (see its ``music_bed_report = {...}`` — every key filled, never
       left ``None``) — so ``report is None`` at this point with
       ``requested=True`` is a state that should be UNREACHABLE by construction,
       and exactly the kind of "should never happen" that silently shipped a bed-
       less ``completed`` master with no trace anywhere in the job's own logs. A
       production run (jazz_lounge.mp3 requested, ``duck_db``/``music_offset_lu``
       set, zero GETs of the asset anywhere in a 3-hour pod log, zero
       ``music_bed_unfetchable`` log line, delivered master with no ``music_bed``
       key at all and narration-silent windows at digital-silence floor) is the
       proof this branch exists to close, regardless of which exact upstream
       mechanism dropped it on that run — a defect that cannot be reproduced does
       not mean a caller downstream should ever again see ``music_bed: null`` for
       a compile that asked for one.
    3. ``ambient_lost > 0`` — one or more requested ``AmbientCue`` URLs came back
       unfetchable (``compile_shots``'s ambient fetch loop). Unlike music, a lost
       ambient cue never raised ``music_bed_report`` off ``None`` at all: the
       report is built ONLY inside ``overlay_audio_bed``'s ``if music is not
       None:`` branch, so an AMBIENT-ONLY compile (the common case — most
       ambient cues ride with no music bed at all) whose bed fails to fetch
       reports ``music_bed: null``, byte-identical to a compile that never asked
       for ambient in the first place. This is the SAME silent-loss shape music
       had before this function existed, just for the sibling bed type — folded
       into the SAME channel (this service has exactly one per-job note field for
       "a continuous bed was requested and something about it did not land")
       rather than inventing a second one.

    ``requested=False`` and ``ambient_lost=0`` (no bed was ever asked for, or
    every requested ambient cue fetched fine) always returns ``report``
    untouched — ``None`` stays ``None``, so nothing changes for the overwhelming
    majority of compiles.
    """
    ambient_incomplete = ambient_lost > 0
    if not requested and not ambient_incomplete:
        return report
    merged = dict(report or {})
    touched = False
    if unfetchable:
        merged.update(requested=True, applied=False, fetch_failed=True)
        touched = True
    if ambient_incomplete:
        merged.update(
            ambient_requested=ambient_requested,
            ambient_lost=ambient_lost,
            ambient_fetch_failed=True,
        )
        touched = True
    if touched:
        return merged
    if requested and report is None:
        log.error(
            "compile_shots.music_bed_lost_unknown_cause",
            url=url[:200] if url else None,
        )
        return {"requested": True, "applied": False, "fetch_failed": False}
    return report


def _extend_narration_audio_sink(
    sink: list[Path | None] | None, render_results: list[tuple[float, Path | None]]
) -> None:
    """Extend ``sink`` with each shot's narration audio path when the caller
    supplied one — a plain guarded extend, split out for the same reason as
    ``_record_narration_audio``: keep the branch off ``compile_shots``' own
    cyclomatic complexity."""
    if sink is not None:
        sink.extend(audio_path for _duration, audio_path in render_results)


async def compile_shots(
    shots: list[Shot],
    settings: Settings,
    music: MusicBed | None = None,
    ambient: list[AmbientCue] | None = None,
    finish: FinishProfile | None = None,
    *,
    captions: list[ShotCaptions] | None = None,
    captions_render: str = "overlay",
    caption_mood: str | None = None,
    captions_karaoke: bool = False,
    caption_look: CaptionLook | None = None,
    emit_shot_timeline: bool = False,
    deterministic: bool = False,
    freeze_pad_cap: bool = False,
    narration_audio_sink: list[Path | None] | None = None,
) -> tuple[Path, float, dict[str, float | bool | None] | None, list[dict[str, object]] | None, int]:
    """Render and concat ``shots`` to a single MP4. Returns (path, total_seconds,
    music_bed_report, shot_timeline, master_fps).

    ``narration_audio_sink``, when given as an (empty) list, is filled in place —
    an OUT-PARAMETER rather than a widened return tuple, since >15 tests already
    unpack this function's 5-tuple positionally. On return it holds one entry per
    shot, INDEX-ALIGNED to ``shots``/``shot_timeline``: the local path this
    compile fetched for that shot's narration source (``None`` for an unnarrated
    shot, or for a shot whose segment came from the content-addressed render
    cache, meaning no raw source was fetched this run). The caller threads this
    straight into ``mastering.post_mux_verify`` so the post-mux narration-presence
    check reuses these files instead of re-fetching a presigned URL that may have
    expired by the time verification runs, at the very end of the compile. The
    referenced files live under this call's ``work_root`` and outlive this
    function's return — the caller owns cleanup (see the docstring below) and
    must not remove it until verification has read these paths.

    When ``music`` or ``ambient`` is supplied, a single audio-only re-encode pass
    (``overlay_audio_bed``) lays the continuous bed over the concatenated video
    after the stream-copy concat. A bed asset that can't be fetched is dropped;
    if every bed asset drops, the concat output is returned unchanged. An
    unfetchable MUSIC bed is additionally reported (``music_bed_report`` gains
    ``requested/applied/fetch_failed``) — a music-driven product whose bed is
    lost delivers a SILENT film, which must not read as a clean success.

    ``finish.grade`` (VC-5), when it resolves to a non-empty ``_grade_vf``, applies
    ONE whole-film colour-grade re-encode over the post-concat, post-CFR bus —
    folded into the bed pass when a bed exists, else a dedicated pass. An absent /
    disabled / all-identity grade leaves the ``-c copy`` fast path byte-identical.
    A ``match_sources`` grade measures the joined film's signalstats (best-effort)
    and biases the grade toward them.

    ``music_bed_report`` (T1) is ``overlay_audio_bed``'s auditable
    VO-relative-leveling report — ``None`` when no music bed was requested, and a
    ``{"requested": True, "applied": False, "fetch_failed": True}`` marker when one
    WAS requested but could not be fetched. (The grade report is assembled
    route-side from the request via ``grade_report``.)

    ``shot_timeline`` is the authoritative per-shot ``[start, end)`` on the final
    mux, keyed by line_id — built only when ``emit_shot_timeline`` is True (which
    also forces the offset probe). ``None`` on the default probe-free fast path.

    ``deterministic`` (the per-job ``deterministic`` render tag, default off) makes
    the whole encode chain bit-reproducible across runs — single-threaded
    ``+bitexact`` video/audio encodes plus ``+bitexact`` container mux at EVERY
    point (per-shot, deband/source-match, concat, xfade, grade/master, caption
    remux) — the substrate for a cross-run frame_md5 assertion. Off leaves every
    command byte-identical to today.

    Per-clip ``audio.mute`` is realised at SOURCE (no ``0:a`` map) when EVERY shot
    of the request is muted — the music-only reel — and post-master otherwise; the
    decision is taken once here, see ``_all_shots_muted``.

    Caller owns cleanup of the returned file's parent directory.
    """
    # VC-3: resolve ONE master cadence up front and thread it via the settings
    # (no renderer-signature churn — every filter helper reads settings.render_fps).
    # Default compile with no declared source rates resolves to render_fps, so the
    # whole downstream filter chain stays byte-identical.
    #
    # ``settings`` is rebound LOCALLY here, so the caller's object still carries the
    # pre-resolution ``render_fps``. That rate is what the master was NOT encoded at
    # whenever the selector moved it, so ``chosen_fps`` is RETURNED rather than left
    # to be re-derived: ``post_mux_verify``'s ``framerate_coherence`` compares the
    # probed ``avg_frame_rate`` against an expected fps within ±0.5, and handing it
    # the caller's stale 30 for a master encoded at 24 reports ``cfr_ok=False`` on a
    # perfectly coherent file — a permanent false red on the one advisory field the
    # mastering module tells consumers to read directly to act on cadence.
    # Per-render observability channel: clear it BEFORE the render so an absent
    # master pass reports None rather than inheriting a stale mode.
    reset_master_normalization_type()
    chosen_fps = _resolve_master_cadence(shots, settings)
    if chosen_fps != settings.render_fps:
        settings = settings.model_copy(update={"render_fps": chosen_fps})
    # WHERE ``audio.mute`` gets realised is a property of the REQUEST, not of a
    # shot, so it is decided ONCE here and threaded down (the same way as
    # ``deterministic``/``freeze_pad_cap``). ``_postclip_filter`` re-derives it from
    # the same shots list, so the two halves cannot disagree. See ``_all_shots_muted``.
    all_muted = _all_shots_muted(shots)
    work_root = Path(settings.work_dir) / f"job-{uuid.uuid7().hex}"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        fractions = _clip_seek_fractions(shots)
        # One indexed segment file per shot; order is fixed up-front so shots can
        # render concurrently and concat_shots (order-preserving, stream-copy)
        # joins them deterministically regardless of which finishes first.
        segment_paths = [work_root / f"seg-{i:04d}{_SEGMENT_EXT}" for i in range(len(shots))]
        # Fan the per-shot renders out concurrently — shots are independent. A
        # local gate bounds how many fetch+encode pipelines run at once (mirrors
        # max_concurrent_ffmpeg_subprocesses) so one big compile can't launch
        # every shot's download+ffmpeg simultaneously; the module-level ffmpeg
        # semaphore still caps total ffmpeg subprocesses across concurrent jobs.
        gate = asyncio.Semaphore(max(1, settings.max_concurrent_ffmpeg_subprocesses))
        # One download per remote URL for the whole job. Constructed HERE, not at
        # module scope: it hands out paths inside ``work_root``, which the
        # ``finally`` below deletes — and this service runs multi-replica off a
        # durable claim queue, so nothing may assume a single long-lived process.
        # Lives for exactly this job; ``aclose`` drains it before the rmtree.
        source_cache = SourceFetchCache(work_root)
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as http:

            async def _render(i: int, shot: Shot) -> tuple[float, Path | None]:
                # Content-addressed cache in front of render_shot: an unchanged
                # shot's segment is reused byte-for-byte across separate compiles
                # (the re-compile loop), so only shots that actually changed
                # re-render. See _render_shot_cached for the guarded fast-path.
                return await _render_shot_cached(
                    shot,
                    segment_paths[i],
                    settings,
                    work_root,
                    http,
                    clip_seek_fraction=fractions[i],
                    gate=gate,
                    deterministic=deterministic,
                    freeze_pad_cap=freeze_pad_cap,
                    all_shots_muted=all_muted,
                    source_cache=source_cache,
                )

            tasks = [asyncio.ensure_future(_render(i, shot)) for i, shot in enumerate(shots)]
            try:
                # THE encode stage, and the reason this module is instrumented at
                # all: shots fan out concurrently here, so this measures the
                # wall-clock of the whole ffmpeg fan-out, not the sum of its parts.
                # Compare against `compile_shots` and the job total to get the
                # fraction of a job that more CPU (or a hardware encoder) could
                # ever address.
                with metrics.stage(metrics.Stage.RENDER_SHOTS):
                    render_results = await asyncio.gather(*tasks)
                durations = [d for d, _audio_path in render_results]
                _extend_narration_audio_sink(narration_audio_sink, render_results)
            except BaseException:
                # Cancel any still-running siblings and drain them before the
                # work_root is torn down, so no orphaned render keeps writing to
                # a deleted scratch dir (avoids "exception never retrieved" noise).
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # The shared downloads are shielded from the per-shot cancels
                # above (that is the whole point of the shield), so they must be
                # cancelled explicitly or they keep writing into the doomed
                # work_root.
                await source_cache.aclose()
                raise
            # Fetch the continuous bed assets (best-effort) before the http
            # client closes; the overlay ffmpeg pass below consumes local paths.
            music_asset: tuple[Path, MusicBed] | None = None
            music_bed_unfetchable = False
            if music is not None:
                music_path = await _fetch_asset(
                    music.url, work_root / "bed-music", http, source_cache
                )
                if music_path is not None:
                    music_asset = (music_path, music)
                else:
                    # LOUD, not silent. Every other degradation in this module
                    # still delivers the thing the caller asked for (a text card
                    # instead of a clip, a dropped overlay); a lost music bed on a
                    # music-driven product delivers a SILENT film that looks like a
                    # success. Flagged in the job result via ``music_bed`` below —
                    # the same channel the bed's own report rides — because that is
                    # the only per-job note channel this service has.
                    music_bed_unfetchable = True
                    log.error("compile_shots.music_bed_unfetchable", url=music.url[:200])
            ambient_assets: list[tuple[Path, AmbientCue]] = []
            ambient_requested_count = len(ambient or [])
            for k, cue in enumerate(ambient or []):
                amb_path = await _fetch_asset(
                    cue.url, work_root / f"bed-amb-{k:02d}", http, source_cache
                )
                if amb_path is not None:
                    ambient_assets.append((amb_path, cue))
                else:
                    # Unlike music (flagged above via music_bed_unfetchable), an
                    # unfetchable ambient cue used to drop out SILENTLY — no log, no
                    # report field, indistinguishable from "no ambient was ever
                    # asked for". On an ambient-only compile (no music) that means
                    # a request whose entire bed failed to fetch shipped a
                    # `completed` master with `music_bed: null`, exactly the same
                    # silent-loss shape music had before ``_flag_unapplied_music_bed``.
                    # Counted, not raised: one lost cue among several is a partial
                    # (and audible) degradation, not grounds to fail the whole
                    # compile — the flag below makes it visible either way.
                    log.error("compile_shots.ambient_cue_unfetchable", url=cue.url[:200])
            ambient_lost_count = ambient_requested_count - len(ambient_assets)
            # Every fetch for this job is done; nothing may still be in flight
            # while the http client below closes or the work_root is removed.
            await source_cache.aclose()
        total_seconds = float(sum(durations))
        # Picture master (VC-2 + 3a): decide banding risk AND exposure drift
        # concurrently, then re-encode each tripped segment ONCE with a combined
        # filter, BEFORE the stream-copy concat (post-concat the video is only
        # ever -c copy'd). Clean shots stay single-pass and byte-identical; a
        # shot tripping both conditions used to cost two sequential decode/encode
        # generations, now costs one. MUST run here — post-render (after
        # _shot_cache_put in the render gather), pre-concat: correcting inside
        # _render_shot_cached would corrupt the content-addressed cache.
        with metrics.stage(metrics.Stage.DEBAND), metrics.stage(metrics.Stage.SOURCE_MATCH):
            await _apply_picture_master(segment_paths, settings, finish, deterministic)
        out_path = work_root / f"compiled{_SEGMENT_EXT}"
        # Per-clip transitions (WS-2): a transition on ANY shot (past the first,
        # whose transition is meaningless) switches the join from the fast
        # stream-copy concat to a re-encoding xfade/acrossfade chain. The chain
        # OVERLAPS adjacent shots, so the delivered total shrinks by the summed
        # overlaps — _join_segments returns that post-overlap total, which then
        # drives the audio-bed length in _finalize_audio. With no transitions the
        # concat fast path is byte-identical to before.
        # A shot requesting gain/mute needs the POST-master time-ranged pass, which
        # is keyed to each shot's absolute bus offset. Only then does _join_segments
        # probe the segment lengths to compute those offsets (the plain path stays
        # probe-free otherwise). Fades don't need offsets — they ride pre-master.
        # Captions (VC-6) also need the per-shot bus offsets to map shot-local
        # cue times onto the film timeline, so OR them into the offset gate.
        #
        # A MUTE only needs offsets when it is realised post-master. On an
        # all-muted request every segment was already silenced at source and
        # ``_postclip_filter`` skips every mute clause, so the offsets would drive a
        # filter guaranteed to be "" — N ffprobe calls for nothing. GAIN is never
        # realised at source, so it always counts. This mirrors the emit condition
        # in ``_postclip_filter`` exactly; captions / ``emit_shot_timeline`` keep
        # their own independent claim on the offsets.
        need_offsets = (
            any(
                shot.audio is not None
                and ((shot.audio.mute and not all_muted) or shot.audio.gain_db != 0.0)
                for shot in shots
            )
            or bool(captions)
            or emit_shot_timeline
        )
        with metrics.stage(metrics.Stage.JOIN):
            total_seconds, shot_offsets = await _join_segments(
                shots,
                segment_paths,
                total_seconds,
                out_path,
                settings,
                need_offsets=need_offsets,
                deterministic=deterministic,
            )
        # Authoritative per-shot timeline (opt-in): zip each shot to its probed bus
        # offset. Built only when requested — the default path stays probe-free and
        # this stays None.
        shot_timeline = _build_shot_timeline(shots, shot_offsets) if emit_shot_timeline else None
        # Whole-film grade (VC-5): resolve the grade chain over the JOINED film.
        # An absent/identity grade yields "" and the finalize path stays -c copy.
        grade_vf = await _resolve_grade_vf(finish, out_path)
        # Captions (VC-6): map shot-local cues onto the bus and render a single
        # libass ASS pass. The ass filter is APPENDED after the grade so both
        # ride the SAME whole-film re-encode (no second encode). Empty when there
        # are no surviving cues, so the -c copy fast path stays byte-identical.
        # Overlay mode does NOT burn — the cues are muxed as a soft mov_text
        # track after finalize instead (see below), so the picture pass stays
        # burn-free (byte-identical when there is also no grade).
        captions_vf = _resolve_captions_vf(
            captions,
            shots,
            shot_offsets,
            settings,
            work_root,
            captions_render,
            caption_mood,
            captions_karaoke,
            caption_look,
        )
        combined_vf = ",".join(x for x in (grade_vf, captions_vf) if x)
        with metrics.stage(metrics.Stage.FINALIZE_AUDIO):
            final_path, music_bed_report = await _finalize_audio(
                out_path,
                work_root,
                total_seconds,
                music_asset,
                ambient_assets,
                settings,
                shots=shots,
                shot_offsets=shot_offsets,
                grade_vf=combined_vf,
                deterministic=deterministic,
            )
        music_bed_report = _flag_unapplied_music_bed(
            music_bed_report,
            requested=music is not None,
            unfetchable=music_bed_unfetchable,
            url=music.url if music is not None else None,
            ambient_requested=ambient_requested_count,
            ambient_lost=ambient_lost_count,
        )
        # Overlay mode (self-gated): mux the same cues as a soft mov_text track
        # after finalize. A no-op returning `final_path` unchanged for every other
        # mode, so the burned_in/none paths stay byte-identical.
        final_path = await _mux_soft_captions(
            final_path,
            captions,
            shots,
            shot_offsets,
            settings,
            work_root,
            captions_render,
            deterministic,
        )
        return final_path, total_seconds, music_bed_report, shot_timeline, chosen_fps
    except BaseException:
        # BaseException (not Exception) so asyncio.CancelledError also tears the
        # scratch dir down: the work-timeout (asyncio.wait_for) and the SIGTERM
        # drain both CANCEL this coroutine, and the caller's _cleanup_job_dir is a
        # no-op when out_path is None — nothing else reclaims work_root. The
        # cancellation still propagates unchanged (re-raise below).
        shutil.rmtree(work_root, ignore_errors=True)
        raise


async def _probe_audio_params(path: Path) -> tuple[int, int] | None:
    """Return ``(sample_rate, channels)`` for an audio file with safe fallbacks.

    Used only by the gap/re-encode concat path to synthesise inter-clip
    silence that matches the narration's sample rate + layout (the concat
    filter requires all inputs to agree). Kokoro WAVs are 24 kHz mono, which
    is also the fallback when ffprobe can't read the stream.
    """
    out = await _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        streams = json.loads(out).get("streams") or []
        if not streams:
            # ffprobe exits 0 with an EMPTY stream list when the file carries no audio
            # stream at all. That is a FACT about the clip, not a measurement failure, and
            # it must not be folded into the 24 kHz/mono fallback below: doing so made an
            # audio-less clip probe byte-identical to real Kokoro narration, so
            # ``_assert_uniform_audio`` — the only guard before the stream-copy join —
            # passed it and the audiobook shipped ``completed`` with a corrupt join /
            # gap silence sized against a clip that has no audio.
            return None
        stream = streams[0]
        # An UNREADABLE field is not 24 kHz/mono. The `or 24000` / `or 1` defaults
        # here used to fabricate exactly the nominal Kokoro shape whenever ffprobe
        # reported something unparseable ("N/A", a missing key, truncated JSON),
        # which made an UNMEASURED clip byte-identical to real narration — so
        # `_assert_uniform_audio` (the ONLY guard before the stream-copy join)
        # passed it and a 48 kHz stereo segment was concat-copied into a 24 kHz
        # master. The mirror case was worse: when CLIP 0 was the one that failed to
        # probe, every genuinely-uniform clip was rejected as "heterogeneous".
        #
        # Both callers run AFTER `_assert_uniform_audio`, so raising fails the
        # compile loudly with a named reason instead of shipping a corrupt join.
        # Contrast the `return None` above, which reports a FACT (this clip has no
        # audio stream); this reports the ABSENCE of a measurement. The two must
        # not share a representation.
        raw_rate = stream.get("sample_rate")
        raw_channels = stream.get("channels")
        sample_rate = int(raw_rate) if raw_rate not in (None, "") else 0
        channels = int(raw_channels) if raw_channels not in (None, "") else 0
        if sample_rate <= 0 or channels <= 0:
            raise FfmpegError(
                f"audio params unreadable for {path.name}: "
                f"sample_rate={raw_rate!r}, channels={raw_channels!r}"
            )
        return sample_rate, channels
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise FfmpegError(f"audio params unreadable for {path.name}: {exc}") from exc


async def _assert_uniform_audio(segment_paths: list[Path]) -> None:
    """Guard both concat paths against heterogeneous clips.

    Both the stream-copy demuxer (``_concat_audio_copy``) and the concat filter
    (``_concat_audio_filter``, which sizes its silence to the FIRST clip only)
    assume every clip agrees on sample rate + channel count. A mismatch would
    silently corrupt the join (copy path) or misalign the gap silence (filter
    path). Probe each clip and raise on the first that diverges from clip 0.
    No-op for uniform input (the prod case) — never changes the emitted command.
    """
    if len(segment_paths) < 2:
        return
    params0 = await _probe_audio_params(segment_paths[0])
    if params0 is None:
        raise FfmpegError("clip 0 carries no audio stream")
    for i, seg in enumerate(segment_paths[1:], start=1):
        params = await _probe_audio_params(seg)
        # ``None`` = the clip has NO audio stream. It cannot "match" clip 0 and must be
        # named as its own condition rather than reported as a rate/channel mismatch.
        if params is None:
            raise FfmpegError(f"clip {i} carries no audio stream")
        if params != params0:
            raise FfmpegError(
                f"heterogeneous audio: clip {i} is {params[0]}Hz/{params[1]}ch, "
                f"expected {params0[0]}Hz/{params0[1]}ch"
            )


def _concat_edge_fade(in_label: str, out_label: str, seconds: float) -> str:
    """One filtergraph node ramping a clip's head up and its tail down by
    ``_SHOT_EDGE_FADE_S``, so a ``concat`` butt-join against ``anullsrc`` silence
    lands on silence at both ends instead of stepping off a non-zero sample.

    A clip too short to hold both ramps (``seconds <= 2 * _SHOT_EDGE_FADE_S``) —
    or one whose duration could not be probed (``0.0``) — gets ``anull`` rather
    than overlapping ramps that would dip the whole clip: the SAME rule
    ``_apply_shot_audio`` applies to a too-short shot. ``anull`` keeps the node
    (and therefore the label) present so the concat inputs stay index-aligned
    whatever the clip lengths are; it is a documented no-op filter.
    """
    fade = _SHOT_EDGE_FADE_S
    if seconds <= 2 * fade:
        return f"{in_label}anull{out_label}"
    return (
        f"{in_label}afade=t=in:st=0:d={fade},"
        f"afade=t=out:st={seconds - fade:.6f}:d={fade}{out_label}"
    )


async def _concat_audio_copy(
    segment_paths: list[Path], out_path: Path, deterministic: bool = False
) -> None:
    """Fast path: stream-copy uniform WAV clips via the ffmpeg concat demuxer.

    Safe because every clip comes from the one TTS backend (same codec /
    sample rate / channels), so no re-encode is needed — mirrors
    ``concat_shots`` for the MP4 case.

    ``deterministic`` inserts the global ``-fflags +bitexact`` before ``-i`` so
    the stream-copy mux strips the container ``creation_time``/``encoder`` tag —
    the only run-to-run noise on this path. Off → command byte-identical to prod.
    """
    await _assert_uniform_audio(segment_paths)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file {shlex.quote(str(p.resolve()))}" for p in segment_paths) + "\n"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        *_bitexact_input_args(deterministic),
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    try:
        await _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


async def _concat_audio_filter(
    segment_paths: list[Path],
    out_path: Path,
    output_format: str,
    gap_ms: int,
    deterministic: bool = False,
) -> None:
    """Re-encode path: concat via filtergraph, optionally inserting silence.

    Taken when the output is MP3 or an inter-clip ``gap_ms`` is requested.
    Silence is an ``anullsrc`` lavfi input sized to the first clip's sample
    rate + channel layout so the concat filter's same-format precondition
    holds.

    ``deterministic`` inserts the global ``-fflags +bitexact`` before the first
    ``-i`` (strips the container tag) AND routes the output codec through
    ``_audiobook_codec_args`` so libmp3lame drops its LAME version stamp. Off →
    command byte-identical to prod.

    ⚠️ **Every clip's edges are ramped when a gap is inserted.** ``concat`` BUTT-JOINS
    its inputs, so a clip whose last sample is non-zero meeting ``anullsrc``'s
    digital silence is a step discontinuity — a click, at every single chapter
    boundary of the book. This is the same defect ``_apply_shot_audio``'s
    ``_SHOT_EDGE_FADE_S`` removes on the VIDEO path at every segment join, and this
    path had no equivalent: a whole delivery path was missing the fix. Same 8 ms
    ramp, below the ~20 ms at which a level ramp is audible as a fade. Applied only
    on the gap path (``gap_ms > 0``): with no gap the clips meet each other rather
    than silence, which is the join the ``wav``/stream-copy path also makes, and
    the command stays byte-identical to prod there.
    """
    await _assert_uniform_audio(segment_paths)
    inputs: list[str] = []
    labels: list[str] = []
    fades: list[str] = []
    idx = 0
    sample_rate, channels, layout, gap_seconds = 24000, 1, "mono", 0.0
    if gap_ms > 0:
        # ⛔ NO FALLBACK. This used to read ``or (24000, 1)`` — the nominal Kokoro
        # shape — so an UNMEASURED clip 0 silently sized every gap at 24 kHz mono.
        # Against real 48 kHz stereo clips that is a concat-filter format mismatch
        # or a mis-timed gap, from an assumption nothing in the output records.
        # ``_assert_uniform_audio`` above already rejected an audio-less clip 0 and
        # ``_probe_audio_params`` RAISES on an unreadable one, so ``None`` here is
        # unreachable by contract — which is exactly why it must fail loudly rather
        # than be papered over by a constant that happens to look like narration.
        probed = await _probe_audio_params(segment_paths[0])
        if probed is None:
            raise FfmpegError(
                f"gap silence cannot be sized: {segment_paths[0].name} reports no audio stream"
            )
        sample_rate, channels = probed
        layout = "mono" if channels == 1 else "stereo"
        gap_seconds = gap_ms / 1000.0
    for i, seg in enumerate(segment_paths):
        inputs += ["-i", str(seg)]
        if gap_ms > 0:
            # One de-click ramp per clip edge, keyed to that clip's OWN probed
            # length so the fade-out lands on its final samples. The label is built
            # from ``idx`` — the real ffmpeg input index — which is NOT the clip
            # index once gap inputs are interleaved between the clips.
            out_label = f"[e{i}]"
            fades.append(_concat_edge_fade(f"[{idx}:a]", out_label, await _ffprobe_duration(seg)))
            labels.append(out_label)
        else:
            labels.append(f"[{idx}:a]")
        idx += 1
        if gap_ms > 0 and i < len(segment_paths) - 1:
            inputs += [
                "-f",
                "lavfi",
                "-t",
                f"{gap_seconds}",
                "-i",
                f"anullsrc=r={sample_rate}:cl={layout}",
            ]
            labels.append(f"[{idx}:a]")
            idx += 1
    concat_node = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
    filtergraph = ";".join([*fades, concat_node])
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        *inputs,
        "-filter_complex",
        filtergraph,
        "-map",
        "[out]",
        *_audiobook_codec_args(output_format, deterministic),
        str(out_path),
    ]
    await _run(cmd)


def _escape_ffmetadata(value: str) -> str:
    """Escape the ffmetadata special characters (``= ; # \\`` + newline)."""
    out = value.replace("\\", "\\\\")
    for ch in ("=", ";", "#"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("\n", " ").replace("\r", " ")


def _write_chapter_metadata(chapters: list[Chapter], total_seconds: float, path: Path) -> None:
    """Write an ffmetadata file: one ``[CHAPTER]`` block per chapter.

    ``TIMEBASE=1/1000`` so START/END are milliseconds. START is the chapter's
    cumulative ``start_ms``; END is the NEXT chapter's start, and the last
    chapter runs to ``total_seconds``. A zero/negative-length span is nudged to
    ≥1 ms (ffmpeg rejects an empty chapter). Chapters are read in the order given.
    """
    total_ms = max(round(total_seconds * 1000), 0)
    lines = [";FFMETADATA1"]
    n = len(chapters)
    for i, ch in enumerate(chapters):
        start = max(ch.start_ms, 0)
        end = chapters[i + 1].start_ms if i + 1 < n else total_ms
        if end <= start:
            end = start + 1
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start}",
            f"END={end}",
            f"title={_escape_ffmetadata(ch.title)}",
        ]
    path.write_text("\n".join(lines) + "\n")


async def _remux_m4b_chapters(
    pre_path: Path, out_path: Path, chapters: list[Chapter], deterministic: bool = False
) -> None:
    """Stream-copy ``pre_path`` (an already-encoded m4b) into ``out_path`` with
    ffmetadata chapter markers. ``-c copy`` → NO audio re-encode, so the
    delivered samples are identical to ``pre_path``; only the container's chapter
    atoms change."""
    total_seconds = await _ffprobe_duration(pre_path)
    meta_path = out_path.with_suffix(".chapters.txt")
    _write_chapter_metadata(chapters, total_seconds, meta_path)
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(pre_path),
        "-i",
        str(meta_path),
        "-map_metadata",
        "1",
        "-c",
        "copy",
        "-f",
        "ipod",
        str(out_path),
    ]
    try:
        await _run(cmd)
    finally:
        meta_path.unlink(missing_ok=True)


def _finite_float(value: object) -> float | None:
    """Coerce a loudnorm JSON value to a FINITE float, else None.

    loudnorm reports a silent input's ``input_i``/``target_offset`` as the
    non-empty strings ``"-inf"``/``"inf"``; those are real strings but not usable
    as ``measured_*`` linear-normalisation params (ffmpeg rejects them), so they
    must be treated as absent."""
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        parsed = float(value)
    else:
        return None
    return parsed if math.isfinite(parsed) else None


async def _acx_program_master(
    raw_path: Path,
    out_path: Path,
    output_format: str,
    *,
    lufs_target: float,
    tp: float,
    lra: float,
    deterministic: bool = False,
) -> None:
    """Two-pass ACX program-loudness master of a stitched book → ``out_path``.

    Pass 1 measures the raw stitch (``loudnorm print_format=json``); pass 2
    applies a LINEAR loudnorm to the ACX target under the true-peak ceiling and
    encodes to the delivery codec at 44.1 kHz. If the measure pass fails, pass 2
    falls back to a single-pass dynamic loudnorm (still lands in-band). This is
    the A12 seam: exactly ONE ``loudnorm``, on the concat-audio path only — it is
    NEVER stacked on ``_master_filter`` (the video bus, a different code path).

    Both passes hand loudnorm a bus a WHOLE number of 100 ms frames long
    (:func:`_loudnorm_frame_align`) and pass 2 bounds the output back to the raw
    stitch's own length, so the book is delivered exactly as long as it went in."""
    # The raw stitch's own length arms the alignment. Pass 2 runs DYNAMIC whenever
    # the measure pass yields nothing usable (the fallback below) or ffmpeg refuses
    # ``linear=true`` — and dynamic loudnorm drops the missing part of a partial
    # last frame, a hole in the book's tail. ``None`` (unprobeable) fails open to
    # the unaligned chain — the audio this path produced before.
    raw_seconds = await _audio_stream_seconds(raw_path)
    align = _loudnorm_frame_align(raw_seconds)
    measure_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(raw_path),
        "-af",
        # Same alignment as pass 2: pass 1 must measure the bus pass 2 normalises.
        f"{align}loudnorm=I={_ffnum(lufs_target)}:TP={_ffnum(tp)}:LRA={_ffnum(lra)}"
        ":print_format=json",
        "-f",
        "null",
        "-",
    ]
    measured: dict[str, object] = {}
    try:
        from src.services import loudness

        stderr = await _run(measure_cmd, capture_stderr=True)
        measured = loudness._extract_json_block(stderr) or {}
    except Exception as exc:  # noqa: BLE001 — degrade to single-pass dynamic, never fail
        # Same channel and the same escalation as the video bus (see
        # ``_record_master_measure_failure``): the ACX master is the ONLY loudness
        # stage on the audiobook path, so a swallowed measure pass here means a
        # paid book was mastered by the dynamic fallback with nothing on the record.
        _record_master_measure_failure("acx_ffmpeg_failed", error=str(exc), path=str(raw_path))
    ln = f"loudnorm=I={_ffnum(lufs_target)}:TP={_ffnum(tp)}:LRA={_ffnum(lra)}"
    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    # Only take the linear two-pass path when every measured value is a FINITE
    # number. A fully-silent stitch (poison/failed-TTS) makes ffmpeg report
    # ``input_i="-inf"`` / ``target_offset="inf"`` — non-empty strings that would
    # slip a bare ``not in (None, "")`` guard and build an out-of-range
    # ``measured_I=-inf`` filter that ffmpeg rejects (0-byte output → job crash).
    # Falling through to the single-pass dynamic loudnorm degrades safely instead.
    if all(_finite_float(measured.get(k)) is not None for k in keys):
        ln += (
            f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true"
        )
    cmd = [
        "ffmpeg",
        "-y",
        *_bitexact_input_args(deterministic),
        "-i",
        str(raw_path),
        "-af",
        f"{align}{ln}",
        # Take the alignment pad back off: the book ends where the raw stitch ends.
        # An OUTPUT ``-t``, deliberately NOT a trailing ``atrim`` (the film
        # master's no-bed shape): a filter node downstream of ``loudnorm`` is what
        # breaks format negotiation on this path — see the ``-resampler`` note
        # below. Audio-only output, so ``-t`` bounds nothing else.
        *(["-t", _ffdur(raw_seconds)] if raw_seconds is not None else []),
        "-ar",
        "44100",
        # ⚠️ THE RESAMPLER, NOT JUST THE RATE. `loudnorm` emits **192 kHz**
        # regardless of its input rate (documented at `_master_filter`;
        # `audio_premaster` hits the same gotcha), so `-ar 44100` above is a
        # 192000 -> 44100 conversion — a NON-INTEGER 147:640 ratio — and it was
        # running on swr's default cubic interpolator, on the one deliverable that
        # is nothing but voice. The FILM master guards its own version of this with
        # an `aformat=sample_rates=48000` node after loudnorm; this path had no such
        # node at all. soxr at precision=28 puts the resampling artifacts below the
        # noise floor.
        #
        # ⚠️ THESE ARE OUTPUT OPTIONS ON PURPOSE — an `aresample=44100:resampler=soxr`
        # node in the `-af` chain does NOT work here and fails the render outright:
        # verified on real ffmpeg (4.4, libsoxr enabled), an explicit `aresample`
        # downstream of `loudnorm` cannot negotiate a channel layout with the wav
        # muxer and dies with "Cannot select channel layout for the link between
        # filters Parsed_aresample_1 and format_out_0_0" -> "Conversion failed!", on
        # BOTH mono and stereo input. As output options they configure the
        # auto-inserted resampler instead, which negotiates fine and — verified —
        # preserves the channel count (a mono ACX stitch stays mono; forcing a
        # layout to make the filter node work would have upmixed it).
        "-resampler",
        "soxr",
        "-precision",
        "28",
        # 4e: this same swr instance also does loudnorm's 192 kHz FLOAT bus ->
        # pcm_s16le for the "wav" output_format (`_AUDIOBOOK_OUTPUT_CODECS`) —
        # swr's default `dither_method` is `none`, so that float->int16 step is
        # plain truncation on the one deliverable that is nothing but voice.
        # Bench (wave 8, real ffmpeg): a -90 dBFS 1 kHz tone at 192 kHz float,
        # resampled to 44.1 kHz/s16 with these exact args, measures `astats`
        # Flat factor 23.5 undithered (long runs of identical quantized
        # samples — the audible staircase) vs 0.016 with `triangular_hp` (the
        # dither randomizes the quantization error instead of correlating it
        # with the signal). Harmless for "mp3"/"m4b" (lossy encoders quantize
        # their own way downstream) but costs nothing to apply unconditionally.
        "-dither_method",
        "triangular_hp",
        *_audiobook_codec_args(output_format, deterministic),
        str(out_path),
    ]
    await _run(cmd)


async def concat_audio(
    clips: list[AudioClip],
    settings: Settings,
    output_format: str = "wav",
    gap_ms: int = 0,
    deterministic: bool = False,
    *,
    acx_master: bool = False,
    acx_lufs_target: float | None = None,
    acx_tp: float | None = None,
    acx_lra: float | None = None,
    chapters: list[Chapter] | None = None,
) -> tuple[Path, float, dict[str, float | bool | None] | None]:
    """Fetch + stitch ``clips`` into a single audio file. Returns (path, seconds,
    acx_report).

    WAV with no gaps takes the fast stream-copy demuxer path; MP3/m4b output or a
    non-zero ``gap_ms`` re-encodes through the concat filter. The output's
    duration is always re-probed (never trusted from the per-clip ``seconds``).
    Caller owns cleanup of the returned file's parent directory.

    ``acx_master`` (opt-in) applies the compile-tail ACX program master (A12
    seam) to the combined book and returns an ``acx_report`` verdict; ``chapters``
    + ``output_format="m4b"`` package a chaptered audiobook container. With
    ``acx_master`` off, ``output_format`` in {wav, mp3}, and no ``chapters``, the
    ffmpeg command is byte-identical to prod (``acx_report`` is None).

    ``deterministic`` (opt-in, default off) makes the stitch bit-reproducible:
    threaded into every branch so the container tag (and, on MP3, the LAME
    version stamp) are stripped. Off → command byte-identical to prod.
    """
    if output_format not in _AUDIOBOOK_OUTPUT_CODECS:
        raise FfmpegError(f"unsupported audio output_format: {output_format!r}")
    work_root = Path(settings.work_dir) / f"audio-{uuid.uuid7().hex}"
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        fetched: list[Path] = []
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as http:
            for i, clip in enumerate(clips):
                dest = work_root / f"clip-{i:04d}"
                fetched.append(await _fetch(clip.audio_url, dest, http))
        out_path = work_root / f"audiobook.{output_format}"
        # m4b + chapters needs a post-encode ``-c copy`` remux to inject the
        # chapter atoms; encode into a pre-mux file first in that case.
        need_remux = output_format == "m4b" and bool(chapters)
        encoded = work_root / f"premux.{output_format}" if need_remux else out_path

        if acx_master:
            # ALWAYS re-encode the raw stitch to PCM via the filter path — the
            # inputs may be AAC/m4b (the combined book's per-chapter files), and a
            # ``-c copy`` demux of AAC into a .wav container is malformed (silent
            # ~30% truncation, rc=0). The filter decodes every codec to clean PCM
            # for the master. gap_ms=0 simply inserts no silence.
            raw = work_root / "raw.wav"
            await _concat_audio_filter(fetched, raw, "wav", gap_ms, deterministic)
            await _acx_program_master(
                raw,
                encoded,
                output_format,
                lufs_target=acx_lufs_target
                if acx_lufs_target is not None
                else settings.render_audiobook_lufs_target,
                tp=acx_tp if acx_tp is not None else settings.render_audiobook_tp,
                lra=acx_lra if acx_lra is not None else settings.render_audiobook_lra,
                deterministic=deterministic,
            )
            raw.unlink(missing_ok=True)
        elif output_format == "wav" and gap_ms == 0:
            await _concat_audio_copy(fetched, encoded, deterministic)
        else:
            await _concat_audio_filter(fetched, encoded, output_format, gap_ms, deterministic)

        if need_remux and chapters is not None:
            await _remux_m4b_chapters(encoded, out_path, chapters, deterministic)
            encoded.unlink(missing_ok=True)

        acx_report: dict[str, float | bool | None] | None = None
        if acx_master:
            from src.services import loudness

            measure = await loudness.acx_measure(str(out_path))
            acx_report = loudness.evaluate_acx(
                measure,
                lufs_min=settings.render_audiobook_lufs_min,
                lufs_max=settings.render_audiobook_lufs_max,
                tp_max=settings.render_audiobook_tp,
                noise_floor_max=settings.render_audiobook_noise_floor_db,
            )
        return out_path, await _ffprobe_duration(out_path), acx_report
    except BaseException:
        # BaseException (not Exception) so asyncio.CancelledError also tears the
        # scratch dir down on a work-timeout / SIGTERM-drain cancel; the
        # cancellation still propagates unchanged (re-raise below).
        shutil.rmtree(work_root, ignore_errors=True)
        raise
