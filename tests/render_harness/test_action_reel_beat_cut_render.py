"""Real-ffmpeg proof that the ``action_reel`` product actually delivers: N short
beat-synced shots cut out of ONE long source clip, under a music bed.

The product shape (new): several ``Shot``s share ONE ``video_url`` and each names
an EXPLICIT source window via ``clip_lead_seconds`` + ``clip_out_seconds``, at a
per-shot ``speed``, with its OWN audio muted (``ShotAudio(mute=True)``) so the
only thing heard is a request-level ``MusicBed``. Nothing about that combination
has ever been confirmed against delivered pixels/samples — only against
filtergraph strings — so this module drives ``renderer.compile_shots`` end to end
and interrogates the delivered mp4 with ffprobe/ffmpeg.

WHICH DEFECT EACH ASSERTION GUARDS — do not weaken any of these without
understanding what stops being covered:

* ``test_action_reel_beat_cuts_land_on_the_intended_windows``
  - **per-shot length** (``_assert_beat_lengths``) guards THE defect this module
    was written for: the own-audio branch of ``render_shot`` sets ``trim=False``
    and emits NO output ``-t``, so a shot's length is decided ENTIRELY by the
    INPUT-side ``-t`` that ``_clip_seek_args`` emits for ``clip_out_seconds``.
    Lose that input ``-t`` (or emit it after ``-i``, or scale it by ``speed``)
    and every beat renders its source to EOF instead of its window. The defect
    ONLY manifests when the source clip HAS an audio stream — a silent clip takes
    the ``anullsrc`` branch, where ``trim`` stays True and an output ``-t`` masks
    the bug — which is why the source synthesised here always carries audio.
  - **total duration** guards the same defect at film level, and specifically
    guards against a total that is "near enough" while individual beats are
    wrong: it is asserted to ~80 ms AND asserted to be nowhere near the
    8x-full-source length a to-EOF regression would produce.
  - **cut positions** guard the join: correct beat LENGTHS with a wrong
    concat/xfade would still put the cuts in the wrong place on the film bus.
  - **frame rate** guards the VC-3 cadence seam: the source runs at 24000/1001
    and the render is pinned to an integer ``render_fps``, so a lost ``fps=``
    fragment delivers a 23.976 (or VFR) master that silently drifts against every
    beat time computed above.
  - **window identity** (luma probe, synthetic source only) guards the one thing
    a duration/position assertion CANNOT see: that each beat shows the source
    span it asked for. Eight beats of the RIGHT LENGTH cut from the WRONG (e.g.
    all from t=0, i.e. a dropped ``-ss``) pass every timing assertion above.

* ``test_action_reel_audio_is_the_bed_not_the_source_clip``
  - guards that ``ShotAudio(mute=True)`` removes the SOURCE clip's own audio from
    the delivered bus while the ``MusicBed`` survives. Both halves matter: a bus
    that is silent everywhere trivially satisfies "no drone audio", and a bus
    that carries the drone under the bed trivially satisfies "has audio".
  - the "no source audio" half is made NON-tautological by a CONTROL render
    (``_render_control_segment``) that renders the same window UNMUTED and with
    NO bed, and proves the same probe reads the source tone loudly there. Without
    that control, "the 400 Hz band is quiet" could be true because the probe
    cannot see the source tone at all.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_clip_out_point.py`` / ``test_clip_seek_duration.py``.

REAL FOOTAGE. The demo corpus (10 x 3840x2160 HEVC drone clips, 24000/1001 fps,
6-18 s, each with a PCM audio stream) is not in the repo and will never exist on
a CI runner, so the DEFAULT source is synthesised by ``_make_reel_source`` to
reproduce the corpus properties that are load-bearing here — the 24000/1001
cadence, a real audio stream, and enough length to cut 8 beats from. Point
``W2V_ACTION_REEL_SOURCE_CLIP`` at a real clip (e.g.
``/Users/georgedekker/Movies/coconut beach watersport videos/C0012.MP4``) to run
the geometry assertions against the real footage instead; the beat layout is
derived from the clip's PROBED duration so any length works, and the two
content-dependent probes (luma window identity, the 400 Hz source tone) are
skipped-per-assertion with an explicit note rather than silently weakened, since
real drone footage carries neither a luma ramp nor a known-frequency tone.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import MusicBed, Shot, ShotAudio
from src.config import get_settings
from src.services import renderer
from tests.render_harness import (
    make_bed_tone,
    probe_band_mean_volume_window,
    probe_duration,
    probe_has_audio,
    probe_mean_volume,
    probe_video_frame_count,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

# Env override: an absolute path to ONE real source clip. Unset (the CI default)
# synthesises the stand-in below.
_SOURCE_ENV = "W2V_ACTION_REEL_SOURCE_CLIP"

# The corpus cadence, verbatim. Load-bearing: it is NOT the render cadence, so
# every window length quantises against a 1001/24000 source grid while the
# delivered film quantises against 1/24 — the exact mismatch a naive
# "duration == window" assertion trips over.
_SRC_FPS_NUM, _SRC_FPS_DEN = 24000, 1001
_SRC_FRAME = _SRC_FPS_DEN / _SRC_FPS_NUM  # 0.0417083s
_SYNTH_SECONDS = 60.0

# Canvas: deliberately small. Frame SIZE is not load-bearing for anything asserted
# here (the corpus is 4K, this is 320x240) and a 4K harness render would cost
# minutes per beat. Cadence, audio presence and length ARE load-bearing and are
# reproduced exactly.
_W, _H = 320, 240
_RENDER_FPS = 24
_FRAME = 1.0 / _RENDER_FPS

# The source's own audio: a low tone, well away from the bed's 8 kHz, so a band
# probe can tell the two apart in the delivered mix. See the module docstring.
_SOURCE_TONE_HZ = 400
_BED_TONE_HZ = 8000
# Cascaded 1-pole lowpasses put the 8 kHz bed ~54 dB down inside the source-tone
# band, so what the low-band probe measures really is the source clip's audio.
_LOW_BAND_HZ, _LOW_BAND_STAGES = 1000.0, 3
_HIGH_BAND_HZ, _HIGH_BAND_STAGES = 6000.0, 2
# A 20 dB band contrast is ~10x in amplitude — far beyond codec/filter leakage,
# far below the ~50 dB separation a correct mix actually produces.
_BAND_CONTRAST_DB = 20.0

# Eight beats, expressed as fractions of the source so the same layout works on a
# 60 s synthetic clip and on a 6-18 s real one. Each beat lives in its own 1/8
# slot, starting 20% into the slot: that leaves a gap of at least 0.35 slot
# between one beat's out-point and the next beat's in-point, which is what makes
# each cut a real jump in source content (and therefore detectable).
_BEATS = 8
_LEAD_FRACTION_OF_SLOT = 0.20
# Window length as a fraction of the slot, and the per-beat speed. Mixed
# deliberately: 1.0s coexist with retimed beats, all inside the 0.5-2.0 band the
# product uses, so the ÷speed arithmetic is exercised in both directions.
_SPAN_FRACTIONS = (0.55, 0.40, 0.60, 0.50, 0.35, 0.40, 0.30, 0.55)
_SPEEDS = (1.0, 1.25, 2.0, 1.0, 0.75, 1.5, 0.5, 2.0)

# Scene-change threshold for cut detection. ffmpeg's ``scene`` value is
# ``min(mafd, |mafd - prev_mafd|)/100`` where mafd is the mean absolute luma
# difference in codewords. On the synthesised ramp: WITHIN a beat consecutive
# frames differ by ~0.18 codewords (score ~0.002), while AT a cut the jump is at
# least ~11 codewords (score ~0.11). 0.03 sits between the two by ~an order of
# magnitude on each side.
_SCENE_THRESHOLD = 0.03
# A cut is a whole frame on the film grid; ±50 ms is a little over one frame at
# 24 fps, which is the finest a cut can be located at all.
_CUT_TOLERANCE_S = 0.05
# Per-beat length: ONE frame period, as tight as the grid allows.
_BEAT_TOLERANCE_S = _FRAME
# Total: the per-beat quantisation errors are ~2 ms each (a window that is an
# exact multiple of the SOURCE frame period lands within 0.1% of an exact
# multiple of the RENDER frame period), so eight of them cannot reach 80 ms.
_TOTAL_TOLERANCE_S = 0.080


# ---------------------------------------------------------------------------
# Source + beat layout
# ---------------------------------------------------------------------------


async def _make_reel_source(path: Path, *, duration: float = _SYNTH_SECONDS) -> Path:
    """Synthesise the CI stand-in for one long drone clip.

    Three properties of the real corpus are reproduced because the assertions
    depend on them:

    1. **24000/1001 fps** — the source grid differs from the render grid.
    2. **An audio stream** — the ``clip_out_seconds`` defect this module guards
       only manifests on the own-audio branch of ``render_shot`` (a silent clip
       is bounded by an output ``-t`` that hides it). The tone is a distinctive
       ``_SOURCE_TONE_HZ`` sine so the audio test can find it (or prove it gone).
    3. **Long enough to cut 8 beats from** — the layout below spreads the beats
       across the whole clip.

    The PICTURE is a linear black→white luma ramp over the whole clip (a fade-in
    spanning the full duration). That is not cosmetic, it is the measuring stick:

    * frame-to-frame it changes by ~0.18 codewords, so nothing INSIDE a beat
      looks like a scene change (a ``testsrc`` pattern would fire constantly);
    * a cut between two beats jumps several seconds of source time, i.e. ~11+
      codewords, which reads unambiguously as a scene change;
    * absolute luma encodes SOURCE TIMECODE, so a single frame's mean luma says
      which part of the source is on screen — the only way to prove a beat shows
      the window it asked for rather than merely lasting the right length.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s={_W}x{_H}:r={_SRC_FPS_NUM}/{_SRC_FPS_DEN}:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={_SOURCE_TONE_HZ}:duration={duration}:sample_rate=48000",
            "-vf",
            f"fade=t=in:st=0:d={duration},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def _resolve_source(tmp_path: Path) -> tuple[Path, bool]:
    """``(clip, is_synthetic)``. Honours ``_SOURCE_ENV`` when set.

    A set-but-missing override is a hard FAILURE, not a skip: it means the runner
    was told to use real footage and could not, which must never read as a pass.
    A real clip too short to hold 8 beats is an explicit SKIP.
    """
    override = os.environ.get(_SOURCE_ENV, "").strip()
    if not override:
        return await _make_reel_source(tmp_path / "reel-source.mp4"), True
    clip = Path(override)
    if not clip.is_file():
        pytest.fail(f"{_SOURCE_ENV}={override!r} does not point at a file")
    src_seconds = await probe_duration(clip)
    # Each beat needs a window of at least a handful of source frames, and the
    # layout gives each beat 1/8 of the clip.
    if src_seconds < _BEATS * 0.5:
        pytest.skip(
            f"{_SOURCE_ENV}={override!r} is {src_seconds:.2f}s — too short to cut "
            f"{_BEATS} beats from"
        )
    return clip, False


def _beat_layout(src_seconds: float) -> list[tuple[float, float, float]]:
    """``[(lead, out, speed), ...]`` — the 8 source windows and their speeds.

    The OUTPUT length of each shot is snapped to a whole number of RENDER frames
    and the speed is then DERIVED from it, which is what the cut planner itself
    does: it quantises each cut boundary to the output grid and reports
    ``speed = (out - lead) / seconds``. Aligning the source SPAN instead — the
    obvious move, and what an earlier draft of this file did — does not survive
    the retime: a span that is a whole number of 1001/24000 source frames divided
    by 1.25 or 0.75 is no longer a whole number of anything, so the renderer, which
    can only emit whole frames, rounds each segment by up to 41.7 ms. Over eight
    shots those roundings are one-sided and sum: the reel came out 131 ms long
    against an 80 ms tolerance, and no renderer change could have fixed it because
    the request was asking for durations that do not exist on the output grid.

    Deriving the speed keeps the 80 ms bound a real statement about the renderer
    rather than about the arithmetic of the fixture.
    """
    slot = src_seconds / _BEATS
    render_frame = 1.0 / _RENDER_FPS
    layout: list[tuple[float, float, float]] = []
    for i in range(_BEATS):
        lead = round(slot * (i + _LEAD_FRACTION_OF_SLOT), 3)
        frames = max(6, round(_SPAN_FRACTIONS[i] * slot / _SRC_FRAME))
        span = round(frames * _SRC_FRAME, 3)
        out = round(lead + span, 3)
        # The span the renderer will actually be handed, after `out` is rounded
        # to the 3dp the request carries — derive the speed from THAT, not from
        # the unrounded span, or the two disagree by up to a millisecond a shot.
        exact_span = out - lead
        out_frames = max(3, round(exact_span / _SPEEDS[i] / render_frame))
        layout.append((lead, out, exact_span / (out_frames * render_frame)))
    return layout


def _reel_shots(source: Path, layout: list[tuple[float, float, float]]) -> list[Shot]:
    """The action_reel request: 8 shots, ONE ``video_url``, muted, retimed.

    ``clip_seek_fraction`` (which ``compile_shots`` derives for shots that SHARE a
    ``video_url``, to pan repeated uses through a recording) is inert here on
    purpose: ``_clip_seek_args`` switches into window mode whenever
    ``clip_out_seconds`` is set and ignores the fraction entirely. If that ever
    stops being true these shots slide off their windows and the luma probe below
    is what catches it.
    """
    return [
        Shot(
            id=f"beat{i}",
            # The delivered on-screen length. Not what bounds the render — the
            # own-audio branch reassigns ``duration`` to the window — but it is
            # what a real caller sends, so send it.
            seconds=round((out - lead) / speed, 3),
            video_url=str(source),
            clip_lead_seconds=lead,
            clip_out_seconds=out,
            speed=speed,
            audio=ShotAudio(mute=True),
        )
        for i, (lead, out, speed) in enumerate(layout)
    ]


def _settings(tmp_path: Path):
    return get_settings().model_copy(
        update={
            "render_width": _W,
            "render_height": _H,
            "render_fps": _RENDER_FPS,
            # Pin the cadence: no shot declares ``target_fps``, so 'auto' already
            # resolves to render_fps, but pinning it makes the frame-rate
            # assertion a statement about the REQUESTED rate rather than about
            # whatever the selector happened to choose.
            "render_cadence_mode": "fixed",
            # The synthetic source is a smooth gradient — exactly what the
            # conditional deband pass exists for. It would fire on every segment
            # and cost a full re-encode each without changing a single timestamp.
            "render_deband": "off",
            "work_dir": str(tmp_path / "work"),
            # The content-addressed shot cache survives across compiles; disable
            # it so one test can never serve another test's segments.
            "shot_cache_max_bytes": 0,
        }
    )


# ---------------------------------------------------------------------------
# Probes (module-local: the shared harness has no scene/rate/luma probe)
# ---------------------------------------------------------------------------


async def _probe_cut_times(path: Path, threshold: float = _SCENE_THRESHOLD) -> list[float]:
    """Film-bus timestamps (seconds) of every detected scene change.

    ``select='gt(scene,T)'`` is ffmpeg's own scene-change detector; ``showinfo``
    then logs one line per SELECTED frame, so the ``pts_time`` values are exactly
    the frames the detector called a cut. The very first frame has no predecessor
    and is never selected, so the returned list holds INTERNAL cuts only — for N
    beats a correct render yields N-1 entries.
    """
    out = await renderer._run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-filter_complex",
            f"[0:v]select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    return [float(m.group(1)) for m in re.finditer(r"pts_time:\s*(\d+\.?\d*)", out)]


async def _probe_avg_frame_rate(path: Path) -> float:
    """The video stream's ``avg_frame_rate`` as a float (0.0 when unparseable)."""
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    num, _, den = out.strip().partition("/")
    try:
        return float(num) / float(den or 1)
    except ValueError, ZeroDivisionError:
        return 0.0


async def _probe_frame_luma(path: Path, at_seconds: float) -> float:
    """Mean luma (YAVG, 0-255) of the single frame at ``at_seconds``.

    ``signalstats`` publishes YAVG as frame metadata and ``metadata=print`` dumps
    it to stdout. One frame only, so this is cheap enough to call per beat.
    """
    out = await renderer._run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{at_seconds:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "signalstats,metadata=print:file=-",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    m = re.search(r"lavfi\.signalstats\.YAVG=(-?\d+\.?\d*)", out)
    assert m is not None, f"signalstats published no YAVG for {path} at {at_seconds:.3f}s"
    return float(m.group(1))


async def _probe_low_band_mean_volume(path: Path, start: float, end: float) -> float:
    """Mean volume (dBFS) over ``[start, end)`` of ONLY the content BELOW
    ``_LOW_BAND_HZ`` — the mirror of the harness's high-band probe.

    Cascaded 1-pole lowpasses put an 8 kHz bed ~54 dB down here, so a level read
    through this filter belongs to the source clip's 400 Hz tone, not to the bed.
    """
    lpf = ",".join([f"lowpass=f={_LOW_BAND_HZ:g}"] * _LOW_BAND_STAGES)
    out = await renderer._run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,{lpf},volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    return float(m.group(1)) if m else float("-inf")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_action_reel_beat_cuts_land_on_the_intended_windows(tmp_path: Path) -> None:
    """8 beats cut from ONE clip: right lengths, right total, right cut points,
    right cadence, and each beat showing the source span it asked for.

    Every number below is measured on the DELIVERED mp4. A regression that lets a
    beat render its source to EOF makes the film ~8x the source length, so it
    fails the total assertion by minutes and the per-beat assertion by seconds —
    this cannot pass on a broken renderer.
    """
    source, is_synthetic = await _resolve_source(tmp_path)
    src_seconds = await probe_duration(source)
    assert await probe_has_audio(source), (
        "the source clip must carry audio — the defect under test only manifests "
        "on render_shot's own-audio branch"
    )
    layout = _beat_layout(src_seconds)
    shots = _reel_shots(source, layout)
    intended = [round((out - lead) / speed, 6) for lead, out, speed in layout]
    intended_total = sum(intended)

    bed = await make_bed_tone(
        tmp_path / "bed.wav", duration=intended_total + 5.0, frequency=_BED_TONE_HZ
    )
    final_path, reported_total, _, _, _ = await renderer.compile_shots(
        shots,
        _settings(tmp_path),
        music=MusicBed(url=str(bed), gain_db=-6.0, fade_in_s=0.5, fade_out_s=0.5),
    )

    # --- 2. Total duration -------------------------------------------------
    delivered_total = await probe_duration(final_path)
    assert delivered_total == pytest.approx(intended_total, abs=_TOTAL_TOLERANCE_S), (
        f"reel is {delivered_total:.3f}s, intended {intended_total:.3f}s — the sum "
        f"of the 8 windows (÷speed) is not what was delivered"
    )
    # The to-EOF regression fence. Eight shots each running to the end of the
    # source would deliver roughly this instead; keep the two claims separate so
    # a future loosening of the tolerance above cannot silently admit it.
    to_eof_total = sum((src_seconds - lead) / speed for lead, _out, speed in layout)
    assert delivered_total < to_eof_total * 0.5, (
        f"reel is {delivered_total:.3f}s — within reach of the {to_eof_total:.3f}s a "
        f"render-to-EOF regression produces; the input -t is not bounding the read"
    )
    assert reported_total == pytest.approx(delivered_total, abs=4 * _FRAME), (
        f"compile_shots reported {reported_total:.3f}s for a {delivered_total:.3f}s file"
    )

    # --- 5. Frame rate -----------------------------------------------------
    rate = await _probe_avg_frame_rate(final_path)
    assert rate == pytest.approx(float(_RENDER_FPS), abs=0.01), (
        f"delivered at {rate:.4f} fps, requested {_RENDER_FPS} — the 24000/1001 "
        f"source cadence leaked through the master"
    )
    frames = await probe_video_frame_count(final_path)
    assert frames / _RENDER_FPS == pytest.approx(delivered_total, abs=2 * _FRAME), (
        f"{frames} frames at {_RENDER_FPS} fps is {frames / _RENDER_FPS:.3f}s but the "
        f"container says {delivered_total:.3f}s — the stream was resampled, not re-timed"
    )

    # --- 3. Cut positions --------------------------------------------------
    expected_cuts = []
    running = 0.0
    for length in intended[:-1]:
        running += length
        expected_cuts.append(running)
    detected = await _probe_cut_times(final_path)
    for i, want in enumerate(expected_cuts):
        nearest = min(detected, key=lambda t: abs(t - want), default=None)
        assert nearest is not None and abs(nearest - want) <= _CUT_TOLERANCE_S, (
            f"no cut within {_CUT_TOLERANCE_S * 1000:.0f}ms of beat {i}→{i + 1} at "
            f"{want:.3f}s (detected cuts: {[round(t, 3) for t in detected]})"
        )
    if is_synthetic:
        # Precision, not just recall — held only on the engineered source, whose
        # picture changes imperceptibly WITHIN a beat by construction. Real drone
        # footage legitimately produces extra scene hits mid-beat.
        assert len(detected) == len(expected_cuts), (
            f"{len(detected)} cuts detected, expected exactly {len(expected_cuts)} "
            f"({[round(t, 3) for t in detected]}) — a spurious cut means the join "
            f"introduced a discontinuity the beat layout does not call for"
        )

    # --- 1. Per-shot length ------------------------------------------------
    # Beat boundaries on the delivered bus: 0, the detected cuts (matched to the
    # intended boundaries above), and the end of the film. The DELTAS are the
    # rendered beat lengths, measured on pixels.
    boundaries = [0.0]
    boundaries += [min(detected, key=lambda t: abs(t - want)) for want in expected_cuts]
    boundaries.append(delivered_total)
    for i, want in enumerate(intended):
        got = boundaries[i + 1] - boundaries[i]
        assert got == pytest.approx(want, abs=_BEAT_TOLERANCE_S), (
            f"beat {i} rendered {got:.3f}s, intended window "
            f"{layout[i][0]:.3f}-{layout[i][1]:.3f}s ÷ speed {layout[i][2]} = "
            f"{want:.3f}s — the source window did not bound this shot"
        )

    # --- Window identity (synthetic source only) ---------------------------
    if not is_synthetic:
        return
    # Calibrate luma→source-time on the SOURCE itself rather than assuming a
    # range/gamma: two probes give the line, and every comparison below is then a
    # measurement against the same encoder's own output.
    y_lo = await _probe_frame_luma(source, src_seconds * 0.15)
    y_hi = await _probe_frame_luma(source, src_seconds * 0.85)
    slope = (y_hi - y_lo) / (src_seconds * 0.70)
    assert slope > 1.0, (
        f"luma ramp too flat to identify a window ({slope:.3f} codewords/s) — the "
        f"synthesised source is wrong, not the renderer"
    )
    for i, (lead, out, _speed) in enumerate(layout):
        at = boundaries[i] + (boundaries[i + 1] - boundaries[i]) / 2.0
        y = await _probe_frame_luma(final_path, at)
        shown = src_seconds * 0.15 + (y - y_lo) / slope
        want_mid = (lead + out) / 2.0
        # ±0.75s: the ramp resolves ~0.27s per codeword here, and the nearest
        # neighbouring window sits >3s away, so this still identifies the window
        # uniquely while tolerating encode noise.
        assert shown == pytest.approx(want_mid, abs=0.75), (
            f"beat {i} shows source t≈{shown:.2f}s but its window is "
            f"{lead:.3f}-{out:.3f}s (midpoint {want_mid:.2f}s) — right length, "
            f"wrong footage: the input -ss did not select this window"
        )


async def _render_control_segment(tmp_path: Path, source: Path, lead: float, out: float) -> Path:
    """The CONTROL: the same source window rendered UNMUTED with NO bed.

    This is what makes the "no source audio in the reel" assertion mean anything.
    It proves the low-band probe DOES read the source clip's tone loudly when the
    tone is present, so a quiet low band in the reel is evidence of the mute
    working — not evidence that the probe is deaf.
    """
    seg = tmp_path / "control-unmuted.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(
                id="control",
                seconds=round(out - lead, 3),
                video_url=str(source),
                clip_lead_seconds=lead,
                clip_out_seconds=out,
            ),
            seg,
            _settings(tmp_path),
            tmp_path,
            http,
        )
    return seg


async def test_action_reel_audio_is_the_bed_not_the_source_clip(tmp_path: Path) -> None:
    """The delivered bus carries the MUSIC BED and not the source clip's audio.

    Always runs on the synthesised source, even when ``_SOURCE_ENV`` points at
    real footage: the assertion needs a source whose audio sits at a KNOWN
    frequency, otherwise "the source's audio is absent" cannot be measured — only
    assumed. Real drone audio is broadband and would be indistinguishable from
    the bed in any band probe.
    """
    source = await _make_reel_source(tmp_path / "reel-source.mp4")
    src_seconds = await probe_duration(source)
    layout = _beat_layout(src_seconds)
    shots = _reel_shots(source, layout)
    intended_total = sum((out - lead) / speed for lead, out, speed in layout)

    bed = await make_bed_tone(
        tmp_path / "bed.wav", duration=intended_total + 5.0, frequency=_BED_TONE_HZ
    )
    final_path, _total, _report, _, _ = await renderer.compile_shots(
        shots,
        _settings(tmp_path),
        music=MusicBed(url=str(bed), gain_db=-6.0, fade_in_s=0.5, fade_out_s=0.5),
    )
    delivered = await probe_duration(final_path)

    # An audio stream exists at all.
    assert await probe_has_audio(final_path), "the reel was delivered with no audio stream"
    # ...and it is not silence. A muted-everything bus (which is what happens if
    # the per-clip mute is applied to the FINISHED bed bus rather than to the
    # shots' own audio) lands near -91 dBFS and fails right here.
    overall = await probe_mean_volume(final_path)
    assert overall > -45.0, (
        f"reel bus measures {overall:.1f} dBFS — silent. The music bed is not "
        f"reaching the delivered file"
    )

    # Measure inside the bed's fades so neither end is being read at fade level.
    win = (1.0, max(1.5, delivered - 1.0))
    reel_high = await probe_band_mean_volume_window(
        final_path, *win, highpass_hz=_HIGH_BAND_HZ, stages=_HIGH_BAND_STAGES
    )
    reel_low = await _probe_low_band_mean_volume(final_path, *win)
    assert reel_high > -60.0, (
        f"the {_BED_TONE_HZ} Hz bed band measures {reel_high:.1f} dBFS — the bed is absent"
    )

    # The control proves the probe can hear the source clip's tone.
    lead, out, _speed = layout[0]
    control = await _render_control_segment(tmp_path, source, lead, out)
    ctl_dur = await probe_duration(control)
    ctl_win = (0.2, max(0.4, ctl_dur - 0.2))
    ctl_low = await _probe_low_band_mean_volume(control, *ctl_win)
    ctl_high = await probe_band_mean_volume_window(
        control, *ctl_win, highpass_hz=_HIGH_BAND_HZ, stages=_HIGH_BAND_STAGES
    )
    assert ctl_low - ctl_high >= _BAND_CONTRAST_DB, (
        f"CONTROL is not a control: an UNMUTED render of the same window measures "
        f"{ctl_low:.1f} dBFS in the source-tone band vs {ctl_high:.1f} dBFS above "
        f"{_HIGH_BAND_HZ:g} Hz — the probe cannot see the source clip's audio, so "
        f"the reel assertion below would prove nothing"
    )

    # ...and the reel inverts that contrast: bed present, source audio gone.
    assert reel_high - reel_low >= _BAND_CONTRAST_DB, (
        f"reel measures {reel_low:.1f} dBFS in the source-tone band vs "
        f"{reel_high:.1f} dBFS in the bed band — the source clip's own audio "
        f"survived audio.mute and is playing under the music bed"
    )
