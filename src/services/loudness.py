"""Read-only loudness / silence measurement of the final mux (observability).

MEASURE-ONLY: nothing in this module changes the audio the compiler emits. It
runs a discarded (``-f null -``) measure pass over an already-finished file and
reports what the delivered audio actually measured — integrated loudness, true
peak, loudness range, how much of the timeline is silence, and the measured
noise floor.

Two pieces:

* ``parse_loudness`` — a PURE, unit-testable parser. It turns an ffmpeg
  ``loudnorm ...:print_format=json`` + ``silencedetect`` stderr dump into a
  small metrics dict, and NEVER raises: garbage/empty in → all-``None`` out.
  Mirrors the parse-and-fallback shape of ``renderer._ffprobe_duration`` /
  ``probe_output_duration``.
* ``final_mux_loudness`` — runs the read-only measure pass through the
  renderer's ffmpeg concurrency gate (``renderer._run`` — never a bare
  ``create_subprocess_exec``, the async-hardening invariant). Best-effort:
  returns ``None`` on ANY failure and never raises, so a measurement hiccup can
  never fail a compile job.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import structlog

from src.services import renderer

log = structlog.get_logger(__name__)

# silencedetect emits `... silence_duration: <float>` per detected silent span.
_SILENCE_DURATION_RE = re.compile(r"silence_duration:\s*([0-9]+(?:\.[0-9]+)?)")
# ffmpeg's input probe prints `  Duration: HH:MM:SS.ss, ...` — the total we
# ratio the summed silence against.
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _extract_json_block(stderr: str) -> dict[str, Any] | None:
    """Return the trailing ``{...}`` JSON object from a loudnorm stderr dump.

    loudnorm can print its stats block more than once (filter re-init on a
    stream change); the LAST complete object is the one covering the whole
    file. Returns ``None`` on anything unparseable."""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(stderr[start : end + 1])
    except ValueError, TypeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_float(value: object) -> float | None:
    """Coerce a loudnorm JSON value (quoted string or number) to float, or None.

    loudnorm reports pure silence as ``"-inf"``; ``float("-inf")`` is a valid
    real measurement so it is kept as-is."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _finite_or_none(value: float | None) -> float | None:
    """Drop a non-finite float (``±inf``/``nan``) to None — keeps a measurement
    JSON-safe and treats an unmeasurable value as absent."""
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _total_duration(stderr: str) -> float | None:
    """Parse the ffmpeg ``Duration: HH:MM:SS.ss`` line to seconds, or None."""
    m = _DURATION_RE.search(stderr)
    if m is None:
        return None
    try:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except ValueError:
        return None


def _silence_ratio(silencedetect_stderr: str) -> float | None:
    """Fraction of the timeline flagged silent: sum(silence_duration) / total.

    ``None`` when no silencedetect output was supplied, no total duration is
    parseable, or no silent spans were detected."""
    if not silencedetect_stderr:
        return None
    total = _total_duration(silencedetect_stderr)
    if total is None or total <= 0:
        return None
    durations = [float(x) for x in _SILENCE_DURATION_RE.findall(silencedetect_stderr)]
    if not durations:
        return None
    return sum(durations) / total


def parse_loudness(
    loudnorm_json_stderr: str, silencedetect_stderr: str = "", astats_stderr: str = ""
) -> dict[str, float | None]:
    """Pure parser: loudnorm+silencedetect+astats stderr → a loudness metrics dict.

    Keys: ``integrated_lufs`` (loudnorm ``input_i``), ``true_peak_dbtp``
    (``input_tp``), ``lra`` (``input_lra``), ``silence_ratio`` (derived from
    ``silencedetect``, or ``None`` when not provided/derivable), and
    ``noise_floor_db`` (the robust windowed floor when the dump carries windowed
    astats levels, else astats' ``Noise floor dB`` summary, or ``None`` when no
    astats output was supplied). NEVER raises — any missing/garbage field falls back to
    ``None``.

    The three stderr arguments are SEPARATE parameters even though
    ``final_mux_loudness`` passes one combined dump to all three: each names WHICH
    filter's output the key beneath it is read from, so a caller that ran only one
    of the filters gets ``None`` for the others rather than a silently-empty
    parse. Both extra filters are pass-through analysers, so all three read the
    ORIGINAL samples — see the filter-order note on ``final_mux_loudness``.

    Every value is FINITE or ``None``: a non-finite reading (loudnorm prints
    ``-inf`` for digital silence) is JSON-unsafe and would poison the JSONB
    completion write — see the comment in the body."""
    data = _extract_json_block(loudnorm_json_stderr) or {}
    return {
        # ⚠️ EVERY float here is scrubbed through ``_finite_or_none`` — the same
        # guard ``acx_measure`` already applies. loudnorm prints ``"input_i":
        # "-inf"`` for a digitally-silent mux, and ``_as_float`` deliberately
        # preserves that as a real measurement. It is NOT JSON-safe: this dict is
        # returned to ``compile.py`` and stored verbatim in ``CompileJob.result``
        # (JSONB). No custom ``json_serializer`` is configured, so ``json.dumps``
        # emits the bare token ``-Infinity`` and postgres rejects the whole write
        # with ``invalid input syntax for type json``. The completion write then
        # retries with the IDENTICAL payload (a retry that varies nothing can
        # never succeed), the row stays ``processing``, the reaper reclaims and
        # RE-RENDERS the job, and it finally reports ``failed`` for an artifact
        # that was actually delivered — and never billed. An unmeasurable value
        # is ABSENT (``None``), never ``±inf``/``nan``.
        "integrated_lufs": _finite_or_none(_as_float(data.get("input_i"))),
        "true_peak_dbtp": _finite_or_none(_as_float(data.get("input_tp"))),
        "lra": _finite_or_none(_as_float(data.get("input_lra"))),
        # ``_silence_ratio`` divides by a total it has already proven > 0 and sums
        # values from a digits-only regex, so it cannot produce inf/nan — scrubbed
        # anyway so NO key on this path can carry a non-finite float.
        "silence_ratio": _finite_or_none(_silence_ratio(silencedetect_stderr)),
        # ``parse_noise_floor`` already clamps astats' ``-inf`` (pure digital
        # silence) to a finite ``_SILENT_NOISE_FLOOR_DB``, so it cannot introduce a
        # non-finite value; scrubbed anyway to keep the invariant above total.
        #
        # WINDOWED levels win when the dump carries them (the robust estimator —
        # ``parse_windowed_noise_floor``); a dump from a plain ``astats`` pass has
        # none and falls back to its ``Overall`` summary line.
        "noise_floor_db": _finite_or_none(
            windowed
            if (windowed := parse_windowed_noise_floor(astats_stderr)) is not None
            else parse_noise_floor(astats_stderr)
        ),
    }


# astats prints `... Noise floor dB: <float|-inf>` once per channel and once for
# the trailing `Overall` block; the Overall block is emitted LAST, so the final
# match is the whole-file noise floor.
_NOISE_FLOOR_RE = re.compile(r"Noise floor dB:\s*(-?(?:inf|[0-9]+(?:\.[0-9]+)?))")


# astats reports pure digital silence as a ``-inf`` noise floor; ``-inf`` is not
# JSON-representable (it would break the AcxReport column), so we clamp it to a
# finite sub-audible floor that still passes any real ACX ceiling.
_SILENT_NOISE_FLOOR_DB = -120.0


def parse_noise_floor(astats_stderr: str) -> float | None:
    """Whole-file measured noise floor (dB) from an ``astats`` stderr dump.

    Returns the LAST ``Noise floor dB`` value (the trailing ``Overall`` block),
    or ``None`` when astats printed no noise-floor line. A ``-inf`` reading
    (pure digital silence) is clamped to ``_SILENT_NOISE_FLOOR_DB`` so the value
    stays finite/JSON-safe while still clearing any real noise-floor ceiling.
    NEVER raises.

    ⚠️ This is astats' whole-file MINIMUM windowed peak — one quiet window decides
    it. That is the right reading for the ACX ceiling it serves (``acx_measure``)
    and the WRONG one for the speech-presence check, which uses
    ``parse_windowed_noise_floor`` instead; see there for why."""
    matches = _NOISE_FLOOR_RE.findall(astats_stderr or "")
    if not matches:
        return None
    value = _as_float(matches[-1])
    if value is None:
        return None
    if value == float("-inf"):
        return _SILENT_NOISE_FLOOR_DB
    return value


#: Length of one noise-floor analysis window. 50 ms is astats' own default
#: short-window length (``length=0.05``), so a windowed peak here is the same
#: quantity astats' ``Noise floor dB`` is the minimum of.
NOISE_FLOOR_WINDOW_SECONDS = 0.05

#: Which percentile of the windowed peak levels is reported as the floor. See
#: ``parse_windowed_noise_floor`` for why this is a low percentile and not the
#: minimum, and for what it costs.
NOISE_FLOOR_PERCENTILE = 5.0

#: ``asetnsamples`` frame length feeding the windowed astats: 50 ms at 48 kHz.
#: Only an upper bound on the frame length — the parser bins by ``pts_time``, so
#: a 96 kHz master (25 ms frames) or a 44.1 kHz one (54 ms) still yields
#: ``NOISE_FLOOR_WINDOW_SECONDS``-wide windows, never sample-rate-dependent ones.
_NOISE_FLOOR_FRAME_SAMPLES = 2400

_WINDOW_PEAK_KEY = "lavfi.astats.Overall.Peak_level"

#: The windowed-noise-floor head of the measure chain. ``asetnsamples`` only
#: re-slices frames (``p=0``: the short tail frame is NOT zero-padded), ``astats``
#: with ``reset=1`` computes its statistics per frame and attaches them as frame
#: metadata, and ``ametadata=mode=print`` logs the one key the parser reads. All
#: three pass the samples through untouched. Every option used here exists on
#: ffmpeg 4.4, 6.1 and 7.1; nothing depends on a summary line only some versions
#: print.
WINDOWED_NOISE_FLOOR_FILTERS = (
    f"asetnsamples=n={_NOISE_FLOOR_FRAME_SAMPLES}:p=0,"
    "astats=metadata=1:reset=1,"
    f"ametadata=mode=print:key={_WINDOW_PEAK_KEY}"
)

# ``ametadata=mode=print`` logs, per frame, ``frame:N pts:N pts_time:T`` and then
# one ``key=value`` line. They are matched as two independent line patterns (not
# one two-line regex) because ffmpeg's ``\r``-terminated progress line is written
# from another thread and can land between them.
_WINDOW_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9]+(?:\.[0-9]+)?)")
_WINDOW_PEAK_RE = re.compile(re.escape(_WINDOW_PEAK_KEY) + r"=\s*(-?(?:inf|[0-9]+(?:\.[0-9]+)?))")


def _window_peaks(stderr: str) -> list[float]:
    """Peak level (dBFS) of each ``NOISE_FLOOR_WINDOW_SECONDS`` window, in order.

    Frames are binned by their ``pts_time`` and a bin's level is the LOUDEST frame
    peak in it. When any frame carries no parseable ``pts_time`` the binning is
    abandoned and every frame is its own window (a degraded but still robust
    reading) rather than guessing a timeline. ``-inf`` (an all-zero window) is
    clamped to ``_SILENT_NOISE_FLOOR_DB``."""
    frames: list[tuple[float | None, float]] = []
    pts: float | None = None
    for line in (stderr or "").splitlines():
        peak = _WINDOW_PEAK_RE.search(line)
        if peak is None:
            stamp = _WINDOW_PTS_RE.search(line)
            if stamp is not None:
                pts = _as_float(stamp.group(1))
            continue
        level = _as_float(peak.group(1))
        if level is None or math.isnan(level) or level == float("inf"):
            pts = None
            continue
        frames.append((pts, max(level, _SILENT_NOISE_FLOOR_DB)))
        pts = None
    if any(stamp is None for stamp, _ in frames):
        return [level for _, level in frames]
    windows: dict[int, float] = {}
    for stamp, level in frames:
        if stamp is None:  # unreachable (checked above); narrows the type
            continue
        index = math.floor(stamp / NOISE_FLOOR_WINDOW_SECONDS + 1e-6)
        windows[index] = max(level, windows.get(index, level))
    return [windows[index] for index in sorted(windows)]


def robust_noise_floor(
    window_levels_db: list[float], percentile: float = NOISE_FLOOR_PERCENTILE
) -> float | None:
    """The ``percentile``-th lowest of ``window_levels_db`` (dB), or ``None`` if empty.

    Rank ``floor(n * percentile / 100)`` of the ascending-sorted levels, with no
    interpolation: the result is always a level some window actually measured.
    With the default 5th percentile and at least twenty windows (1 s of audio)
    the lowest window can never be the answer, so one outlier — however far down —
    cannot move it; a 12 s clip tolerates twelve. Under twenty windows it degrades
    to the minimum; there is no robust statistic of a file that short."""
    if not window_levels_db:
        return None
    ordered = sorted(window_levels_db)
    rank = min(len(ordered) - 1, int(len(ordered) * percentile / 100.0))
    return ordered[max(rank, 0)]


def parse_windowed_noise_floor(stderr: str) -> float | None:
    """ROBUST noise floor (dB) of a file, from a ``WINDOWED_NOISE_FLOOR_FILTERS`` dump.

    THE ESTIMATOR: cut the file into ``NOISE_FLOOR_WINDOW_SECONDS`` (50 ms)
    windows, take each window's peak level, and report the
    ``NOISE_FLOOR_PERCENTILE``-th (5th) percentile of those peaks. It is the same
    kind of quantity as astats' ``Noise floor dB`` — a whole-file minimum of a
    short-window peak — with the minimum replaced by a low percentile, so the
    numbers stay on the scale ``SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB`` was
    calibrated on (measured below).

    WHY THE MINIMUM WAS WRONG. A minimum is decided by ONE window. The
    speech-presence check asks "does the program rise above its own floor?", and a
    hiss-only master answers no only while its floor IS its level. One quiet window
    — astats reports it as ``Noise floor count: 1`` — drops the minimum 9-39 dB
    under the bed, the gap turns positive, and a master with no speech in it reads
    as "speech present". Measured 2026-10-05 on ffmpeg 7.1.5: 13 of 300 AAC encodes
    of white noise (~4%) did exactly that. A short tail frame, an encoder
    priming/flush frame or a single dropout is enough; none of them is evidence of
    speech. A percentile needs 5% of the timeline to be quiet before the floor
    moves, which is what "the program has pauses" actually means.

    WHY THE 5TH AND NOT A HIGHER ONE — measured 2026-10-06, ffmpeg 4.4, AAC-in-MP4,
    gap = ``integrated_lufs - floor`` against the 1.5 dB threshold:

                                              astats min    p5      p10     p20
      white hiss @ -34 dBFS (no speech)          -6.30     -5.93   -6.05   -6.33
      1 kHz tone (worst no-speech reading)       -3.33     -3.10   -3.10   -3.11
      bursts + pink bed (legit)                  +5.85     +6.93   +6.50   +5.70
      real TTS speech, one unbroken sentence    +28.30    +29.06   +7.45   -0.22
      the same over a pink bed 3 dB down        +10.45    +10.44   +4.04   -0.58
      bursts 1.8 s on / 0.2 s off + pink bed     -5.10    +12.46   -5.01   -5.25

    The percentile is the fraction of the timeline that must be PAUSE before the
    floor is read from a pause. Dense narration pauses for little of it: at the
    20th percentile a real spoken sentence is convicted as hiss, and at the 10th it
    keeps under a quarter of its margin. The 5th reads within 0.25 dB of astats'
    minimum on every legitimate fixture and within 1.3 dB on every steady one, and
    still discards the twelve lowest windows of a 12 s clip. WHICH WAY IT DRIFTS: a
    narration with pauses in under 5% of its 50 ms windows is judged on its quietest
    SPEECH windows instead, i.e. toward a smaller gap. (The last row is a case
    astats' own minimum got wrong in that same direction; it is pinned by
    ``test_dense_narration_with_short_pauses_is_not_convicted``.)

    ON PRODUCTION FFMPEG (7.1.5, measured 2026-10-06 in the deployed image, 200
    seeds x 12 s/6 s of the white hiss above): the minimum let 11 of 400 hiss
    encodes through, this floor 0 of 400 (worst gap -5.94 dB); 0 of 310 speech
    fixtures (TTS clean and over pink beds, dense narration, real masters) were
    convicted by either. The cost is margin, as predicted above: TTS over a pink
    bed fell from ~+19 dB to a worst +4.87 dB (threshold 1.5).

    Returns ``None`` when the dump carries no windowed levels. NEVER raises."""
    return robust_noise_floor(_window_peaks(stderr))


def program_to_noise_floor_db(
    integrated_lufs: float | None, noise_floor_db: float | None
) -> float | None:
    """How far the PROGRAM sits above the file's own measured NOISE FLOOR, in dB.

    ``integrated_lufs - noise_floor_db``, or ``None`` when either input is absent
    (an unmeasured input yields no verdict — never a fabricated 0.0).

    WHY THIS DIFFERENCE AND NOT EITHER NUMBER ALONE. Neither term separates speech
    from steady hiss on its own, and neither does ``silence_ratio``: hiss is a
    constant low-level signal, so ``silencedetect``'s -50 dB gate never flags it,
    and loudnorm normalises a hiss-only master to the same ``integrated_lufs`` as a
    narrated one. The DIFFERENCE does separate them, and it is LEVEL-INVARIANT —
    gain applied to the whole file moves both terms equally, so a hiss master that
    the loudness master lifted to -14 LUFS reads the same as the raw one.

    The mechanism is dynamics, not level. astats derives its noise floor from the
    QUIETEST part of the signal; real narration has pauses, so its floor sits far
    below its program level. A steady signal IS its own floor, so the difference
    collapses toward zero (slightly NEGATIVE in practice, because loudnorm's
    K-weighting attenuates the broadband content astats measured unweighted).

    Units are deliberately mixed (LUFS above dBFS) and that is sound here: both are
    referenced to digital full scale and only their DIFFERENCE is used, so the
    common reference cancels. The residual K-weighting offset is folded into the
    measured separation that ``mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB``
    cites, rather than corrected for analytically."""
    if integrated_lufs is None or noise_floor_db is None:
        return None
    return integrated_lufs - noise_floor_db


def evaluate_acx(
    measure: dict[str, float | None] | None,
    *,
    lufs_min: float,
    lufs_max: float,
    tp_max: float,
    noise_floor_max: float,
) -> dict[str, float | bool | None]:
    """PURE ACX-compliance verdict over an ``acx_measure`` dict.

    ``pass`` is True iff integrated loudness is within ``[lufs_min, lufs_max]``,
    true peak ≤ ``tp_max``, AND noise floor ≤ ``noise_floor_max``. A missing
    measurement (``None``) fails its gate — an unmeasurable master is NOT a
    passing master (mirrors ``mastering.py``'s no-audio-must-not-silently-pass
    guard). NEVER raises."""
    m = measure or {}
    integrated = m.get("integrated_lufs")
    true_peak = m.get("true_peak_dbtp")
    noise_floor = m.get("noise_floor_db")
    loud_ok = integrated is not None and lufs_min <= integrated <= lufs_max
    tp_ok = true_peak is not None and true_peak <= tp_max
    # A silent/near-silent master reads a very LOW (good) noise floor but is not a
    # real book; the loudness gate above already rejects it, so noise-floor only
    # needs the ≤ ceiling here.
    floor_ok = noise_floor is not None and noise_floor <= noise_floor_max
    return {
        "integrated_lufs": integrated,
        "true_peak_dbtp": true_peak,
        "noise_floor_db": noise_floor,
        "pass": bool(loud_ok and tp_ok and floor_ok),
    }


async def acx_measure(path: str) -> dict[str, float | None] | None:
    """Measure a mastered audiobook file's ACX metrics (read-only, best-effort).

    One discarded pass runs ``astats`` (noise floor) + ``loudnorm print_format``
    (integrated loudness + true peak) over ``path``. astats passes the signal
    through unchanged, so both filters measure the ORIGINAL file; ``-f null -``
    throws the output away. Returns ``{integrated_lufs, true_peak_dbtp,
    noise_floor_db}`` or ``None`` on any failure — never raises, never fails a
    job."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        path,
        "-af",
        "astats=metadata=0,loudnorm=I=-20:TP=-3:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = await renderer._run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("acx_measure.measure_failed", path=path, error=str(exc))
        return None
    try:
        data = _extract_json_block(stderr) or {}
        return {
            # A ``-inf`` integrated/true-peak means a silent file: not JSON-safe
            # and not a valid master, so drop to None (fails the ACX gate).
            "integrated_lufs": _finite_or_none(_as_float(data.get("input_i"))),
            "true_peak_dbtp": _finite_or_none(_as_float(data.get("input_tp"))),
            "noise_floor_db": parse_noise_floor(stderr),
        }
    except Exception as exc:  # noqa: BLE001 — parser is defensive, but never fail a job
        log.warning("acx_measure.parse_failed", path=path, error=str(exc))
        return None


async def final_mux_loudness(path: str) -> dict[str, float | None] | None:
    """Measure the delivered file's loudness + silence (read-only, best-effort).

    Runs a single discarded silencedetect/loudnorm pass over ``path`` through
    the ffmpeg concurrency gate and parses the stderr. The measurement target
    (``I=-14``) is only the reference loudnorm prints its stats against; the
    ``-f null -`` output is thrown away, so this NEVER alters the delivered
    audio. Returns ``None`` on any failure — it must never fail a job.

    ⚠️ FILTER ORDER IS LOAD-BEARING: ``silencedetect`` MUST come BEFORE
    ``loudnorm``. ``loudnorm`` is not a pass-through analyser — even with
    ``print_format=json`` its single pass NORMALIZES the samples it forwards, so
    a downstream ``silencedetect`` would see audio already lifted toward -14
    LUFS and quiet passages that really are below the -50 dB gate in the
    DELIVERED file get pushed above it (``silence_ratio`` under-reports exactly
    the quiet masters the metric exists to catch). ``silencedetect`` IS
    pass-through (it only logs, never touches samples), so with it first BOTH
    filters measure the original file — the same reasoning as ``acx_measure``'s
    ``astats`` ordering. loudnorm's ``input_i``/``input_tp``/``input_lra`` JSON
    that ``parse_loudness`` reads is unaffected by the swap.

    ``astats`` LEADS THE CHAIN for exactly the same reason, and it is here rather
    than in a second pass because it costs nothing: it is a pass-through analyser
    (it only annotates and prints), so with it first it measures the ORIGINAL
    samples and every filter behind it sees the signal it would have seen anyway.
    It runs WINDOWED (``WINDOWED_NOISE_FLOOR_FILTERS``: 50 ms frames, per-frame
    peak printed by ``ametadata``) and the floor is a low percentile of those
    windows, not astats' own whole-file minimum — ``parse_windowed_noise_floor``
    says why. ``asetnsamples`` re-slices frames without padding and changes no
    sample, so ``silencedetect`` and ``loudnorm`` behind it read exactly what they
    read before (verified: identical ``input_i``/``input_tp``/``input_lra`` and
    silence spans with and without it).
    That floor is the second measurement the speech-presence check
    needs — no threshold on ``integrated_lufs`` or ``silence_ratio`` alone can
    separate steady hiss from speech, because hiss never trips the -50 dB silence
    gate and normalises to the same loudness as narration. Same placement and same
    argument as ``acx_measure``, which has run astats+loudnorm in ONE pass since it
    was written."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        path,
        "-af",
        (
            f"{WINDOWED_NOISE_FLOOR_FILTERS},"
            "silencedetect=n=-50dB:d=0.3,loudnorm=I=-14:TP=-1.5:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = await renderer._run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("final_mux_loudness.measure_failed", path=path, error=str(exc))
        return None
    try:
        metrics = parse_loudness(stderr, stderr, stderr)
        # ONLY the windowed estimate, never ``parse_loudness``'s summary-line
        # fallback: this chain runs astats with ``reset=1``, so its end-of-run
        # ``Noise floor dB`` summary covers the LAST 50 ms window alone and is not a
        # whole-file reading on any ffmpeg version. No windows → no floor (``None``,
        # which the speech-presence check treats as unmeasured, not as a verdict).
        metrics["noise_floor_db"] = _finite_or_none(parse_windowed_noise_floor(stderr))
        return metrics
    except Exception as exc:  # noqa: BLE001 — parser is defensive, but never fail a job
        log.warning("final_mux_loudness.parse_failed", path=path, error=str(exc))
        return None
