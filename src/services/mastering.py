"""Encode / delivery MASTER — the canonical home of contract C11 (aspect A12).

Composes AFTER the shipped audio master (``renderer._master_filter`` /
``renderer.master_audio``); it never touches or re-derives loudness — the
audio master stays exactly as VC-1 shipped it. This module adds the PICTURE
side of the deliverable master:

* **BT.709 colorspace tagging** — an additive ``colr`` atom on every per-shot
  encode (``colorspace_tag_args``). Empirically it survives both downstream
  ``-c copy`` steps (concat demuxer + audio remux), so per-shot tagging is
  sufficient — no re-assert on the master remux.
* **content-sized CRF / bitrate ladder** (``encode_ladder``)
  whose DEFAULT cell is ``render_crf`` with no VBV — byte-identical to the
  pre-master encode. VBV ``-maxrate``/``-bufsize`` are emitted only when
  configured.
* **conditional deband** (``banding_risk`` + ``deband_filter``) — a smooth
  gradient that would otherwise band is debanded; a clean shot stays a single
  pass and is byte-identical.
* **post-mux soft-gate** (``post_mux_verify``) — a read-only verification of the
  delivered file (colorspace tag present, loudness in range, framerate coherent,
  banding low) that REPORTS a pass/fail + which stages would need a rerun, but
  never fails the render.

Every ffmpeg-touching function here is best-effort / never-raise, exactly like
``loudness.py`` and ``measure.py``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import shutil
import tempfile
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel

from src.api.schemas import VIOLATION_CONTRACT_KEYS, MasteringViolation, NarrationPresenceClip
from src.services.banding_spatial import _median_line as _median_line
from src.services.banding_spatial import spatial_banding_factor as spatial_banding_factor

# The PURE derived metric only — no measurement is performed here.
# ``post_mux_verify`` REUSES the caller's ``final_mux_loudness`` result and does
# not re-measure, and this import keeps that true: the difference is computed
# from numbers already in hand. Safe at module level despite ``loudness``
# importing ``renderer``, because every ``renderer`` import of ``mastering`` is
# function-local (there is no cycle to break).
from src.services.loudness import program_to_noise_floor_db

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from src.config import Settings

log = structlog.get_logger(__name__)

# Fleet default integrated loudness target (LUFS) for the delivery master — the
# fallback used when a Settings object does not carry ``master_lufs``. Mirrors
# the pydantic default in ``src.config.Settings``; kept as a single named
# constant so the gate and any future fallback can't drift apart.
DEFAULT_MASTER_LUFS = -14.0

# Worst-frame banding score at or above which the delivered file still reads as
# banded. Named (rather than inlined in ``_banding_gate_pass``) so the enforce-mode
# violation can quote the SAME number the gate compared against — a violation that
# reports a ceiling the gate did not use is worse than no violation at all.
BANDING_GATE_CEILING = 0.15

# Below this WHOLE-FRAME (pixel-free) ``banding_score``, a spatial re-measure
# (a streamed ffmpeg decode pass + a pure-Python sampled-line staircase pass, see
# ``measure.measure_spatial_factors_streamed``) is not worth paying for: the
# score is already far under ``BANDING_GATE_CEILING`` and no spatial evidence
# could push it back UP (``spatial_factor`` only ever attenuates, never
# amplifies — see ``banding_score``). Callers (``renderer._deband_decide``,
# ``mastering.master_verify``) gate the extra decode on this so the added
# cost is paid ONLY on a frame that could plausibly cross the gate. Set to
# BANDING_GATE_CEILING / 3: comfortably above 0 (skips the vast majority of
# frames, which score exactly 0 by construction — duplicate/static dwell
# frames) and comfortably below the ceiling (nothing that could gate gets
# skipped).
BANDING_SPATIAL_PREFILTER = BANDING_GATE_CEILING / 3

# Maximum accepted deviation (LU) of the delivered integrated loudness from the
# configured ``master_lufs`` target before the loudness check clears the gate.
#
# ⚠️ TUNABLE, PROVENANCE: NONE. n=0. Unlike ``AV_COVERAGE_TOLERANCE_SECONDS``
# below and ``BANDING_GATE_CEILING`` above, this number has never been measured
# against a fleet distribution: 3.0 LU is an EBU-R128-flavoured convention, not
# an observation of this pipeline's masters. That is precisely why the
# ``loudness`` gate is tiered TUNABLE in ``_GATE_TIERS`` and its kill switch
# ``mastering_gate_loudness_enabled`` defaults to **False** — arming an
# unmeasured ceiling is how this project previously built a signal that would
# have fired on 96.7% of masters. Before flipping that switch on, measure the
# ``integrated_lufs`` distribution over a fleet sample and record the n HERE.
LOUDNESS_GATE_TOLERANCE_LU = 3.0

# How far the delivered AUDIO may outrun the delivered PICTURE before the master
# is defective (seconds).
#
# WHY 0.5 AND NOT ZERO: a correct master really does end a little audio-long. AAC
# carries an encoder delay plus a tail pad to fill its final 1024-sample frame
# (~23ms at 44.1kHz), and the mux rounds to the container timebase on top. A
# delivered promo measured on 2026-09-10 shows video 58.200s against audio
# 58.361s — 0.161s of legitimate padding on a master with nothing wrong with it.
# Gating at 0.0 would reject every AAC delivery in the fleet.
#
# PROVENANCE (C12): measured, 2026-09-10. n=1 clean master for the legitimate-
# padding edge (0.161s) and n=2 real truncations for the defect edge (8.8s
# smallest, 39.6s largest), all cited concretely below. This is the ONLY armed
# threshold in this module with a measured two-sided separation, which is why
# ``av_coverage`` is the gate that was already enforcing before the tiering
# below existed.
#
# WHY NOT LARGER: the same day's measurements put the SMALLEST real truncation at
# 8.8s (135.513 video / 144.293 audio) and the largest at 39.6s. 0.5s sits ~3x
# above the observed legitimate padding and ~17x below the smallest real defect,
# so both edges have an order of magnitude of headroom and the choice is not
# balanced on a knife edge.
#
# A module CONSTANT, deliberately, in the shape of LOUDNESS_GATE_TOLERANCE_LU
# above — not a Settings field. A configurable tolerance is a configurable way to
# turn this gate off, and "picture covers narration" is not a per-job preference.
AV_COVERAGE_TOLERANCE_SECONDS = 0.5

# SPEECH PRESENCE — the minimum dB by which the delivered PROGRAM must sit above
# the delivered file's OWN measured noise floor
# (``loudness.program_to_noise_floor_db`` = ``integrated_lufs -
# noise_floor_db``).
#
# THE DEFECT IT NAMES: a production ``website_review`` master shipped carrying no
# speech at all — only faint hiss — and every existing signal passed it. #224
# fixed the CAUSE (a sample-format mismatch surviving a stream-copy concat), but
# ``_assert_uniform_segment_audio`` is a PRE-concat guard: nothing measured the
# FINISHED master for the presence of speech, so any future path to a silent or
# hissing master ships just as quietly.
#
# WHY NOT A THRESHOLD ON ``integrated_lufs`` OR ``silence_ratio``. Neither can
# separate hiss from speech, and this is the whole reason the check needs a second
# measurement. Hiss is a CONSTANT low-level signal: ``silencedetect``'s -50 dB
# gate never flags a sample of it, so ``silence_ratio`` reads ~0 exactly as a
# wall-to-wall narration does; and the loudness master normalises a hiss-only
# file to the same ``integrated_lufs`` as a narrated one. The discriminator is
# DYNAMICS — astats derives its noise floor from the quietest part of the signal,
# real narration has pauses, and a steady signal IS its own floor.
#
# PROVENANCE: MEASURED 2026-09-16 against real ffmpeg over synthesized fixtures
# (this repo carries no audio fixtures at all). FLEET n=0 — which is precisely why
# this gate is NOT ARMED (see ``_GATE_ARMED_DEFAULTS``).
# Gap = integrated_lufs - noise_floor_db, in dB, measured BOTH as raw PCM and
# re-encoded to AAC-in-MP4 (the container this pipeline actually delivers; AAC's
# coding of broadband noise moves the floor a little, so the delivered format was
# measured rather than assumed):
#
#                                              floor    program      PCM     AAC
#   NO-SPEECH side (steady signals — the defect shape):
#     white noise @ -34 dBFS                  -31.26     -36.71    -1.85   -5.45
#     the same, normalised to -14 LUFS         -8.62     -14.74    -5.74   -6.12
#     1 kHz tone @ -34 dBFS                   -51.82     -55.05    -2.80   -3.23
#     pink                                                         -9.49   -9.95
#     brown                                                       -10.95  -10.59
#     blue -7.59 (PCM) | violet -8.84 (PCM)
#     WORST (i.e. highest) no-speech reading over all ten:          -1.85
#   SPEECH side (program with pauses):
#     narration bursts @ -20 dB                                    +87.7
#     the same, normalised to -14 LUFS       -120.00     -14.09   +105.9  +105.9
#     narration + pink music bed 6 dB down    -36.68     -14.10    +22.6  +22.58
#     narration + tonal music bed 10 dB down  -22.05     -13.96    +8.09   +8.09
#     ADVERSARIAL: narration only 3 dB over
#     a full-timeline pink bed — the hardest
#     legitimate master constructed           -21.86     -16.97    +4.66   +4.89
#     WORST legitimate reading over all five:                       +4.66
#
# So the two populations are separated by the open interval (-1.85, +4.66), 6.5 dB
# wide, in BOTH containers. 1.5 sits 3.35 dB above the worst no-speech reading and
# 3.16 dB below the worst legitimate one — near its midpoint (1.40) and near the
# PHYSICAL asymptote:
# a perfectly steady signal's program level equals its own noise floor, i.e. a gap
# of 0, pushed slightly negative because loudnorm's K-weighting attenuates
# broadband content that astats measured unweighted. The number is therefore
# anchored to a property of the signal, not to a fleet percentile.
#
# WHICH WAY IT DRIFTS: a no-speech master that carries ANY transient (a sting, a
# whoosh) reads a larger gap and is MISSED. That is the safe direction — this
# project's law is that drift must cause a miss, never a false conviction of a
# paid deliverable.
SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB = 1.5

# ⚠️ SUPERSEDED for the defect that motivated it. ``speech_presence`` above
# answers "does this master contain speech ANYWHERE" from a single pair of
# loudness statistics, and it has a hole its own author did not anticipate:
# ``program_to_noise_floor_db`` is ``integrated_lufs - noise_floor_db``, and on a
# DIGITALLY SILENT master both terms come from the same all-zero-sample signal.
# astats reports a ``-inf`` floor (clamped to a finite sentinel upstream), and the
# clamp does not track the sign of the gap it manufactures — on the master that
# actually shipped broken (one of four narration clips entirely absent, 68%
# silence) the two terms diverge enough that the gap reads ~+103.8 dB, i.e. "the
# program sits 103.8 dB above its own floor" — comfortably PASSING a gate meant to
# catch exactly this file. A single reduced statistic over the WHOLE master cannot
# tell "every clip present" from "one clip missing, the rest present" — both
# shapes contain plenty of real speech, so the aggregate margin looks fine either
# way. It is left in place (still reported, still not armed) because it is cheap,
# harmless, and useful telemetry for the SEPARATE hiss-vs-speech shape it was
# built for — but the missing-narration defect this file exists to catch is
# handled by ``narration_presence_report`` below, which checks each shot's own
# clip against its own timeline slot instead of asking one number to speak for
# the whole file.
NARRATION_PRESENCE_SAMPLE_RATE = 8000
NARRATION_PRESENCE_AXCORRELATE_BLOCK_SIZE = 1024

# How far past a shot's probed [start, end) the master-audio window extends on
# EACH side before correlating, in seconds.
#
# ⛔ CALIBRATION DEFECT, FOUND AND FIXED 2026-09-17: this used to be 0.5 s,
# reasoned (never measured) to absorb AAC encoder delay (~24 ms, see
# ``AV_COVERAGE_TOLERANCE_SECONDS``) and the probed bus-offset's own rounding.
# On REAL production audio (see ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``
# for the full table) that padding did not "dilute a little" — it materially
# degraded the RMS reading for EVERY clip whose duration did not exactly match
# its window, false-negativing two of four genuinely-present clips on a clean
# production master (-21.03 dB, -20.23 dB, both under the old -20.0 dB
# threshold). The effect was NOT proportional to the padding's share of the
# window (which would make it worse for short clips and better for long ones —
# the opposite of what was measured): even a 50 ms margin on each side (vs the
# old 0.5 s) still degraded a 2.5 s clip's reading from -7.75 dB (its own exact
# span) to -21.36 dB. ``axcorrelate``'s ``fast`` (FFT-based) algorithm reads a
# mismatched clip/window duration as spurious low-correlation samples across
# the WHOLE output, not just the non-overlapping tail, so ANY padding — not
# just a large one — corrupts an RMS taken over the correlated stream.
#
# THE FIX: 0.0 — correlate against the shot's own probed [start, end) exactly,
# no padding. This is also what makes the statistic LENGTH-INVARIANT: with no
# padding, there is nothing for a longer clip's window to dilute more of.
# ``axcorrelate``'s own block-based lag search (``NARRATION_PRESENCE_AXCORRELATE_BLOCK_SIZE``
# = 1024 samples = 128 ms at 8 kHz, >5x the measured AAC delay) still finds the
# alignment inside an exact window without needing extra samples either side —
# confirmed on all 7 real clips below, including the shortest (2.29 s) and a
# genuinely present clip on the BROKEN production master.
NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS = 0.0

# ⛔ #234 — LAG TOLERANCE, added 2026-09-18. ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS``
# fixed the DILUTION defect (widening the window degrades the mean RMS over the
# whole correlated stream), but left the "own" measurement pinned to exactly one
# offset: the probed [start, end), no search. The docstring below's "DRIFT-
# SENSITIVE" finding (a genuinely present clip loses ~9 dB of margin under 100 ms
# of drift) is a SEPARATE axis from window width — searching several candidate
# windows of the SAME width (start shifted, duration unchanged) and taking the
# best reading does not dilute anything (each candidate is still exact-width),
# it just tolerates the clip/window not lining up perfectly. Searched only for
# the shot's OWN window — see ``NARRATION_PRESENCE_IDENTITY_MARGIN_DB`` for why
# the NEIGHBOUR comparison below deliberately does NOT get this same tolerance.
NARRATION_PRESENCE_LAG_SEARCH_SECONDS = 0.150
NARRATION_PRESENCE_LAG_STEP_SECONDS = 0.030

# ⛔ #234 — IDENTITY MARGIN, added 2026-09-18. The "ABSENT SIDE, NOT CONTENT-AWARE"
# finding below measured that a real present clip correlated against an ADJACENT
# shot's exact window reads -16.7 to -23.3 dB — comfortably above the -25.0 dB
# absolute floor, so the old statistic could not tell "this shot's own audio" from
# "audio that belongs one slot over". Fix: also correlate the clip against its
# immediate NEIGHBOUR shot(s)' own windows — at offset 0 ONLY, no lag search (the
# neighbour check must not get to search into the current shot's own territory,
# which is exactly what happens on a zero-gap timeline if lag search is applied to
# both sides) — and require the shot's own (lag-tolerant) reading to beat the best
# neighbour reading by this margin before calling it present.
#
# CALIBRATED 2026-09-18 on the SAME 3 real production masters as the presence
# table below, this time measuring OWN (lag-tolerant, ±0.150s) vs NEIGHBOUR
# (exact window, no lag) for all 4 narration lines on each:
#
#   master   line   truth      own dB     neighbour dB   margin
#   b00d     L1     present    -10.68       -20.18        +9.50
#   b00d     L2     present    -13.92       -16.51         +2.58   <- worst present margin
#   b00d     L3     present    -14.64       -19.40         +4.77
#   b00d     L4     present    -12.53       -23.30        +10.77
#   aeae     L1     present    -10.47       -20.83        +10.36
#   aeae     L2     present    -13.66       -16.41         +2.75
#   aeae     L3     ABSENT      -90.0       -19.16        -71.16
#   aeae     L4     present    -12.48        -90.0        +77.52
#   f82      L1     ABSENT     -31.31        -90.0        +58.69  <- fails the ABSOLUTE
#   f82      L2     ABSENT      -90.0       -33.14        -56.86     floor regardless
#   f82      L3     ABSENT      -90.0        -90.0          0.0      of margin (AND, not OR)
#   f82      L4     unmeasured (window past this master's own end — excluded, not scored)
#
# The worst REAL PRESENT margin is +2.58 dB (b00d L2); every ABSENT line either
# fails the -25.0 dB absolute floor on its own (f82 L1) or has a deeply negative
# margin (aeae L3, f82 L2/L3). 1.5 dB leaves >1 dB of headroom below the worst
# present case while staying clear of every absent one — both checks are
# required (AND): a clip that beats its neighbour but is itself below the floor
# (f82 L1) is still ABSENT.
NARRATION_PRESENCE_IDENTITY_MARGIN_DB = 1.5

# A non-finite (``-inf``) RMS level — a window and a clip that share NOT ONE
# correlated sample — clamped to a finite JSON-safe sentinel, same shape as
# ``loudness._SILENT_NOISE_FLOOR_DB``.
_NARRATION_PRESENCE_SILENT_RMS_DB = -90.0

# THE MEASURE: cross-correlate the fetched SOURCE narration clip against the
# delivered master's own audio at that shot's (padded) timeline window, using
# ffmpeg's native ``axcorrelate`` filter (present on both the dev box's ffmpeg
# 4.4 and the production pod's 7.1.5 — verified 2026-09-17), then read the
# correlated stream's RMS level (dB) via ``astats``. Chosen over a loudness
# statistic (the ``speech_presence`` gate above) because it answers the actual
# question — "is THIS clip's content in THIS window" — instead of "does the
# whole file contain speech somewhere", and over a hand-rolled numpy
# cross-correlation because it needed no new dependency (this repo carries
# neither numpy nor scipy) and runs through the SAME ffmpeg binary every other
# measure in this module already shells out to — no new version surface.
#
# PROVENANCE — RECALIBRATED 2026-09-17 on REAL production audio, replacing a
# synthetic-only claim ("-0.2 to -6 dB present, -32 to -38 dB absent") that did
# not hold once measured against real narration: two of four genuinely-present
# clips on a CLEAN production master read on the wrong side of the old -20.0 dB
# threshold (-21.03 dB, -20.23 dB) — both the two LONGER (~5 s) clips, because
# the OLD padded-window statistic degraded with clip duration (see
# ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS`` for the padding-artifact root
# cause and the fix: correlate against the exact [start, end) window, no
# padding). Measured against ffmpeg 4.4 (dev; ffmpeg 7.1.5 on the production
# pod carries the same ``axcorrelate``/``astats`` filters, verified
# version-stable 2026-09-17) with the FIXED (unpadded) statistic, over 7 REAL
# clips — 4 from a clean production run
# (``01a0aa27-f309-76b5-986e-fb82e88d29a1``) and 3 from the broken production
# master (``01a0aeae-db0d-76e8-97b9-bce533b6e752`` / ``master_aeae.mp4``) whose
# 4th clip is the real defect this gate exists to catch:
#
#   clip                          truth     old (padded) RMS   NEW (unpadded) RMS
#   good shot-000 (2.29s)         present         -7.75 dB           -7.75 dB
#   good clip2 (2.53s)            present        -18.16 dB           -6.63 dB
#   good clip3 (4.96s)            present        -21.03 dB           -7.92 dB
#   good clip4 (4.73s)            present        -20.23 dB           -6.51 dB
#   broken line1                  present        -24.26 dB          -13.66 dB
#   broken line2                  ABSENT            -inf               -inf
#   broken line3                  present        -32.94 dB          -12.48 dB
#
# Under the NEW statistic, every real present clip reads -13.66 dB or higher;
# the one real absent clip reads a clamped -90.0 dB (``-inf``, see
# ``_NARRATION_PRESENCE_SILENT_RMS_DB``).
#
# ⚠️ MEASURED LIMITATIONS (2026-09-17) — BOTH FIXED 2026-09-18 (#234), read on:
#
# PRESENT SIDE, DRIFT-SENSITIVE (FIXED): the worst real present clip (-13.66 dB)
# sat 11.3 dB clear of -25.0 dB, but that margin assumed the window lined up
# with the clip. Shifting the correlation window by +0.10 s (simulating drift
# between the probed [start, end) and where the content actually sits in the
# delivered master) dropped the WEAKEST real clip to -22.88 dB — only ~2.1 dB of
# margin left, not 11.3. Accumulated AAC encoder delay from concat joins
# (~24 ms per join, see ``AV_COVERAGE_TOLERANCE_SECONDS``) compounds shot over
# shot, so a shot late in a long video was the case most exposed to this.
# FIX: ``NARRATION_PRESENCE_LAG_SEARCH_SECONDS`` — the own-window measurement now
# searches ±0.150s of offset (same-width candidate windows, so no dilution) and
# keeps the best reading, absorbing realistic drift instead of losing margin to it.
#
# ABSENT SIDE, NOT CONTENT-AWARE (FIXED): the only REAL absent case measured was
# digital silence (-inf) — there was no real measurement of the gate's actual
# target defect, the WRONG content landing in a shot's slot. Correlating a real
# present clip against the WRONG (adjacent shot's) window read -16.5 to -23.3 dB,
# comfortably ABOVE -25.0 dB — i.e. classified PRESENT regardless of which slot
# the audio actually belonged to. FIX: ``NARRATION_PRESENCE_IDENTITY_MARGIN_DB`` —
# the shot's own (lag-tolerant) reading must now beat its neighbour(s)' exact-
# window reading by 1.5 dB before being called present; see that constant's
# docstring for the full real-master calibration table (worst present margin
# +2.58 dB, every absent line either fails the absolute floor or reads deeply
# negative on this margin).
#
# STILL UNARMED — NOT a lag/identity limitation, a SAMPLE SIZE one: this repo's
# own precedent (``_GATE_ARMED_DEFAULTS["colorspace"]``, ``["speech_presence"]``)
# requires an ~n=120 fleet sample before arming any new threshold — twice, arming
# on a smaller sample built "a signal that fires on everything" (120/120
# duplicate frames; a 96.7% deband ceiling). This fix raises calibration from
# n=1 master / 7 clips to n=3 masters / 11 measurable lines, ALL classified
# correctly — real progress, but still nowhere near n=120. Left UNARMED
# (``_GATE_ARMED_DEFAULTS["narration_presence"] = False``) pending fleet volume;
# arming is an explicit owner decision, not something this fix does on its own.
NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB = -25.0

_RMS_LEVEL_DB_RE = re.compile(r"RMS level dB:\s*(-?(?:inf|[0-9]+(?:\.[0-9]+)?))")

# VC-W14/V3 — TRUE-PEAK CEILING, in dBTP.
#
# ``true_peak_dbtp`` has been MEASURED and PUBLISHED on every report since the
# gate was written (``final_mux_loudness`` measures it, ``post_mux_verify``
# re-exports it, ``_unmeasured_metrics`` tracks its absence) and was NEVER
# COMPARED TO ANYTHING. No violation could be constructed from it at any value.
# A master that clips delivered, and said so on the wire, and nothing read it.
#
# 0.0 dBTP IS DIGITAL FULL SCALE, NOT A TUNED CEILING — which is exactly why this
# gate is STRUCTURAL and needs no fleet distribution to be trustworthy. There is
# no amount of intersample clipping at which shipping the master is the right
# call, and the number is not a judgement about degree: above 0 dBTP a downstream
# lossy transcode reconstructs a waveform that cannot be represented and
# distorts.
#
# ⚠️ IT IS ALSO NOT THE MASTER TARGET. ``settings.master_tp`` (-1.0) is where the
# limiter AIMS, and ``renderer._LIMITER_TP_HEADROOM_DB`` deliberately aims
# ``alimiter`` 0.3 dB BELOW it to absorb a measured 0.07-0.13 dB of intersample
# overshoot. Gating at ``master_tp`` would therefore convict masters the limiter
# handled exactly as designed — a degree judgement against a moving target, i.e.
# the tunable shape this project has repeatedly been burned by. Gating at 0.0
# convicts only a master where the limiter DID NOT RUN or did not hold, which is
# breakage.
TRUE_PEAK_CEILING_DBTP = 0.0

# ---------------------------------------------------------------------------
# GATE TIERS + PER-GATE KILL SWITCHES
# ---------------------------------------------------------------------------
#
# The governing failure mode of this pipeline is that a green run proves
# nothing: it once delivered a 218.47s film against a 30.0s order with every
# stage green. ``gate_pass`` was computed correctly and then decided nothing,
# because enforcement was one whole-gate opt-in (``mastering_gate_mode``) that
# no deployment ever set. This block replaces that single switch with a TIERED,
# per-gate one. The tiering IS the safety design; it is not bookkeeping.
#
# STRUCTURAL — armed by default. These detect ABSENCE or BREAKAGE, not degree,
#   so they need no fleet calibration to be trustworthy: there is no ratio of
#   blank frames, no size of A/V gap, and no amount of missing verification at
#   which shipping the master is the right call. A structural gate can be wrong
#   about a given file (a probe lies), but it cannot be MISCALIBRATED.
#
# TUNABLE — switch wired, DEFAULT OFF. These are quality JUDGEMENTS about
#   degree, decided against a ceiling, and their fleet distributions are
#   UNMEASURED. Arming an uncalibrated ceiling is exactly how this project once
#   built a "signal" that would have fired on 96.7% of masters, and how the
#   duplicate-frame gate convicted 120 of 120 deliveries. Default OFF until the
#   n is measured and cited at the constant (see ``LOUDNESS_GATE_TOLERANCE_LU``).
#
# An UNKNOWN stage falls through to ``tunable``/not-armed on purpose: drift in
# this map must cause a MISSED gate, never a fleet-wide false positive.
#: The tier vocabulary. A closed Literal, not a str, so a typo in
#: ``_GATE_TIERS`` or a new construction site inventing a third tier is a TYPE
#: error rather than a value the consumer silently cannot interpret.
GateTier = Literal["structural", "tunable"]

_DEFAULT_GATE_TIER: GateTier = "tunable"

_GATE_TIERS: dict[str, GateTier] = {
    # structural — absence/breakage
    "av_coverage": "structural",  # the picture ends before the narration does
    "dead_frame": "structural",  # the delivered picture is blank/uniform
    "banding_measurement": "structural",  # the picture was never measured at all
    "verify": "structural",  # no report exists — verification could not run
    "colorspace": "structural",  # the BT.709 tag is absent, not "too small"
    # VC-W14/V3. The master clipped: above 0 dBTP the limiter did not run or did
    # not hold. Full scale is a property of the format, not a tuned ceiling.
    "true_peak": "structural",
    # VC-W14/V6. The job REQUESTED VO-relative bed levelling/ducking and the
    # narration measurement could not be made, so the bed gain and the ducker's
    # open threshold are both literal fallbacks. Absence of the calibration, not
    # a degree of mis-levelling.
    "bed_calibration": "structural",
    # VC-W14/V4. Pass-1 loudness measurement CRASHED, so the master silently fell
    # back to dynamic loudnorm — audibly different gain riding, and until now
    # reported with ``gate_pass=True`` and no note at all. Absence of the
    # measurement, not a degree of anything.
    "measure": "structural",
    # The delivered master carries no speech — only a steady bed of noise. ABSENCE
    # of the deliverable's content, not a degree of quality: a narration film whose
    # narration is missing is undeliverable however well it is normalised. (The
    # THRESHOLD is measured, but what it decides is presence/absence — the same
    # shape as ``dead_frame``, which also compares a measured ratio and is
    # structural because what it reports is a blank picture.)
    "speech_presence": "structural",
    # The delivered master is missing one or more shots' narration entirely —
    # content-aware, per-shot ABSENCE (the clip the compiler was given never
    # appears in the delivered audio), not a degree of quality. Supersedes
    # ``speech_presence`` for the defect that motivated both: see
    # ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB`` for why a whole-file loudness
    # statistic cannot see this and a per-clip cross-correlation can.
    "narration_presence": "structural",
    # tunable — degree, against a ceiling
    "loudness": "tunable",
    # Cadence coherence is decided against ``_CFR_FRAME_UNIFORMITY`` — an uncited
    # ratio, i.e. a degree judgement, not presence/absence.
    "cfr": "tunable",
}

# Per-gate kill switch: the ``Settings`` field (and therefore the env var, upper
# -cased) that arms or disarms each gate. A bad threshold is backed out by
# setting one of these to ``false`` in gitops — never by a code change and a
# rollout. Absent from this map ⇒ the stage cannot be armed at all.
_GATE_KILL_SWITCHES: dict[str, str] = {
    "av_coverage": "mastering_gate_av_coverage_enabled",
    "dead_frame": "mastering_gate_dead_frame_enabled",
    "banding_measurement": "mastering_gate_banding_measurement_enabled",
    "verify": "mastering_gate_verify_enabled",
    "colorspace": "mastering_gate_colorspace_enabled",
    "true_peak": "mastering_gate_true_peak_enabled",
    "measure": "mastering_gate_measure_enabled",
    "bed_calibration": "mastering_gate_bed_calibration_enabled",
    "speech_presence": "mastering_gate_speech_presence_enabled",
    "narration_presence": "mastering_gate_narration_presence_enabled",
    "loudness": "mastering_gate_loudness_enabled",
    "cfr": "mastering_gate_cfr_enabled",
}

# The default each switch carries, duplicated here so ``gate_armed`` still
# answers correctly for a settings object that predates the fields (test doubles,
# a ``SimpleNamespace``, a pickled older Settings) instead of silently reading
# every gate as disarmed — the fail-open direction that made the old
# ``mastering_gate_mode`` decide nothing. Must agree with ``src/config.py``;
# ``tests/unit/test_av_quality_wave13.py`` pins the agreement.
_GATE_ARMED_DEFAULTS: dict[str, bool] = {
    # ARMED. Structural, and measured two-sided (2026-09-10) — see
    # AV_COVERAGE_TOLERANCE_SECONDS. Was already enforcing in BOTH modes before
    # the tiering existed; this preserves that exactly.
    "av_coverage": True,
    # ARMED. Structural: a uniform-luma delivery is a dead render.
    "dead_frame": True,
    # NOT ARMED — a MEASURED REFUSAL, and the one place this wave deliberately
    # stopped short of the brief. It is STRUCTURAL by shape (it reports the
    # ABSENCE of any picture measurement, not a degree) and failing it closed on
    # ``gate_pass`` stays. But arming it does NOT convict a property of the
    # MASTER — it convicts a property of the PROBE: the signalstats pass could
    # not run. That rate has never been measured (n=0), and unlike
    # ``av_coverage``/``dead_frame`` a fire here says nothing about whether the
    # delivered film is good, so a transient ffmpeg failure over a clean master
    # would destroy a paid deliverable with a remediation (re-render) that
    # addresses nothing.
    #
    # EVIDENCE IT FIRES ON UNRELATED CONDITIONS: arming it broke FOUR pre-existing
    # tests that have nothing to do with picture quality (a backpressure test, an
    # idempotency-replay test, an object-key sanitisation test, a bed-mix limiter
    # test) purely because their fixture mp4 is not probe-able. That is the
    # "a signal that fires on everything is not a signal" shape, and the honest
    # answer is to measure the fleet probe-failure rate before arming, not to
    # arm and find out in production.
    #
    # ``verify`` below IS armed and covers the unambiguous half of the same
    # concern: the report is absent ENTIRELY, verification did not run at all.
    "banding_measurement": False,
    # ARMED. Structural: a crashed verification proved nothing about the master,
    # and ``degraded_report()`` makes that indistinguishable from a clean pass.
    "verify": True,
    # NOT ARMED, and this is a MEASURED refusal rather than caution. colorspace
    # is structural by nature (the tag is present or it is not), but the same
    # n=120 fleet sample that retired the duplicate-frame and banding gates
    # measured ``colorspace_ok`` FALSE on 23.3% of completed masters. That is an
    # upstream TAGGING regression to fix at its source; arming it here would
    # fail ~1 delivery in 4 on a defect this service does not cause and cannot
    # remediate, which is the 96.7% trap at a quarter scale. Flip this switch on
    # AFTER the tagging fix lands and the rate is re-measured to ~0.
    "colorspace": False,
    # ARMED (VC-W14/V3). Structural. PROVENANCE: the ceiling is DIGITAL FULL
    # SCALE, not a fleet percentile — see TRUE_PEAK_CEILING_DBTP for why gating
    # at 0.0 rather than at ``master_tp`` is what keeps this out of the tunable
    # class. The delivered master runs through ``alimiter`` aimed 0.3 dB below
    # ``master_tp``, so exceeding 0.0 means that limiter did not run or did not
    # hold; and ``_true_peak_violation`` carries the same unsatisfiable-gate
    # guard as loudness (mastering OFF ⇒ no limiter ⇒ no gate), so the one
    # configuration that can legitimately clip is excluded rather than convicted.
    "true_peak": True,
    # NOT ARMED (VC-W14/V4) — and this is a REASONED refusal, not caution.
    # Structural by shape: pass-1 measurement failed, so the requested two-pass
    # linear master was never possible and the file shipped with dynamic gain
    # riding. But unlike ``av_coverage`` or ``dead_frame``, the resulting file is
    # a SHIPPABLE film — loudnorm still normalised it, just dynamically — and the
    # pass-1 failure rate over this fleet is n=0, UNMEASURED. Arming would refuse
    # paid deliverables at an unknown rate over a defect a viewer may not hear.
    # It still clears ``gate_pass`` and publishes its violation, which is the
    # whole of the defect V4 names: the degradation used to be INVISIBLE
    # (``gate_pass=True``, no note). Arm once the rate exists.
    "measure": False,
    # NOT ARMED (VC-W14/V6) — a deliberate refusal in the shape of
    # ``banding_measurement``. It is STRUCTURAL and it DOES clear ``gate_pass``
    # (that is the whole fix: it was telemetry that decided nothing), but arming
    # it would convict the PROBE, not the master: the condition is a
    # ``final_mux_loudness`` failure over the narration bus, whose fleet rate is
    # n=0 — unmeasured. A transient probe failure would then destroy a paid
    # deliverable whose mix may be perfectly acceptable at the literal fallback
    # levels. Flip on once the probe-failure rate is measured; until then the
    # violation is REPORTED with its tier, which is how that n gets collected.
    "bed_calibration": False,
    # NOT ARMED, and deliberately so even though it is STRUCTURAL and its
    # threshold has a measured two-sided separation (see
    # SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB). The separation was measured over
    # SYNTHESIZED fixtures; the FLEET distribution of
    # ``program_to_noise_floor_db`` over real masters is n=0. Arming a ceiling
    # whose fleet rate is unknown is exactly how this project twice built a signal
    # that fired on everything (120 of 120 duplicate frames; a 96.7% deband
    # ceiling), and a master that IS narrated but reads a narrow gap — a
    # wall-to-wall music bed, a dense sound design — would be destroyed for a
    # defect it does not have.
    #
    # ⛔ A GATE MUST NEVER DESTROY A PAID DELIVERABLE. It REPORTS with its tier and
    # clears ``gate_pass``, which is how the fleet n needed to arm it gets
    # collected — the same path ``colorspace`` and ``bed_calibration`` are on.
    "speech_presence": False,
    # NOT ARMED, same shape as ``speech_presence`` beside it and for the same
    # reason: STRUCTURAL (content is present or it is not) with a threshold
    # measured on real production audio (see
    # ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``). #234 (2026-09-18) fixed the
    # two measurement defects that USED to also block arming this
    # (drift-sensitivity and content-blindness — see that constant's "MEASURED
    # LIMITATIONS" section) and raised calibration to n=3 masters / 11 measurable
    # lines, all classified correctly. Still NOT ARMED on sample size alone: not
    # the n=120 this project requires before arming a threshold at all (the same
    # bar ``colorspace``/``speech_presence`` are held to, for the same reason —
    # see those entries). It ALREADY clears ``gate_pass`` and appears in
    # ``report.violations`` with its tier while this is false, which is how the
    # n gets collected before anyone flips it on.
    "narration_presence": False,
    # NOT ARMED. Tunable + PROVENANCE: NONE (n=0). See
    # LOUDNESS_GATE_TOLERANCE_LU.
    "loudness": False,
    # NOT ARMED. Tunable; ``_CFR_FRAME_UNIFORMITY`` is uncited. Unlike its
    # disarmed siblings this one does not even REPORT a violation while off — see
    # ``_cfr_violation``.
    "cfr": False,
}


def gate_tier(stage: str) -> GateTier:
    """The tier of a gate stage — ``"structural"`` or ``"tunable"``.

    Part of the published report contract: every entry in the report's
    ``violations`` carries its tier so the orchestrator can tell a structurally
    undeliverable master from one that merely missed a tunable ceiling, WITHOUT
    re-deriving intent by string-matching ``rerun_stages``. An unknown stage
    reads as ``tunable`` — drift must under-report severity, never over-report."""
    return _GATE_TIERS.get(stage, _DEFAULT_GATE_TIER)


def gate_armed(settings: Settings, stage: str) -> bool:
    """Whether a violation of ``stage`` must FAIL the compile rather than annotate it.

    Three inputs, in precedence order:

    1. ``mastering_gate_mode='enforce'`` — the whole-gate opt-in. Still honoured,
       and it arms EVERY stage including the tunable and unmapped ones. It is now
       an escalation, not the only way in.
    2. The stage's own kill switch (``_GATE_KILL_SWITCHES``), read off
       ``Settings`` / env. This is the back-out path: a bad threshold is disarmed
       by config, never by a code change.
    3. ``_GATE_ARMED_DEFAULTS`` — used when the settings object does not carry
       the field at all, so a test double or an older Settings reads the real
       default instead of fail-open ``False``.

    A stage with no kill switch can only ever be armed by (1)."""
    if gate_enforced(settings):
        return True
    switch = _GATE_KILL_SWITCHES.get(stage)
    if switch is None:
        return False
    return bool(getattr(settings, switch, _GATE_ARMED_DEFAULTS.get(stage, False)))


class MasteringGateViolation(BaseModel):
    """One gating check that CLEARED ``gate_pass``, with the numbers behind it.

    Carried on ``MasteringGateError`` AND — since the gate arming — published on
    the report itself as ``violations`` (see :func:`violations_payload`).
    ``value``/``limit`` are None for the non-numeric checks (colorspace tag,
    unmeasurable loudness, absent report) — ``detail`` is always populated and is
    what reaches the operator as the job's ``error``.

    ``tier`` is REQUIRED and has no default on purpose: it is half of the
    published contract, and a default would let a new construction site ship an
    unclassified violation that the consumer reads as merely tunable."""

    stage: str
    metric: str
    value: float | None = None
    limit: float | None = None
    tier: GateTier
    detail: str


#: The exact key set published per entry in the report's ``violations``, RE-EXPORTED
#: from the model that defines it (``schemas.MasteringViolation``) rather than
#: retyped here — a second hand-kept copy of a contract is a second place to drift.
#: ``detail`` is deliberately NOT in it (it is prose for the operator's ``error``
#: string, and a consumer that parses it is re-deriving intent from a message
#: again — the exact defect ``violations`` exists to end).
_VIOLATION_CONTRACT_KEYS = VIOLATION_CONTRACT_KEYS


def violations_payload(violations: list[MasteringGateViolation]) -> list[dict[str, object]]:
    """Project violations onto the published contract shape.

    Produced by DUMPING :class:`schemas.MasteringViolation`, so the wire shape is
    whatever that model declares — five keys, in declaration order, ``tier``
    validated against its two-value Literal on the way out. It was a hand-built
    ``dict[str, Any]`` until the ``Any`` ratchet refused it; the model is the
    better shape and the JSON is byte-identical.

    PLAIN DICTS, NOT MODEL INSTANCES, ON PURPOSE. This payload goes into the
    report dict that ``compile_job_store`` persists to a **JSONB** column, and a
    ``BaseModel`` is not JSON-serialisable there. ``MasteringReport.violations``
    is typed as the model, so the entries are re-validated at the response
    boundary — the model is enforced at both ends and the middle stays storable.

    ``mode="json"`` so ``None`` stays ``null`` and the values are wire types,
    never Python objects that happen to round-trip in tests.

    WHY THIS EXISTS AT ALL: this list was built every run (one entry per cleared
    check, with the measured value and the ceiling it breached) and then never
    placed in the returned dict, so the only thing a consumer could see was
    ``rerun_stages`` — a list of bare stage NAMES. The orchestrator therefore
    re-derived the gate's intent by string-matching those names, which cannot
    distinguish a stage that "might help" (``"cfr"`` is appended advisorily) from
    one that failed a gate, and carries neither the measured value nor the limit.
    """
    return [
        MasteringViolation(
            stage=v.stage, metric=v.metric, value=v.value, limit=v.limit, tier=v.tier
        ).model_dump(mode="json")
        for v in violations
    ]


class MasteringGateError(RuntimeError):
    """A cleared mastering gate under ``mastering_gate_mode='enforce'``.

    Raised ONLY in enforce mode — in advisory mode (the default) ``post_mux_verify``
    still never raises. The message names every offending metric with its measured
    value and the ceiling it breached, so the failed job's ``error`` says WHICH
    check failed and by how much rather than "mastering gate failed"."""

    def __init__(self, violations: list[MasteringGateViolation]) -> None:
        self.violations = violations
        joined = "; ".join(v.detail for v in violations)
        super().__init__(f"mastering gate failed (enforce mode): {joined}")

    def violations_payload(self) -> list[dict[str, object]]:
        """The same contract shape the report publishes, for the FAILING path.

        An armed structural gate RAISES instead of returning, so the report that
        would have carried ``violations`` never reaches the caller — the failure
        path is the one where a consumer most needs the structured verdict rather
        than a parsed error string."""
        return violations_payload(self.violations)


def gate_enforced(settings: Settings) -> bool:
    """Whether a cleared gate must FAIL the render rather than annotate it."""
    mode = getattr(settings, "mastering_gate_mode", "advisory") or "advisory"
    return str(mode).strip().lower() == "enforce"


# ---------------------------------------------------------------------------
# Colorspace tagging (pure)
# ---------------------------------------------------------------------------

# Recognised colorspaces → the (primaries, transfer, matrix) triplet ffmpeg
# wants for ``-color_primaries``/``-color_trc``/``-colorspace``. bt709 is the
# fleet default (HD delivery); the others are here so a per-job MasteringProfile
# can retarget without a code change. An unknown / empty value emits NO tags
# (the pre-master behaviour), so a bad value degrades to "untagged", never fails.
_COLORSPACE_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "bt709": ("bt709", "bt709", "bt709"),
    "bt601": ("smpte170m", "smpte170m", "smpte170m"),
    "bt2020": ("bt2020", "bt2020-10", "bt2020nc"),
    "srgb": ("bt709", "iec61966-2-1", "bt709"),
}
_COLORSPACE_DISABLED = {"", "none", "off", "false", "0"}


def normalize_colorspace(raw: str | None) -> str:
    """Lower-case + trim a colorspace value; ``""`` when unset/disabled/unknown."""
    val = (raw or "").strip().lower()
    if val in _COLORSPACE_DISABLED:
        return ""
    return val if val in _COLORSPACE_TRIPLETS else ""


def colorspace_tag_args(settings: Settings) -> list[str]:
    """Additive ``-color_primaries/-color_trc/-colorspace`` args for the configured
    ``render_colorspace`` (default ``bt709``), or ``[]`` when disabled/unknown.

    This is the ONE intended delta versus the pre-master encode: an additive
    ``colr`` atom on the per-shot stream. It carries no re-encode cost and — per
    the empirical check — survives the two downstream ``-c copy`` muxes."""
    triplet = _COLORSPACE_TRIPLETS.get(
        normalize_colorspace(getattr(settings, "render_colorspace", ""))
    )
    if triplet is None:
        return []
    primaries, trc, matrix = triplet
    return [
        "-color_primaries",
        primaries,
        "-color_trc",
        trc,
        "-colorspace",
        matrix,
    ]


# H.264 VUI code points (ITU-T H.264 Table E-3/E-4/E-5) for the ffmpeg colour
# NAMES in ``_COLORSPACE_TRIPLETS``. ``h264_metadata`` takes NUMBERS only — it
# rejects ``colour_primaries=bt709`` outright ("Invalid argument"), so the names
# must be mapped here rather than passed through.
_H264_VUI_CODES: dict[str, int] = {
    "bt709": 1,
    "smpte170m": 6,
    "bt2020": 9,
    "bt2020-10": 14,
    "bt2020nc": 9,
    "iec61966-2-1": 13,
}


def colorspace_bsf_args(settings: Settings) -> list[str]:
    """``-bsf:v h264_metadata=...`` that STAMPS the colour triplet into the
    delivered H.264 SPS VUI, or ``[]`` when disabled/unknown.

    ⛔ **``colorspace_tag_args`` ALONE DOES NOT WORK ON THE PRODUCTION FFMPEG.**
    Measured on ffmpeg **7.1.5** (the exact prod build: ``python:3.14-slim`` +
    Debian trixie ``ffmpeg``), encoding with this repo's verbatim
    ``_video_codec_args``::

        -c:v libx264 -pix_fmt yuv420p -color_primaries bt709 -color_trc bt709
        -colorspace bt709 -preset medium -crf 20 -r 30 -g 60 -keyint_min 30 -vsync cfr

        => color_space=bt709  color_transfer=unknown  color_primaries=unknown

    Only the MATRIX lands. ffmpeg >= 5 propagates the filtergraph frame's colour
    properties onto the encoder context, which clobbers the user-supplied
    ``-color_primaries``/``-color_trc`` (the source frames are UNSPECIFIED); the
    matrix survives because the libx264 wrapper writes ``vui.i_colmatrix``
    separately. On ffmpeg **4.4** the same command tags all three, which is why
    this was invisible locally and false on every delivered master.

    **This is the whole of the 100%-false ``colorspace_ok`` fleet rate, and it is
    at the ENCODE — not at a downstream ``-c:v copy``.** Also measured on 7.1.5:
    a file tagged correctly at the encode keeps all three through an audio-only
    ``-c:v copy`` pass AND through a further ``-c copy`` remux. So fixing the
    encode is sufficient; the copy branches need nothing.

    A bitstream filter is used rather than a ``setparams`` video filter (which
    also works, measured) because it is ONE central addition to
    ``_video_codec_args`` — it needs no surgery on the ``-vf`` / ``-filter_complex``
    chain of every encode site, and it applies identically on the hardware
    encoder branches, which have no filtergraph of their own to hang a filter on.
    It rewrites the SPS only; the coded picture data is untouched.
    """
    triplet = _COLORSPACE_TRIPLETS.get(
        normalize_colorspace(getattr(settings, "render_colorspace", ""))
    )
    if triplet is None:
        return []
    primaries, trc, matrix = triplet
    codes = [_H264_VUI_CODES.get(name) for name in (primaries, trc, matrix)]
    if any(code is None for code in codes):
        # A triplet naming a colour we have no VUI code for: emit nothing rather
        # than a half-stamp. `colorspace_tag_args` still runs, as today.
        return []
    prim_code, trc_code, matrix_code = codes
    return [
        "-bsf:v",
        f"h264_metadata=colour_primaries={prim_code}"
        f":transfer_characteristics={trc_code}"
        f":matrix_coefficients={matrix_code}",
    ]


# ---------------------------------------------------------------------------
# Banding (pure)
# ---------------------------------------------------------------------------

# A smooth gradient that risks 8-bit banding: a WIDE luma span carried with very
# LOW temporal difference (no dithering noise to break up the ramp). Slide /
# screen / AI-image backgrounds are the usual offenders.
_BANDING_YDIF_MAX = 3.0
_BANDING_RANGE_MIN = 60.0


# ---------------------------------------------------------------------------
# Banding — SPATIAL term: the pure pixel detector lives in
# ``src.services.banding_spatial`` (stdlib-only, so the settle pass can run it in
# lightweight worker interpreters). Re-exported here for callers and tests.
# ---------------------------------------------------------------------------

# The [0, 1] multiplier scale: a raw spatial factor at/above this value earns
# the FULL multiplier (1.0); below it the multiplier falls off linearly to 0.
# DELIBERATELY UNCHANGED by the 2026-10-21 recalibration (one variable at a
# time). Against the recalibrated detector it sits ~4x above the worst live
# negative frame (0.0123, max over 2,367 exhaustively decoded frames of the
# three generic masters; 0.0210 = ~2.4x through the 1920 cap) and ~15x below the weakest positive (0.77, x264
# posterized vignette). Raising it would buy negative margin at no positive
# cost on this corpus — a separate decision, alongside ``BANDING_GATE_CEILING``.
_BANDING_SPATIAL_SCALE = 0.05


def banding_score(stats: dict[str, float], *, spatial_factor: float | None = None) -> float:
    """A 0..1 score from averaged signalstats. Higher = wide luma range carried
    with a low frame-to-frame delta. PURE. ``0.0`` when unmeasured.

    ⛔ **WITHOUT ``spatial_factor`` THIS SCORE HAS NO SPATIAL TERM.** Banding
    is a SPATIAL artifact — visible quantisation steps between ADJACENT pixels
    in what should be a smooth gradient. Nothing in the whole-frame formula
    inspects a frame's own pixel neighbourhood for that structure (note 5 adds
    the optional pixel-level multiplier; only frames a caller decoded get it).
    The whole-frame formula is

        span       = clamp((YMAX - YMIN - 60) / (255 - 60), 0, 1)   # luma RANGE
        smoothness = clamp((3.0 - YDIF) / 3.0, 0, 1)                # low TEMPORAL delta
        score      = span * smoothness

    i.e. "wide luma range" times "near-static shot" — both real signals, and
    both CONFOUNDED with banding often enough to have looked like a working
    heuristic, but neither one is banding. ``YDIF`` is a TEMPORAL
    (frame-to-frame) difference used here only as a proxy for dither; a
    genuinely still, richly detailed, perfectly clean shot scores identically
    to a smooth banded ramp, because the two are indistinguishable through
    ``(YMIN, YMAX, YDIF)`` alone. This is also why deband — a SPATIAL filter —
    cannot move this score: it was never wired to anything spatial in the
    first place.

    Measured 2026-08-26 with the real render harness + ``measure.measure_shot_frames``
    over real ffmpeg output, 60 frames each:

        material                                          max score  frac >= 0.15
        smooth 8-bit gradient (banded by construction)         0.810        51.7%
        same gradient + heavy dither (unbandable)               0.000         0.0%
        testsrc2 (textured, no banding)                         0.248        80.0%  FALSE POSITIVE
        mandelbrot (dense detail, near-static, no banding)      0.440        71.7%  FALSE POSITIVE
        life (noisy, high motion)                                0.000         0.0%
        flat grey                                                0.000         0.0%

    (Reproduced locally on a different ffmpeg/libx264 build: mandelbrot maxed
    at 0.533 with 41.7% of frames over ceiling — same false positive, different
    magnitude. See ``tests/render_harness/test_banding_score_false_positive_on_static_detail.py``,
    a KNOWN-DEFECT test that pins this.)

    And on a real delivered master: median ``max_banding_before`` was **0.9128**
    against the **0.15** ceiling (98.7% of frames over it), SSIM before/after
    deband **0.9998** (the filter barely touched the picture), and the score went
    UP on 13 of 13 segments — exactly what you'd expect from a spatial filter
    being unable to move a score that has no spatial term to move.

    ``BANDING_GATE_CEILING = 0.15`` was hardcoded in the original feature PR
    (``219baba``, #73) with a docstring claim of being "default-tuned" and **no
    measurement cited**. Four later commits touched it (see the corrections
    below); all four fixed *satisfiability* of the gate against hand-built
    fixture dicts — none re-derived the ceiling against real content, and no
    test before the one referenced above ever ran real ffmpeg output through
    this function. Treat ``0.15`` as unvalidated, not as a tuned threshold.

    ⚠️ THREE corrections. The first two made the gate that reads this UNSATISFIABLE;
    the third (2026-09-02) closed the spatial-term gap the block above describes:

    1. ``YDIF == 0`` scores 0.0 — it is UNMEASURED, not "perfectly smooth".
       ``YDIF`` is a TEMPORAL (frame-to-frame) difference used here only as a
       proxy for dither, and a frame identical to its predecessor has none to
       report. signalstats emits ``YDIF=0`` for every DUPLICATE frame and always
       for frame 0. Fleet masters run ~90% duplicate frames, so on the old
       formula any textured duplicate scored >= 0.25 — over the 0.15 ceiling —
       *whatever* its gradient, which is the mechanism behind ``gate_pass=false``
       on 96.7% of masters and the endless ``rerun_stages=["deband"]`` → "no
       stage was re-run; degrading". And the remediation could never work:
       ``deband`` is a SPATIAL filter, so it cannot move a TEMPORAL metric.
    2. The span-INDEPENDENT ``+ smoothness * 0.25`` floor is gone. It put the
       score at up to 0.25 — above the ceiling — for a frame whose span factor is
       exactly 0, i.e. a verdict that did not depend on the gradient the score
       claims to measure, and a 0.25 discontinuity at ``_BANDING_RANGE_MIN``.

    3. ✅ THE SPATIAL TERM, added 2026-09-02. Banding is a SPATIAL artifact and
       this function had no spatial input at all — it read "wide luma range x
       near-static shot", i.e. a stillness detector. `YHIGH`-`YLOW` (signalstats'
       10th/90th luma percentiles, already parsed, no extra ffmpeg cost) supplies
       one: a ramp spreads pixels evenly so the 10-90 band covers ~90% of the
       range, while detailed content concentrates them and the ratio collapses.

       Measured, real production functions over real ffmpeg output:

           material                       old max -> new max   fires @0.15
           smooth 8-bit gradient           0.845  ->  0.694    YES  (correct)
           gradient + heavy dither         0.000  ->  0.000    no   (correct)
           testsrc2 (textured)             0.251  ->  0.199    marginal
           mandelbrot (detail, static)     0.530  ->  0.071    no   (FIXED)
           life (noisy, high motion)       0.000  ->  0.000    no   (correct)
           flat grey                       0.000  ->  0.000    no   (correct)

       And on five segments of a real delivered master, where the old score fired
       on 99.4-99.6% of frames: four now score 0.000 and one 0.184.

       ⚠️ `BANDING_GATE_CEILING` is DELIBERATELY UNCHANGED at 0.15. The spatial
       term alone achieves the separation, and moving two things at once would
       make any regression unattributable. The numbers above are the measured
       basis for a future ceiling review — 0.25 would also separate this corpus
       cleanly — but that is a second, separate decision.

    4. ✅ CORRUPT PERCENTILES FAIL CLOSED, 2026-09-02. A non-positive
       ``YHIGH - YLOW`` is now UNMEASURED (``uniformity = 1.0``), exactly like
       the absent-key case, instead of clamping to 0 and scoring the frame 0.0.
       Measured on the pre-fix code:

           input                                                            score
           absent percentiles  {"YMIN":0,"YMAX":255,"YDIF":0.5}            0.8333  FLAGGED
           corrupt/transposed  {..., "YLOW":200, "YHIGH":50}               0.0000  not flagged

       The 90th percentile cannot sit below the 10th, so the second dict cannot
       come from a healthy parse — yet it reported a CLEAN master, fail-OPEN, on
       strictly less information than the absent case beside it. Both fail closed
       now. See the comment at the branch itself.

    ⚠️ TWO KNOWN MISSES, MEASURED AND ACCEPTED (2026-09-02) — documented, not
    fixed. Both score under the 0.15 ceiling:

        material                                     stats                                            score
        partial ramp (sky band ~30%, detail below)   {"YMIN":0,"YMAX":255,"YDIF":0.5,
                                                      "YLOW":70,"YHIGH":150}                          0.0820  MISSED
        limited-range 8-bit gradient                 {"YMIN":80,"YMAX":180,"YDIF":0.5,
                                                      "YLOW":88,"YHIGH":172}                          0.1206  MISSED

    Neither is fixable from these inputs. No WHOLE-FRAME percentile statistic can
    separate the first from clean dense detail: mandelbrot — detailed, unbanded,
    correctly suppressed above — scores **0.071**, and the partial ramp scores
    **0.082**. They are the same number. Catching banding confined to part of a
    frame needs a SPATIALLY LOCALIZED measurement (per-tile / per-region stats),
    which signalstats' whole-frame aggregates cannot provide at any ceiling.

    The second one is a ceiling/exponent question, not a formula question. Its
    three variants, with ``span * smoothness = 0.1709``:

        uniformity exponent   score    fires @0.15
        0 (pre-spatial-term)  0.1709   YES   <- the only variant that catches it
        1 (linear)            0.1436   no
        2 (shipped)           0.1206   no

    ⚠️ The handoff for this change said the LINEAR variant clears the ceiling at
    0.1709. Re-derived here: **0.1709 is the exponent-0 score** — the formula
    with no spatial term at all — and linear is 0.1436, which does NOT clear
    0.15. So the spatial term cost this detection at BOTH exponents, and only
    the ceiling (or a per-region measurement) can recover it.

    Squaring is still what bought the mandelbrot fix (linear 0.194 -> squared
    0.071, see 3. above), so the exponent is paying for a real separation. That
    trade — and the 0.25 figure in 3. — is the input to a future ceiling review.
    That review is NOT performed here, and ``BANDING_GATE_CEILING`` stays 0.15.

    The score is now continuous, monotone in the span, zero wherever the evidence
    is absent, and responsive to the artifact it is named for.

    5. ✅ TRUE SPATIAL TERM (pixel neighbourhood, not whole-frame percentiles),
       added 2026-09-18. Everything above this point — including note 3's
       ``YHIGH``/``YLOW`` "spatial term" — is still a WHOLE-FRAME aggregate: it
       never looks at any pixel's neighbours, so it cannot tell "wide range,
       static, and actually stepped" from "wide range, static, and smooth or
       a hard edge". Measured on two real delivered masters: it fires on 6
       shots (0.5614/0.1678/0.2661/0.2032/0.2093/0.1926) with ZERO visible or
       measurable banding on inspection — a mostly-white page with a small
       logo, and the hard edge between a white panel and a grey letterbox.
       ``spatial_factor``, an OPTIONAL 0..1 confidence from
       ``spatial_banding_factor`` (native-resolution staircase detection —
       wide flat plateaus separated by slope-corrected steps >= 2 levels,
       recalibrated 2026-10-21 after the first version scored every smooth
       8-bit gradient as banded; see the module comment above that function),
       is folded in as a MULTIPLIER via
       ``min(1.0, spatial_factor / _BANDING_SPATIAL_SCALE)``: at or above the
       scale it leaves the whole-frame score untouched (1.0x), and below it
       attenuates toward 0 as the pixel evidence for real staircase structure
       disappears. ``spatial_factor=None`` (the default — no frame decode was
       paid for this frame) leaves this function BYTE-IDENTICAL to before.

       ⚠️ A MAX OVER FRAMES MUST ONLY BE REPORTED OVER CORRECTED FRAMES.
       ``post_mux_verify`` reports ``max(banding_score(f, factor))`` across
       every frame, but only decodes a few. Correcting just the raw-worst frame
       hands the max to the next-worst UNcorrected frame at its raw value —
       that is how a showcase master still reported 0.2661 after this term
       landed. ``_settle_banding_spatial_factors`` decodes until the reported
       max is a corrected frame (or no un-decoded frame's raw score exceeds it,
       which is exact because the multiplier only attenuates).

       On the two real masters above, folding in the measured spatial factor
       drops all 6 false positives to 0.0002-0.037 combined score — none
       within an order of magnitude of ``BANDING_GATE_CEILING``. On a
       synthesized 8/16/32-level posterized ramp (the material this gate
       exists to catch, encoded through this service's own
       libx264/crf=18/yuv420p path) the spatial factor is 0.70-1.00,
       leaving the whole-frame score UNCHANGED (multiplier saturates at
       1.0). Full labelled-set numbers: PR description / commit message.
       (Those 2026-09-18 numbers are for the FIRST detector; the 2026-10-21
       recalibration's labelled set lives in
       ``tests/unit/test_banding_spatial_calibration.py`` and the module
       comment above ``spatial_banding_factor``.)
    """
    if not stats:
        return 0.0
    ydif = stats.get("YDIF")
    if ydif is None:
        return 0.0
    ydif = float(ydif)
    # Duplicate frame / frame 0: no temporal signal was measured at all (see 1. above).
    if ydif <= 0.0:
        return 0.0
    yrange = float(stats.get("YMAX", 0.0)) - float(stats.get("YMIN", 0.0))
    if yrange < _BANDING_RANGE_MIN or ydif >= _BANDING_YDIF_MAX:
        return 0.0
    # yrange in [_BANDING_RANGE_MIN, 255] → span factor in [0, 1];
    # ydif in (0, _BANDING_YDIF_MAX) → smoothness factor in (0, 1).
    span = min(1.0, (yrange - _BANDING_RANGE_MIN) / (255.0 - _BANDING_RANGE_MIN))
    smoothness = max(0.0, (_BANDING_YDIF_MAX - ydif) / _BANDING_YDIF_MAX)
    # THE SPATIAL TERM. `YHIGH`/`YLOW` are signalstats' 10th/90th luma percentiles,
    # already parsed by `measure.parse_signalstats_per_frame` (its regex captures
    # every `lavfi.signalstats.*` key), so this costs NO extra ffmpeg work.
    #
    # A banded RAMP spreads its pixels evenly across its luma range, so the
    # 10-90 band covers nearly all of it. Detailed content concentrates most
    # pixels in a narrow band with a few bright/dark outliers stretching
    # YMIN/YMAX, so the ratio collapses. Measured over 60 frames each:
    #     smooth gradient  0.914   <- the case deband exists for
    #     testsrc2         0.899
    #     mandelbrot       0.390   <- dense detail, near-static, NO banding
    #     life             0.078
    # SQUARED, and that is measured too, not stylistic: linear leaves
    # mandelbrot at 0.194 — still over the 0.15 ceiling — while squared puts it
    # at 0.071 and leaves the true gradient at 0.694.
    #
    # ⚠️ ABSENT is UNMEASURED, not zero. `.get(key, 0.0)` on both percentiles
    # would make a stats dict that carries no `YHIGH`/`YLOW` score exactly 0.0 —
    # i.e. a measurement gap would SILENTLY DISARM the gate on every frame, the
    # fail-open direction. `min_luma_spread` two functions down already refuses
    # that trade explicitly ("a measurement gap is never counted as zero
    # spread"). So when the percentiles are absent, fall back to the
    # pre-spatial-term score (uniformity 1.0) and keep flagging. Real
    # signalstats always emits both, so this path is reachable only from a
    # synthetic stats dict or a truncated parse.
    #
    # ⚠️ A NON-POSITIVE `YHIGH - YLOW` IS UNMEASURED TOO, for exactly the same
    # reason, and this is the branch that used to disagree with the one above.
    # The 90th percentile cannot sit below the 10th: a healthy signalstats parse
    # can never produce it, so `YHIGH < YLOW` means the values are corrupt or
    # transposed — that is BAD INPUT, not a clean picture. Clamping the negative
    # difference to 0 (the old `max(0.0, ...)`) made uniformity 0 and the whole
    # score 0.0, i.e. a corrupt parse reported a CLEAN master and disarmed the
    # gate — fail-OPEN on strictly LESS information than the absent-key case
    # three lines up, which fails CLOSED. Measured on the pre-fix code:
    #     {"YMIN":0,"YMAX":255,"YDIF":0.5}                       -> 0.8333 FLAGGED
    #     {..., "YLOW":200, "YHIGH":50}  (corrupt/transposed)    -> 0.0000 not flagged
    # Both branches now fall back to `uniformity = 1.0` and keep flagging, so the
    # policy "absent/impossible spatial evidence is UNMEASURED, never zero" holds
    # uniformly. The UPPER clamp at 1.0 stays: `p90 - p10 > yrange` is a
    # legitimate rounding artifact of averaged per-frame stats, it still means
    # "the band covers the whole range", and it correctly stays flagged.
    if "YHIGH" not in stats or "YLOW" not in stats:
        uniformity = 1.0
    else:
        p90_p10 = float(stats["YHIGH"]) - float(stats["YLOW"])
        uniformity = min(1.0, p90_p10 / yrange) if p90_p10 > 0.0 else 1.0
    score = span * smoothness * uniformity * uniformity
    if spatial_factor is not None:
        # See note 5 above: no evidence of real pixel-level staircase
        # structure (factor near 0) attenuates toward 0; at/above the scale
        # the whole-frame score passes through unchanged.
        score *= min(1.0, max(0.0, spatial_factor) / _BANDING_SPATIAL_SCALE)
    return round(min(1.0, score), 4)


def banding_risk(stats: dict[str, float], *, threshold: float = BANDING_GATE_CEILING) -> bool:
    """True when ONE frame's (or one average's) banding score crosses ``threshold``.

    Default-tuned so genuinely clean / textured content (the common case) stays
    a single-pass byte-identical encode; only a smooth-gradient shot trips.

    ⚠️ Feed this per-frame stats, not a segment average — see
    ``banding_risk_frames``, which is what the deband trigger uses.
    """
    return banding_score(stats) >= threshold


def banding_risk_frames(
    frames: list[dict[str, float]], *, threshold: float = BANDING_GATE_CEILING
) -> bool:
    """True when the WORST frame of a segment reads as banded → apply deband.

    This is the deband TRIGGER, and it deliberately mirrors the gate
    (``_banding_gate_pass``: ``max(scores) < BANDING_GATE_CEILING``) frame-for-
    frame and constant-for-constant.

    ⚠️ They used to disagree, and the disagreement made the gate UNSATISFIABLE.
    The trigger read ``measure_shot_stats`` — signalstats AVERAGED over the whole
    segment — while the gate read ``measure_shot_frames`` and took the per-frame
    MAX, both against 0.15. A segment banded in a minority of frames therefore
    averaged BELOW the trigger (deband never ran) and maxed ABOVE the gate
    (``gate_pass=False``, ``rerun_stages=["deband"]``). Re-running the named stage
    re-measured the same average and again did nothing, so the remediation the
    gate asked for could not possibly satisfy it: under
    ``mastering_gate_mode=enforce`` that is a permanently-failing render, and
    under advisory a permanent false red.

    Same whole-file-vs-per-frame class already fixed once in this service on the
    GATE side; this is the TRIGGER side of the same bug. Costs nothing extra:
    ``measure_shot_frames`` is the same single discarded ffmpeg pass, parsed
    per-frame instead of averaged.
    """
    if not frames:
        return False
    return max(banding_score(f) for f in frames) >= threshold


def _worst_luma_spread(frames: list[dict[str, float]]) -> float | None:
    """The SMALLEST per-frame intra-frame luma spread (``YMAX - YMIN``) across the
    delivered file, or ``None`` when no frame carried both luma keys. PURE.

    Per-frame (not whole-file-averaged): a master that is blank for only PART of
    its length averages to a wide spread and escapes a mean-based check, but its
    worst frame still collapses to ~0. Mirrors the per-frame ``banding_score``
    gate just above — a localized defect must survive to the gate, not be averaged
    away. Frames missing ``YMIN``/``YMAX`` are skipped (a measurement gap is never
    treated as dead)."""
    spreads = [float(f["YMAX"]) - float(f["YMIN"]) for f in frames if "YMAX" in f and "YMIN" in f]
    return min(spreads) if spreads else None


def _dead_frame_ratio(frames: list[dict[str, float]], floor: float) -> float | None:
    """Fraction of MEASURED frames whose intra-frame luma spread is below ``floor``,
    or ``None`` when no frame carried both luma keys. PURE.

    The denominator is measured frames only, so a partial measurement gap cannot
    dilute the ratio toward "alive"."""
    spreads = [float(f["YMAX"]) - float(f["YMIN"]) for f in frames if "YMAX" in f and "YMIN" in f]
    if not spreads:
        return None
    return sum(1 for s in spreads if s < floor) / len(spreads)


def _dead_frame(ratio: float | None, min_ratio: float) -> bool:
    """True when ENOUGH of the master is uniform/blank to call it DEAD. PURE.

    ``duplicate_frame_ratio`` CANNOT discriminate a dead master from a legitimate
    static screen (both ~1.0 duplicate). Intra-frame luma spread can: a blank/dead
    frame has spread ~0, a static UI screen (black text on white) ~200.

    Gated on a RATIO, not on the single worst frame. The previous rule was
    ``min(spreads) < floor`` — one frame below the floor failed the entire film,
    and a fade-to-black, a cut-to-black between scenes, or a dark title card all
    produce such frames legitimately. ``signalstats`` measures EVERY frame (no
    ``select``/fps filter in ``_signalstats_cmd``), so at 30fps a 1s fade is ~1.4%
    of a 72s master and a 2s fade ~2.8%, while the case this gate exists to catch —
    a master blank for PART of its length — sits far above the default 10%
    (half-blank = 50%). The worst-frame value is still REPORTED as ``luma_spread``;
    it just no longer gates on its own.

    Returns ``False`` when ``ratio`` is ``None`` — a measurement gap must NEVER be
    gated as dead. A ``min_ratio`` of 0.0 restores the old any-single-frame rule.
    """
    if ratio is None:
        return False
    if min_ratio <= 0.0:
        return ratio > 0.0
    return ratio >= min_ratio


# ⛔ THERE IS NO ``_duplicate_frames`` PREDICATE ANY MORE, AND THERE MUST NOT BE
# ONE AGAIN. The duplicate-frame ratio is TELEMETRY, never a gate.
#
# DUPLICATE FRAMES ARE INTENTIONAL DWELL, not a defect. The pipeline holds each
# slide/still for a configured dwell (6500ms) and Playwright's capture runs at a
# hardcoded 25fps, so a *correct* master is overwhelmingly duplicate frames by
# construction. That is a standing project decision, not a tuning opinion.
#
# MEASURED over n=120 completed masters, the WHOLE fleet, 2026-08-27→09-11:
#
#     duplicate_frame_ratio   min .643 / p10 .834 / p50 .896 / p90 .997 / max .998
#     fired against the 0.10 ceiling on   120 of 120 masters (100%)
#
# The MINIMUM observed ratio is 6.4x the ceiling. A check that convicts every
# single delivery is not a gate: it is the reason ``gate_pass`` meant nothing,
# because a verdict that is always False carries no information at all.
#
# It was deleted rather than neutralised with ``ceiling=1.0``. A disabled-but-
# present predicate reads as a tuning accident and invites the next session to
# "re-arm" it; there is no ceiling at which this measurement gates correctly,
# because the ratio measures a FEATURE of the format. ``duplicate_frame_ratio``
# is still probed and still published in the mastering report (it is genuinely
# useful telemetry — it is how the dwell/fps relationship is observed), and
# ``settings.render_duplicate_frame_ratio_ceiling`` went with the predicate so
# nothing is left to re-arm. If a future "the picture must MOVE" gate is wanted,
# it needs a measurement that can tell held dwell from a frozen render — which
# this ratio, by construction, cannot.


def deband_filter() -> str:
    """The ffmpeg ``deband`` filter fragment applied to a shot that trips
    ``banding_risk``. PURE. Conservative thresholds — enough to break an 8-bit
    ramp without smearing real detail.

    ⚠️ ``settings`` WAS A DEAD PARAMETER and is gone (2026-09-15). It took a
    ``Settings`` and returned a constant string, which reads as "the thresholds
    are configurable" — they are not, and no knob for them exists in
    ``config.py``. The honest signature is the one that admits the constant.
    Deriving the thresholds from settings instead was the alternative and was
    NOT taken: there is no measurement anywhere in this repo relating a
    ``1thr``/``range``/``blur`` value to a delivered picture, so a knob would
    ship four more unvalidated constants (see ``BANDING_GATE_CEILING``'s history
    in ``banding_score``). Add the knob when there is a measurement to set it
    from."""
    return "deband=1thr=0.02:2thr=0.02:3thr=0.02:4thr=0.02:range=16:blur=1:coupling=0"


def deband_enabled(settings: Settings) -> bool:
    """Whether the conditional deband stage runs at all. ``auto``/``on`` measure
    per-seg and deband tripped shots; ``off``/``false`` skips measurement so the
    encode is unconditionally single-pass."""
    mode = (getattr(settings, "render_deband", "auto") or "auto").strip().lower()
    return mode in {"auto", "on", "true", "1", "yes"}


def source_match_enabled(settings: Settings) -> bool:
    """Whether the T9 per-shot source-matching stage runs at all. ``auto``/``on``
    measure each rendered segment and nudge outlier shots toward the film's median
    luma; ``off``/``false`` skips the pass so segments stay single-pass encodes.
    Additionally gated per-request on ``finish.grade.match_sources``."""
    mode = (getattr(settings, "render_source_match", "auto") or "auto").strip().lower()
    return mode in {"auto", "on", "true", "1", "yes"}


# ---------------------------------------------------------------------------
# CRF / bitrate ladder (pure)
# ---------------------------------------------------------------------------

# Per-content CRF nudges from the configured default. flat/detail keep the
# default (sharp text); motion tolerates +1 (footage hides it, saves bitrate).
# By construction the DEFAULT/unknown cell is the render_crf itself → identical.
_CRF_DELTA_BY_CLASS: dict[str, int] = {"unknown": 0, "flat": 0, "detail": 0, "motion": 1}


def _render_crf_default() -> int:
    """``render_crf``'s declared default, read from the field rather than
    restated.

    ⚠️ The fallback below used to be the literal ``20``. ``config.py`` moved
    ``render_crf``'s default to **18** (the near-transparent cell for the 2K
    canvas this fleet renders at) and this restatement was not moved with it, so
    a ``Settings``-shaped stub with no ``render_crf`` silently laddered a
    DIFFERENT cell than the real service. Reading the field default cannot
    drift."""
    from src.config import Settings

    return int(Settings.model_fields["render_crf"].default)


def encode_ladder(
    settings: Settings, content: str = "unknown", *, intermediate: bool = False
) -> tuple[int, list[str]]:
    """Return ``(crf, vbv_args)`` for one encode.

    The DEFAULT cell (``content='unknown'`` / no VBV configured / not
    ``intermediate``) returns exactly ``settings.render_crf`` and ``[]`` — a
    pixel-identical, byte-identical encode versus the pre-master path.
    ``-maxrate``/``-bufsize`` are emitted ONLY when ``render_bitrate_mode``
    selects a VBV mode and ``render_vbv_maxrate`` is set; a per-content CRF
    nudge applies only when a non-default content class is passed in (the
    per-shot encode passes ``unknown`` → default, staying identical; the ladder
    is content-aware for a future measured path).

    ``intermediate=True`` asks for the cell of a re-encode whose output is a
    TEMP file, not the deliverable: ``render_crf - render_intermediate_crf_delta``
    (floored at 0), and **no VBV**.

    ⛔ WHY IT EXISTS. The delivered picture passed through THREE x264
    generations, every one of them calling this function with the same default
    cell: the segment encode, the deband/source-match re-encode
    (``renderer._deband_segment`` / ``_regrade_segment`` / ``_polish_segment``),
    and the whole-film grade or bed pass. Generations 2 and 3 re-quantise
    material x264 has ALREADY quantised at that exact cell, so re-running them
    at the delivery CRF throws detail away a second and third time for nothing.
    Running the intermediate lower makes that pass near-transparent to
    generation 1 — the picture handed to the final encode is the one generation
    1 produced, rather than a degraded copy of it.

    The VBV args are dropped on the intermediate deliberately, not as an
    oversight: ``-maxrate``/``-bufsize`` model a DELIVERY constraint (a player's
    decode buffer). Imposing a delivery bitrate ceiling on a temp file would cap
    exactly the extra bits the lower CRF exists to keep, i.e. it would silently
    undo this. The delivered encode still gets its full VBV cell.

    ⚠️ The delta is a CONFIGURED knob (``render_intermediate_crf_delta``,
    default 6), not a threshold derived from delivered-picture measurement. 6
    CRF points is x264's rule-of-thumb ~2x bitrate step; it is a
    quality-headroom choice, and setting the knob to 0 restores the previous
    behaviour byte-for-byte. It is deliberately NOT applied on the hardware
    encoder branches, which are bitrate-driven and ignore CRF entirely —
    ``renderer._video_codec_args`` documents that limitation at the branch."""
    base_crf = int(getattr(settings, "render_crf", _render_crf_default()))
    crf = min(51, max(0, base_crf + _CRF_DELTA_BY_CLASS.get(content, 0)))
    if intermediate:
        delta = int(getattr(settings, "render_intermediate_crf_delta", 0) or 0)
        return max(0, crf - delta), []
    vbv: list[str] = []
    mode = (getattr(settings, "render_bitrate_mode", "auto") or "auto").strip().lower()
    maxrate = (getattr(settings, "render_vbv_maxrate", "") or "").strip()
    if mode in {"vbv", "cbr", "capped"} and maxrate:
        bufsize = (getattr(settings, "render_vbv_bufsize", "") or "").strip()
        vbv = ["-maxrate", maxrate, "-bufsize", bufsize or _double_rate(maxrate)]
    return crf, vbv


def _double_rate(rate: str) -> str:
    """Derive a sane ``-bufsize`` (~2x ``-maxrate``) when one is not configured.

    Accepts an ffmpeg rate string (``10M``, ``8000k``, ``6000000``); on anything
    unparseable it echoes the maxrate back (bufsize == maxrate is a valid VBV)."""
    rate = rate.strip()
    suffix = ""
    if rate and rate[-1] in "kKmMgG":
        suffix = rate[-1]
        num = rate[:-1]
    else:
        num = rate
    try:
        return f"{round(float(num) * 2)}{suffix}"
    except ValueError:
        return rate


# ---------------------------------------------------------------------------
# ffmpeg-touching verification (best-effort / never-raise)
# ---------------------------------------------------------------------------


async def _none() -> None:
    """A ready-made ``None`` awaitable, so the concurrent probe round in
    ``post_mux_verify`` keeps a FIXED five-element shape whether or not a
    colorspace is expected. Building the gather list conditionally would make the
    tuple unpack positional-dependent — the kind of edit that silently assigns
    ``dup_ratio`` to ``cfr_ok`` when someone later adds a probe."""
    return


async def _probe_colorspace(path: Path | str) -> tuple[str, str, str] | None:
    """ffprobe the delivered file's ``(primaries, transfer, matrix)`` colour triplet,
    or ``None`` on any probe/parse failure. Best-effort — never raises (mirrors
    ``probe_dimensions``).

    WHY THE WHOLE TRIPLET AND NOT JUST ``color_space``: this probe feeds the gate,
    and the gate compares it against ``_COLORSPACE_TRIPLETS[render_colorspace]`` —
    the exact three values ``colorspace_tag_args`` told the encoder to write. The
    single ``color_space`` field is the AVColorSpace MATRIX name, which is a
    DIFFERENT vocabulary from the colorspace NAME the setting carries: bt601
    encodes matrix ``smpte170m`` and bt2020 encodes ``bt2020nc``, so a name-vs-
    matrix comparison could only ever succeed for bt709 and reported a correctly
    tagged bt601/bt2020/srgb master as ``colorspace_ok=False`` forever (re-running
    re-emitted the identical tag, so it never cleared). Reading all three also
    keeps ``srgb`` and ``bt709`` distinguishable: they share matrix AND primaries
    ``bt709`` and differ ONLY in transfer (``iec61966-2-1`` vs ``bt709``), so a
    matrix-only probe cannot tell an sRGB deck from a Rec.709 one.

    Values are returned verbatim (ffmpeg's own names, incl. the literal
    ``"unknown"`` an untagged stream reports) — normalising them here would let an
    untagged master masquerade as a tagged one."""
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_primaries,color_transfer,color_space",
                # KEYED output (no ``nokey=1``): ffprobe emits these fields in ITS
                # own order. MEASURED, not assumed: it prints
                # ``color_space, color_transfer, color_primaries`` — the exact
                # REVERSE of the order requested above — so positional parsing into
                # ``(primaries, transfer, matrix)`` would transpose PRIMARIES and
                # MATRIX. Transfer sits in the middle either way and is the one
                # component a swap leaves intact. (An earlier note here said
                # "transfer and primaries"; that named the wrong pair.)
                # The swap is INVISIBLE for three of the four legal values, which
                # is what makes it dangerous: bt709 and bt601 have uniform
                # triplets, and srgb's primaries and matrix are both ``bt709``.
                # Only bt2020 (``bt2020`` primaries vs ``bt2020nc`` matrix) would
                # mis-read — reintroducing a gate unsatisfiable for exactly one
                # legal value, the same defect class this probe was fixed for.
                # Keying the output removes the ordering dependency entirely.
                "-of",
                "default=noprint_wrappers=1",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering._probe_colorspace.failed", path=str(path), error=str(exc))
        return None
    fields: dict[str, str] = {}
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    try:
        return (
            fields["color_primaries"],
            fields["color_transfer"],
            fields["color_space"],
        )
    except KeyError:
        # A stream that reported none of the three — treat as unprobed, not as a
        # mismatch, so the gate degrades to None rather than failing a good master.
        return None


async def av_stream_seconds(path: Path | str) -> tuple[float | None, float | None]:
    """The delivered master's ``(video_seconds, audio_seconds)``, probed PER STREAM.

    THE MEASUREMENT THE COMPILER NEVER TOOK. ``probe_output_duration`` reports a
    single scalar reduced with ``min(stream, container)``, so a master whose
    picture stops before its narration reports ONE number that agrees with the
    plan, and the gap is arithmetically invisible to every consumer of it. Two
    numbers, compared, is the only shape that can see it.

    ``None`` on either side means UNMEASURED, never zero — the gate declines to
    fire on a gap rather than failing a master it could not read (the same
    no-gate-on-a-measurement-gap policy as ``dead_frame`` / ``duplicate_frames``),
    and ``_unmeasured_metrics`` reports that it declined.

    Per-stream fallbacks, because a duration TAG is not guaranteed:
    * video — ``renderer._video_stream_packet_end`` measures the picture from
      packet timestamps. NOT ``nb_frames / fps``: sources here are VFR.
    * audio — the container duration, which is the MAXIMUM over all streams. That
      can only ever over-report the audio, and over-reported audio SHRINKS the
      measured gap, so the fallback can bias this gate towards silence but never
      towards a false accusation."""
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(out)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail on the probe
        log.warning("mastering.av_stream_seconds.failed", path=str(path), error=str(exc))
        return None, None
    if not isinstance(payload, dict):
        return None, None
    streams = payload.get("streams")
    streams = streams if isinstance(streams, list) else []
    container = renderer._duration_field(payload.get("format"))

    def _first(codec_type: str) -> dict[str, Any] | None:
        for stream in streams:
            if isinstance(stream, dict) and stream.get("codec_type") == codec_type:
                return stream
        return None

    video_stream, audio_stream = _first("video"), _first("audio")
    video_seconds: float | None = None
    if video_stream is not None:
        tagged = renderer._duration_field(video_stream)
        measured = (
            tagged
            if tagged > 0.0
            else await renderer._video_stream_packet_end(Path(path), container)
        )
        video_seconds = measured if measured > 0.0 else None
    audio_seconds: float | None = None
    if audio_stream is not None:
        tagged = renderer._duration_field(audio_stream)
        measured = tagged if tagged > 0.0 else container
        audio_seconds = measured if measured > 0.0 else None
    return video_seconds, audio_seconds


# One sample of slack on a packet interval: container timebases round.
_AUDIO_PTS_GAP_TOLERANCE_SAMPLES = 1


def audio_pts_gaps(packet_pts: Sequence[int], sample_rate: int) -> tuple[int, float]:
    """PURE: ``(count, total_ms)`` of the holes in an audio stream's packet timeline.

    ``packet_pts`` are the stream's packet timestamps IN SAMPLES, in order. Every
    packet of a constant-frame-size codec (AAC 1024, ALAC 4096) starts exactly one
    frame after the previous one, so the nominal interval is the most common one
    and any LONGER interval is time the track has no samples for — a hole a player
    renders as a dropout or skips, desyncing everything after it.

    ⛔ Measured on the INTERVAL, not on ``pts[n] + duration[n] != pts[n+1]``. An MP4
    has no way to store a gap: the muxer writes it as a longer duration on the
    packet BEFORE it, so that sum closes exactly and reports a clean stream. On
    production master 01a10e19 it found 0 discontinuities across two holes (5.4 ms
    and 89.1 ms) that the interval reading names: 1282 and 5302 against 1024.

    A SHORTER interval is not a hole (a short last frame before a stream-copy join)
    and is ignored, as is anything within ``_AUDIO_PTS_GAP_TOLERANCE_SAMPLES``.
    """
    deltas = [b - a for a, b in itertools.pairwise(packet_pts)]
    if not deltas or sample_rate <= 0:
        return 0, 0.0
    nominal = Counter(deltas).most_common(1)[0][0]
    excess = [d - nominal for d in deltas if d - nominal > _AUDIO_PTS_GAP_TOLERANCE_SAMPLES]
    return len(excess), round(sum(excess) * 1000.0 / sample_rate, 3)


async def audio_pts_gap_report(path: Path | str) -> tuple[int | None, float | None]:
    """``audio_pts_gaps`` of the delivered master's first audio stream.

    READ-ONLY and ADVISORY: it gates nothing and no switch reads it. ``(None,
    None)`` means UNMEASURED (probe failed, or the master has no audio stream) —
    never "zero gaps". Never raises, so it is safe in ``post_mux_verify``'s bare
    ``gather``."""
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,time_base:packet=pts",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(out)
        streams = payload.get("streams") or []
        if not streams:
            # No audio stream at all (a video-only master). That is UNMEASURED, and
            # it is an expected input — not a probe failure to warn about.
            return None, None
        stream = streams[0]
        sample_rate = int(stream["sample_rate"])
        num, den = (int(x) for x in str(stream["time_base"]).split("/", 1))
        pts = sorted(
            int(p["pts"]) * num * sample_rate // den
            for p in payload["packets"]
            if isinstance(p, dict) and "pts" in p
        )
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering.audio_pts_gaps.failed", path=str(path), error=str(exc))
        return None, None
    if len(pts) < 2:
        return None, None
    return audio_pts_gaps(pts, sample_rate)


#: A video packet interval counts as irregular when it differs from the nominal
#: (most common) one by MORE than this fraction of it. Half a frame, because the
#: defect is quantised and the noise is not: a missing slot is a whole extra
#: frame, timebase rounding is a sliver of one. Delivered master 01a11032
#: (2026-10-06, 30 fps, timebase 1/15360): 950 intervals of 512 ticks and ONE of
#: 1024 (pts 20.233333 -> 20.300000, 66.667 ms against 33.333) — an excess of
#: 1.0 nominal, twice this bar. The coarsest timebase in play (Matroska, 1 ms)
#: jitters a 33.333 ms interval by under 1 ms, i.e. 0.03 nominal, a sixteenth of
#: it. Nothing legitimate lands between the two.
_VIDEO_PTS_GAP_TOLERANCE_FRAMES = 0.5

#: How many irregular intervals ``video_pts_gaps`` reports the POSITION of. The
#: count is always complete; the positions are a diagnostic, and a genuinely VFR
#: file would otherwise put one entry per frame into the report.
_VIDEO_PTS_GAP_POSITIONS_MAX = 5


def video_pts_gaps(
    packet_pts: Sequence[int], time_base: Fraction
) -> tuple[int, list[dict[str, float]]]:
    """PURE: ``(count, first_positions)`` of the irregular intervals in a video
    stream's presentation timeline.

    ``packet_pts`` are the stream's packet timestamps in ``time_base`` ticks, in
    ANY order: packets arrive in decode order, which B-frames reorder, so they are
    sorted here and a reordered-but-regular stream reads as clean. A CFR stream
    then steps by one identical interval; the nominal interval is the most common
    one, and every interval off it by more than
    ``_VIDEO_PTS_GAP_TOLERANCE_FRAMES`` of it is counted — LONGER (a frame slot
    with no frame: the picture holds for two frames, a visible stutter) and
    SHORTER (two frames crowded into one slot) alike, since both break the cadence
    ``cfr_ok`` vouches for.

    ⛔ Measured on the pts INTERVAL, never on the declared packet duration — the
    same trap as ``audio_pts_gaps``. On master 01a11032 all 952 packets declare
    ``duration=512`` including the one before the hole, so a duration-uniformity
    scan of the WHOLE file (``_frame_durations_are_uniform`` reads those) reports
    100% uniform across a missing frame.

    Each position is ``{"at_seconds": pts of the frame BEFORE the irregular
    interval, "interval_ms": the interval}``, in presentation order.
    """
    ordered = sorted(packet_pts)
    deltas = [b - a for a, b in itertools.pairwise(ordered)]
    if not deltas:
        return 0, []
    nominal = Counter(deltas).most_common(1)[0][0]
    if nominal <= 0:
        # Most frames share a timestamp: there is no cadence to measure against.
        return len(deltas), []
    tolerance = nominal * _VIDEO_PTS_GAP_TOLERANCE_FRAMES
    seconds = float(time_base)
    irregular = [(i, d) for i, d in enumerate(deltas) if abs(d - nominal) > tolerance]
    positions = [
        {
            "at_seconds": round(ordered[i] * seconds, 6),
            "interval_ms": round(d * seconds * 1000.0, 3),
        }
        for i, d in irregular[:_VIDEO_PTS_GAP_POSITIONS_MAX]
    ]
    return len(irregular), positions


async def video_pts_gap_report(
    path: Path | str,
) -> tuple[int | None, list[dict[str, float]] | None]:
    """``video_pts_gaps`` over EVERY packet of the delivered master's first video
    stream (a demux, no decode: ~1 packet line per frame).

    ``(None, None)`` means UNMEASURED (probe failed, no video stream, or fewer
    than two timestamped packets) — never "zero gaps". Never raises, so it is safe
    in ``post_mux_verify``'s bare ``gather``."""
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=time_base:packet=pts",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(out)
        num, den = (int(x) for x in str(payload["streams"][0]["time_base"]).split("/", 1))
        time_base = Fraction(num, den)
        pts = [int(p["pts"]) for p in payload["packets"] if isinstance(p, dict) and "pts" in p]
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering.video_pts_gaps.failed", path=str(path), error=str(exc))
        return None, None
    if len(pts) < 2:
        return None, None
    return video_pts_gaps(pts, time_base)


def _cfr_verdict(rate_coherent: bool | None, video_pts_gap_count: int | None) -> bool | None:
    """PURE: ``cfr_ok`` from the two independent readings of cadence.

    ``rate_coherent`` (``framerate_coherence``) reads the stream HEADERS plus the
    declared durations of the first seconds; it cannot see a hole in the timeline.
    ``video_pts_gap_count`` reads every pts interval; it cannot see a stream that
    is perfectly regular at the wrong rate. CFR needs both:

    * either one MEASURED bad -> ``False`` (one bad reading is a verdict, whatever
      the other could or could not see);
    * both measured good -> ``True``;
    * otherwise ``None``. In particular coherent headers over an UNMEASURED
      timeline are NOT ``True``: that is exactly the blind pass master 01a11032
      got, and ``None`` puts ``cfr_ok`` in the report's ``unmeasured`` channel.
    """
    if rate_coherent is False or (video_pts_gap_count is not None and video_pts_gap_count > 0):
        return False
    if rate_coherent is None or video_pts_gap_count is None:
        return None
    return True


def _av_gap_seconds(video_seconds: float | None, audio_seconds: float | None) -> float | None:
    """PURE: how many seconds of narration play over no picture. ``None`` when
    either side is unmeasured, or when the master carries no audio stream at all
    (a music-less silent reel has nothing to outrun the picture)."""
    if video_seconds is None or audio_seconds is None:
        return None
    return audio_seconds - video_seconds


def _av_coverage_violation(gap_seconds: float | None) -> MasteringGateViolation | None:
    """PURE: the violation for a master whose audio outruns its picture, or None.

    Fires strictly ABOVE :data:`AV_COVERAGE_TOLERANCE_SECONDS`, so the measured
    0.161s of legitimate AAC tail padding on a good promo master is not a defect.
    An unmeasured gap (``None``) never fires — no-gate-on-a-gap, as everywhere
    else in this module. A NEGATIVE gap (picture outlasts audio, e.g. a tail card
    held after the last line) is not this defect and never fires either."""
    if gap_seconds is None or gap_seconds <= AV_COVERAGE_TOLERANCE_SECONDS:
        return None
    return MasteringGateViolation(
        stage="av_coverage",
        tier=gate_tier("av_coverage"),
        metric="av_gap_seconds",
        value=gap_seconds,
        limit=AV_COVERAGE_TOLERANCE_SECONDS,
        detail=(
            f"av_coverage: audio outruns video by {gap_seconds:.3f}s, over the "
            f"{AV_COVERAGE_TOLERANCE_SECONDS:.3f}s tolerance — the master would play "
            f"narration over a picture that has already ended"
        ),
    )


#: Fraction of sampled frames that must share one duration for the stream to read
#: as constant-frame-rate in :func:`_frame_durations_are_uniform`. Not 1.0: ffprobe
#: reports a blank per-frame duration for the last frame or two of a stream as an
#: ordinary tail artifact, and one such frame must not flip a clean master to VFR.
#:
#: ⚠️ ADVISORY, PROVENANCE: NONE. n=0 — 0.98 is the "one or two tail frames out of
#: ~100 sampled" arithmetic above, not a fleet measurement, and post-#218 fleet
#: ``cfr_ok=False`` rate was unmeasured when this was written. Measured since:
#: 37/37 delivered masters ``cfr_ok=True`` (2026-10-05, after #230). By default
#: nothing it feeds can gate: ``cfr_ok`` only appends ``"cfr"`` to
#: ``rerun_stages``. It becomes a ``violations`` entry ONLY when
#: ``mastering_gate_cfr_enabled`` is true (default false; see ``_cfr_violation``)
#: — ``mastering_gate_mode='enforce'`` alone does not arm it. There is still no
#: remediation stage, so arming it fails the job rather than repairing cadence.
_CFR_FRAME_UNIFORMITY = 0.98

#: The per-frame duration keys ffprobe may emit, NEWEST SPELLING FIRST.
#:
#: ⛔ ``pkt_duration_time`` was REMOVED in ffmpeg 7.0 (renamed ``duration_time``).
#: Asking for a key ffprobe does not know is not an error — it prints nothing and
#: exits 0 — so on the production build (7.1.5) the probe below returned ZERO
#: lines, ``_frame_durations_are_uniform`` took its empty-``durations`` branch,
#: and ``cfr_ok`` published ``null`` on every single master while looking
#: perfectly healthy. Measured on the live pod: with the old verbatim args, 0
#: lines; with ``frame=duration_time``, 30 lines for a 30-frame input.
#:
#: Both names are requested and whichever one PARSES is used, so this works on a
#: 7.x build and on a pre-7.0 build without version detection. ``Dockerfile``
#: additionally asserts the ffmpeg major version at build time so a distro bump
#: that renames these again fails the build instead of silently re-nulling the
#: metric.
_CFR_DURATION_KEYS = ("duration_time", "pkt_duration_time")

#: How many seconds from the head of the file to sample for frame durations. A full
#: decode of a delivered master costs far more than this advisory metric is worth;
#: a VFR stream is VFR from the start, so the head is a sufficient witness.
#:
#: ⚠️ Sufficient for a stream that is VFR THROUGHOUT, and only for that. A single
#: hole past the sample (master 01a11032: 20.2 s in) is outside it, and would pass
#: even inside it, because these are DECLARED durations and the frame before a hole
#: declares a normal one. ``video_pts_gaps`` reads the whole timeline for that.
_CFR_SAMPLE_SECONDS = 10


def _parse_frame_durations(probe_output: str) -> list[str]:
    """Pull the per-frame durations out of a ``key=value`` ffprobe dump, taking
    the FIRST key in :data:`_CFR_DURATION_KEYS` that actually yielded values.

    PURE. Preferring one key over merging both matters: a build that emits both
    spellings would otherwise contribute two lines per frame, which is harmless
    for the uniformity ratio but doubles ``len(durations)`` and makes the log
    read as twice the sampled frames. ``N/A`` is dropped — ffprobe prints it for
    the tail frames of a stream, and it is not a duration."""
    by_key: dict[str, list[str]] = {key: [] for key in _CFR_DURATION_KEYS}
    for line in probe_output.splitlines():
        key, sep, value = line.strip().partition("=")
        value = value.strip()
        if not sep or key not in by_key or not value or value == "N/A":
            continue
        by_key[key].append(value)
    for key in _CFR_DURATION_KEYS:
        if by_key[key]:
            return by_key[key]
    return []


async def _frame_durations_are_uniform(path: Path | str) -> bool | None:
    """Whether the head of the stream is CFR *frame-for-frame*. ``None`` on probe failure.

    ⛔ WHY THIS EXISTS. ``framerate_coherence`` used to answer CFR purely from
    ``r_frame_rate`` vs ``avg_frame_rate``, and that comparison CANNOT answer it.
    ``avg_frame_rate`` is not a cadence reading — the container computes it as
    frames/duration, so it carries the duration's own rounding. Measured on three
    delivered masters (2026-09-13): each probed ``r_frame_rate=30/1`` against
    ``avg_frame_rate=8678400/289403`` = 29.9873, i.e. 0.0127 apart — over the 0.01
    ceiling, so each was reported VFR — while each was in fact CFR frame-for-frame,
    565 of 567 frames at exactly ``pkt_duration_time=0.033333``. That false negative
    is the bulk of the live ``cfr_ok=false`` rate.

    ⚠️ And the gap could NOT be closed by widening the ceiling: the rounding
    artifact measured 0.042% while a genuine NTSC 29.97-vs-30 mismatch is 0.1%,
    so any tolerance separating them would sit inside a 2.4x band and be tuned to
    the sample rather than to the property. Measure the property instead.
    """
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"%+{_CFR_SAMPLE_SECONDS}",
                "-show_entries",
                "frame=" + ",".join(_CFR_DURATION_KEYS),
                "-of",
                # ``nokey=1`` is NOT usable here: two keys are requested and their
                # values would arrive interleaved and indistinguishable. Keep the
                # keys and split on them below.
                "default=noprint_wrappers=1",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering.frame_durations.failed", path=str(path), error=str(exc))
        return None

    durations = _parse_frame_durations(out)
    if not durations:
        return None
    dominant = Counter(durations).most_common(1)[0][0]
    return durations.count(dominant) / len(durations) >= _CFR_FRAME_UNIFORMITY


async def framerate_coherence(
    path: Path | str, expected_fps: int | float | None = None
) -> bool | None:
    """Probe ``r_frame_rate`` vs ``avg_frame_rate`` (and, if given, ``expected_fps``)
    to report whether the delivered stream is frame-rate coherent (a proxy for
    CFR). Probe-only — it NEVER changes cadence and does not touch any cadence
    field (VC-3 owns those). Returns ``None`` on any probe failure.

    ⚠️ This is ONE of the two readings ``cfr_ok`` is built from, not ``cfr_ok``
    itself: it cannot see a hole in the pts timeline. ``post_mux_verify`` combines
    it with ``video_pts_gap_report`` through ``_cfr_verdict``."""
    from src.services import renderer

    try:
        out = await renderer._run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering.framerate_coherence.failed", path=str(path), error=str(exc))
        return None
    rates = [_parse_rate(line) for line in out.strip().splitlines() if line.strip()]
    rates = [r for r in rates if r is not None]
    if len(rates) < 2:
        return None
    r_rate, avg_rate = rates[0], rates[1]
    coherent = abs(r_rate - avg_rate) <= 0.01
    if not coherent:
        # The headers disagree — that is a QUESTION, not a verdict. Ask the frames.
        uniform = await _frame_durations_are_uniform(path)
        if uniform is None:
            return None
        coherent = uniform
    if expected_fps is not None and coherent:
        coherent = abs(avg_rate - float(expected_fps)) <= 0.5
    return coherent


async def measured_fps(path: Path | str) -> float | None:
    """Probe the delivered master's ``avg_frame_rate`` as a float, or ``None`` on
    any probe/parse failure. Read-only, best-effort — never raises (mirrors
    ``probe_dimensions``). The measured counterpart to the declared render
    cadence (VC-3); advisory, does not gate."""
    from src.services import renderer

    try:
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
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering.measured_fps.failed", path=str(path), error=str(exc))
        return None
    lines = [line for line in out.strip().splitlines() if line.strip()]
    return _parse_rate(lines[0]) if lines else None


async def _count_frames(path: Path | str, *, decimate: bool) -> int | None:
    """Count frames sent to the null muxer, optionally through ``mpdecimate``.

    Without ``decimate`` this is the source frame count; with it, the count of
    frames that SURVIVED mpdecimate (near-duplicate frames dropped). Parsed from
    ffmpeg's ``frame=`` progress tail. Best-effort — ``None`` on any failure."""
    from src.services import renderer

    vf = ["-vf", "mpdecimate"] if decimate else []
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path), *vf, "-an", "-f", "null", "/dev/null"]
    try:
        stderr = await renderer._run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering._count_frames.failed", path=str(path), error=str(exc))
        return None
    matches = re.findall(r"frame=\s*(\d+)", stderr)
    return int(matches[-1]) if matches else None


async def _count_packets(path: Path | str) -> int | None:
    """Demux-only frame count via ``ffprobe -count_packets`` — NO decode.

    For a CFR H.264/mp4 delivered master one video packet is one frame, so this
    is the same number the plain ``_count_frames(decimate=False)`` full decode
    produced, without decoding a single frame. ``nb_read_packets`` (counted by
    walking the index) is used deliberately over ``stream=nb_frames``, which is
    frequently ``N/A`` for mp4. Best-effort — ``None`` on any failure or a
    non-numeric result, so the caller can fall back to the real decode rather
    than gate a delivered master on a wrong count."""
    from src.services import renderer

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        out = await renderer._run(cmd)
    except Exception as exc:  # noqa: BLE001 — best-effort metric, never fail a job
        log.warning("mastering._count_packets.failed", path=str(path), error=str(exc))
        return None
    for line in (out or "").splitlines():
        token = line.strip()
        if token.isdigit():
            value = int(token)
            return value if value > 0 else None
    log.warning("mastering._count_packets.unparsable", path=str(path), output=(out or "")[:200])
    return None


async def duplicate_frame_ratio(path: Path | str) -> float | None:
    """Best-effort fraction of the delivered master that is duplicate (padded)
    frames, via an ``mpdecimate`` diff pass. ``0.0``..``1.0`` — higher means more
    of the film is held/duplicate frames (e.g. a source shorter than its narration
    padded with clones).

    ⛔ THIS IS TELEMETRY. IT DOES NOT GATE, and it must not be made to gate again:
    the frames are intentional dwell, and the ratio was measured at 0.643-0.998
    across 120 of 120 completed masters — see the comment block where
    ``_duplicate_frames`` used to live. It is published in the mastering report as
    ``duplicate_frame_ratio`` and read by humans and dashboards only; nothing in
    ``post_mux_verify`` compares it to anything. ``None`` on any probe/parse
    failure — the probe NEVER raises."""
    # The TOTAL is a pure count — it never needed the pixels. A demux-only
    # ffprobe packet walk replaces what used to be a second full decode of the
    # whole master (packets == frames for a CFR H.264/mp4 delivery). Correctness
    # beats speed here: this ratio decides whether a film is called "frozen
    # video", so an unparsable/empty probe falls back to the real decode instead
    # of producing a wrong denominator.
    # The packet walk and the mpdecimate decode read the SAME immutable file and
    # do not depend on each other, so they overlap. Both are best-effort (each
    # returns None rather than raising), which is what makes a bare ``gather``
    # safe here — no sibling is left orphaned by a propagating exception. The
    # sequential fallback below is unchanged and still only fires when the
    # packet walk came back unusable.
    total, kept = await asyncio.gather(
        _count_packets(path),
        # The mpdecimate survivor count MUST stay a decode — mpdecimate is a
        # pixel comparison. Do not try to fuse the two passes into one ``split``
        # filtergraph: the ``frame=`` progress parser takes the LAST match and
        # cannot separate two interleaved null-output counters.
        _count_frames(path, decimate=True),
    )
    if total is None:
        total = await _count_frames(path, decimate=False)
    if not total or kept is None:
        return None
    dropped = max(0, total - kept)
    return round(dropped / total, 4)


def _parse_rate(rate: str) -> float | None:
    """Parse an ffprobe ``num/den`` rational (or a bare number) to fps, or None."""
    rate = rate.strip()
    if not rate or rate == "0/0":
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            d = float(den)
            return float(num) / d if d else None
        return float(rate)
    except ValueError, ZeroDivisionError:
        return None


def _loudness_violation(
    integrated_lufs: float | None, settings: Settings
) -> MasteringGateViolation | None:
    """The loudness half of the mastering gate: the violation it produces, or None.

    UNSATISFIABLE-GATE GUARD, and it is the whole reason this is a function:
    ``renderer._loudnorm_filter`` returns ``""`` and the loudness master pass is
    SKIPPED ENTIRELY when ``master_enabled`` is False — which any caller sets per
    job via ``tags={"mastering": "off"}``. A raw mix measures around -21 LUFS
    against the -12.5 default target, i.e. far outside the 3 LU tolerance, so this
    gate used to convict a CORRECT render and name ``loudness`` as the remediation
    for a stage the job itself switched off. Re-running that stage normalizes nothing and
    re-measures the identical number, so ``rerun_stages=["loudness"]`` could never
    be satisfied and ``mastering_gate_mode=enforce`` failed the job permanently.
    With mastering off there is NO violation at all — including the unmeasurable
    verdict, which names the same disabled stage and is equally unsatisfiable. The
    measured value is still REPORTED; only the GATING is suppressed.

    With mastering ON, two verdicts:

    - ``None`` measured (``final_mux_loudness`` failed, no audio stream, or loudnorm
      printed no usable ``input_i`` — including the ``-inf`` of a digitally-silent
      mux, which ``parse_loudness`` scrubs to ``None`` because ``-Infinity`` is not
      JSON-safe) → gate fails CLOSED: an unmeasured master is not a clean one.
    - measured further than ``LOUDNESS_GATE_TOLERANCE_LU`` from ``master_lufs`` →
      the real, remediable finding.
    """
    if not bool(getattr(settings, "master_enabled", True)):
        return None
    target_lufs = float(getattr(settings, "master_lufs", DEFAULT_MASTER_LUFS))
    if integrated_lufs is None:
        return MasteringGateViolation(
            stage="loudness",
            tier=gate_tier("loudness"),
            metric="integrated_lufs",
            limit=target_lufs,
            detail=(
                "loudness: integrated_lufs is unmeasurable (target "
                f"{target_lufs:.1f} LUFS +/- {LOUDNESS_GATE_TOLERANCE_LU:.1f} LU)"
            ),
        )
    if abs(integrated_lufs - target_lufs) <= LOUDNESS_GATE_TOLERANCE_LU:
        return None
    return MasteringGateViolation(
        stage="loudness",
        tier=gate_tier("loudness"),
        metric="integrated_lufs",
        value=integrated_lufs,
        limit=target_lufs,
        detail=(
            f"loudness: integrated {integrated_lufs:.2f} LUFS is "
            f"{abs(integrated_lufs - target_lufs):.2f} LU off target "
            f"{target_lufs:.2f} (max {LOUDNESS_GATE_TOLERANCE_LU:.2f} LU)"
        ),
    )


def _speech_presence_violation(
    integrated_lufs: float | None, noise_floor_db: float | None
) -> MasteringGateViolation | None:
    """The master carries speech, or it does not: the violation, or None.

    Compares ``loudness.program_to_noise_floor_db`` (the delivered program level
    minus the delivered file's OWN measured noise floor) against
    ``SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB``. Both inputs come from the ONE
    ``final_mux_loudness`` pass the caller already ran, so this costs no extra
    probe — ``astats`` rides that pass exactly as it does ``acx_measure``'s.

    NO-GATE-ON-A-GAP: an absent ``integrated_lufs`` or ``noise_floor_db`` yields
    ``None``, not a conviction. This is the one place it differs from
    ``_loudness_violation``, which fails CLOSED on an unmeasurable
    ``integrated_lufs`` — and deliberately, because that verdict ALREADY names the
    identical condition. A second violation over the same missing number would
    report one probe failure twice and name a remediation (``speech_presence``)
    for a measurement, not for the master.

    NO ``master_enabled`` GUARD, unlike loudness and true peak. Those two gate on
    an ABSOLUTE level that only the loudness master pass can put in range, so with
    mastering off they convict a correct render for a stage the job itself
    switched off. This metric is a DIFFERENCE of two levels, so it is invariant
    under any gain applied to the whole file: a raw, unmastered narration mix
    reads the same gap as a mastered one (measured: bursty narration at -20 dB
    reads +87.7, and +105.9 after normalising the very same file to -14 LUFS).
    There is no configuration in which switching mastering off can manufacture
    this violation, so there is nothing to suppress."""
    gap = program_to_noise_floor_db(integrated_lufs, noise_floor_db)
    if gap is None or gap >= SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB:
        return None
    return MasteringGateViolation(
        stage="speech_presence",
        tier=gate_tier("speech_presence"),
        metric="program_to_noise_floor_db",
        value=gap,
        limit=SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB,
        detail=(
            f"speech_presence: the delivered program sits {gap:.2f} dB above the "
            f"master's own noise floor (integrated {integrated_lufs:.2f} LUFS vs "
            f"astats noise floor {noise_floor_db:.2f} dB), under the minimum "
            f"{SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB:.2f} dB — the audio reads "
            "as a steady bed of noise with no speech in it, not as narration"
        ),
    )


async def _decode_audio_pcm16(source: Path, dest: Path, *, sample_rate: int) -> bool:
    """Decode ``source``'s audio (any container) to a mono 16-bit PCM WAV at
    ``sample_rate``, written to ``dest``. Returns False on ANY failure — missing
    audio stream, unreadable file, ffmpeg error — never raises.

    This is a full, ordinary DECODE (not a stream copy), which is what makes it
    safe over a discontiguous-pts AAC track: ffmpeg's decoder walks the real
    container timeline and fills any gap with silence as part of normal decoding.
    A ``-c:a copy`` extraction would instead concatenate the coded frames with no
    regard for gaps between their timestamps — the collapse this function exists
    to avoid. Called ONCE per input (the master's full audio, and each fetched
    narration clip); windows into the decoded master are then sliced in pure
    Python by :func:`_slice_pcm16_wav`, so no window ever re-seeks the container."""
    from src.services import renderer

    try:
        await renderer._run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-sample_fmt",
                "s16",
                "-f",
                "wav",
                str(dest),
            ]
        )
    except Exception as exc:  # noqa: BLE001 — best-effort measure, never fail a job
        log.warning("narration_presence.decode_failed", source=str(source), error=str(exc))
        return False
    # A bare WAV header with zero sample bytes is not a usable decode (e.g. the
    # source carried no audio stream at all) — ffmpeg still exits 0 for that.
    # Threaded off the loop (ASYNC240) — a stat of a file this function just
    # finished writing is a trivial, one-shot check, not a hot-loop concern.
    return await asyncio.to_thread(lambda: dest.exists() and dest.stat().st_size > 44)


def _slice_pcm16_wav(src: Path, dest: Path, *, start: float, duration: float) -> bool:
    """Byte-slice ``[start, start+duration)`` out of a mono 16-bit PCM WAV into
    ``dest``. Pure Python (``wave``) — no ffmpeg process, no container re-seek, so
    it cannot be fooled by the discontiguous-pts case ``_decode_audio_pcm16``'s
    docstring names: ``src`` was already fully decoded once onto a flat,
    uniformly-sampled timeline, so this only indexes samples. Clamped to the
    source's own frame count; returns False (leaves ``dest`` unwritten) for an
    empty or out-of-range window, or any read/write failure — never raises."""
    try:
        with wave.open(str(src), "rb") as reader:
            sample_rate = reader.getframerate()
            total_frames = reader.getnframes()
            start_frame = max(0, min(total_frames, round(start * sample_rate)))
            end_frame = max(0, min(total_frames, round((start + duration) * sample_rate)))
            if end_frame <= start_frame:
                return False
            reader.setpos(start_frame)
            frames = reader.readframes(end_frame - start_frame)
            params = reader.getparams()
        with wave.open(str(dest), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
    except Exception as exc:  # noqa: BLE001 — best-effort measure, never fail a job
        log.warning("narration_presence.slice_failed", src=str(src), error=str(exc))
        return False
    return True


async def _axcorrelate_rms_db(clip: Path, window: Path) -> float | None:
    """Cross-correlate two mono PCM WAVs with ffmpeg's ``axcorrelate`` filter and
    return the correlated stream's RMS level in dB (``astats``), or ``None`` on
    any failure. See :data:`NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB` for the
    measured separation this reads against."""
    from src.services import renderer

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(clip),
        "-i",
        str(window),
        "-filter_complex",
        (
            f"[0:a][1:a]axcorrelate=size={NARRATION_PRESENCE_AXCORRELATE_BLOCK_SIZE}:"
            "algo=fast[c];[c]astats=metadata=0[o]"
        ),
        "-map",
        "[o]",
        "-f",
        "null",
        "-",
    ]
    try:
        stderr = await renderer._run(cmd, capture_stderr=True)
    except Exception as exc:  # noqa: BLE001 — best-effort measure, never fail a job
        log.warning("narration_presence.correlate_failed", error=str(exc))
        return None
    matches = _RMS_LEVEL_DB_RE.findall(stderr or "")
    if not matches:
        return None
    value = _as_float(matches[-1])
    if value is None:
        return None
    if value == float("-inf"):
        return _NARRATION_PRESENCE_SILENT_RMS_DB
    return value


def _narration_presence_lag_offsets(lag_search: bool) -> list[float]:
    """The set of start-offsets ``_narration_clip_correlation_db`` tries, in
    seconds. ``lag_search=False`` (used for the NEIGHBOUR comparison) is exactly
    ``[0.0]`` — the neighbour check must never search INTO the current shot's
    own territory, which zero-lag windows abutting on a tight timeline would
    otherwise invite. ``lag_search=True`` (the shot's own window) returns every
    multiple of ``NARRATION_PRESENCE_LAG_STEP_SECONDS`` in
    ``[-NARRATION_PRESENCE_LAG_SEARCH_SECONDS, +NARRATION_PRESENCE_LAG_SEARCH_SECONDS]``."""
    if not lag_search:
        return [0.0]
    lag = NARRATION_PRESENCE_LAG_SEARCH_SECONDS
    step = NARRATION_PRESENCE_LAG_STEP_SECONDS
    offsets: list[float] = []
    steps = round(lag / step)
    for i in range(-steps, steps + 1):
        offsets.append(round(i * step, 6))
    return offsets


async def _narration_clip_correlation_db(
    local_audio_path: Path,
    master_pcm: Path,
    master_duration: float,
    start: float,
    end: float,
    work_dir: Path,
    index: int,
    *,
    lag_search: bool = True,
) -> float | None:
    """One shot's correlation measurement against its ALREADY-LOCAL narration
    source — the file ``compile_shots`` itself fetched (and threaded through, see
    ``narration_presence_report``'s docstring for why this never re-fetches):
    decode it, slice the padded master window, correlate. ``None`` on any failure
    along the way (undecodable audio, an empty window) — no-gate-on-a-gap, the
    same policy every other measure in this module follows.

    ``lag_search`` (default ``True``): try every offset
    :func:`_narration_presence_lag_offsets` returns — each candidate window has
    the SAME width (``end - start``, plus ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS``
    on each side, unchanged), only its start position shifts, so this cannot
    reintroduce the dilution defect a WIDER window caused (see
    ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS``) — and return the BEST (highest)
    reading found, tolerating realistic drift between the probed [start, end) and
    where the content actually landed in the delivered master. Pass
    ``lag_search=False`` for a NEIGHBOUR-window comparison (see
    ``NARRATION_PRESENCE_IDENTITY_MARGIN_DB``): that check must stay pinned to
    the neighbour's own exact window, never searching toward the current shot."""
    src_pcm = work_dir / f"src-{index}.wav"
    if not await _decode_audio_pcm16(
        local_audio_path, src_pcm, sample_rate=NARRATION_PRESENCE_SAMPLE_RATE
    ):
        return None
    margin = NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS
    window_pcm = work_dir / f"win-{index}.wav"
    best_db: float | None = None
    for offset in _narration_presence_lag_offsets(lag_search):
        window_start = max(0.0, start + offset - margin)
        window_end = min(master_duration, end + offset + margin)
        if window_end <= window_start:
            continue
        if not _slice_pcm16_wav(
            master_pcm, window_pcm, start=window_start, duration=window_end - window_start
        ):
            continue
        db = await _axcorrelate_rms_db(src_pcm, window_pcm)
        if db is not None and (best_db is None or db > best_db):
            best_db = db
    return best_db


def _narration_entry_window(entry: Mapping[str, object] | None) -> tuple[float, float] | None:
    """``(start, end)`` out of a ``shot_timeline`` entry, or ``None`` when the
    entry is absent or carries no usable window — the same permissive parsing
    :func:`_narration_presence_entry` applies to its own entry, reused here for
    neighbour entries."""
    if entry is None:
        return None
    raw_start = entry.get("start")
    raw_end = entry.get("end")
    start = float(raw_start) if isinstance(raw_start, (int, float)) else 0.0
    end = float(raw_end) if isinstance(raw_end, (int, float)) else 0.0
    if end <= start:
        return None
    return start, end


async def _narration_presence_entry(
    index: int,
    entry: Mapping[str, object],
    shot: object,
    master_pcm: Path,
    master_duration: float,
    work_dir: Path,
    local_audio_path: Path | None,
    neighbor_entries: Sequence[Mapping[str, object] | None] = (),
) -> NarrationPresenceClip | None:
    """One ``shot_timeline`` entry → its typed per-clip report, or ``None`` when
    the shot carries no narration at all (nothing to measure, not even an
    unmeasured entry — a purely visual shot is not a presence question).
    Split out of :func:`narration_presence_report`'s loop body to keep that
    function's branching legible.

    ``local_audio_path`` is the local copy of this shot's narration clip that
    ``compile_shots`` already downloaded while rendering — ``None`` when no such
    copy survived to verification time (e.g. this shot's segment was served by
    the content-addressed render cache, so the raw source was never fetched this
    compile). Existence is confirmed by the ATTEMPT: ``_narration_clip_correlation_db``
    → ``_decode_audio_pcm16`` never raises on a missing/unreadable file, it just
    returns ``False`` (same best-effort contract as every other decode in this
    module) — so a stale or absent path reports UNMEASURED
    (``correlation_db=None, present=None``, the SAME gap convention as an
    undecodable clip) — never as "missing", and never re-fetched.

    ``neighbor_entries`` (#234): the immediate adjacent ``shot_timeline``
    entries (typically ``[prev, next]``, either of which may be ``None`` at a
    timeline edge). The SAME clip is also correlated against each neighbour's
    OWN window (``lag_search=False`` — see ``NARRATION_PRESENCE_IDENTITY_MARGIN_DB``
    for why), and ``present`` now additionally requires the shot's own reading to
    beat the best neighbour reading by that margin — line-identity-awareness, not
    just "is some audio here"."""
    audio_url = getattr(shot, "audio_url", None)
    if not audio_url or not isinstance(audio_url, str):
        return None
    window = _narration_entry_window(entry)
    correlation_db: float | None = None
    neighbor_db: float | None = None
    if window is not None and local_audio_path is not None:
        start, end = window
        correlation_db = await _narration_clip_correlation_db(
            local_audio_path, master_pcm, master_duration, start, end, work_dir, index
        )
        if correlation_db is not None:
            for neighbor_entry in neighbor_entries:
                neighbor_window = _narration_entry_window(neighbor_entry)
                if neighbor_window is None:
                    continue
                nb_start, nb_end = neighbor_window
                nb_db = await _narration_clip_correlation_db(
                    local_audio_path,
                    master_pcm,
                    master_duration,
                    nb_start,
                    nb_end,
                    work_dir,
                    index,
                    lag_search=False,
                )
                if nb_db is not None and (neighbor_db is None or nb_db > neighbor_db):
                    neighbor_db = nb_db
    present: bool | None = None
    if correlation_db is not None:
        present = correlation_db >= NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB
        if present and neighbor_db is not None:
            present = (correlation_db - neighbor_db) >= NARRATION_PRESENCE_IDENTITY_MARGIN_DB
    raw_line_id = entry.get("line_id")
    line_id = raw_line_id if isinstance(raw_line_id, str) else None
    return NarrationPresenceClip(
        line_id=line_id,
        correlation_db=correlation_db,
        present=present,
        neighbor_correlation_db=neighbor_db,
    )


async def narration_presence_report(
    master_path: Path | str,
    shot_timeline: Sequence[Mapping[str, object]] | None,
    shots: Sequence[object] | None,
    settings: Settings,
    narration_audio_paths: Sequence[Path | None] | None = None,
) -> tuple[list[dict[str, object]], MasteringGateViolation | None]:
    """Per-shot content-aware check: does every shot's SOURCE narration clip the
    compiler was given actually appear in the delivered master, at that shot's
    own timeline window? Replaces a whole-file loudness statistic
    (``_speech_presence_violation``) with a measurement of the actual defect: a
    production master shipped with one of its four narration clips completely
    ABSENT and 68% silence, and no existing signal noticed — ``integrated_lufs -
    noise_floor_db`` read ~+103.8 dB (comfortably "speech present") because
    digital silence makes both terms degenerate from the SAME zero signal, and a
    single number over the whole file cannot separate "every clip present" from
    "one clip missing, three present" — both contain plenty of real speech.

    THE MEASURE (see ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB`` for the full
    provenance, including the real-production calibration table): for each shot
    that carries an ``audio_url``, decode its ALREADY LOCAL source clip
    (``narration_audio_paths[index]`` — see below) and the master's audio to
    8 kHz mono PCM, cross-correlate the clip against the master's EXACT
    [start, end) window for that shot (``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS``
    — no padding, see that constant for why: even a small padded window
    measurably degrades the RMS reading and, on real audio, degrades it worse
    for longer clips, which is what makes the unpadded statistic the
    LENGTH-INVARIANT fix), and read the correlated stream's RMS level via
    ``astats``. ffmpeg's own ``axcorrelate`` filter is BED-ROBUST by
    construction (a continuous bed correlates with neither the clip nor itself
    at any meaningful rate — proven by ``test_a_missing_clip_is_still_flagged_with_a_music_bed``
    and ``test_a_complete_master_with_a_bed_flags_nothing``) because it
    correlates the SHAPE of the two signals, not their aggregate loudness. It
    is also VERSION-STABLE: the filter and its output shape are identical on
    ffmpeg 4.4 (dev) and 7.1.5 (the production pod, verified 2026-09-17).

    ⛔ NEVER RE-FETCHES. This check runs at the very end of a compile, after the
    final mux — exactly when a job's presigned narration-clip URLs are most
    likely EXPIRED (a proven production failure: a compile retried past 60
    minutes got 403 on its 1-hour presigned inputs). Re-fetching here would make
    the check fail to read an expired-but-perfectly-fine clip and either skip it
    (no-gate-on-a-gap) or, worse, read as evidence of absence for content that
    genuinely shipped. ``compile_shots`` has ALREADY downloaded every narration
    clip once, during ``compile_shots``/``_resolve_shot_assets``, to render the
    segments — ``narration_audio_paths`` is that same local copy, threaded
    through unchanged (index-aligned to ``shots``/``shot_timeline``, via
    ``compile_shots``' ``narration_audio_sink`` out-parameter). A shot whose
    local copy did not survive to verification time (``None``, or a path that no
    longer exists — e.g. this shot's segment was served by the content-addressed
    render cache, so the raw source was never fetched THIS compile) is reported
    UNMEASURED, never re-fetched and never reported as missing — see
    :func:`_narration_presence_entry`.

    NO-GATE-ON-A-GAP throughout: a clip that has no local copy, or could not be
    decoded, or windowed reports ``correlation_db=None, present=None`` and is
    EXCLUDED from the violation — an unmeasurable clip is not evidence the
    master is broken, unlike every other check in this module.

    Returns ``(per_clip_report, violation)``. ``violation`` is a SINGLE
    ``MasteringGateViolation`` naming every shot found missing (never one per
    shot — the report already carries the per-clip detail), or ``None`` when
    every measured clip was found. ``([], None)`` when there is nothing to check
    (no shot_timeline, no shots, or the master's own audio could not be decoded
    at all — the latter is already named by other gates in this module,
    ``av_coverage``/``verify``, so this function stays silent rather than
    reporting the same absence a second time under a different name).

    Run sequentially (no fan-out, no client, nothing to bound): compile jobs
    carry a handful of narrated shots at most, and every clip is already local
    disk I/O plus an ffmpeg decode — there is no network round-trip left to
    parallelize."""
    if not shot_timeline or not shots:
        return [], None
    master = Path(master_path)
    work_dir = Path(tempfile.mkdtemp(prefix="narration-presence-"))
    per_clip: list[NarrationPresenceClip] = []
    missing: list[str] = []
    worst_db: float | None = None
    try:
        master_pcm = work_dir / "master.wav"
        if not await _decode_audio_pcm16(
            master, master_pcm, sample_rate=NARRATION_PRESENCE_SAMPLE_RATE
        ):
            return [], None
        with wave.open(str(master_pcm), "rb") as reader:
            master_duration = reader.getnframes() / float(reader.getframerate())

        for index, entry in enumerate(shot_timeline):
            if index >= len(shots):
                break
            local_audio_path = (
                narration_audio_paths[index]
                if narration_audio_paths is not None and index < len(narration_audio_paths)
                else None
            )
            neighbor_entries = [
                shot_timeline[index - 1] if index > 0 else None,
                shot_timeline[index + 1] if index + 1 < len(shot_timeline) else None,
            ]
            clip = await _narration_presence_entry(
                index,
                entry,
                shots[index],
                master_pcm,
                master_duration,
                work_dir,
                local_audio_path,
                neighbor_entries,
            )
            if clip is None:
                continue
            per_clip.append(clip)
            correlation_db = clip.correlation_db
            if clip.present is False:
                line_id = clip.line_id
                missing.append(line_id if line_id is not None else f"shot[{index}]")
                if correlation_db is not None and (worst_db is None or correlation_db < worst_db):
                    worst_db = correlation_db
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    per_clip_payload = [c.model_dump(mode="json") for c in per_clip]
    if not missing:
        return per_clip_payload, None
    names = ", ".join(missing)
    detail = (
        f"narration_presence: narration is missing from the delivered master for shot(s) {names}"
    )
    if worst_db is not None:
        detail += (
            f" (worst correlation {worst_db:.2f} dB RMS, min "
            f"{NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB:.2f} dB)"
        )
    violation = MasteringGateViolation(
        stage="narration_presence",
        tier=gate_tier("narration_presence"),
        metric="correlation_rms_db",
        value=worst_db,
        limit=NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB,
        detail=detail,
    )
    return per_clip_payload, violation


def _true_peak_violation(
    true_peak_dbtp: float | None, settings: Settings
) -> MasteringGateViolation | None:
    """VC-W14/V3 — the true-peak half of the gate: the violation, or None.

    ``true_peak_dbtp`` was measured and published on every report and compared to
    NOTHING. This is that comparison.

    Three verdicts, in the shape ``_loudness_violation`` already established:

    - mastering OFF (``master_enabled`` False, which any job sets via
      ``tags={"mastering": "off"}``) → **no violation at all**, measured value
      still reported. UNSATISFIABLE-GATE GUARD: with the master pass skipped
      there is no ``alimiter`` in the chain, so a raw mix legitimately peaks
      wherever it peaks, and naming ``loudness`` as the remediation for a stage
      the job itself switched off fails a correct render permanently under an
      armed gate. This is the exact trap ``_loudness_violation`` documents.
    - ``None`` measured → **no violation**. NO-GATE-ON-A-GAP, deliberately the
      opposite of ``_loudness_violation``'s fail-closed: an unmeasured peak is
      already reported through ``unmeasured``, and failing closed here would
      convict every silent reel and every master whose loudness probe timed out
      on a condition that says nothing about clipping.
    - measured above :data:`TRUE_PEAK_CEILING_DBTP` → the real finding.
    """
    if not bool(getattr(settings, "master_enabled", True)):
        return None
    if true_peak_dbtp is None:
        return None
    if true_peak_dbtp <= TRUE_PEAK_CEILING_DBTP:
        return None
    return MasteringGateViolation(
        stage="true_peak",
        tier=gate_tier("true_peak"),
        metric="true_peak_dbtp",
        value=true_peak_dbtp,
        limit=TRUE_PEAK_CEILING_DBTP,
        detail=(
            f"true_peak: delivered {true_peak_dbtp:.2f} dBTP exceeds full scale "
            f"{TRUE_PEAK_CEILING_DBTP:.2f} dBTP — the master limiter did not hold"
        ),
    )


def measure_violation(*, measure_failed: bool, settings: Settings) -> MasteringGateViolation | None:
    """VC-W14/V4 — the pass-1-measurement half of the gate: the violation, or None.

    When ``_measure_master_loudnorm`` crashes, the render FAILS OPEN by design:
    ``_master_filter`` is built with ``measured=None`` and the master runs
    single-pass ``loudnorm``, i.e. DYNAMIC gain riding instead of the one constant
    gain a two-pass linear master applies. That file is audibly different — and it
    shipped with ``gate_pass=True`` and no note, indistinguishable in the report
    from a master that measured cleanly. ``master_measure_failed`` was recorded
    and published, and NOTHING consumed it.

    ``settings`` is taken for the same unsatisfiable-gate guard the other two
    carry: with ``master_enabled`` False no measurement pass exists to fail, so
    the flag cannot be set and the gate cannot name a stage the job switched off.

    PUBLIC (no leading underscore) because the flag is recorded by the RENDER and
    read at the compile route — the only place that has it — rather than inside
    :func:`post_mux_verify`, which never sees the ContextVar. Same placement as
    the ``verify`` gate, which is likewise raised at the route.
    """
    if not measure_failed:
        return None
    if not bool(getattr(settings, "master_enabled", True)):
        return None
    return MasteringGateViolation(
        stage="measure",
        tier=gate_tier("measure"),
        metric="master_measure_failed",
        detail=(
            "measure: pass-1 loudness measurement failed, so the master ran "
            "single-pass DYNAMIC loudnorm instead of the two-pass linear master "
            "— the delivered gain riding is not the one that was requested"
        ),
    )


def bed_calibration_violation(
    music_bed: Mapping[str, object] | None, settings: Settings
) -> MasteringGateViolation | None:
    """VC-W14/V6 — the bed-calibration half of the gate: the violation, or None.

    ``music_was_measured`` and ``overlay_audio_bed``'s ``vo_measure_failed`` were
    TELEMETRY WITH NO GATE: one published on the report, the other a log line, and
    nothing read either. Meanwhile ``vo_lufs=None`` sends BOTH
    ``renderer._offset_gain_db`` and ``renderer._vo_relative_threshold`` to their
    literal fallbacks, so a job that ASKED for a bed sat N LU under the narration
    got an arbitrary absolute gain, and the ducker's open threshold became a
    constant the narration bus may never cross — the compressor never opens and
    the bed runs flat through every VO gap. On a warning.

    Decided from ``vo_measure_failed``, NOT from ``vo_lufs is None``:
    ``vo_lufs`` is legitimately None on every job that requested no VO-relative
    knobs, so gating on it would fire on the entire default population — the
    "a signal that fires on everything is not a signal" shape. The renderer
    publishes the conjunction (calibration wanted AND unmeasurable) because it is
    the only place that knows both halves.

    NOT ARMED by default (see ``_GATE_ARMED_DEFAULTS``) — it still clears
    ``gate_pass`` and appears in ``violations``, which is the actual defect being
    fixed here.
    """
    if not bool(getattr(settings, "master_enabled", True)):
        return None
    if not music_bed:
        return None
    if not bool(music_bed.get("vo_measure_failed")):
        return None
    return MasteringGateViolation(
        stage="bed_calibration",
        tier=gate_tier("bed_calibration"),
        metric="vo_measure_failed",
        detail=(
            "bed_calibration: VO-relative bed levelling/ducking was requested but "
            "the narration bus could not be measured — the bed gain and the "
            "sidechain open threshold both fell back to literals, so the mix "
            "under the narration is uncalibrated"
        ),
    )


def apply_route_violations(
    report: dict[str, object],
    violations: Sequence[MasteringGateViolation | None],
    settings: Settings,
) -> dict[str, object]:
    """Fold route-built violations into ``report``, then RAISE if any is armed.

    THE ORDER IS THE CONTRACT. Every non-``None`` violation is folded FIRST —
    ``gate_pass`` cleared, the stage named, the structured entry published —
    and only then does arming decide whether delivery is also refused. A
    DISARMED gate therefore still degrades the report instead of vanishing,
    which is the exact defect this wave closes: ``master_measure_failed`` and
    ``vo_measure_failed`` were both recorded, published, and consumed by
    nothing.

    ``None`` entries are skipped, so a clean compile passes its report through
    unchanged and raises nothing.
    """
    for violation in violations:
        report = fold_violation(report, violation)
    armed = [v for v in violations if v is not None and gate_armed(settings, v.stage)]
    if armed:
        raise MasteringGateError(armed)
    return report


def fold_violation(
    report: dict[str, object], violation: MasteringGateViolation | None
) -> dict[str, object]:
    """Fold a ROUTE-BUILT violation into an already-returned mastering report.

    ``post_mux_verify`` builds ``violations``/``rerun_stages``/``gate_pass`` from
    what it can probe on the delivered file. Two findings are NOT probeable from
    the file — ``measure`` (pass-1 loudness measurement crashed) and
    ``bed_calibration`` (the narration bus could not be measured during the bed
    mix) — because both are self-reported by the RENDER and ride out of it on a
    ContextVar / the music-bed report. The compile route is the only place that
    holds both the report and those facts.

    Without this, a DISARMED such gate would vanish entirely: the route can raise
    when armed, but an unarmed violation has nowhere to go and the report would
    still say ``gate_pass=True`` with no note — exactly the "measured and then
    never used" defect this wave exists to close. So the fold happens regardless
    of arming, and arming decides only whether the route then RAISES.

    ``None`` returns the report unchanged (identity), so nothing moves on any
    compile that produced no such finding.
    """
    if violation is None:
        return report
    existing = report.get("violations")
    violations = list(existing) if isinstance(existing, list) else []
    stages = report.get("rerun_stages")
    rerun_stages = list(stages) if isinstance(stages, list) else []
    return {
        **report,
        "gate_pass": False,
        "rerun_stages": [*rerun_stages, violation.stage],
        "violations": [*violations, *violations_payload([violation])],
    }


def _banding_violation(frames: list[dict[str, float]]) -> MasteringGateViolation | None:
    """The banding half of the mastering gate: the violation it produces, or None.

    ⛔ **THE BANDING SCORE IS ADVISORY-ONLY. IT DOES NOT GATE, AT ANY CEILING.**
    Only the MEASUREMENT-GAP verdict below survives; the ``deband`` score verdict
    was removed from the gate on 2026-09-11. Two independent reasons, and the
    first alone is disqualifying:

    1. THE SCORE HAS NO SPATIAL TERM THAT CAN CARRY A GATE. ``banding_score``'s
       own docstring is an eight-paragraph record of this: banding is a SPATIAL
       artifact, the inputs are whole-frame signalstats aggregates, and the two
       documented-and-accepted misses (partial ramp 0.0820, limited-range
       gradient 0.1206) are arithmetically INSEPARABLE from clean dense detail
       (mandelbrot 0.071 vs the partial ramp's 0.082 — the same number). A score
       that cannot separate the defect from clean content cannot be calibrated
       into a trustworthy verdict at 0.15, at 0.25, or at any other value. This
       is a MEASUREMENT problem; only a per-tile/per-region measurement fixes it.
    2. MEASURED over n=120 completed masters, the whole fleet, 2026-08-27→09-11:
       worst-frame banding was min 0 / p10 .146 / p50 .642 / p90 .959 / max 1.00,
       and the 0.15 ceiling fired on **89.2%** of them. A verdict that convicts
       nine deliveries in ten is not a quality signal; it is what made
       ``gate_pass`` uninformative.

    The score is STILL measured and STILL published (``banding_scores``,
    ``per_shot[].banding``), and it still drives the deband TRIGGER
    (``banding_risk_frames`` / ``bucket_banding`` — a spatial filter applied to a
    suspect shot costs one extra pass and harms nothing). It just no longer
    decides whether a delivered master is acceptable. ``BANDING_GATE_CEILING``
    survives as the TRIGGER threshold, which is what it was always measured
    against; re-arming it as a gate needs a new measurement, not a new number.

    ONE remaining verdict — and separating it from the score verdict is what let
    an unmeasured master pass before:

    - NO FRAMES → ``banding_measurement``. ``measure_shot_frames`` swallows any ffmpeg
      failure and returns ``[]``, and ``_banding_gate_pass([])`` is ``True`` while
      ``luma_spread``/``dead_ratio`` are both ``None`` and their gates skip a
      measurement gap by design — so one transient probe failure used to return
      ``gate_pass=True``: the strongest verdict in the report, on zero evidence.
      Loudness already fails CLOSED on exactly this condition; the asymmetry was the
      bug. The stage is deliberately NOT ``deband``: the remediation is to MEASURE
      again, and re-running a filter cannot fix a probe that died.
    HISTORY, so this is not re-litigated: there used to be a second verdict here,
    ``deband`` (measured and over the ceiling), plus an UNSATISFIABLE-GATE GUARD
    suppressing it whenever the job disabled debanding
    (``tags={"deband":"off"}`` → ``deband_enabled`` False), because naming a
    remediation stage the render never runs fails a correct master forever under
    enforce. Both the verdict and its guard are gone: with the score itself
    advisory there is nothing left to suppress, and the deband-off case needs no
    special handling. ``banding_measurement`` never depended on
    ``deband_enabled`` — its remediation is to MEASURE again, which runs no
    filter — so dropping ``settings`` from the signature loses nothing.

    ``banding_measurement`` is KEPT GATING on purpose, and it is not the check the
    n=120 measurement condemns: all 120 masters produced frames, so it fired on
    NONE of them. It closes a real fail-open — with ``frames == []`` every
    picture-side signal (``luma_spread``, ``dead_frame_ratio``) is ``None`` and
    skipped by the no-gate-on-a-gap policy, so a dead signalstats probe would
    otherwise return ``gate_pass=True`` on zero picture evidence, which is the
    strongest verdict in the report backed by nothing.
    """
    if not frames:
        return MasteringGateViolation(
            stage="banding_measurement",
            tier=gate_tier("banding_measurement"),
            metric="banding_scores",
            detail=(
                "banding: the per-frame signalstats pass produced no frames — the "
                "delivered master is UNMEASURED, not proven clean"
            ),
        )
    # MEASURED AND OVER THE TRIGGER THRESHOLD: no violation, by decision. The
    # worst-frame score is reported, the deband filter already ran on the shots
    # that tripped the trigger, and residual score is NOT evidence of a bad
    # master (deband is spatial, the score's spatial term is a whole-frame
    # percentile ratio, and SSIM before/after deband on a real master was 0.9998
    # — the filter barely moves the picture, so "residual banding" here mostly
    # means "the score was never able to see this shot correctly"). See the
    # docstring above for the n=120 measurement and the inseparability argument.
    return None


def _cfr_violation(
    cfr_ok: bool | None,
    settings: Settings,
    *,
    measured: float | None,
    video_pts_gap_count: int | None = None,
) -> MasteringGateViolation | None:
    """The cadence gate's verdict — ``None`` unless its switch is ON **and** the
    master measured ``cfr_ok is False``.

    Mirrors the colorspace check in everything but one deliberate respect:

    * same trigger shape — only a MEASURED ``False`` fires. ``cfr_ok is None``
      (the probe returned nothing usable) is a measurement gap, never a
      violation, exactly as ``colorspace_ok is None`` is; it is surfaced through
      the report's ``unmeasured`` channel instead.
    * same violation shape — non-numeric, so ``value``/``limit`` stay ``None``
      and ``detail`` carries the reason.
    * DIFFERENT while disarmed: colorspace records its violation and clears
      ``gate_pass`` even with its switch off. cfr does NOT — with
      ``mastering_gate_cfr_enabled`` false this returns ``None`` and the report
      is byte-identical to the one published before the switch existed
      (``cfr_ok`` advisory, ``"cfr"`` in ``rerun_stages``, nothing else).

    The switch itself is read here, not ``gate_armed``: ``gate_armed`` is also
    true under ``mastering_gate_mode='enforce'``, and going through it would make
    enforce mode start failing compiles on cadence with this switch OFF — a
    behaviour change the switch's default must not carry. Once the violation
    exists, ``gate_armed("cfr")`` is what raises, as for every other stage."""
    if cfr_ok is not False:
        return None
    switch = _GATE_KILL_SWITCHES["cfr"]
    if not bool(getattr(settings, switch, _GATE_ARMED_DEFAULTS["cfr"])):
        return None
    expected = getattr(settings, "render_fps", None)
    # ``cfr_ok`` is False for either of two reasons (``_cfr_verdict``); name the
    # pts one when it is there, since the rates in the base text can both look fine.
    gaps = f"; {video_pts_gap_count} irregular video pts interval(s)" if video_pts_gap_count else ""
    return MasteringGateViolation(
        stage="cfr",
        tier=gate_tier("cfr"),
        metric="cfr_ok",
        detail=(
            f"cfr: delivered master is not constant-frame-rate against "
            f"render_fps {expected!r} (measured_fps {measured!r}){gaps}"
        ),
    )


def _banding_gate_pass(scores: list[float]) -> bool:
    """The delivered file's worst banding score is under the deband TRIGGER
    threshold (i.e. any score-visible banding was already handled).

    ⚠️ NAMED ``_banding_gate_pass`` FOR HISTORY ONLY — banding no longer gates
    (see ``_banding_violation``). This is the shared TRIGGER predicate: it backs
    ``banding_risk_frames`` (which shot gets the deband filter) and
    ``bucket_banding``'s per-shot ``banding_residual_over_trigger`` flag, both
    advisory/remedial.
    Nothing it returns reaches ``gate_pass``."""
    return not scores or max(scores) < BANDING_GATE_CEILING


def banding_advisories(scores: list[float]) -> list[dict[str, object]]:
    """The residual-banding breach as a published ADVISORY, or ``[]``. PURE.

    WHY: run ``01a0ab23-6c47-76a2-9d53-f16220e4792d`` delivered ``needs_deband``
    on all three shots (banding up to 0.5935) with ``gate_pass: true`` and
    ``violations: []`` — the per-shot flag and the verdict disagreed, and nothing
    at the report's top level said the breach was seen and deliberately not gated.

    Same predicate as the per-shot flag (``_banding_gate_pass``), so ANY shot with
    ``needs_deband`` implies an entry here: the whole-file max is >= every shot's
    max. Same five-key shape as a violation (``MasteringViolation``), tier
    ``tunable`` because the 0.15 trigger is uncalibrated on the current formula.

    ⛔ NOT a ``violations`` entry, and it must never become one. Every
    ``violations`` entry is a check that CLEARED ``gate_pass`` — this repo's model
    and the orchestrator's ``mastering_gate_degradations`` both say so — and
    ``mastering_gate_mode=enforce`` arms every stage that lands there, so moving it
    would fail compiles on an ungated score. Same reasoning as ``cfr_ok``."""
    if _banding_gate_pass(scores):
        return []
    return [
        MasteringViolation(
            stage="banding_residual",
            metric="banding_scores",
            value=max(scores),
            limit=BANDING_GATE_CEILING,
            tier="tunable",
        ).model_dump(mode="json")
    ]


def bucket_banding(
    timed_frames: list[tuple[float | None, dict[str, float]]],
    shot_timeline: list[dict[str, object]],
    *,
    spatial_factors: dict[float, float] | None = None,
) -> list[dict[str, object]]:
    """Bucket per-frame banding onto each shot's ``[start, end)`` window → one
    ``{"line_id", "banding", "needs_deband"}`` entry per shot. PURE.

    For every shot in ``shot_timeline`` (each carries ``line_id``, ``start``,
    ``end`` seconds on the final bus), collect the timed frames whose ``pts_time``
    is not ``None`` and falls in ``[start, end)``, and take the WORST (max)
    ``banding_score`` — a localized band in a few frames stays visible instead of
    being averaged away. A shot with no in-window measured frames scores ``0.0``.
    ``banding_residual_over_trigger`` reuses the SAME predicate as the whole-file
    banding gate (``_banding_gate_pass``) so the per-shot and whole-file
    thresholds can never drift. NEVER raises.

    ⚠️ IT IS PUBLISHED UNDER TWO NAMES, AND THAT IS DELIBERATE.
    ``banding_residual_over_trigger`` is the honest one: this is measured on the
    DELIVERED file, i.e. AFTER the conditional deband pass already ran, so a
    true value says "residual banding is still over the trigger threshold", not
    "a deband is owed". The old name ``needs_deband`` reads as an unmet ACTION
    and is how ``rerun_stages=["deband"]`` → "no stage was re-run; degrading"
    kept being chased. It is kept as an exact alias because there is a LIVE
    consumer: ``w2v_orchestrator``'s ``demo_video/encode_master.py`` names it in
    the critic prompt AND pins it in a structured-output JSON schema, whose test
    asserts the per-shot property set EXACTLY. Dropping it here would break that
    repo. Retire the alias in a coordinated change once the orchestrator reads
    the new name.

    ``spatial_factors`` (optional, ``{pts_time: factor}``) folds in the
    pixel-neighbourhood banding term (see ``spatial_banding_factor``) for
    whichever frames the caller already paid to re-measure — typically each
    shot's own worst frame, via ``post_mux_verify``'s bounded spatial pass.
    A ``pts_time`` absent from the map scores exactly as before (``None``
    ``spatial_factor`` — ``banding_score`` leaves it unmultiplied)."""
    result: list[dict[str, object]] = []
    spatial = spatial_factors or {}
    for shot in shot_timeline:
        start = _as_float(shot.get("start"))
        end = _as_float(shot.get("end"))
        scores = [
            banding_score(metrics, spatial_factor=spatial.get(pts))
            for pts, metrics in timed_frames
            if pts is not None and start is not None and end is not None and start <= pts < end
        ]
        banding = max(scores, default=0.0)
        over_trigger = not _banding_gate_pass([banding])
        result.append(
            {
                "line_id": shot.get("line_id"),
                "banding": banding,
                "banding_residual_over_trigger": over_trigger,
                # Back-compat alias — same value, same object. See the docstring.
                "needs_deband": over_trigger,
            }
        )
    return result


# Spatial-refinement bounds. Frames whose RAW score clears
# ``BANDING_SPATIAL_PREFILTER`` are measured worst-first in at most a few
# streamed ffmpeg passes (``measure.measure_spatial_factors_streamed``): one
# decode of the master per pass, only the selected frames converted + piped,
# one gray frame in memory at a time. This replaced (2026-10-21) a per-frame
# seek-and-decode loop capped at
# 96 extra decodes, which was the binding constraint on normal films: the three
# 2026-10-20 generic masters carry 694-890 prefilter frames (static ken-burns
# dwells, near-identical raw scores), so the budget ran out every time and the
# reported score stayed the RAW upper bound (0.259/0.357/0.328) although every
# corrected frame was <= 0.062.
#
# Both bounds below are ceilings, not the normal path; exceeding either
# leaves the unmeasured frames RAW, i.e. the reported score is an UPPER bound
# (never lower than the true corrected max) — it can leave a false positive
# standing, never hide banding. Live 2026-10-21 (generic 01a15bda-305d, prod
# node, 2560x1440 30s): #254's serial 2-pass settle took 42.6s of the 60s
# budget, so anything much longer fell back to the upper bound.
#
# What it costs, measured 2026-10-21 locally on a synthetic clean 2560x1440 120s
# clip (3,396 prefilter frames — every one must be measured, see
# ``_banding_spatial_frontier``): one full decode through the spatial filter is
# ~11s; the pure-Python detector ~15-20ms/frame. The serial 2-pass settle took
# 64s (round 1 = a WHOLE extra decode to measure 16 frames, then a serial
# measure the pipe waited on). Now the detector runs on
# ``_banding_spatial_workers()`` worker interpreters while ffmpeg keeps
# decoding, and frames that can no longer be any group's max are SKIPPED as
# results land (``_BandingPruner`` — exact, the factor only attenuates), so the
# detector's work is proportional to the frames that can still matter.
#
# The candidate cap is a pathological bound only (10 min at 30 fps); the budget
# is what bounds wall time.
_BANDING_SPATIAL_MAX_CANDIDATES = 18000
_BANDING_SPATIAL_BUDGET_SECONDS = 60.0
# Pass 1 streams each group's top-16 raw frames; pass 2 the WHOLE remaining
# frontier. ffmpeg's cost is per SELECTED frame (scale + gray + a ~2 MB PGM
# down the pipe; decoding the rest is cheap), so pass 1 is cheap even on a
# clean film and settles a GENUINELY banded one on its own: measured 2026-10-21
# on the synthetic banded 120s clip, 16 frames in 2.8s, vs 51s to pipe all
# 3,400 prefilter frames and prune them. Do not fold it into pass 2.
_BANDING_SPATIAL_FIRST_ROUND = 16
# Worker interpreters for the detector: the pod's REAL CPU budget (cgroup quota,
# not ``os.cpu_count`` — worker4 has 12 CPUs, the pod a 10-core limit) minus 2
# (ffmpeg's decode threads + the event loop), capped. Other compiles' encodes
# share the same limit; CPU is compressible, so contention throttles, never fails.
_BANDING_SPATIAL_MAX_WORKERS = 8
_BANDING_SPATIAL_RESERVED_CPUS = 2


def _banding_groups(
    timed_frames: list[tuple[float | None, dict[str, float]]],
    shot_timeline: list[dict[str, object]] | None,
) -> list[list[tuple[float, dict[str, float]]]]:
    """PURE. The frame sets whose MAX ``banding_score`` is reported: the whole
    file first, then each shot's ``[start, end)`` window in ``shot_timeline``
    order (same windowing as ``bucket_banding``). Frames without a
    ``pts_time`` cannot be addressed by a spatial decode and are left out —
    exactly as before (the whole-file max still reads them, uncorrected)."""
    timed = [(pts, m) for pts, m in timed_frames if pts is not None]
    groups = [timed]
    for shot in shot_timeline or []:
        start = _as_float(shot.get("start"))
        end = _as_float(shot.get("end"))
        if start is None or end is None:
            continue
        groups.append([(pts, m) for pts, m in timed if start <= pts < end])
    return groups


def _banding_spatial_frontier(
    timed_frames: list[tuple[float | None, dict[str, float]]],
    shot_timeline: list[dict[str, object]] | None,
    factors: dict[float, float],
    attempted: set[float],
    *,
    per_group: int = 1,
) -> list[float]:
    """PURE. The next ``pts_time`` values worth a spatial decode.

    A group's reported score is ``max(banding_score(m, spatial_factor=f))``
    over its frames, where an un-decoded frame scores RAW (``f is None``). The
    spatial factor only ever ATTENUATES, so for every frame ``corrected <=
    raw``. Let ``best`` be the max over the group's ALREADY-ATTEMPTED frames
    (corrected where the decode succeeded, raw where it failed). Any
    unattempted frame with ``raw <= best`` cannot change the group's max, and
    any with ``raw < BANDING_SPATIAL_PREFILTER`` is far under the ceiling
    either way. So the frontier is, per group, the top ``per_group``
    unattempted frames by raw score with ``raw > best`` and ``raw >=
    PREFILTER``. An EMPTY frontier means every group's reported max is the
    exact corrected max (or provably under the prefilter).

    This is the fix for the reported false positive: the old candidate set
    (each group's single raw-worst frame) corrected the worst frame DOWN and
    then let the next-worst, never-decoded frame become the reported max at
    its raw value (a showcase master reported 0.2661 vs the 0.15 ceiling).

    Order: whole-file group first, then shots; deduped; ``per_group=1`` with
    nothing attempted is exactly the historical candidate set."""
    frontier: list[float] = []
    seen: set[float] = set()
    for group in _banding_groups(timed_frames, shot_timeline):
        best = 0.0
        pending: list[tuple[float, float]] = []
        for pts, m in group:
            if pts in attempted:
                best = max(best, banding_score(m, spatial_factor=factors.get(pts)))
            else:
                raw = banding_score(m)
                if raw >= BANDING_SPATIAL_PREFILTER:
                    pending.append((raw, pts))
        pending.sort(key=lambda t: t[0], reverse=True)
        for raw, pts in pending[:per_group]:
            if raw <= best:
                break
            if pts not in seen:
                seen.add(pts)
                frontier.append(pts)
    return frontier


def _banding_spatial_candidates(
    timed_frames: list[tuple[float | None, dict[str, float]]],
    shot_timeline: list[dict[str, object]] | None,
) -> list[float]:
    """PURE. The FIRST refinement round: the whole-file worst frame plus each
    shot's own worst frame, only when it clears ``BANDING_SPATIAL_PREFILTER``.
    ``_settle_banding_spatial_factors`` continues from here."""
    return _banding_spatial_frontier(timed_frames, shot_timeline, {}, set(), per_group=1)


class _BandingPruner:
    """Per-group running corrected maxima, updated as spatial factors land.

    ``needed(pts)``: could measuring this frame still change ANY group's
    reported max? Only if its RAW score beats that group's current best over
    measured frames (the factor only attenuates, so ``corrected <= raw``) — the
    same test ``_banding_spatial_frontier`` applies, evaluated mid-pass. A
    skipped frame has ``raw <= best <= final max``, so the result stays EXACT."""

    def __init__(
        self,
        timed_frames: list[tuple[float | None, dict[str, float]]],
        shot_timeline: list[dict[str, object]] | None,
        factors: dict[float, float],
    ) -> None:
        self._stats = {pts: m for pts, m in timed_frames if pts is not None}
        self._raw = {pts: banding_score(m) for pts, m in self._stats.items()}
        self._groups_of: dict[float, list[int]] = {}
        groups = _banding_groups(timed_frames, shot_timeline)
        for g, group in enumerate(groups):
            for pts, _ in group:
                self._groups_of.setdefault(pts, []).append(g)
        self._best = [0.0] * len(groups)
        for pts, factor in factors.items():
            self.record(pts, factor)

    def record(self, pts: float, factor: float) -> None:
        score = banding_score(self._stats[pts], spatial_factor=factor)
        for g in self._groups_of.get(pts, ()):
            self._best[g] = max(self._best[g], score)

    def needed(self, pts: float) -> bool:
        raw = self._raw[pts]
        return any(raw > self._best[g] for g in self._groups_of.get(pts, ()))


def _banding_spatial_workers() -> int:
    """Worker interpreters for the detector (see the constants above); 1 = serial."""
    from src.services import measure

    cpus = measure.available_cpus()
    return max(1, min(_BANDING_SPATIAL_MAX_WORKERS, cpus - _BANDING_SPATIAL_RESERVED_CPUS))


def _init_banding_worker(paths: list[str]) -> None:
    """Worker-interpreter initializer: the parent's ``sys.path`` so ``src`` is
    importable. Self-contained (runs against the worker's own globals)."""
    import sys

    sys.path[:] = paths


def _loop_spawns_safely_with_subinterpreters() -> bool:
    """True when the running event loop (if any) is a stdlib asyncio loop, whose
    subprocesses go through ``subprocess.Popen`` (no Python code in the child).
    uvloop's ``Loop`` is not a ``BaseEventLoop`` and is NOT safe: see
    ``_banding_executor``. No running loop: nothing spawns through one -> True."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return True
    return isinstance(loop, asyncio.BaseEventLoop)


def _banding_executor(workers: int) -> Executor | None:
    """An ``InterpreterPoolExecutor`` (own GIL per worker, stdlib-only module
    import, no process fork of a multi-GiB pod) proven by one probe call, or
    ``None`` -> the serial thread path. Never raises."""
    if workers <= 1:
        return None
    if not _loop_spawns_safely_with_subinterpreters():
        # uvloop (or any non-stdlib loop): its fork-based spawn runs CPython's
        # after-fork-child hook, which segfaults the ffmpeg CHILD while worker
        # interpreters exist (2026-10-21: every master, signal 11). Serial is
        # slower but correct; ``asgi.main`` pins the stdlib loop so prod is safe.
        log.warning("banding.spatial_pool_unavailable", error="event loop is not stdlib asyncio")
        return None
    import sys
    from concurrent.futures import InterpreterPoolExecutor

    from src.services import banding_spatial

    pool: Executor | None = None
    try:
        pool = InterpreterPoolExecutor(
            max_workers=workers, initializer=_init_banding_worker, initargs=(list(sys.path),)
        )
        probe = bytes(range(16)) * 16
        rows, cols = banding_spatial.spatial_banding_lines(probe, 16, 16)
        got = pool.submit(banding_spatial.measure_lines_in_worker, rows, cols, 16, 16).result(30)
        if got != banding_spatial.spatial_factor_from_lines(rows, cols, 16, 16):
            raise RuntimeError(f"worker probe mismatch: {got!r}")
    except Exception as exc:  # noqa: BLE001 — fall back to serial, never fail a job
        log.warning("banding.spatial_pool_unavailable", error=str(exc))
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        return None
    return pool


async def _settle_banding_spatial_factors(
    path: Path | str,
    timed_frames: list[tuple[float | None, dict[str, float]]],
    shot_timeline: list[dict[str, object]] | None,
    *,
    workers: int | None = None,
) -> dict[float, float]:
    """Spatial factors until every reported max (whole file and each shot) is
    the EXACT corrected max.

    Pass 1 streams each group's top ``_BANDING_SPATIAL_FIRST_ROUND`` raw-score
    frames; pass 2+ the whole remaining frontier (``_banding_spatial_frontier``,
    worst-first under the candidate cap), each in ONE ffmpeg pass, the detector
    in parallel on
    ``workers`` interpreters (default ``_banding_spatial_workers()``; 1 =
    serial), skipping frames that can no longer be any group's max as results
    land (``_BandingPruner``). A pass that lands nothing stops the loop. The spatial factor only ever
    LOWERS a score, so an empty frontier proves exactness. Bounds
    (``_BANDING_SPATIAL_MAX_CANDIDATES`` frames, ``_BANDING_SPATIAL_BUDGET_SECONDS``
    across ALL passes): past either, unmeasured frames stay RAW — an UPPER bound,
    never lower than the truth. Best-effort: a failed pass keeps what earlier
    passes measured (``{}`` if none), never a job failure."""
    from src.services import measure

    if not any(
        pts is not None and banding_score(m) >= BANDING_SPATIAL_PREFILTER for pts, m in timed_frames
    ):
        return {}
    loop = asyncio.get_running_loop()
    raw_by_pts = {pts: banding_score(m) for pts, m in timed_frames if pts is not None}
    all_pts = [pts for pts, _ in timed_frames if pts is not None]
    factors: dict[float, float] = {}
    pruner = _BandingPruner(timed_frames, shot_timeline, factors)
    n_workers = _banding_spatial_workers() if workers is None else max(1, workers)
    executor = _banding_executor(n_workers)
    # The budget clock starts once the pool is READY: worker start-up (~20ms
    # locally, slower on a loaded CI runner; the probe has its own 30s bound) is
    # not settle work, and counting it let a slow start eat a pass's budget.
    started = loop.time()
    deadline = started + _BANDING_SPATIAL_BUDGET_SECONDS
    per_group = _BANDING_SPATIAL_FIRST_ROUND
    passes = 0
    reason = "exact"
    try:
        while frontier := _banding_spatial_frontier(
            timed_frames, shot_timeline, factors, set(factors), per_group=per_group
        ):
            room = _BANDING_SPATIAL_MAX_CANDIDATES - len(factors)
            remaining = deadline - loop.time()
            if room <= 0 or remaining <= 0:
                reason = "candidate_cap" if room <= 0 else "budget"
                break
            frontier.sort(key=lambda p: raw_by_pts[p], reverse=True)
            got = await measure.measure_spatial_factors_streamed(
                path,
                all_pts,
                set(frontier[:room]),
                budget_seconds=remaining,
                executor=executor,
                window=2 * n_workers,
                needed=pruner.needed,
                on_factor=pruner.record,
            )
            passes += 1
            new = {p: f for p, f in got.items() if p not in factors}
            factors.update(new)
            for p, f in new.items():
                pruner.record(p, f)  # idempotent for a real pass; fakes may skip on_factor
            # A wanted frame neither measured nor pruned (``needed`` only ever
            # turns False as maxima rise) means the pass ended early: budget
            # expiry mid-pass (something landed) or a failed pass (nothing).
            # Stop either way; unmeasured frames stay RAW.
            if any(p not in factors and pruner.needed(p) for p in frontier[:room]):
                reason = "budget" if new else "pass_failed"
                break
            per_group = len(timed_frames)  # pass 2+: the whole frontier at once
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    exact = reason == "exact"
    log.info(
        "banding.spatial_settled",
        path=str(path),
        exact=exact,
        passes=passes,
        workers=n_workers if executor is not None else 1,
        measured=len(factors),
        seconds=round(loop.time() - started, 2),
    )
    if not exact:
        # Upper bound, not exact: an unmeasured frame may still be the reported
        # max somewhere. Fails toward a (possibly false) advisory, never toward
        # hiding banding.
        log.info(
            "banding.spatial_refinement_incomplete",
            path=str(path),
            reason=reason,
            candidates=sum(1 for r in raw_by_pts.values() if r >= BANDING_SPATIAL_PREFILTER),
            measured=len(factors),
        )
    return factors


async def post_mux_verify(
    path: Path | str,
    settings: Settings,
    *,
    loudness: dict[str, Any] | None = None,
    shot_timeline: list[dict[str, object]] | None = None,
    shots: Sequence[object] | None = None,
    narration_audio_paths: Sequence[Path | None] | None = None,
) -> dict[str, Any]:
    """Read-only soft-gate over the delivered mux → a MasteringReport-shaped dict.

    GATING checks — each one that fails clears ``gate_pass`` AND names its
    remediation in ``rerun_stages``. There are FOUR, plus one evidence check:
    the BT.709 tag is present (``colorspace``), the (reused, not re-measured)
    loudness sits near target or is measurable at all (``loudness``), the master
    is not dead/blank (``dead_frame``), the picture covers the narration
    (``av_coverage``), and the picture was MEASURED AT ALL
    (``banding_measurement`` — an empty signalstats pass, which leaves every
    picture-side signal ``None``).

    ⛔ TWO CHECKS WERE REMOVED FROM THIS LIST ON 2026-09-11, measured over n=120
    completed masters (the whole fleet, 2026-08-27→09-11), because between them
    they made ``gate_pass`` carry no information at all:

    * ``duplicate_frames`` — fired on **120 of 120** (100%), minimum ratio 0.643
      against a 0.10 ceiling. Duplicate frames are intentional dwell. The
      predicate and its setting are GONE, not widened.
    * ``deband`` (worst-frame banding score) — fired on **89.2%**, from a score
      with no usable spatial term. Advisory now; see ``_banding_violation``.

    Both metrics are still measured and still published. ``colorspace_ok`` fails
    on 23.3% of masters, which is an upstream TAGGING regression to fix at the
    source — it still clears ``gate_pass`` and still reports a ``violations``
    entry, but it is deliberately NOT ARMED (see ``_GATE_ARMED_DEFAULTS``):
    arming a 23.3% fire rate on a defect this service neither causes nor can
    remediate would fail one delivery in four. It was not widened to hide it
    either.

    A gating check NEVER fires for a stage THIS JOB DISABLED. ``loudness``
    (``tags={"mastering":"off"}`` → ``master_enabled`` False) names a remediation
    the render skipped outright, so gating on it demands a re-run of a stage that
    does not run and cannot move the number — a correct render fails forever under
    enforce. See ``_loudness_violation``. (The same guard existed for ``deband``
    and went with the verdict it protected.) The metrics stay REPORTED either way;
    only the gating is suppressed.

    ADVISORY-ONLY signals — REPORTED but NEVER gating (never a ``violations``
    entry, never clear ``gate_pass``), because no stage of this pipeline can
    remediate them and a frame-rate-incoherent-but-playable master is
    suboptimal, not unshippable: ``cfr_ok``, ``measured_fps``, ``luma_spread``,
    ``true_peak_dbtp``, ``duplicate_frame_ratio`` and ``banding_scores`` /
    ``per_shot[].banding``. Read ``cfr_ok is False`` directly to act on cadence;
    do NOT infer it from ``gate_pass``. ``cfr_ok is False`` DOES append
    ``"cfr"`` to ``rerun_stages`` (3d, emission only — see the block above the
    ``rerun_stages`` list): a plain list append, never a ``violations`` entry,
    so cadence joins the "stages that might help" vocabulary WITHOUT joining
    the gating set above. The ONE exception is opt-in:
    ``mastering_gate_cfr_enabled`` (default OFF) turns a measured
    ``cfr_ok is False`` into a ``violations`` entry that fails the compile; off,
    nothing here changes (see ``_cfr_violation``).

    ⚠️ **THIS FUNCTION RAISES BY DEFAULT NOW (WAVE-13, owner decision).** The
    long-standing "soft gate never fails the render" property is GONE for the
    STRUCTURAL tier. A violation of ``av_coverage``, ``dead_frame`` or
    ``banding_measurement`` raises ``MasteringGateError`` under the DEFAULT
    settings, because the governing failure mode of this pipeline was that a
    green run proved nothing: ``gate_pass`` was computed correctly and then
    decided nothing, since the only enforcement path was a whole-gate
    ``mastering_gate_mode='enforce'`` that no deployment ever set.

    Arming is PER GATE (``gate_armed`` / ``_GATE_ARMED_DEFAULTS``) and each gate
    has its own kill switch in ``Settings``, so a threshold that turns out to be
    wrong is disarmed with one env var and no code change. The TUNABLE tier
    (``loudness``) and the measured-refusal ``colorspace`` are wired and DEFAULT
    OFF: they are still REPORTED in ``violations`` with their tier, which is how
    the fleet distribution needed to arm them gets collected.

    A probe failure still degrades the individual check to ``None`` — an UNPROBED
    check is not a violation, and no-gate-on-a-gap is unchanged. The one thing
    that IS a violation is measuring NOTHING AT ALL
    (``banding_measurement``), because that leaves every picture-side gate
    "passing" on zero evidence. ``mastering_gate_mode='enforce'`` remains
    available as an escalation that arms every stage at once.
    ``integrated_lufs``/``true_peak_dbtp`` are REUSED from the caller's
    ``final_mux_loudness`` result — this function does not re-measure loudness."""
    from src.services import measure

    loud = loudness or {}
    integrated_lufs = _as_float(loud.get("integrated_lufs"))
    true_peak_dbtp = _as_float(loud.get("true_peak_dbtp"))
    # Third reused number from the SAME ``final_mux_loudness`` pass — astats rides
    # that pass as a pass-through analyser, so the noise floor costs no extra probe
    # and no extra decode. ``None`` for a caller whose loudness dict predates the
    # key, which yields no speech-presence verdict rather than a false one.
    noise_floor_db = _as_float(loud.get("noise_floor_db"))
    speech_gap_db = program_to_noise_floor_db(integrated_lufs, noise_floor_db)

    colorspace_ok: bool | None = None
    expected_cs = normalize_colorspace(getattr(settings, "render_colorspace", ""))

    # ONE round of concurrent measurement over the delivered master.
    #
    # These are five INDEPENDENT read-only probes of the same immutable file, and
    # they used to run strictly serially — ``master_verify`` measured 26.5s mean
    # in production (24% of total job time), which was simply their sum. Two of
    # them are full decodes (signalstats, and mpdecimate inside
    # ``duplicate_frame_ratio``); the rest are header probes. Serial, the wall
    # time is decode+decode+probes. Concurrent, it is max(decode, decode)+probes.
    #
    # This is a SCHEDULING change only: every verdict below is computed from the
    # same values, in the same order, so the returned report is byte-identical.
    # The ffmpeg semaphore (``MAX_CONCURRENT_FFMPEG_SUBPROCESSES``, 3 in prod)
    # still bounds real parallelism, and it sits idle at this point in the job —
    # verification runs after the render fan-out has finished, so every slot is
    # free. That is the same argument #187 used to unserialise the deband and
    # source-match measure passes.
    #
    # A bare ``gather`` (no ``return_exceptions``) is deliberate and safe: every
    # one of these five is best-effort by contract and returns None/[] rather
    # than raising, so there is no exception to propagate and therefore no
    # sibling left running orphaned. If any of them is ever made to raise, this
    # must become a TaskGroup — the reason is recorded here so that change is not
    # made silently.
    (
        probed,
        timed_frames,
        rate_coherent,
        measured,
        dup_ratio,
        av_and_gaps,
    ) = await asyncio.gather(
        _probe_colorspace(path) if expected_cs else _none(),
        measure.measure_shot_frames_timed(path),
        framerate_coherence(path, getattr(settings, "render_fps", None)),
        measured_fps(path),
        duplicate_frame_ratio(path),
        # A/V coverage. Joins the concurrent round for the same reason as the other
        # five (the render fan-out is finished, every slot is idle) and honours the
        # same contract: it returns (None, None) on any failure rather than raising,
        # so the bare ``gather`` stays safe.
        #
        # Paired (an inner ``gather``, not a seventh argument — ``asyncio.gather``
        # is only typed to six) with the holes in the audio packet timeline:
        # header-only, read-only, advisory, and ``(None, None)`` on any failure.
        #
        # ...and with the holes in the VIDEO packet timeline, the half of ``cfr_ok``
        # that ``framerate_coherence`` cannot see (``_cfr_verdict``).
        asyncio.gather(
            av_stream_seconds(path), audio_pts_gap_report(path), video_pts_gap_report(path)
        ),
    )
    (
        (video_seconds, audio_seconds),
        (audio_pts_gap_count, audio_pts_gap_ms_total),
        (video_pts_gap_count, video_pts_gap_positions),
    ) = av_and_gaps
    cfr_ok = _cfr_verdict(rate_coherent, video_pts_gap_count)
    av_gap = _av_gap_seconds(video_seconds, audio_seconds)
    # Compare LIKE WITH LIKE: the probed ``(primaries, transfer, matrix)`` against
    # the very triplet ``colorspace_tag_args`` handed the encoder for this
    # ``render_colorspace``. ``expected_cs`` is a colorspace NAME and the probe
    # returns ffmpeg's per-component names; comparing the name to the matrix (what
    # this line used to do) is a category error that only ever agreed for bt709.
    expected_triplet = _COLORSPACE_TRIPLETS.get(expected_cs)
    # Whether the colorspace check was even ASKED FOR. `colorspace_ok is None`
    # carries two different facts and the `unmeasured` channel below must not let
    # them share a value: "no `render_colorspace` is configured, so there is
    # nothing to check and the probe never ran" (the check is OFF) versus "a
    # colorspace WAS requested and the probe failed" (the check could not run).
    # Only the second is a measurement gap. Conflating them would make `unmeasured`
    # non-empty on the ordinary disabled path — i.e. every `mastering: off` job
    # would report an unverified master — which is the mirror-image false signal of
    # the fail-open it exists to close.
    colorspace_gate_active = bool(expected_cs) and expected_triplet is not None
    if colorspace_gate_active:
        colorspace_ok = probed == expected_triplet if probed is not None else None

    # Score banding PER FRAME and keep the WORST frame's score, not the segment
    # mean — a localized band in a few frames must survive to the gate instead of
    # being averaged away across a mostly-clean shot. On clean content every frame
    # scores ~0 → max ~0, so the gate result is unchanged.
    # ONE discarded signalstats decode for BOTH the whole-file gate and the
    # per-shot bucketer. ``parse_signalstats_per_frame_timed`` performs the SAME
    # ``_FRAME_MARKER_RE`` split and the SAME ``_block_metrics`` per block as its
    # untimed sibling (including the no-marker aggregate degrade); it only adds
    # ``pts_time``. So the untimed list is EXACTLY ``[m for _, m in timed]`` and is
    # derived here rather than paid for with a second full decode of the delivered
    # master — that duplicate pass was ~a quarter of ``master_verify``'s wall time.
    frames = [m for _, m in timed_frames]

    # Spatial re-measure (``_settle_banding_spatial_factors``). The reported
    # score is a MAX over all frames, so every frame that could still be a
    # group's max is corrected (worst first, streamed passes) — correcting
    # only the raw-worst frame let the next-worst, never-decoded frame become
    # the reported max at its RAW value (showcase master: 0.2661 vs 0.15, a
    # false positive).
    # Nothing flagged: no decode. Bounds exceeded: an UPPER bound, never lower.
    spatial_factors = await _settle_banding_spatial_factors(path, timed_frames, shot_timeline)

    scores = (
        [
            max(
                (
                    banding_score(m, spatial_factor=spatial_factors.get(pts))
                    for pts, m in timed_frames
                ),
                default=0.0,
            )
        ]
        if frames
        else []
    )

    # DEAD/BLANK-master gate signal: the WORST per-frame intra-frame luma spread
    # (YMAX-YMIN) over the delivered file. ~0 → a uniform/blank (dead) region;
    # ~200 → a legitimate static UI screen. duplicate_frame_ratio can't tell those
    # apart; this spread can. Per-frame (reusing the ``frames`` pass above, no
    # extra ffmpeg run) so a master that is blank for only PART of its length —
    # which averages to a wide spread — is still caught. ``None`` when no frame
    # carried luma data.
    luma_spread = _worst_luma_spread(frames)

    # PER-SHOT banding (S20, additive): buckets the SAME timed measure pass read
    # above so the encode-critic can populate a
    # ``per_shot:[{line_id,banding,needs_deband}]`` schema. Read-only + additive —
    # the whole-file gate above (``scores`` / ``banding_scores``) is byte-identical
    # whether or not boundaries are supplied. Only bucketed when the caller passed
    # shot boundaries; otherwise ``per_shot`` is [].
    per_shot: list[dict[str, object]] = []
    if shot_timeline:
        per_shot = bucket_banding(timed_frames, shot_timeline, spatial_factors=spatial_factors)

    # ADVISORY, BY DECISION — ``cfr_ok is False`` deliberately does NOT clear
    # ``gate_pass`` and does NOT add a ``violations`` entry, unlike colorspace /
    # deband / loudness / dead_frame below. Two reasons, both load-bearing:
    #   1. No remediation exists. Every GATING signal names a stage that can fix
    #      it; cadence is owned by VC-3 at encode time and this probe is
    #      explicitly read-only (see ``framerate_coherence``). There is no
    #      "re-run cadence" stage, so gating here would wedge a run on something
    #      nothing downstream can repair.
    #   2. It is not an unshippable defect. A stream whose r_frame_rate and
    #      avg_frame_rate differ by >0.01, or whose avg sits >0.5 off
    #      ``render_fps``, still decodes and plays everywhere — unlike a blank
    #      master or an untagged colorspace. The gate is already noisy in the
    #      field; adding a non-actionable contributor makes ``gate_pass`` less
    #      usable.
    #
    # ``"cfr"`` IS still appended to ``rerun_stages`` below (2026-09-13, 3d) when
    # ``cfr_ok is False`` — emission only, no arming. #218 made CFR real
    # (``-vsync cfr`` now unconditional in production, not just under
    # ``deterministic``), so ``cfr_ok`` finally has a live producer worth a
    # consumer; before this it was a dead key on the report. This does NOT touch
    # ``gate_pass``/``violations``/``degraded`` — it is a plain list append, so a
    # consumer that already treats ``rerun_stages`` as "stages that MIGHT help"
    # (rather than "the gate failed") gains cadence as an option. Fleet
    # cfr_ok=False rates have not been re-measured since #218 shipped, so this
    # stays advisory until they are.
    # ``duplicate_frame_ratio`` is advisory for the SAME reason and then some, and
    # this note records a round trip so it is not made a third time. The rationale
    # originally ended "...which is exactly why ``duplicate_frame_ratio`` is
    # advisory too"; W1-3 OVERTURNED that and armed the ratio on the argument that
    # a 0.9197 master is "nine times its own ceiling" and reads as a frozen video.
    # The fleet measurement settled it: across 120 of 120 completed masters the
    # ratio ran 0.643-0.998, i.e. the W1-3 exemplar was not an outlier, it was the
    # MEDIAN, because the dwell-plus-25fps capture makes duplicates the format. The
    # gate and its per-deployment ceiling carve-out are gone (2026-09-11) rather
    # than re-tuned, since no ceiling separates dwell from a frozen render.
    # The signal is not lost: ``cfr_ok`` is returned tri-state (True coherent /
    # False incoherent / None unprobed) in the payload and declared on
    # MasteringReport, so a caller that DOES want to act on cadence reads it
    # directly. If cadence ever becomes remediable, this is the line to change.
    # (``cfr_ok``, ``measured`` and ``dup_ratio`` are measured in the concurrent
    # round above; this comment stays here because it documents how ``cfr_ok`` is
    # USED by the gate below, not how it is obtained.)
    # VC-3 measured-cadence observability (advisory; never gates the render).

    rerun_stages: list[str] = []
    # 3d — emission only, see the ADVISORY comment above: "cfr" is a plain list
    # append, never a ``violations`` entry, so it can never clear ``gate_pass``
    # or count toward the armed-gate set (``gate_armed``) / ``degraded``.
    if cfr_ok is False:
        rerun_stages.append("cfr")
    # One entry per cleared check, carrying the measured value + the ceiling it
    # breached. Built unconditionally (it is cheap and keeps the two modes on ONE
    # code path) but only ever READ under enforce — in advisory mode it is
    # discarded and the returned payload is byte-identical to before.
    violations: list[MasteringGateViolation] = []
    gate_pass = True
    # ``MASTERING_GATE_CFR_ENABLED`` (default OFF). ``"cfr"`` is already in
    # ``rerun_stages`` from the unconditional advisory append above, so it is NOT
    # appended a second time here; with the switch off ``_cfr_violation`` is
    # ``None`` and this block is a no-op.
    cfr_violation = _cfr_violation(
        cfr_ok, settings, measured=measured, video_pts_gap_count=video_pts_gap_count
    )
    if cfr_violation is not None:
        violations.append(cfr_violation)
        gate_pass = False
    if colorspace_ok is False:
        rerun_stages.append("colorspace")
        violations.append(
            MasteringGateViolation(
                stage="colorspace",
                tier=gate_tier("colorspace"),
                metric="colorspace_ok",
                detail=(
                    f"colorspace: delivered (primaries, transfer, matrix) {probed!r} "
                    f"is not the {expected_cs!r} triplet {expected_triplet!r}"
                ),
            )
        )
        gate_pass = False
    # BANDING IS ADVISORY. This can only ever return the MEASUREMENT-GAP verdict
    # (``frames == []``); the worst-frame score contributes nothing to the gate.
    #
    # Loudness is a soft observation here (the audio master already targets it);
    # it flags a gross miss (>3 LU from target) as a rerun candidate without
    # failing, since its switch defaults off.
    #
    # VC-W14/V3 adds ``_true_peak_violation`` — the comparison ``true_peak_dbtp``
    # never had. Measured on the DELIVERED mux by the same ``final_mux_loudness``
    # call that produced ``integrated_lufs``, so it costs no extra probe.
    #
    # ONE LOOP, not N copies of the same four lines: the order of the tuple IS
    # the published order of ``violations`` / ``rerun_stages``.
    #
    # ``_speech_presence_violation`` joins the loop on the same footing: it reads
    # the noise floor measured by that SAME ``final_mux_loudness`` call and
    # compares it to the program level, which is the only pair of numbers that can
    # tell a narrated master from one carrying nothing but hiss (see
    # SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB for the measured separation and
    # for why no single-number threshold can). It is STRUCTURAL and NOT ARMED.
    #
    # ``_av_coverage_violation`` ALSO joins here (moved in from its own
    # standalone ``if`` — see below) even though it is not a quality judgement
    # with a tunable ceiling like its loop-mates: the loop already treats
    # structural and tunable violations uniformly, and what makes it
    # always-enforced is ``_GATE_ARMED_DEFAULTS``, not a second code path.
    #
    # ``narration_violation`` (content-aware, per-shot — see
    # ``narration_presence_report``) is the one ASYNC check in this function, so
    # it is awaited just above the loop rather than constructed inline in the
    # tuple like its sync loop-mates.
    narration_presence, narration_violation = await narration_presence_report(
        path, shot_timeline, shots, settings, narration_audio_paths
    )
    for _violation in (
        _banding_violation(frames),
        _loudness_violation(integrated_lufs, settings),
        _true_peak_violation(true_peak_dbtp, settings),
        _speech_presence_violation(integrated_lufs, noise_floor_db),
        _av_coverage_violation(av_gap),
        narration_violation,
    ):
        if _violation is not None:
            rerun_stages.append(_violation.stage)
            violations.append(_violation)
            gate_pass = False
    # DEAD/BLANK-master gate: a uniform (near-zero luma spread) delivered file is a
    # dead render even though every other signal (dup ratio, cfr, loudness) looks
    # fine. Only fires on measured luma data, never on a measurement gap.
    dead_floor = float(getattr(settings, "render_dead_frame_luma_spread_floor", 8.0))
    dead_min_ratio = float(getattr(settings, "render_dead_frame_min_ratio", 0.10))
    dead_ratio = _dead_frame_ratio(frames, dead_floor)
    if _dead_frame(dead_ratio, dead_min_ratio):
        rerun_stages.append("dead_frame")
        # ``_dead_frame`` only fires on a MEASURED ratio (a gap never gates), so the
        # None branch here is unreachable — it exists so the formatter stays total.
        measured_dead = dead_ratio if dead_ratio is not None else 0.0
        violations.append(
            MasteringGateViolation(
                stage="dead_frame",
                tier=gate_tier("dead_frame"),
                metric="dead_frame_ratio",
                value=measured_dead,
                limit=dead_min_ratio,
                detail=(
                    f"dead_frame: {measured_dead:.4f} of frames sit below luma-spread "
                    f"floor {dead_floor:.1f}, over the max ratio {dead_min_ratio:.4f}"
                ),
            )
        )
        gate_pass = False
    # NO DUPLICATE-FRAME GATE. ``dup_ratio`` is measured above and published in
    # the report below, and it contributes NOTHING to ``gate_pass`` or to
    # ``rerun_stages``. Duplicate frames are intentional dwell (6500ms hold x
    # Playwright's 25fps capture), the W1-3 gate fired on 120 of 120 measured
    # masters with a MINIMUM ratio of 0.643 against its 0.10 ceiling, and the
    # predicate was deleted outright rather than widened — see the block where
    # ``_duplicate_frames`` used to live for the measurement and the reasoning.

    # A/V COVERAGE and NARRATION PRESENCE are both folded into the append loop
    # above now (``_av_coverage_violation`` / ``narration_violation``) — see the
    # comment there. ``av_coverage`` still fails the compile in BOTH modes
    # (see ``_GATE_TIERS`` / ``_GATE_ARMED_DEFAULTS``), which is decided by
    # arming, not by a separate code path here.

    # ENFORCE: the soft-gate becomes a hard one. Raised INSTEAD of returning, so a
    # caller cannot mistake a failed master for a delivered one, and the message
    # names the metric + value + ceiling rather than a generic "gate failed".
    #
    # ARMED PER GATE (``gate_armed``), not per mode. Every STRUCTURAL gate is armed
    # by default and each carries its own kill switch, so a bad threshold is backed
    # out in config rather than by a code change and a rollout; the TUNABLE ones are
    # wired and default OFF because their fleet distributions are unmeasured. See
    # ``_GATE_TIERS`` / ``_GATE_ARMED_DEFAULTS``. ``mastering_gate_mode='enforce'``
    # still arms everything, as an escalation.
    #
    # When ANY armed gate fires, EVERY violation is reported, not just the armed
    # ones: an operator handed a failed master should see the full list in one pass
    # rather than one defect per re-run, and the tunable entries are the evidence
    # for whether their switch should be flipped on.
    armed_violations = [v for v in violations if gate_armed(settings, v.stage)]
    if armed_violations:
        log.warning(
            "mastering.gate_enforced_failure",
            rerun_stages=rerun_stages,
            violations=[v.detail for v in violations],
            armed_stages=sorted({v.stage for v in armed_violations}),
            enforced_by="mode" if gate_enforced(settings) else "stage",
        )
        raise MasteringGateError(violations)

    return {
        "colorspace_ok": colorspace_ok,
        "integrated_lufs": integrated_lufs,
        "true_peak_dbtp": true_peak_dbtp,
        # BOTH the raw floor and the difference the gate decided on, published on
        # the clean path too — the same reasoning as ``video_stream_seconds`` /
        # ``audio_stream_seconds`` beside ``av_gap_seconds``. A consumer that can
        # see -14.09 LUFS over a -36.68 dB floor can tell a comfortably narrated
        # master from one sitting 2 dB over the line, which the single reduced
        # scalar never could; and a ``null`` here is the caller's visible signal
        # that no speech-presence verdict was reachable this run.
        #
        # ``noise_floor_db`` is deliberately NOT added to ``unmeasured``: a missing
        # floor is, in every case this pipeline can produce, the SAME loudness-probe
        # failure that ``integrated_lufs`` already names there, and a second name
        # for one gap reads as two unverified inputs. The gap is not hidden — it is
        # the ``null`` on these two keys.
        "noise_floor_db": noise_floor_db,
        "program_to_noise_floor_db": speech_gap_db,
        # THE CUTOFF THIS RUN COMPARED AGAINST, published unconditionally for the
        # same reason as ``dead_frame_min_ratio`` below: a consumer must be able to
        # read the threshold off the wire rather than hardcode a second copy of it
        # in another repo.
        "speech_presence_min_program_over_floor_db": (SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB),
        # Content-aware per-shot presence: one entry per shot that carried an
        # ``audio_url`` and was actually measured (fetch/decode/window failures are
        # simply absent, per no-gate-on-a-gap), each ``{line_id, correlation_db,
        # present}``. Empty when ``shot_timeline``/``shots`` were not supplied.
        "narration_presence": narration_presence,
        # THE CUTOFF THIS RUN COMPARED AGAINST, published unconditionally — same
        # reasoning as ``speech_presence_min_program_over_floor_db`` beside it.
        "narration_presence_min_correlation_db": (NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB),
        "banding_scores": scores,
        "per_shot": per_shot,
        "cfr_ok": cfr_ok,
        "measured_fps": measured,
        "duplicate_frame_ratio": dup_ratio,
        "luma_spread": luma_spread,
        "dead_frame_ratio": dead_ratio,
        # THE CUTOFF THIS RUN ACTUALLY COMPARED AGAINST, published unconditionally
        # — on the clean path too, not only when ``dead_frame`` fires.
        #
        # WHY: ``dead_frame_ratio`` was reported and its CUTOFF was not, so the
        # orchestrator duplicated the number as its own
        # ``shared/mastering_gate.py:_DEAD_FRAME_MIN_RATIO = 0.10``. The only
        # on-wire statement of it was the ``limit`` of a ``dead_frame`` violation,
        # which exists only once this module has ALREADY decided — by which point
        # the number is too late to be useful to anyone re-deriving the verdict.
        # Retuning ``render_dead_frame_min_ratio`` here therefore left the
        # orchestrator's blank-picture verdict comparing against a stale 0.10:
        # two repos, one threshold, no link.
        #
        # ``dead_floor`` is deliberately NOT published alongside it: the ratio is
        # the number the gate decides on, and the floor is an input to the ratio
        # this module has already applied. Publishing both would invite a consumer
        # to re-measure rather than read the verdict.
        "dead_frame_min_ratio": dead_min_ratio,
        # The two numbers whose absence WAS the defect, published separately and
        # alongside the gap the gate decided on. Reported even when the gate passed:
        # a consumer that can see 58.200 against 58.361 can tell "clean" from "0.4s
        # and drifting", which a single reduced scalar never could.
        "video_stream_seconds": video_seconds,
        "audio_stream_seconds": audio_seconds,
        "av_gap_seconds": av_gap,
        # ADVISORY, read-only: how many holes the audio packet timeline has and how
        # much time they cover (``audio_pts_gaps``). No gate and no switch reads
        # them; ``null`` = unmeasured, never "none".
        "audio_pts_gap_count": audio_pts_gap_count,
        "audio_pts_gap_ms_total": audio_pts_gap_ms_total,
        # The pts-interval half of ``cfr_ok`` (``video_pts_gaps``): how many video
        # frame intervals are off the nominal one, and where the first few are.
        # NOT advisory the way the audio pair is — a count above zero is what makes
        # ``cfr_ok`` False. ``null`` = unmeasured, never "none".
        "video_pts_gap_count": video_pts_gap_count,
        "video_pts_gaps": video_pts_gap_positions,
        # ``duplicate_frame_advisory`` was REMOVED here (W1-3) — a second boolean
        # restating a ``dup_ratio > ceiling`` comparison, with zero consumers. The
        # comparison it restated is itself gone now (2026-09-11): the ratio above
        # is the whole of what this report says about duplicate frames.
        "gate_pass": gate_pass,
        "rerun_stages": rerun_stages,
        # THE structured verdict. One entry per cleared check, each carrying the
        # metric, the measured value, the ceiling it breached, and its TIER — so a
        # consumer reads the gate's intent instead of re-deriving it by
        # string-matching stage names in ``rerun_stages`` (which cannot tell an
        # advisory "this stage might help" append like ``"cfr"`` from a real gate
        # failure, and carries no numbers at all). Built every run since the gate
        # was written; it was simply never placed in this dict.
        #
        # ``enforced_stages`` WAS REMOVED HERE and must not come back. It published
        # ``sorted(_ALWAYS_ENFORCED_STAGES)`` — a module constant, identical on
        # every response, which the consumer already mirrored by name as its own
        # hardcoded fallback. A key whose value can never differ from the reader's
        # default is not a contract, it is a cross-repo negotiation that cannot
        # fail; and per-violation ``tier`` below answers the question it was asked
        # to answer (is this master structurally undeliverable?) for the specific
        # violations this run actually produced.
        "violations": violations_payload(violations),
        # Signals that were measured over their threshold and deliberately NOT
        # gated — so ``gate_pass: true`` + ``violations: []`` can no longer hide a
        # master whose shots all flagged ``needs_deband``. See ``banding_advisories``.
        "advisories": banding_advisories(scores),
        # Which checks could NOT be evaluated. `gate_pass` alone conflates "every
        # check ran and none was violated" with "the checks that RAN found nothing",
        # and a consumer cannot tell those apart from the metric values (a `None`
        # means unmeasured, but nothing said which Nones the gate actually cared
        # about). The no-gate-on-a-gap policy itself is deliberate and stays —
        # failing closed on a probe timeout would reject good masters — but a gap
        # that is invisible to the caller is the fail-open half of it.
        "unmeasured": _unmeasured_metrics(
            duplicate_frame_ratio=dup_ratio,
            dead_frame_ratio=dead_ratio,
            luma_spread=luma_spread,
            measured_fps=measured,
            integrated_lufs=integrated_lufs,
            true_peak_dbtp=true_peak_dbtp,
            # `framerate_coherence` has no disabled path — it runs on every
            # verification — so `cfr_ok is None` unambiguously means the probe or
            # the rate parse failed. Advisory-but-unmeasured is exactly the state
            # this channel names; `measured_fps`, `luma_spread` and
            # `true_peak_dbtp` above are advisory too and were always listed.
            cfr_ok=cfr_ok,
            # ONLY when the check was actually on. `_CHECK_DISABLED` is not None,
            # so a disabled colorspace gate contributes nothing here — see
            # `colorspace_gate_active` above for why "off" and "could not run" must
            # not collapse.
            colorspace_ok=colorspace_ok if colorspace_gate_active else _CHECK_DISABLED,
            # OFF vs COULD-NOT-RUN, the same distinction ``colorspace_ok`` makes one
            # line up. A master with a video stream and NO audio stream (a silent
            # reel) has nothing that could outrun the picture — the check is off, and
            # listing it would report every silent master as unverified. A probe that
            # came back with neither number is the real gap.
            av_gap_seconds=(
                _CHECK_DISABLED if video_seconds is not None and audio_seconds is None else av_gap
            ),
        ),
    }


class _CheckDisabled:
    """Sentinel for a check that was never asked for, as distinct from one that was
    asked for and could not be measured. Deliberately NOT ``None``: ``None`` is the
    wire value for "unmeasured", and the whole point of this type is that a switched-
    off check must not enter the ``unmeasured`` channel and read as an unverified
    master."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_CHECK_DISABLED"


_CHECK_DISABLED = _CheckDisabled()


def _unmeasured_metrics(**metrics: float | bool | _CheckDisabled | None) -> list[str]:
    """Names of the verification inputs that WERE requested and could not be
    measured, sorted.

    Empty means every requested input was measured — which is the only state in
    which ``gate_pass=True`` means "verified". A non-empty list next to
    ``gate_pass=True`` means "no violation among the checks that RAN", and the
    caller can now tell.

    Pass ``_CHECK_DISABLED`` for a check the job switched off. "The check was off"
    and "the check could not run" are different facts about the master and must not
    share a value: the first says nothing was expected, the second says something
    was expected and is missing. Only the second belongs here.
    """
    return sorted(name for name, value in metrics.items() if value is None)


def _as_float(value: object) -> float | None:
    """Coerce a loudness value (number or quoted string) to float, or None."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def degraded_report() -> dict[str, Any]:
    """A best-effort MasteringReport for when verification itself could not run —
    ``gate_pass=False`` so the caller can surface that the master was unverified,
    without ever failing the render."""
    return {
        "colorspace_ok": None,
        "integrated_lufs": None,
        "true_peak_dbtp": None,
        # Same key set as ``post_mux_verify`` so the two report shapes agree.
        # ``None``, not 0.0: nothing was measured, so the master's noise floor and
        # its program-over-floor margin are UNKNOWN. A 0.0 here would read as a
        # measured verdict of "steady noise, no speech".
        "noise_floor_db": None,
        "program_to_noise_floor_db": None,
        # None, NOT the module constant — the same reasoning as
        # ``dead_frame_min_ratio`` below: this function holds no ``Settings`` and
        # nothing was compared, so restating the cutoff would claim a comparison
        # that never happened.
        "speech_presence_min_program_over_floor_db": None,
        # Same reasoning as ``speech_presence_min_program_over_floor_db`` above:
        # verification did not run, so nothing was measured and nothing was
        # compared against.
        "narration_presence": [],
        "narration_presence_min_correlation_db": None,
        "banding_scores": [],
        "per_shot": [],
        "cfr_ok": None,
        "measured_fps": None,
        "duplicate_frame_ratio": None,
        "luma_spread": None,
        # None, not 0.0: verification did not RUN, so the dead-frame fraction is unknown.
        # Reporting 0.0 here would read as "measured, and the film is clean" — the exact
        # absence-of-evidence-as-evidence confusion that let an audio-less clip pass the
        # audiobook uniformity guard.
        "dead_frame_ratio": None,
        # None, NOT the 0.10 default. This function holds no ``Settings`` — it
        # cannot know the cutoff a deployment configured, and restating the
        # module default here would re-create exactly the duplicated constant
        # that publishing the key exists to eliminate. The consumer prefers a
        # reported value and falls back to its own when absent, which on a report
        # where nothing was measured at all is the correct outcome: there is no
        # ``dead_frame_ratio`` to compare it against either.
        "dead_frame_min_ratio": None,
        # Same reasoning as `dead_frame_ratio`: verification did not run, so the
        # picture and narration lengths are UNKNOWN, not equal. A 0.0 gap here would
        # read as "measured, and the picture covers the audio" — the exact claim the
        # degraded report has no evidence for.
        "video_stream_seconds": None,
        "audio_stream_seconds": None,
        "av_gap_seconds": None,
        "audio_pts_gap_count": None,
        "audio_pts_gap_ms_total": None,
        "video_pts_gap_count": None,
        "video_pts_gaps": None,
        "gate_pass": False,
        "rerun_stages": ["verify"],
        # Same key set as `post_mux_verify` (the two report shapes must agree, as
        # `unmeasured` below already does) — so a consumer never has to special-case
        # which report it received. An ABSENT report is itself a STRUCTURAL
        # violation: verification did not run, so nothing at all is known about the
        # master, and a report with `violations: []` here would read as "checked,
        # nothing wrong". The caller decides what to do with it via
        # `gate_armed(settings, "verify")`; this function holds no settings and only
        # states the finding.
        "violations": violations_payload(
            [
                MasteringGateViolation(
                    stage="verify",
                    metric="post_mux_verify",
                    tier=gate_tier("verify"),
                    detail=(
                        "verify: post_mux_verify did not run, so no gate was "
                        "evaluated — the master is unverified, not clean"
                    ),
                )
            ]
        ),
        # Nothing was measured, so nothing can be over a threshold.
        "advisories": [],
        # Verification did not RUN, so EVERY gate input is unmeasured — same key set
        # ``post_mux_verify`` reports, so the two report shapes agree and a consumer
        # does not have to special-case which one it received. Omitting it here would
        # have made the degraded report read as "nothing was missed".
        "unmeasured": _unmeasured_metrics(
            duplicate_frame_ratio=None,
            dead_frame_ratio=None,
            luma_spread=None,
            measured_fps=None,
            integrated_lufs=None,
            true_peak_dbtp=None,
            cfr_ok=None,
            # Unconditionally unmeasured here, with no `_CHECK_DISABLED` branch:
            # `post_mux_verify` can distinguish "off" from "could not run" because it
            # holds `settings`. This function does not run verification AT ALL and
            # takes no settings, so it cannot claim the colorspace check was
            # deliberately off. Reporting it as unmeasured is the honest answer —
            # the direction that over-names a gap rather than hiding one.
            colorspace_ok=None,
            # Unconditionally unmeasured for the same reason as `colorspace_ok` above:
            # without having probed the file this function cannot know whether the
            # master even HAS an audio stream, so it cannot claim the check was off.
            av_gap_seconds=None,
        ),
    }
