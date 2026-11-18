"""Real-ffmpeg proof that the speech-presence gate FIRES on a hiss-only master.

THE DEFECT: a production ``website_review`` master shipped with no speech at all
— only faint hiss — and nothing noticed. #224 fixed the cause (a sample-format
mismatch surviving a stream-copy concat) and ``_assert_uniform_segment_audio``
guards that concat, but it is a PRE-concat guard. Nothing measured the FINISHED
master for the presence of speech, so any other route to a silent or hissing
master ships just as quietly.

WHY A SECOND MEASUREMENT IS NEEDED, proven here rather than asserted: steady hiss
never trips ``silencedetect``'s -50 dB gate, so ``silence_ratio`` reads the same
as a wall-to-wall narration; and the loudness master normalises a hiss-only file
to the SAME ``integrated_lufs`` as a narrated one. Both of those are measured
below on the very same fixtures, so the "no single threshold can do this" claim
is a measurement in this file and not a comment in another one.

There is no hiss fixture in this repo (there are no audio fixture files at all),
so both masters are SYNTHESIZED here with ffmpeg at the levels of the real
defect: ~-34 dBFS of ``anoisesrc`` hiss versus a narration-shaped burst signal.

Gated at MODULE level and tagged ``render`` — the same shape as
``test_silence_ratio_render.py`` and ``test_audio_level_render.py``. ``render``
is SELECTED in CI (``addopts`` deselects only ``integration``) and the archetype
installs ffmpeg via ``extra_apt``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.services import loudness, mastering, renderer
from src.services.loudness import final_mux_loudness

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_DURATION = 12.0

#: THE DEFECT SHAPE. White noise at amplitude 0.02 measures ~-34 dBFS — the level
#: of the real "faint hiss" master, versus ~-20 dBFS for real speech.
#:
#: SEEDED. ``anoisesrc`` defaults to ``seed=-1`` — a new random sequence per run —
#: and a measurement of a different file every run is not a regression test. Seed 2
#: is the single fixed file the three original tests below measure.
#:
#: HISTORY, because the seed used to hide a real defect. The floor was astats'
#: ``Noise floor dB``, a whole-file MINIMUM, and on ffmpeg 7.1.5 (prod; the CI test job's 6.1.1 also flaked) the gap it gave
#: off the AAC encode of this noise was BIMODAL across sequences. Measured
#: 2026-10-05 over 300 unseeded encodes: 278 read -5.8..-5.4 dB (gate fires), 9
#: fired outside the asserted band, and 13 (~4%) read +3..+25 dB because a single
#: astats window (``Noise floor count: 1``) read 9-39 dB under the bed — the gate
#: did NOT fire. That turned main red on 139db21 with no code change (-36.7 LUFS vs
#: a -41.14 dB floor = +4.44 dB). ffmpeg 4.4 never showed it (60 of 60 in
#: -6.5..-6.0). #241 pinned the seed, which fixed the test and not the gate.
#:
#: The floor is now a low percentile of 50 ms windowed peaks
#: (``loudness.parse_windowed_noise_floor``), which one window cannot move, and
#: ``test_it_fires_on_every_hiss_sequence`` below measures a spread of seeds
#: instead of one. Measured 2026-10-06 inside the PRODUCTION image (ffmpeg 7.1.5),
#: seeds 0..199 at 12 s and 6 s: the old whole-file minimum let 11 of 400 hiss
#: encodes through (8/200 at 12 s, 3/200 at 6 s; gap up to +33.2 dB); the windowed
#: floor caught 400 of 400 (gap -6.86..-5.94 dB) with 0 of 310 speech fixtures
#: convicted. On 4.4 both catch every seed. The outlier itself is reproduced on
#: any version by ``test_one_quiet_window_does_not_pass_a_hiss_master``.
_HISS_SEED = 2
_HISS_SEEDS = tuple(range(30))
_SEED_SWEEP_DURATION = 6.0


def _hiss_src(seed: int, duration: float) -> str:
    return f"anoisesrc=color=white:amplitude=0.02:sample_rate=48000:duration={duration}:seed={seed}"


_HISS_SRC = _hiss_src(_HISS_SEED, _DURATION)

#: A NARRATION-SHAPED master: a 180 Hz tone gated 1.2 s on / 0.8 s off, which is
#: the property that matters — program with PAUSES for astats to find a floor in.
#: (A ``lavfi sine`` with pauses, as the sibling render tests use; nothing here
#: depends on it being a voice, only on it being intermittent.)
_NARRATION_TONE = f"sine=frequency=180:duration={_DURATION}:sample_rate=48000"
_NARRATION_GATE = f"aevalsrc='0.35*lt(mod(t\\,2)\\,1.2)':d={_DURATION}:s=48000"


async def _encode(dest: Path, args: list[str]) -> Path:
    """Encode a real AAC-in-MP4 master — the container the pipeline delivers."""
    await renderer._run(["ffmpeg", "-y", "-hide_banner", *args, "-c:a", "aac", str(dest)])
    return dest


async def _hiss_master(tmp_path: Path) -> Path:
    return await _encode(tmp_path / "hiss.mp4", ["-f", "lavfi", "-i", _HISS_SRC])


async def _narrated_master(tmp_path: Path, *, normalise: bool = True) -> Path:
    """Narration-shaped bursts, run through the SAME loudness master the pipeline
    applies, so the comparison is against a real delivered master and not a raw mix."""
    args = [
        "-f", "lavfi", "-i", _NARRATION_TONE,
        "-f", "lavfi", "-i", _NARRATION_GATE,
        "-filter_complex", "[0][1]amultiply" + (",loudnorm=I=-14:TP=-1.5" if normalise else ""),
    ]  # fmt: skip
    return await _encode(tmp_path / "narrated.mp4", args)


@pytest.mark.asyncio
async def test_it_fires_on_a_hiss_only_master(tmp_path: Path) -> None:
    """FIRES ON BAD INPUT — the acceptance for this gate.

    A real encode of nothing but ~-34 dBFS hiss, measured by the real
    ``final_mux_loudness`` pass, produces a ``speech_presence`` violation."""
    measured = await final_mux_loudness(str(await _hiss_master(tmp_path)))
    assert measured is not None
    assert measured["noise_floor_db"] is not None, "astats must ride the measure pass"

    gap = measured["integrated_lufs"], measured["noise_floor_db"]
    violation = mastering._speech_presence_violation(*gap)
    assert violation is not None, f"gate did not fire on hiss: {measured}"
    assert violation.stage == "speech_presence"
    assert violation.tier == "structural"
    assert violation.value is not None
    assert violation.value < mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB
    # Measured 2026-09-16: -1.85 dB as raw PCM, -5.45 dB re-encoded to AAC (the
    # encoder's handling of broadband noise moves the floor, and the exact value
    # depends on bitrate). Banded generously and one-sidedly on purpose: what must
    # hold is that a steady bed of noise reads a NEGATIVE margin — its program
    # level does not rise above its own floor — not the third decimal of an
    # encoder version.
    assert -12.0 <= violation.value <= -1.0, violation.value


@pytest.mark.parametrize("seed", _HISS_SEEDS)
@pytest.mark.asyncio
async def test_it_fires_on_every_hiss_sequence(tmp_path: Path, seed: int) -> None:
    """NOT ONE LUCKY SEED. The gate must convict hiss whatever sequence the noise
    source rolled — the miss this replaces was a per-sequence one (see
    ``_HISS_SEED``). Same band as the single-seed test above."""
    master = await _encode(
        tmp_path / f"hiss_{seed}.mp4", ["-f", "lavfi", "-i", _hiss_src(seed, _SEED_SWEEP_DURATION)]
    )
    measured = await final_mux_loudness(str(master))
    assert measured is not None
    violation = mastering._speech_presence_violation(
        measured["integrated_lufs"], measured["noise_floor_db"]
    )
    assert violation is not None, f"seed {seed}: gate did not fire on hiss: {measured}"
    assert violation.value is not None
    assert -12.0 <= violation.value <= -1.0, (seed, violation.value)


@pytest.mark.asyncio
async def test_one_quiet_window_does_not_pass_a_hiss_master(tmp_path: Path) -> None:
    """THE OUTLIER, MADE ON PURPOSE. 0.2 s of the hiss is pulled 30 dB down — four
    quiet 50 ms windows in 240, the shape that let ~4% of hiss masters through on
    ffmpeg 7.1.5. The gate still fires, and the reading does not move.

    The second half is what makes this able to fail: the plain MINIMUM over the very
    same measured windows reads a positive gap and would have passed the file."""
    master = await _encode(
        tmp_path / "hiss_dropout.mp4",
        ["-f", "lavfi", "-i", _HISS_SRC, "-af", "volume=enable='between(t,5,5.2)':volume=-30dB"],
    )
    measured = await final_mux_loudness(str(master))
    assert measured is not None
    lufs, floor = measured["integrated_lufs"], measured["noise_floor_db"]
    assert lufs is not None and floor is not None
    violation = mastering._speech_presence_violation(lufs, floor)
    assert violation is not None, f"one quiet window passed a hiss master: {measured}"
    assert violation.value is not None
    assert -12.0 <= violation.value <= -1.0, violation.value

    stderr = await renderer._run(
        ["ffmpeg", "-hide_banner", "-i", str(master),
         "-af", loudness.WINDOWED_NOISE_FLOOR_FILTERS, "-f", "null", "-"],
        capture_stderr=True,
    )  # fmt: skip
    windows = loudness._window_peaks(stderr)
    assert len(windows) >= 230, len(windows)
    assert loudness.robust_noise_floor(windows) == pytest.approx(floor)
    assert min(windows) < floor - 15.0, "the fixture carries no quiet window"
    assert mastering._speech_presence_violation(lufs, min(windows)) is None


@pytest.mark.asyncio
async def test_dense_narration_with_short_pauses_is_not_convicted(tmp_path: Path) -> None:
    """THE OTHER DIRECTION — a robust floor must not start convicting speech.

    Narration that is on for 1.8 s of every 2 s, over a quiet pink bed: only ~10%
    of the timeline is pause. A floor read from too HIGH a percentile lands in the
    narration itself and calls this hiss (measured: the 10th percentile reads
    -5.0 dB here, the 5th +12.5 dB). This is the guard on ``NOISE_FLOOR_PERCENTILE``."""
    gate = f"aevalsrc='0.35*lt(mod(t\\,2)\\,1.8)':d={_DURATION}:s=48000"
    bed = f"anoisesrc=color=pink:amplitude=0.01:sample_rate=48000:duration={_DURATION}:seed=5"
    master = await _encode(
        tmp_path / "dense.mp4",
        [
            "-f", "lavfi", "-i", _NARRATION_TONE,
            "-f", "lavfi", "-i", gate,
            "-f", "lavfi", "-i", bed,
            "-filter_complex",
            "[0][1]amultiply[n];[n][2]amix=inputs=2:normalize=0,loudnorm=I=-14:TP=-1.5",
        ],
    )  # fmt: skip
    measured = await final_mux_loudness(str(master))
    assert measured is not None
    lufs, floor = measured["integrated_lufs"], measured["noise_floor_db"]
    assert lufs is not None and floor is not None
    assert mastering._speech_presence_violation(lufs, floor) is None, measured
    assert lufs - floor > 6.0, (lufs, floor)


@pytest.mark.asyncio
async def test_it_does_not_fire_on_a_normal_narrated_master(tmp_path: Path) -> None:
    """PASSES ON GOOD INPUT — without this the test above proves only that the
    predicate can return non-None, which a gate that fires on everything also does."""
    measured = await final_mux_loudness(str(await _narrated_master(tmp_path)))
    assert measured is not None
    assert measured["noise_floor_db"] is not None

    gap = measured["integrated_lufs"], measured["noise_floor_db"]
    violation = mastering._speech_presence_violation(*gap)
    assert violation is None, f"gate fired on a normal narrated master: {measured}"


@pytest.mark.asyncio
async def test_neither_loudness_nor_silence_ratio_can_tell_them_apart(
    tmp_path: Path,
) -> None:
    """WHY THE SECOND MEASUREMENT EXISTS, measured rather than asserted.

    Both masters are put through the same loudness master. Afterwards:

    * ``integrated_lufs`` agrees to within a fraction of a LU — a loudness gate
      reads the hiss master as dead on target.
    * ``silence_ratio`` flags NOTHING on the hiss master, because steady hiss is a
      constant signal that never falls under silencedetect's -50 dB gate.

    So a threshold on either one, at ANY value, either misses the hiss master or
    convicts the narrated one. Only the program-over-noise-floor margin separates
    them, and it does so by a wide margin on the same two files."""
    hiss_raw = await _hiss_master(tmp_path)
    hiss = await _encode(
        tmp_path / "hiss_mastered.mp4",
        ["-i", str(hiss_raw), "-af", "loudnorm=I=-14:TP=-1.5"],
    )
    narrated = await _narrated_master(tmp_path)

    bad = await final_mux_loudness(str(hiss))
    good = await final_mux_loudness(str(narrated))
    assert bad is not None and good is not None
    bad_lufs, good_lufs = bad["integrated_lufs"], good["integrated_lufs"]
    assert bad_lufs is not None and good_lufs is not None

    # 1. LOUDNESS CANNOT DISCRIMINATE: both land on the -14 LUFS target.
    assert abs(bad_lufs - good_lufs) < 1.5, (bad_lufs, good_lufs)

    # 2. SILENCE RATIO CANNOT DISCRIMINATE: the hiss master reports no silence at
    #    all (None = silencedetect found no span), exactly like continuous speech.
    assert bad["silence_ratio"] is None

    # 3. THE MARGIN CAN, and by more than 20 dB on these two files.
    bad_gap = mastering._speech_presence_violation(bad_lufs, bad["noise_floor_db"])
    good_gap = mastering._speech_presence_violation(good_lufs, good["noise_floor_db"])
    assert bad_gap is not None and bad_gap.value is not None
    assert good_gap is None
    good_margin = good_lufs - (good["noise_floor_db"] or 0.0)
    assert good_margin - bad_gap.value > 20.0, (bad_gap.value, good_margin)
