"""The speech-presence noise floor is a low PERCENTILE of windowed peaks, not a minimum.

THE DEFECT: the floor used to be astats' ``Noise floor dB`` — a whole-file
MINIMUM. On ffmpeg 7.1.5 about 4% of white-noise hiss-only encodes carry one
window reading 9-39 dB under the bed (``Noise floor count: 1``); that one window
became the floor, ``integrated_lufs - floor`` turned positive, and a master with
no speech in it PASSED the speech-presence check as "speech present".

These are PURE tests over synthetic ``ametadata=mode=print`` dumps — the shape
``loudness.WINDOWED_NOISE_FLOOR_FILTERS`` makes ffmpeg log. The real-ffmpeg proof
is ``tests/render_harness/test_speech_presence_render.py``.
"""

from __future__ import annotations

import math

import pytest

from src.services import loudness, mastering, renderer
from src.services.loudness import parse_loudness

_KEY = "lavfi.astats.Overall.Peak_level"
_WINDOW = 0.05

#: White hiss @ -34 dBFS, as ``final_mux_loudness`` reads it off an AAC master.
_HISS_LUFS = -36.71
_HISS_PEAK = -31.0


def _dump(levels: list[float | str], *, step: float = _WINDOW, summary: str | None = None) -> str:
    """An ffmpeg stderr dump carrying one printed window per entry of ``levels``."""
    lines = ["  Duration: 00:00:12.00, start: 0.000000, bitrate: 70 kb/s"]
    for index, level in enumerate(levels):
        lines.append(
            f"[Parsed_ametadata_2 @ 0x1] frame:{index:<4} pts:{index * 2400:<7} "
            f"pts_time:{index * step:.6g}"
        )
        lines.append(f"[Parsed_ametadata_2 @ 0x1] {_KEY}={level}")
    if summary is not None:
        lines.append(f"[Parsed_astats_1 @ 0x2] Noise floor dB: {summary}")
        lines.append("[Parsed_astats_1 @ 0x2] Noise floor count: 1.000000")
    return "\n".join(lines) + "\n"


def _hiss(n: int = 240) -> list[float | str]:
    """``n`` windows of a steady bed: peaks wander ±0.5 dB, deterministically."""
    return [round(_HISS_PEAK + 0.5 * math.sin(i * 1.7), 6) for i in range(n)]


def _with_outlier(levels: list[float | str], drop_db: float = 30.0) -> list[float | str]:
    out = list(levels)
    out[len(out) // 2] = _HISS_PEAK - drop_db
    return out


# --------------------------------------------------------------------------
# THE FIX — one quiet window cannot move the floor
# --------------------------------------------------------------------------


def test_a_hiss_only_dump_with_one_quiet_window_still_fires_the_gate() -> None:
    """FAILS BEFORE THE FIX. The dump carries what astats' whole-file minimum reads
    on such a file (``Noise floor dB: -61``, count 1) AND the per-window peaks. The
    minimum makes the gap +24 dB — "speech present"; the percentile reads the bed."""
    dump = _dump(_with_outlier(_hiss()), summary=f"{_HISS_PEAK - 30.0:.6f}")
    floor = parse_loudness("", "", dump)["noise_floor_db"]

    assert floor is not None
    assert floor == pytest.approx(_HISS_PEAK, abs=0.6), "one window moved the floor"
    violation = mastering._speech_presence_violation(_HISS_LUFS, floor)
    assert violation is not None, f"hiss-only master passed as speech (floor {floor})"
    assert violation.stage == "speech_presence"
    assert violation.value == pytest.approx(_HISS_LUFS - floor)


@pytest.mark.parametrize("drop_db", [9.0, 30.0, 39.0, 200.0])
def test_one_outlier_window_does_not_move_the_floor(drop_db: float) -> None:
    """The observed outliers sit 9-39 dB under the bed; none of them — nor an
    absurd one — changes the reading by more than the bed's own ±0.5 dB wander."""
    clean = loudness.parse_windowed_noise_floor(_dump(_hiss()))
    dirty = loudness.parse_windowed_noise_floor(_dump(_with_outlier(_hiss(), drop_db)))
    assert clean is not None and dirty is not None
    assert abs(clean - dirty) < 0.1, (clean, dirty)
    windows = loudness._window_peaks(_dump(_with_outlier(_hiss(), drop_db)))
    assert min(windows) < clean - 8.0, "the fixture carries no outlier"


def test_the_minimum_would_have_passed_the_same_windows() -> None:
    """The estimator, not the fixture, is what convicts: the plain minimum over the
    SAME windows reads a +24 dB gap and lets the hiss master through."""
    windows = loudness._window_peaks(_dump(_with_outlier(_hiss())))
    assert mastering._speech_presence_violation(_HISS_LUFS, min(windows)) is None
    assert (
        mastering._speech_presence_violation(_HISS_LUFS, loudness.robust_noise_floor(windows))
        is not None
    )


def test_a_dozen_quiet_windows_in_240_still_do_not_move_it() -> None:
    """5% of the timeline has to be quiet before the floor follows it down."""
    levels = _hiss()
    for index in range(11):
        levels[index * 20] = -80.0
    floor = loudness.parse_windowed_noise_floor(_dump(levels))
    assert floor == pytest.approx(_HISS_PEAK, abs=0.6)


# --------------------------------------------------------------------------
# it still finds a real floor — narration with pauses stays "speech"
# --------------------------------------------------------------------------


def test_narration_with_pauses_reads_its_pauses_as_the_floor() -> None:
    """1.2 s on / 0.8 s off: 40% of the windows are the bed, so the floor is the bed."""
    levels: list[float | str] = [-9.0 if (i % 40) < 24 else -45.0 for i in range(240)]
    floor = loudness.parse_windowed_noise_floor(_dump(levels))
    assert floor == pytest.approx(-45.0)
    assert mastering._speech_presence_violation(-14.1, floor) is None


def test_digitally_silent_pauses_clamp_to_the_finite_sentinel() -> None:
    """astats prints ``-inf`` for an all-zero window; the floor stays JSON-safe."""
    levels: list[float | str] = [-9.0 if (i % 40) < 24 else "-inf" for i in range(240)]
    floor = parse_loudness("", "", _dump(levels))["noise_floor_db"]
    assert floor is not None and math.isfinite(floor)
    assert floor == loudness._SILENT_NOISE_FLOOR_DB


def test_pauses_under_five_percent_are_not_the_floor() -> None:
    """The documented cost: the floor is read from the 5th-percentile window, so 4%
    of pause does not reach it. Pinned so a percentile change is a visible diff."""
    levels: list[float | str] = [-9.0] * 250
    for index in range(10):  # 10 of 250 = 4%
        levels[index * 25] = -45.0
    assert loudness.parse_windowed_noise_floor(_dump(levels)) == pytest.approx(-9.0)
    for index in range(10, 14):  # 14 of 250 = 5.6%
        levels[index * 17 + 1] = -45.0
    assert loudness.parse_windowed_noise_floor(_dump(levels)) == pytest.approx(-45.0)


# --------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------


def test_frames_are_binned_into_50ms_windows_by_pts_time() -> None:
    """A 96 kHz master gives 25 ms frames. Two frames share one window and the
    window's level is the LOUDER peak — so the window count, and with it what
    fraction of the timeline a percentile means, does not depend on sample rate."""
    half: list[float | str] = []
    for i in range(240):
        half += [-31.0, -60.0] if i % 2 else [-60.0, -31.0]
    windows = loudness._window_peaks(_dump(half, step=0.025))
    assert len(windows) == 240
    assert set(windows) == {-31.0}


def test_a_progress_line_between_the_two_printed_lines_is_tolerated() -> None:
    """ffmpeg's ``\\r``-terminated progress line comes from another thread and can
    land between ``frame:… pts_time:…`` and the ``key=value`` line."""
    dump = (
        "[Parsed_ametadata_2 @ 0x1] frame:0    pts:0       pts_time:0\n"
        "size=N/A time=00:00:00.00 bitrate=N/A speed=   0x    \r"
        f"[Parsed_ametadata_2 @ 0x1] {_KEY}=-27.981668\n"
        "[Parsed_ametadata_2 @ 0x1] frame:1    pts:2400    pts_time:0.05\n"
        "[silencedetect @ 0x3] silence_start: 0.01\n"
        f"[Parsed_ametadata_2 @ 0x1] {_KEY}=-30.708945\n"
    )
    assert loudness._window_peaks(dump) == [-27.981668, -30.708945]


def test_frames_without_a_timestamp_are_each_their_own_window() -> None:
    """No ``pts_time`` → no guessed timeline; every frame counts once."""
    dump = "".join(f"[Parsed_ametadata_2 @ 0x1] {_KEY}={level}\n" for level in (-30.0, -31.0))
    assert loudness._window_peaks(dump) == [-30.0, -31.0]


def test_nan_windows_are_dropped_not_read_as_levels() -> None:
    dump = _dump([-30.0, "nan", -31.0])
    assert loudness._window_peaks(dump) == [-30.0, -31.0]


@pytest.mark.parametrize("stderr", ["", "no astats here", "[astats] Noise floor dB: -40.0\n"])
def test_no_windows_is_none_never_a_fabricated_floor(stderr: str) -> None:
    assert loudness.parse_windowed_noise_floor(stderr) is None
    assert loudness.robust_noise_floor([]) is None


def test_under_twenty_windows_degrades_to_the_minimum() -> None:
    """Stated limit, pinned: a sub-second file has no robust floor."""
    assert loudness.robust_noise_floor([-30.0] * 18 + [-70.0]) == -70.0
    assert loudness.robust_noise_floor([-30.0] * 19 + [-70.0]) == -30.0


def test_parse_loudness_prefers_windows_over_the_astats_summary() -> None:
    dump = _dump(_hiss(), summary="-70.000000")
    assert parse_loudness("", "", dump)["noise_floor_db"] == pytest.approx(_HISS_PEAK, abs=0.6)


# --------------------------------------------------------------------------
# the measure pass
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_measure_pass_runs_the_windowed_chain_ahead_of_the_analysers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The windowed astats leads, and ``silencedetect`` still precedes ``loudnorm``
    (the load-bearing order ``final_mux_loudness`` documents)."""
    seen: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen.append(cmd)
        return _dump(_with_outlier(_hiss())) + '{ "input_i" : "-36.71" }'

    monkeypatch.setattr(renderer, "_run", fake_run)
    measured = await loudness.final_mux_loudness("x.mp4")

    chain = seen[0][seen[0].index("-af") + 1]
    assert chain.startswith(loudness.WINDOWED_NOISE_FLOOR_FILTERS + ",")
    assert "reset=1" in chain and "metadata=1" in chain and f"key={_KEY}" in chain
    assert chain.index("ametadata") < chain.index("silencedetect") < chain.index("loudnorm")
    assert measured is not None
    assert measured["noise_floor_db"] == pytest.approx(_HISS_PEAK, abs=0.6)
    assert (
        mastering._speech_presence_violation(
            measured["integrated_lufs"], measured["noise_floor_db"]
        )
        is not None
    )


@pytest.mark.asyncio
async def test_the_measure_pass_never_falls_back_to_the_reset_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``reset=1`` astats' end-of-run ``Noise floor dB`` covers the LAST window
    only. If no windows were printed the floor is unmeasured — never that line."""

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        return '[Parsed_astats_1 @ 0x2] Noise floor dB: -44.000897\n{ "input_i" : "-36.71" }'

    monkeypatch.setattr(renderer, "_run", fake_run)
    measured = await loudness.final_mux_loudness("x.mp4")
    assert measured is not None
    assert measured["integrated_lufs"] == pytest.approx(-36.71)
    assert measured["noise_floor_db"] is None
