"""Real-ffmpeg proof that ``silence_ratio`` measures the DELIVERED signal.

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same shape as
``test_audio_level_render.py``.

The defect these guard: ``final_mux_loudness`` ran
``loudnorm=...:print_format=json,silencedetect=n=-50dB:d=0.3``. ``loudnorm`` is
NOT a pass-through analyser — even in its print-stats single pass it normalizes
the samples it forwards, so ``silencedetect`` saw audio already lifted toward
-14 LUFS. A quiet master's genuinely-below--50 dB passages were pushed above the
gate and ``silence_ratio`` under-reported exactly the files the metric exists to
catch. ``silencedetect`` is itself pass-through, so putting it FIRST makes both
filters measure the original signal.

No MinIO/compile involved — these measure a synthesised file directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.services import renderer
from src.services.loudness import _silence_ratio, final_mux_loudness, parse_loudness

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

# A deliberately QUIET master (~-42 LUFS integrated): 1.0 s of tone, 1.5 s of a
# genuinely sub--50 dBFS passage, 1.0 s of tone. Total 3.5 s → the true silence
# ratio is 1.5/3.5. The tone is quiet enough that loudnorm wants a large positive
# gain, which is what used to lift the middle passage above the -50 dB gate.
_LOUD_AMPLITUDE = 0.1
_QUIET_AMPLITUDE = 0.001
_TOTAL_SECONDS = 3.5
_QUIET_SECONDS = 1.5

# The two filter chains, verbatim: the buggy ordering and the shipped one.
_LOUDNORM_FIRST = "loudnorm=I=-14:TP=-1.5:print_format=json,silencedetect=n=-50dB:d=0.3"
_SILENCEDETECT_FIRST = "silencedetect=n=-50dB:d=0.3,loudnorm=I=-14:TP=-1.5:print_format=json"


def _sine(amplitude: float, seconds: float) -> str:
    return f"sine=frequency=440:duration={seconds}:sample_rate=48000,volume={amplitude}"


async def _make_quiet_master(tmp_path: Path) -> Path:
    """Synthesise the quiet-with-a-silent-middle master described above."""
    out = tmp_path / "quiet-master.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            _sine(_LOUD_AMPLITUDE, 1.0),
            "-f",
            "lavfi",
            "-i",
            _sine(_QUIET_AMPLITUDE, _QUIET_SECONDS),
            "-f",
            "lavfi",
            "-i",
            _sine(_LOUD_AMPLITUDE, 1.0),
            "-filter_complex",
            "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    )
    return out


async def _measure(path: Path, chain: str) -> str:
    """Run one discarded measure pass with ``chain`` and return its stderr."""
    return await renderer._run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", chain, "-f", "null", "-"],
        capture_stderr=True,
    )


async def test_silence_ratio_reports_a_genuinely_quiet_passage(tmp_path: Path) -> None:
    """The shipped measure pass finds the sub--50 dBFS middle of a quiet master."""
    master = await _make_quiet_master(tmp_path)
    result = await final_mux_loudness(str(master))
    assert result is not None
    ratio = result["silence_ratio"]
    assert ratio is not None, "silence_ratio must not be None — the passage IS silent"
    assert ratio == pytest.approx(_QUIET_SECONDS / _TOTAL_SECONDS, abs=0.05)


async def test_loudnorm_first_ordering_hides_the_silence(tmp_path: Path) -> None:
    """The ordering is LOAD-BEARING, not incidental.

    Same file, same filters, only the order swapped: with ``loudnorm`` first its
    normalization lifts the quiet passage above the -50 dB gate and silencedetect
    reports strictly LESS silence than the truth (in practice: none at all)."""
    master = await _make_quiet_master(tmp_path)
    buggy = _silence_ratio(await _measure(master, _LOUDNORM_FIRST)) or 0.0
    shipped = _silence_ratio(await _measure(master, _SILENCEDETECT_FIRST)) or 0.0
    assert shipped == pytest.approx(_QUIET_SECONDS / _TOTAL_SECONDS, abs=0.05)
    assert buggy < shipped, (
        "loudnorm-first must under-report the silence; if this ever stops being "
        "true the ordering guard in final_mux_loudness has lost its reason to exist"
    )


async def test_reordering_preserves_the_loudnorm_measurement(tmp_path: Path) -> None:
    """Putting silencedetect first must not shift loudnorm's numbers.

    silencedetect only logs — it forwards the samples untouched — so the
    ``input_i``/``input_tp``/``input_lra`` the caller parses are identical under
    either ordering. This is what makes the fix safe for the loudness metrics."""
    master = await _make_quiet_master(tmp_path)
    before = parse_loudness(await _measure(master, _LOUDNORM_FIRST))
    after = parse_loudness(await _measure(master, _SILENCEDETECT_FIRST))
    for key in ("integrated_lufs", "true_peak_dbtp", "lra"):
        assert before[key] is not None
        assert after[key] == pytest.approx(before[key])
