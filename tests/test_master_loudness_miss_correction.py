"""The post-master loudness MISS correction (``renderer._correct_master_loudness_miss``).

At the configured target pair loudnorm runs its DYNAMIC mode on essentially every
film, and that mode starts at 0 dB of gain when the film OPENS on silence, then
only creeps up while speech plays. Three prod masters landed -20.73/-20.73/-21.69
LUFS against -14. The correction re-renders pass 2 ONCE with a constant gain plus
the master limiter, and ONLY when the master missed by more than the mastering
gate's own tolerance. These tests pin both halves: a miss corrects, a master that
landed is untouched — and the real-ffmpeg test proves the defect is reproduced
BEFORE it proves the fix (a correction that never fires would pass a fix-only test).
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.api.schemas import MusicBed
from src.config import get_settings
from src.services import mastering, renderer

_TOL = mastering.LOUDNESS_GATE_TOLERANCE_LU


def _loudnorm_json(input_i: str) -> str:
    """Real shape of loudnorm's ``print_format=json`` stderr block."""
    return (
        "[Parsed_loudnorm_0 @ 0x55]\n{\n"
        f'\t"input_i" : "{input_i}",\n\t"input_tp" : "-1.20",\n'
        '\t"input_lra" : "3.10",\n\t"input_thresh" : "-31.00",\n'
        '\t"normalization_type" : "dynamic",\n\t"target_offset" : "0.00"\n}\n'
    )


# --------------------------------------------------------------------------
# 1. The decision function
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("measured", "expected"),
    [
        (-21.69, 7.69),  # the prod miss: corrects to target
        (-20.73, 6.73),
        (-9.0, -5.0),  # a miss on the HOT side corrects down
        (-14.52, None),  # the prod masters that landed: untouched
        (-14.0, None),
        (-14.0 - _TOL, None),  # exactly at the gate's tolerance: untouched
        (-14.0 + _TOL, None),
        (None, None),  # unmeasurable: never guess
        (-math.inf, None),  # digital silence
        (-40.0, None),  # beyond the cap: pathological, not an under-ridden bus
    ],
)
def test_correction_fires_only_on_a_miss(measured: float | None, expected: float | None) -> None:
    gain = renderer._loudness_correction_db(measured, get_settings())
    if expected is None:
        assert gain is None
    else:
        assert gain == pytest.approx(expected)


def test_correction_never_fires_with_mastering_off() -> None:
    settings = get_settings().model_copy(update={"master_enabled": False})
    assert renderer._loudness_correction_db(-21.69, settings) is None


def test_correction_chain_is_constant_gain_then_the_master_limiter() -> None:
    settings = get_settings()
    chain = renderer._loudness_correction_chain(7.69, settings)
    assert chain.startswith("volume=7.69dB,alimiter=")
    assert f"limit={renderer._ffnum(renderer._limiter_linear_limit(settings))}" in chain
    assert chain.endswith(":level=disabled")


# --------------------------------------------------------------------------
# 2. The orchestration, with ffmpeg faked
# --------------------------------------------------------------------------


def _fake_run(monkeypatch: pytest.MonkeyPatch, readings: list[str], *, fail: bool = False):
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(cmd)
        if renderer._DELIVERED_LOUDNESS_FILTER in cmd:
            return _loudnorm_json(readings.pop(0))
        if fail:
            raise RuntimeError("ffmpeg exit=1")
        Path(cmd[-1]).write_bytes(b"corrected")
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    renderer.reset_master_normalization_type()
    return cmds


def _builder(calls: list[tuple[str, Path]]):
    def build(extra: str, dest: Path) -> list[str]:
        calls.append((extra, dest))
        return ["ffmpeg", "-filter_complex", f"[0:a]master,{extra}[aout]", str(dest)]

    return build


async def test_a_miss_rerenders_once_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "master.mp4"
    out.write_bytes(b"original")
    cmds = _fake_run(monkeypatch, ["-21.69", "-14.20"])
    calls: list[tuple[str, Path]] = []

    await renderer._correct_master_loudness_miss(out, get_settings(), _builder(calls))

    assert len(calls) == 1, "the correction must re-render exactly once"
    assert calls[0][0].startswith("volume=7.69dB,alimiter=")
    assert calls[0][1] != out, "re-render must not overwrite the master in place"
    assert out.read_bytes() == b"corrected"
    assert len(cmds) == 3  # measure, re-render, re-measure — never a loop
    assert renderer.get_master_loudnorm_evidence() == {
        "loudness_correction_db": 7.69,
        "loudness_pre_correction_lufs": -21.69,
        "loudness_post_correction_lufs": -14.2,
    }


async def test_a_master_that_landed_is_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "master.mp4"
    out.write_bytes(b"original")
    cmds = _fake_run(monkeypatch, ["-14.52"])
    calls: list[tuple[str, Path]] = []

    await renderer._correct_master_loudness_miss(out, get_settings(), _builder(calls))

    assert calls == []
    assert len(cmds) == 1  # the measure only
    assert out.read_bytes() == b"original"
    assert "loudness_correction_db" not in renderer.get_master_loudnorm_evidence()


async def test_a_failed_correction_ships_the_uncorrected_master(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "master.mp4"
    out.write_bytes(b"original")
    _fake_run(monkeypatch, ["-21.69"], fail=True)

    await renderer._correct_master_loudness_miss(out, get_settings(), _builder([]))

    assert out.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [out], "the failed re-render left a temp file"
    assert "loudness_correction_db" not in renderer.get_master_loudnorm_evidence()


# --------------------------------------------------------------------------
# 3. Real ffmpeg: reproduce the defect, then prove the fix
# --------------------------------------------------------------------------

_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="real-ffmpeg test: ffmpeg/ffprobe not on PATH",
)

_DUR = 15.0


def _ffmpeg(*args: str) -> str:
    exe = shutil.which("ffmpeg")
    assert exe is not None
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell, resolved exe
        [exe, "-hide_banner", "-nostats", "-y", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stderr


def _ebur128(path: Path) -> tuple[float, float]:
    """(integrated LUFS, TRUE peak dBTP) of ``path``'s audio, via ebur128."""
    err = _ffmpeg("-i", str(path), "-map", "0:a", "-af", "ebur128=peak=true", "-f", "null", "-")
    summary = err[err.rfind("Summary:") :]
    integrated = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+) dBFS", summary)
    assert integrated is not None
    assert peak is not None
    return float(integrated.group(1)), float(peak.group(1))


def _speech_like(path: Path, leading_silence: float) -> None:
    """A speech-like bus that OPENS on ``leading_silence`` of digital silence.

    Pink-noise "speech" with a short 150 Hz plosive every 250 ms (a speech-like
    crest, so loudnorm refuses ``linear`` exactly as it does in prod), levelled so
    pass 1 measures ~-23 LUFS after the bus compressor (prod: ~-22.85)."""
    raw = path.with_suffix(".wav")
    gate = f"gte(t,{leading_silence * _DUR:.3f})"
    _ffmpeg(
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:sample_rate=48000:amplitude=0.25:d={_DUR}",
        "-filter_complex",
        f"[0:a]volume='if({gate.replace(',', chr(92) + ',')}\\,1\\,0)':eval=frame[n];"
        f"aevalsrc='0.5*sin(2*PI*150*t)*if({gate}*lt(mod(t,0.25),0.012),1,0)':s=48000:d={_DUR}[p];"
        "[n][p]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo[mix]",
        "-map", "[mix]", str(raw),
    )  # fmt: skip
    integrated, _ = _ebur128(raw)
    _ffmpeg(
        "-f", "lavfi", "-i", f"color=black:s=160x120:r=25:d={_DUR}",
        "-i", str(raw),
        "-af", f"volume={-16.0 - integrated:.2f}dB",
        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(path),
    )  # fmt: skip


@pytest.mark.render
@_needs_ffmpeg
async def test_a_film_opening_on_silence_is_corrected_to_target(tmp_path: Path) -> None:
    settings = get_settings()
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _speech_like(src, leading_silence=0.65)
    renderer.reset_master_normalization_type()

    await renderer.master_audio(src, out, settings)

    record = renderer.get_master_loudnorm_evidence()
    # THE DEFECT, reproduced: the dynamic master missed the target by more than
    # the gate's tolerance. Without this the fix-half below could pass vacuously.
    pre = record.get("loudness_pre_correction_lufs")
    assert isinstance(pre, float)
    assert settings.master_lufs - pre > _TOL
    # THE FIX: one constant gain landed it within tolerance, under the 0 dBTP gate.
    integrated, true_peak = _ebur128(out)
    assert abs(integrated - settings.master_lufs) <= _TOL
    assert true_peak <= mastering.TRUE_PEAK_CEILING_DBTP
    assert record.get("loudness_correction_db") == pytest.approx(settings.master_lufs - pre)


@pytest.mark.render
@_needs_ffmpeg
async def test_a_film_that_landed_is_not_corrected(tmp_path: Path) -> None:
    settings = get_settings()
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _speech_like(src, leading_silence=0.10)
    renderer.reset_master_normalization_type()

    await renderer.master_audio(src, out, settings)

    assert "loudness_correction_db" not in renderer.get_master_loudnorm_evidence()
    integrated, _ = _ebur128(out)
    assert abs(integrated - settings.master_lufs) <= _TOL


def _music_like(path: Path) -> None:
    """A steady tonal bed (a C-major triad) as long as the film, so it is never
    looped — the bed's own level is not what is under test."""
    _ffmpeg(
        "-f", "lavfi", "-i",
        f"aevalsrc='0.2*sin(2*PI*261.63*t)+0.2*sin(2*PI*329.63*t)+0.2*sin(2*PI*392*t)'"
        f":s=48000:d={_DUR}",
        "-af", "aformat=channel_layouts=stereo", str(path),
    )  # fmt: skip


@pytest.mark.render
@_needs_ffmpeg
async def test_a_film_opening_on_silence_under_a_music_bed_lands_on_target(
    tmp_path: Path,
) -> None:
    """The MUSIC-BED path. A film with music never reaches ``master_audio``: the
    master is folded into ``overlay_audio_bed``'s own re-encode, which carries its
    own copy of the miss correction (``_bed_pass2_cmd``). The no-bed tests above
    cannot see a defect there.

    Narration opens on ~3 s of silence with the bed already playing under it.
    Both numbers are MEASURED off the delivered file with ebur128 — never read
    back from the report the code under test wrote.

    ⚠️ THIS PINS THE OUTCOME, NOT THE CORRECTION FIRING. Measured when written
    (ffmpeg 4.4): the master landed -14.3 LUFS / -1.2 dBTP with NO
    ``loudness_correction_db`` recorded — the bed fills the opening, so the bus
    is not digital silence there and dynamic loudnorm does not under-ride it
    (9.75 s of narration silence gave -13.0 / -1.1, also uncorrected). So the
    correction is deliberately not asserted either way: the contract is that
    the delivered file is on target, whichever branch gets it there."""
    settings = get_settings()
    src = tmp_path / "in.mp4"
    bed = tmp_path / "bed.wav"
    out = tmp_path / "out.mp4"
    _speech_like(src, leading_silence=3.0 / _DUR)
    _music_like(bed)
    renderer.reset_master_normalization_type()

    await renderer.overlay_audio_bed(
        src,
        out,
        total_seconds=_DUR,
        music=(bed, MusicBed(url="bed")),
        ambient=[],
        settings=settings,
    )

    integrated, true_peak = _ebur128(out)
    assert abs(integrated - settings.master_lufs) <= _TOL, (integrated, true_peak)
    # The repo's gate ceiling (0.0), not ``master_tp``: the gate convicts only a
    # limiter that did not hold. CI (test job, ffmpeg 6.1.1 — the prod image is 7.1.5) measured -0.9 dBTP here, which is
    # 0.4 dB over the limiter's -1.3 aim and more than the 0.07-0.13 dB overshoot
    # mastering.py documents — unexplained, and inside the ceiling.
    assert true_peak <= mastering.TRUE_PEAK_CEILING_DBTP, (integrated, true_peak)
