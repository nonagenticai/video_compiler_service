"""ACX audiobook master + m4b chapter tests.

Pure tests (``evaluate_acx``, ``parse_noise_floor``, ``_write_chapter_metadata``,
command byte-identity) run without ffmpeg.

The five end-to-end proofs — that the ACX master actually lands a real file in
the ACX loudness window, that a silent stitch degrades instead of crashing, that
the m4b carries chapter markers, and that the combined concat does not truncate
AAC inputs — wore ``@pytest.mark.integration`` until 2026-09-07. That marker is
DESELECTED by ``addopts`` and re-selected nowhere in CI, so those five ran on no
machine at all: reverting the truncation fix (``-c copy`` mis-muxing AAC-in-wav
and dropping ~30% of the book) back to its original bug would have been a green
diff. They need nothing CI lacks — only real ffmpeg/ffprobe, which the archetype
installs via ``extra_apt: ffmpeg`` — so they now wear ``render``, the marker this
repo uses for real-encode tests and which nothing deselects, plus the same
``shutil.which`` module guard the render harness uses.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import Chapter
from src.config import Settings, get_settings
from src.services import renderer
from src.services.loudness import acx_measure, evaluate_acx, parse_noise_floor

#: Real-ffmpeg proofs: tagged ``render`` (SELECTED in CI — `addopts` deselects
#: only ``integration``) and skipped only where the binaries genuinely are not
#: installed. Not ``integration``: that marker is a hole, not a gate.
_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on runner (CI installs them via extra_apt: ffmpeg)",
)

# ACX gate bounds used across the pure tests (mirror the service defaults).
_GATES = {"lufs_min": -23.0, "lufs_max": -18.0, "tp_max": -3.0, "noise_floor_max": -60.0}


# --------------------------------------------------------------------------- #
# evaluate_acx — the pure compliance verdict
# --------------------------------------------------------------------------- #
def test_evaluate_acx_passes_in_band() -> None:
    report = evaluate_acx(
        {"integrated_lufs": -20.0, "true_peak_dbtp": -3.5, "noise_floor_db": -70.0}, **_GATES
    )
    assert report["pass"] is True
    assert report["integrated_lufs"] == -20.0


@pytest.mark.parametrize(
    "measure",
    [
        {"integrated_lufs": -17.0, "true_peak_dbtp": -3.5, "noise_floor_db": -70.0},  # too loud
        {"integrated_lufs": -24.0, "true_peak_dbtp": -3.5, "noise_floor_db": -70.0},  # too quiet
        {"integrated_lufs": -20.0, "true_peak_dbtp": -2.0, "noise_floor_db": -70.0},  # TP hot
        {"integrated_lufs": -20.0, "true_peak_dbtp": -3.5, "noise_floor_db": -55.0},  # noisy
        {"integrated_lufs": None, "true_peak_dbtp": -3.5, "noise_floor_db": -70.0},  # unmeasured
        None,  # whole measure missing
    ],
)
def test_evaluate_acx_fails_each_gate(measure: dict[str, float | None] | None) -> None:
    """A single out-of-spec (or missing) metric fails the whole verdict — an
    unmeasurable master is NOT a passing master."""
    assert evaluate_acx(measure, **_GATES)["pass"] is False


# --------------------------------------------------------------------------- #
# parse_noise_floor — Overall block, -inf clamp
# --------------------------------------------------------------------------- #
def test_parse_noise_floor_takes_overall_last_block() -> None:
    stderr = (
        "[astats] Noise floor dB: -41.2\n"  # per-channel
        "[astats] Overall\n"
        "[astats] Noise floor dB: -63.7\n"  # Overall (last) — the whole-file value
    )
    assert parse_noise_floor(stderr) == pytest.approx(-63.7)


def test_parse_noise_floor_inf_clamped_finite() -> None:
    # Pure digital silence → -inf → clamped to a finite sub-audible floor so it
    # stays JSON-safe and still clears any real ceiling.
    v = parse_noise_floor("[astats] Noise floor dB: -inf\n")
    assert v is not None
    assert v == pytest.approx(-120.0)
    assert v <= -60.0


def test_parse_noise_floor_absent_is_none() -> None:
    assert parse_noise_floor("no astats output here") is None


# --------------------------------------------------------------------------- #
# _write_chapter_metadata — cumulative END, last runs to total
# --------------------------------------------------------------------------- #
def test_write_chapter_metadata_offsets(tmp_path: Path) -> None:
    chapters = [
        Chapter(title="Intro", start_ms=0),
        Chapter(title="Chapter 1", start_ms=5000),
        Chapter(title="Chapter 2", start_ms=12000),
    ]
    out = tmp_path / "meta.txt"
    renderer._write_chapter_metadata(chapters, total_seconds=20.0, path=out)
    text = out.read_text()
    assert text.startswith(";FFMETADATA1")
    assert text.count("[CHAPTER]") == 3
    # END of ch i is the start of ch i+1; the last chapter runs to total (20000ms).
    assert "START=0\nEND=5000" in text
    assert "START=5000\nEND=12000" in text
    assert "START=12000\nEND=20000" in text
    assert "title=Chapter 1" in text


def test_write_chapter_metadata_escapes_specials(tmp_path: Path) -> None:
    out = tmp_path / "meta.txt"
    renderer._write_chapter_metadata(
        [Chapter(title="A=B; #1", start_ms=0)], total_seconds=1.0, path=out
    )
    text = out.read_text()
    assert "title=A\\=B\\; \\#1" in text


def test_write_chapter_metadata_zero_length_nudged(tmp_path: Path) -> None:
    # Two chapters at the same offset would make a zero-length span; nudge to ≥1ms.
    out = tmp_path / "meta.txt"
    renderer._write_chapter_metadata(
        [Chapter(title="a", start_ms=1000), Chapter(title="b", start_ms=1000)],
        total_seconds=2.0,
        path=out,
    )
    assert "START=1000\nEND=1001" in out.read_text()


# --------------------------------------------------------------------------- #
# byte-identity: acx_master off + no chapters ⇒ legacy ffmpeg command
# --------------------------------------------------------------------------- #
async def test_concat_command_byte_identical_when_acx_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no acx_master and no chapters, the WAV-copy command carries NO
    loudnorm / -ar 44100 / ipod — byte-identical to the legacy stitch."""
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(cmd)
        return ""

    async def fake_fetch(url: str, dest: Path, http: object, **_kw: object) -> Path:
        dest.write_bytes(b"x")
        return dest

    async def fake_probe(path: Path) -> float:
        return 10.0

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_assert_uniform_audio", lambda paths: _noop())

    from src.api.schemas import AudioClip

    clips = [AudioClip(id="c0", audio_url="http://m/0.wav")]
    _out, _dur, acx = await renderer.concat_audio(clips, get_settings(), "wav", 0)
    joined = " ".join(cmds[-1])
    assert "loudnorm" not in joined
    assert "44100" not in joined
    assert "ipod" not in joined
    assert acx is None


async def _noop() -> None:
    return None


# --------------------------------------------------------------------------- #
# Integration proofs (real ffmpeg)
# --------------------------------------------------------------------------- #
async def _synth_clip(path: Path, freq: int, seconds: float) -> None:
    """A -6 dB tone with a silent tail — approximates speech + pauses so the
    noise floor reads clean (a pure tone reads a high, failing noise floor).
    Runs through ``renderer._run`` (the ffmpeg gate), like every other test."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={seconds}:sample_rate=24000",
            "-af",
            "volume=-6dB,apad=pad_dur=0.6",
            "-ac",
            "1",
            "-t",
            f"{seconds + 0.6}",
            str(path),
        ]
    )


async def _probe_chapters(path: Path) -> list[dict[str, Any]]:
    """ffprobe the chapter list of ``path`` (through the ffmpeg gate)."""
    stdout = await renderer._run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_chapters",
            str(path),
        ]
    )
    chapters = json.loads(stdout).get("chapters", [])
    return list(chapters)


def _chapter_ms(start: int | str, time_base: str) -> int:
    """A ffprobe chapter ``start`` in milliseconds, whatever ``time_base`` it uses."""
    return round(Fraction(int(start)) * Fraction(time_base) * 1000)


def _as_float(value: object) -> float:
    """Narrow a measured metric to a real float for typed comparisons."""
    assert isinstance(value, (int, float)), value
    return float(value)


def _settings_with_workdir(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"work_dir": str(tmp_path)})


@pytest.mark.render
@_needs_ffmpeg
async def test_acx_master_lands_in_band_and_neg_control(tmp_path: Path) -> None:
    """The ACX master lands the combined book in the ACX window; the SAME raw
    input measured WITHOUT the master is out of band (proves the master is causal)."""
    c0, c1 = tmp_path / "c0.wav", tmp_path / "c1.wav"
    await _synth_clip(c0, 180, 2.5)
    await _synth_clip(c1, 240, 2.5)
    from src.api.schemas import AudioClip

    clips = [AudioClip(id="c0", audio_url=str(c0)), AudioClip(id="c1", audio_url=str(c1))]
    settings = _settings_with_workdir(tmp_path)

    _out, dur, report = await renderer.concat_audio(clips, settings, "wav", 0, acx_master=True)
    assert report is not None
    assert report["pass"] is True, report
    assert -23.0 <= _as_float(report["integrated_lufs"]) <= -18.0
    assert _as_float(report["true_peak_dbtp"]) <= -3.0
    assert _as_float(report["noise_floor_db"]) <= -60.0
    assert dur > 0

    # NEG control: measure the RAW (unmastered) stitch — a -6 dB tone reads well
    # below the ACX floor, so it must NOT be in band. If this passed, the master
    # above would be proving nothing.
    raw_out, _dur2, raw_report = await renderer.concat_audio(clips, settings, "wav", 0)
    assert raw_report is None
    raw_measure = await acx_measure(str(raw_out))
    assert raw_measure is not None
    raw_verdict = evaluate_acx(raw_measure, **_GATES)
    assert raw_verdict["pass"] is False, raw_measure


@pytest.mark.render
@_needs_ffmpeg
async def test_acx_master_silent_stitch_degrades_not_crashes(tmp_path: Path) -> None:
    """Regression: a fully-silent stitch makes ffmpeg report measured_I=-inf /
    offset=inf. The linear two-pass must be SKIPPED (non-finite) and the master
    must fall back to single-pass dynamic loudnorm — producing a real file, not a
    0-byte crash."""
    silent = tmp_path / "s.wav"
    await renderer._run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "2", str(silent)]
    )
    from src.api.schemas import AudioClip

    clips = [AudioClip(id="s", audio_url=str(silent))]
    settings = _settings_with_workdir(tmp_path)
    # Must not raise (the old guard built measured_I=-inf → ffmpeg rejected it).
    out, dur, report = await renderer.concat_audio(clips, settings, "wav", 0, acx_master=True)
    assert out.exists() and out.stat().st_size > 0
    assert dur > 0
    # A silent book cannot be ACX-compliant (loudness out of band), so pass=False,
    # but the job COMPLETED rather than crashing.
    assert report is not None and report["pass"] is False


@pytest.mark.render
@_needs_ffmpeg
async def test_m4b_has_chapter_markers(tmp_path: Path) -> None:
    """output_format=m4b + chapters ⇒ the delivered m4b carries N chapter markers
    at the requested cumulative offsets."""
    c0, c1 = tmp_path / "c0.wav", tmp_path / "c1.wav"
    await _synth_clip(c0, 180, 2.0)
    await _synth_clip(c1, 240, 2.0)
    from src.api.schemas import AudioClip

    clips = [AudioClip(id="c0", audio_url=str(c0)), AudioClip(id="c1", audio_url=str(c1))]
    chapters = [Chapter(title="One", start_ms=0), Chapter(title="Two", start_ms=2600)]
    settings = _settings_with_workdir(tmp_path)

    out, _dur, report = await renderer.concat_audio(
        clips, settings, "m4b", 0, acx_master=True, chapters=chapters
    )
    assert out.suffix == ".m4b"
    assert report is not None and report["pass"] is True

    got = await _probe_chapters(out)
    assert len(got) == 2
    assert got[0]["tags"]["title"] == "One"
    assert got[1]["tags"]["title"] == "Two"
    # The chapter start is written as 2600 at TIMEBASE=1/1000, but the
    # timebase ffprobe REPORTS back is the demuxer's choice: ffmpeg <=7 returns
    # 1/1000, ffmpeg 9 returns the audio track's timescale (1/44100 -> 114660).
    # Compare the instant, not the raw integer.
    assert _chapter_ms(got[1]["start"], got[1]["time_base"]) == 2600


async def _encode_m4b(src: Path, dst: Path) -> None:
    """Encode a wav to a real AAC m4b (the per-chapter file shape the combined
    ACX concat actually receives from the orchestrator)."""
    await renderer._run(
        ["ffmpeg", "-y", "-i", str(src), "-c:a", "aac", "-b:a", "128k", "-f", "ipod", str(dst)]
    )


@pytest.mark.render
@_needs_ffmpeg
async def test_acx_master_m4b_inputs_not_truncated(tmp_path: Path) -> None:
    """Regression: the combined ACX concat gets AAC/m4b per-chapter files. The raw
    stitch must be RE-ENCODED to PCM (not `-c copy`, which mis-muxes AAC-in-wav
    and silently drops ~30% of the book). Assert the mastered output keeps the
    full summed duration."""
    w0, w1 = tmp_path / "w0.wav", tmp_path / "w1.wav"
    await _synth_clip(w0, 180, 2.5)  # ~3.1s each (2.5 tone + 0.6 pad)
    await _synth_clip(w1, 240, 2.5)
    m0, m1 = tmp_path / "m0.m4b", tmp_path / "m1.m4b"
    await _encode_m4b(w0, m0)
    await _encode_m4b(w1, m1)
    from src.api.schemas import AudioClip

    clips = [AudioClip(id="m0", audio_url=str(m0)), AudioClip(id="m1", audio_url=str(m1))]
    settings = _settings_with_workdir(tmp_path)
    _out, dur, report = await renderer.concat_audio(
        clips, settings, "m4b", 0, acx_master=True, chapters=[Chapter(title="X", start_ms=0)]
    )
    # Full book ≈ 2*3.1s ≈ 6.2s; the truncation bug produced ~4.4s. Allow AAC pad.
    assert dur >= 5.8, f"book truncated: {dur}s"
    assert report is not None and report["pass"] is True


@pytest.mark.render
@_needs_ffmpeg
async def test_m4b_without_master_still_chaptered(tmp_path: Path) -> None:
    """m4b + chapters without acx_master still produces a chaptered container
    (chapter path is independent of the master)."""
    c0 = tmp_path / "c0.wav"
    await _synth_clip(c0, 200, 2.0)
    from src.api.schemas import AudioClip

    clips = [AudioClip(id="c0", audio_url=str(c0))]
    settings = _settings_with_workdir(tmp_path)
    out, _dur, report = await renderer.concat_audio(
        clips, settings, "m4b", 0, chapters=[Chapter(title="Solo", start_ms=0)]
    )
    assert report is None  # no master requested
    assert len(await _probe_chapters(out)) == 1


# --------------------------------------------------------------------------- #
# loudnorm frame alignment on the ACX master (the #242 fix, audiobook path)
# --------------------------------------------------------------------------- #
#: A programme length that is NOT a whole number of loudnorm's 100 ms frames.
_OFF_GRID_SECONDS = 5.0375


def _capture_acx_cmds(
    monkeypatch: pytest.MonkeyPatch, raw_seconds: float | None
) -> list[list[str]]:
    """Fake ``_run`` + the raw-stitch probe; returns the captured ffmpeg argvs
    (``[measure, encode]``). The measure pass returns no JSON, so pass 2 is the
    DYNAMIC fallback — the mode the tail hole belongs to."""
    cmds: list[list[str]] = []

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        cmds.append(cmd)
        return ""

    async def fake_seconds(path: Path) -> float | None:
        return raw_seconds

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_audio_stream_seconds", fake_seconds)
    return cmds


async def test_acx_master_frame_aligns_both_loudnorm_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dynamic ``loudnorm`` drops the missing part of a partial last 100 ms frame
    (ffmpeg 7.x; see ``_loudnorm_frame_align``). #242 padded the FILM master's bus
    to whole frames; the audiobook master ran a bare ``loudnorm`` on both passes."""
    cmds = _capture_acx_cmds(monkeypatch, _OFF_GRID_SECONDS)
    await renderer._acx_program_master(
        tmp_path / "raw.wav", tmp_path / "out.wav", "wav", lufs_target=-19.0, tp=-3.0, lra=12.0
    )
    measure, encode = cmds
    align = renderer._loudnorm_frame_align(_OFF_GRID_SECONDS)
    assert align.endswith("apad=whole_dur=5.1,"), align
    for cmd in (measure, encode):
        af = cmd[cmd.index("-af") + 1]
        # Alignment sits directly in front of the ONE loudnorm, on both passes —
        # pass 1 must measure the bus pass 2 normalises.
        assert af.startswith(f"{align}loudnorm="), af
        assert af.count("loudnorm=") == 1
        # Nothing rides downstream of loudnorm in the graph: a filter node there
        # is what breaks format negotiation on this path (see the ``-resampler``
        # comment in ``_acx_program_master``). The pad is removed by ``-t``.
        assert "," not in af.split("loudnorm=", 1)[1], af
        assert "atrim" not in af.split("loudnorm=", 1)[1], af
    # The pad is bounded back to the true programme length by an OUTPUT ``-t``.
    assert encode.count("-t") == 1
    t_at = encode.index("-t")
    assert t_at > encode.index("-i") + 1, "-t must be an output option, not an input one"
    assert encode[t_at + 1] == renderer._ffdur(_OFF_GRID_SECONDS)
    # The measure pass writes nothing, so it carries no bound.
    assert "-t" not in measure


async def test_acx_master_unprobeable_length_fails_open_unaligned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No length ⇒ no pad AND no ``-t`` (a pad with nothing to bound it would
    lengthen the book). Same fail-open as the film master: today's audio."""
    cmds = _capture_acx_cmds(monkeypatch, None)
    await renderer._acx_program_master(
        tmp_path / "raw.wav", tmp_path / "out.wav", "wav", lufs_target=-19.0, tp=-3.0, lra=12.0
    )
    measure, encode = cmds
    for cmd in (measure, encode):
        assert cmd[cmd.index("-af") + 1].startswith("loudnorm=")
    assert "-t" not in encode


@pytest.mark.render
@_needs_ffmpeg
@pytest.mark.parametrize("mode", ["linear", "dynamic"])
async def test_acx_master_keeps_off_grid_programme_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Real ffmpeg: a programme that is NOT a multiple of 100 ms comes out of the
    ACX master the length it went in, to within one output audio sample-frame
    block (1024 samples @ 44.1 kHz) — neither short (the dropped partial frame)
    nor long (the alignment pad leaking past ``-t``). ``dynamic`` forces the
    fallback by blanking the measure pass, the mode the hole belongs to."""
    raw = tmp_path / "raw.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={_OFF_GRID_SECONDS}:sample_rate=24000",
            "-af",
            "volume=-6dB",
            "-ac",
            "1",
            str(raw),
        ]
    )
    raw_seconds = await renderer._audio_stream_seconds(raw)
    assert raw_seconds is not None
    assert abs(raw_seconds - _OFF_GRID_SECONDS) < 1e-3
    assert abs(raw_seconds * 10 - round(raw_seconds * 10)) > 0.1, "fixture must be off the grid"

    out = tmp_path / "master.wav"
    with monkeypatch.context() as scoped:
        if mode == "dynamic":
            from src.services import loudness

            # Scoped to the master: ``acx_measure`` below parses with the same helper.
            scoped.setattr(loudness, "_extract_json_block", lambda _stderr: {})
        await renderer._acx_program_master(raw, out, "wav", lufs_target=-19.0, tp=-3.0, lra=12.0)
    out_seconds = await renderer._audio_stream_seconds(out)
    assert out_seconds is not None
    assert abs(out_seconds - raw_seconds) <= 1024 / 44100, (raw_seconds, out_seconds)
    # Still a real master, not a muted/padded file: loudness lands on target.
    measure = await acx_measure(str(out))
    assert measure is not None
    assert abs(_as_float(measure["integrated_lufs"]) - (-19.0)) <= 1.5, measure
