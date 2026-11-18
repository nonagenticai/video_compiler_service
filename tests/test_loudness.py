"""Pure-unit tests for the loudness stderr parser + best-effort measure pass.

``parse_loudness`` needs no ffmpeg. ``final_mux_loudness`` execs ffmpeg and wore
``@pytest.mark.integration`` until 2026-09-07 — a marker ``addopts`` DESELECTS
and CI re-selects nowhere, so the only test that actually runs ffmpeg over a
real file and reads the measurement back executed on no machine at all. It needs
nothing CI lacks (the archetype installs ffmpeg via ``extra_apt``), so it now
wears ``render``, which nothing deselects, plus a ``shutil.which`` guard.
"""

from __future__ import annotations

import shutil
from typing import cast

import pytest

from src.services import renderer
from src.services.loudness import final_mux_loudness, parse_loudness

#: Real-ffmpeg guard. ``render`` is SELECTED in CI (`addopts` deselects only
#: ``integration``); the skipif fires only where the binary genuinely is absent.
_needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on runner (CI installs it via extra_apt: ffmpeg)",
)

# A real ``loudnorm ...:print_format=json`` stderr dump (banner + trailing JSON).
_LOUDNORM_STDERR = """\
[Parsed_loudnorm_0 @ 0x7fb8d5f04080]
{
\t"input_i" : "-21.76",
\t"input_tp" : "-17.93",
\t"input_lra" : "0.10",
\t"input_thresh" : "-31.76",
\t"output_i" : "-13.96",
\t"output_tp" : "-10.14",
\t"output_lra" : "0.00",
\t"output_thresh" : "-23.96",
\t"normalization_type" : "dynamic",
\t"target_offset" : "-0.04"
}
"""

# A real ``silencedetect`` stderr dump over a 3.02s file with one 1.0449s silence.
_SILENCEDETECT_STDERR = """\
  Duration: 00:00:03.02, start: 0.000000, bitrate: 51 kb/s
[silencedetect @ 0x7fbbf6a04740] silence_start: 1
[silencedetect @ 0x7fbbf6a04740] silence_end: 2.0449 | silence_duration: 1.0449
"""


def test_parse_loudness_real_block_yields_floats() -> None:
    result = parse_loudness(_LOUDNORM_STDERR)
    assert result["integrated_lufs"] == pytest.approx(-21.76)
    assert result["true_peak_dbtp"] == pytest.approx(-17.93)
    assert result["lra"] == pytest.approx(0.10)
    # No silencedetect supplied → silence_ratio unknown.
    assert result["silence_ratio"] is None


def test_parse_loudness_garbage_all_none() -> None:
    for junk in ["", "not json at all", "{broken", "}{", "[Parsed] no braces here"]:
        result = parse_loudness(junk)
        assert result == {
            "integrated_lufs": None,
            "true_peak_dbtp": None,
            "lra": None,
            "silence_ratio": None,
            # astats' noise floor — the second measurement the speech-presence
            # gate needs. An EXACT dict comparison on purpose: a new key must be
            # stated here, because this payload is stored verbatim in a JSONB
            # column and read by another repo.
            "noise_floor_db": None,
        }


def test_parse_loudness_silence_ratio_computed() -> None:
    result = parse_loudness(_LOUDNORM_STDERR, _SILENCEDETECT_STDERR)
    # 1.0449s silence / 3.02s total.
    assert result["silence_ratio"] == pytest.approx(1.0449 / 3.02)


def test_parse_loudness_silence_ratio_none_without_duration() -> None:
    # silence lines but no parseable Duration → ratio unknown, not a crash.
    only_silence = "[silencedetect] silence_end: 2.0 | silence_duration: 1.0"
    assert parse_loudness(_LOUDNORM_STDERR, only_silence)["silence_ratio"] is None


def test_parse_loudness_silence_ratio_zero_when_no_silence() -> None:
    # Duration present but no silence spans → None (nothing detected), not 0.0.
    only_duration = "  Duration: 00:00:03.02, start: 0.000000, bitrate: 51 kb/s"
    assert parse_loudness(_LOUDNORM_STDERR, only_duration)["silence_ratio"] is None


def test_parse_loudness_handles_inf_silence() -> None:
    # loudnorm reports pure silence as "-inf". THIS ASSERTION USED TO PIN THE BUG:
    # it required the parser to return -inf, which is not JSON-safe and killed the
    # JSONB completion write (see tests/unit/test_loudness_json_safe.py). A reading
    # that cannot be persisted is ABSENT, not a measurement.
    inf_block = '{ "input_i" : "-inf", "input_tp" : "-inf", "input_lra" : "0.00" }'
    result = parse_loudness(inf_block)
    assert result["integrated_lufs"] is None
    assert result["true_peak_dbtp"] is None
    assert result["lra"] == pytest.approx(0.0)


def test_parse_loudness_takes_last_json_block() -> None:
    # Filter re-init prints multiple blocks; the LAST covers the whole file.
    doubled = (
        '{ "input_i" : "-inf", "input_tp" : "-inf", "input_lra" : "0.00" }\n'
        '{ "input_i" : "-14.20", "input_tp" : "-2.10", "input_lra" : "5.50" }\n'
    )
    result = parse_loudness(doubled)
    assert result["integrated_lufs"] == pytest.approx(-14.20)
    assert result["true_peak_dbtp"] == pytest.approx(-2.10)


@pytest.mark.asyncio
async def test_final_mux_loudness_returns_none_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed/raising measure pass degrades to None — never raises, never fails a job."""

    async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise renderer.FfmpegError("boom")

    monkeypatch.setattr(renderer, "_run", boom)
    assert await final_mux_loudness("/no/such/file.mp4") is None


@pytest.mark.asyncio
async def test_final_mux_loudness_parses_captured_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """final_mux_loudness reads STDERR (capture_stderr=True) and parses it."""
    combined = _SILENCEDETECT_STDERR + "\n" + _LOUDNORM_STDERR
    seen: dict[str, object] = {}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen["capture_stderr"] = capture_stderr
        seen["cmd"] = cmd
        return combined

    monkeypatch.setattr(renderer, "_run", fake_run)
    result = await final_mux_loudness("/tmp/final.mp4")
    assert seen["capture_stderr"] is True  # must read stderr, not stdout
    assert result is not None
    assert result["integrated_lufs"] == pytest.approx(-21.76)
    assert result["silence_ratio"] == pytest.approx(1.0449 / 3.02)


@pytest.mark.asyncio
async def test_final_mux_loudness_puts_silencedetect_before_loudnorm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FILTER ORDER IS LOAD-BEARING: ``silencedetect`` must precede ``loudnorm``.

    ``loudnorm`` is not a pass-through analyser — its single pass NORMALIZES the
    samples it forwards, so a downstream ``silencedetect`` measures audio already
    lifted toward -14 LUFS and misses quiet passages that really are below the
    -50 dB gate in the delivered file. ``silencedetect`` IS pass-through, so
    first-in-chain both filters measure the ORIGINAL signal (same reasoning as
    ``acx_measure``'s ``astats`` ordering). Proven end-to-end against real ffmpeg
    in ``tests/render_harness/test_silence_ratio_render.py``."""
    seen: dict[str, object] = {}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        seen["cmd"] = cmd
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    await final_mux_loudness("/tmp/final.mp4")
    cmd = cast(list[str], seen["cmd"])
    af: str = cmd[cmd.index("-af") + 1]
    assert "silencedetect" in af and "loudnorm" in af
    assert af.index("silencedetect") < af.index("loudnorm"), (
        f"silencedetect must come FIRST (loudnorm alters the samples it forwards): {af}"
    )
    # The loudnorm JSON the parser reads must still be requested.
    assert "print_format=json" in af


@pytest.mark.render
@_needs_ffmpeg
@pytest.mark.asyncio
async def test_final_mux_loudness_measures_real_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Full path: exec ffmpeg over a real tone file and measure it. Needs ffmpeg."""
    out = tmp_path / "tone.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "aac",
            str(out),
        ]
    )
    result = await final_mux_loudness(str(out))
    assert result is not None
    assert result["integrated_lufs"] is not None
