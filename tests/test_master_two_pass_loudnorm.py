"""The two-pass LINEAR master, and the things that can silently make it inert.

⚠️ WHY THIS FILE EXISTS. The headline of the master chain's second revision is
``linear=true`` — ONE constant gain across the film instead of ffmpeg's
single-pass dynamic mode, whose continuous gain re-derivation over a
mostly-speech bus is the audible "digital" gain riding that was reported. Every
test that existed before this file mocks ``renderer._run`` with a stub returning
``""``, so ``_measure_master_loudnorm`` always found no JSON, always returned
``None``, and every assertion about the master chain was made against the
FAIL-OPEN single-pass branch. The linear path had never executed anywhere.

Two independent ways the fix can be inert, both covered here:

1. Pass 1 returns nothing usable, so pass 2 never gets ``measured_*`` at all
   (that was the state of the whole suite).
2. Pass 1 succeeds, pass 2's filter string says ``linear=true``, and **ffmpeg
   runs dynamic anyway** — it drops ``linear`` when the measured offset is out of
   range, with no warning and no change to its exit code. A filter-STRING
   assertion cannot see that; only reading the mode back off pass-2's stderr can.

The stderr fixtures below are real ffmpeg loudnorm output shapes, not invented
ones — the parse is only worth as much as the text it was proven against.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import AmbientCue, MusicBed
from src.config import get_settings
from src.services import renderer
from tests.conftest import is_delivered_master_measure

# Real ffmpeg pass-1 output: the `print_format=json` stats block, preceded by the
# filter-graph banner ffmpeg always emits ahead of it. `_extract_json_block` takes
# the LAST `{...}`, so the banner has to be here for the test to prove anything
# about a real stderr rather than a bare JSON string.
_PASS1_JSON_STDERR = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'compiled.mp4':
  Stream #0:1(und): Audio: aac (LC), 48000 Hz, stereo, fltp, 128 kb/s
Stream mapping:
  Stream #0:1 (aac) -> aformat
[Parsed_loudnorm_1 @ 0x5591b4a0e400]
{
\t"input_i" : "-22.51",
\t"input_tp" : "-4.68",
\t"input_lra" : "6.20",
\t"input_thresh" : "-32.78",
\t"output_i" : "-14.01",
\t"output_tp" : "-1.31",
\t"output_lra" : "6.10",
\t"output_thresh" : "-24.28",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.01"
}
"""

# A digitally-silent bus. loudnorm reports `-inf`/`inf` here — NON-EMPTY strings
# that slip any `not in (None, "")` guard, and which build a `measured_I=-inf`
# filter that ffmpeg rejects outright (0-byte output, dead render). This is the
# exact payload the non-finite bail exists for, in its real shape.
_PASS1_SILENT_STDERR = """\
[Parsed_loudnorm_1 @ 0x5591b4a0e400]
{
\t"input_i" : "-inf",
\t"input_tp" : "-inf",
\t"input_lra" : "0.00",
\t"input_thresh" : "-inf",
\t"output_i" : "-inf",
\t"output_tp" : "-inf",
\t"output_lra" : "0.00",
\t"output_thresh" : "-inf",
\t"normalization_type" : "dynamic",
\t"target_offset" : "inf"
}
"""


def _pass2_stderr(kind: str) -> str:
    """Pass 2's end-of-run loudnorm summary (the DEFAULT, non-json print format).

    ``Normalization Type`` is the line that says which mode actually ran — the
    only place ffmpeg reports it. The wide column spacing is ffmpeg's own."""
    return f"""\
[Parsed_loudnorm_1 @ 0x5591b4a0e400]
Parsed_loudnorm_1
Input Integrated:    -22.5 LUFS
Input True Peak:      -4.7 dBTP
Input LRA:             6.2 LU
Input Threshold:     -32.8 LUFS

Output Integrated:   -14.0 LUFS
Output True Peak:     -1.3 dBTP
Output LRA:            6.1 LU
Output Threshold:    -24.3 LUFS

Normalization Type:   {kind}
Target Offset:        +0.0 LU
"""


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pass1: str = _PASS1_JSON_STDERR,
    pass2: str = _pass2_stderr("Linear"),
    has_audio: bool = True,
) -> dict[str, Any]:
    """Fakes whose ``_run`` returns REAL ffmpeg stderr, discriminating pass 1 from
    pass 2 by ``-f null -`` (the measure pass writes no file).

    This is the whole difference from the pre-existing fakes: theirs returned
    ``""``, which made every master test exercise the fail-open branch.
    """
    captured: dict[str, Any] = {"cmds": [], "measure_cmds": [], "encode_cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        captured["cmds"].append(cmd)
        if cmd[-1] == "-" and "null" in cmd:
            captured["measure_cmds"].append(cmd)
            return pass1
        captured["encode_cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return pass2

    async def fake_probe(path: Path) -> float:
        return 3.0

    async def fake_has_audio(path: Path) -> bool:
        return has_audio

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    renderer.reset_master_normalization_type()
    return captured


# --------------------------------------------------------------------------
# 1. Pass 1 parse + pass 2 carries the measurements
# --------------------------------------------------------------------------


async def test_measure_master_loudnorm_parses_real_pass1_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four measurements pass 2 needs, parsed out of real ffmpeg stderr and
    returned as STRINGS (they go straight into a filter arg — reformatting them
    through float would change the text ffmpeg sees)."""
    _install(monkeypatch)
    measured = await renderer._measure_master_loudnorm([], "[0:a]anull[a]", "[a]")
    assert measured == {
        "input_i": "-22.51",
        "input_tp": "-4.68",
        "input_lra": "6.20",
        "input_thresh": "-32.78",
    }


async def test_master_audio_second_pass_is_linear_with_measured_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE TEST THIS BRANCH SHIPPED WITHOUT. Pass 1 measures, pass 2 carries
    ``linear=true`` plus all four ``measured_*`` values. Nothing in the suite
    asserted this before: the old fakes returned ``""`` from ``_run``, so pass 1
    always failed and pass 2 always ran the single-pass dynamic fallback."""
    captured = _install(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    # Pass 1 ran, and it measured through the compressor (it is upstream of
    # loudnorm in delivery, so measuring without it measures a signal that never
    # reaches pass 2).
    assert len(captured["measure_cmds"]) == 1
    measure = " ".join(captured["measure_cmds"][0])
    assert "print_format=json" in measure
    assert "acompressor=" in measure

    pass2 = " ".join(captured["encode_cmds"][-1])
    assert "linear=true" in pass2
    assert "measured_I=-22.51" in pass2
    assert "measured_TP=-4.68" in pass2
    assert "measured_LRA=6.20" in pass2
    assert "measured_thresh=-32.78" in pass2
    # print_format=json belongs to the MEASURE pass only — pass 2 must not print.
    assert "print_format=json" not in pass2


def test_master_filter_without_measurements_is_the_dynamic_fallback() -> None:
    """The fail-open half of the contract: no measurements ⇒ no ``linear``, no
    ``measured_*``, and the chain still renders (a measurement hiccup must never
    fail a paid compile)."""
    chain = renderer._master_filter(get_settings(), None)
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11.0" in chain
    assert "linear" not in chain
    assert "measured_" not in chain


# --------------------------------------------------------------------------
# 1b. The non-finite guard, on a real -inf payload
# --------------------------------------------------------------------------


async def test_measure_master_loudnorm_bails_on_silent_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A digitally-silent bus measures ``-inf``. Left to run it builds
    ``measured_I=-inf``, which ffmpeg rejects with "Result too large" and takes
    the whole render with it. The bail returns None ⇒ single-pass fallback."""
    _install(monkeypatch, pass1=_PASS1_SILENT_STDERR)
    assert await renderer._measure_master_loudnorm([], "[0:a]anull[a]", "[a]") is None


async def test_master_audio_silent_bus_falls_back_to_dynamic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end: the -inf measurement must not reach the pass-2 filter string."""
    captured = _install(monkeypatch, pass1=_PASS1_SILENT_STDERR, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    pass2 = " ".join(captured["encode_cmds"][-1])
    assert "-inf" not in pass2
    assert "linear=true" not in pass2
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11.0" in pass2


@pytest.mark.parametrize(
    "stderr",
    [
        "",
        "no json here at all",
        # A complete object that is MISSING one required key.
        '{"input_i": "-22.5", "input_tp": "-4.7", "input_lra": "6.2"}',
        # Present but not a number.
        '{"input_i": "n/a", "input_tp": "-4.7", "input_lra": "6.2", "input_thresh": "-32.8"}',
    ],
)
async def test_measure_master_loudnorm_returns_none_on_unusable_pass1(
    monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    """Every unusable pass-1 shape fails OPEN to ``None``, never raises."""
    _install(monkeypatch, pass1=stderr)
    assert await renderer._measure_master_loudnorm([], "[0:a]anull[a]", "[a]") is None


async def test_measure_master_loudnorm_returns_none_on_ffmpeg_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4b: the ACTUAL cited defect — every prior "unusable pass1" case above is a
    malformed-but-returned STRING; none of them exercise the ``except Exception``
    branch (renderer.py's real trigger is an ffmpeg non-zero exit, i.e. ``_run``
    RAISING, not returning odd text). Prove the raise path fails open too."""

    async def raising_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        raise RuntimeError("ffmpeg exited 1")

    monkeypatch.setattr(renderer, "_run", raising_run)
    assert await renderer._measure_master_loudnorm([], "[0:a]anull[a]", "[a]") is None


async def test_master_audio_surfaces_normalization_type_when_pass1_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """4b end-to-end: pass 1 (measurement) RAISES — not malformed JSON, a real
    ffmpeg failure — and the render must still (a) not fail the compile
    [fail-open, unchanged] and (b) still surface the resulting downgrade via
    ``get_master_normalization_type()``, off pass 2's own stderr summary, so it
    reaches the job's mastering report and is visible/rejectable downstream.
    Mutate this test's pass-2 kind from "Dynamic" to "Linear" (revert the fix's
    read-back) and it goes RED, because the fallback pass 2 for a real 1x1 pixel
    input genuinely cannot honour linear with no measurements."""
    captured: dict[str, Any] = {"cmds": [], "measure_cmds": [], "encode_cmds": []}

    async def flaky_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        captured["cmds"].append(cmd)
        if cmd[-1] == "-" and "null" in cmd:
            captured["measure_cmds"].append(cmd)
            raise RuntimeError("ffmpeg exited 1 on the measure pass")
        captured["encode_cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return _pass2_stderr("Dynamic")

    async def fake_has_audio(path: Path) -> bool:
        return True

    monkeypatch.setattr(renderer, "_run", flaky_run)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    renderer.reset_master_normalization_type()

    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    # Fails open: the render completes despite pass 1 raising.
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    pass2 = " ".join(captured["encode_cmds"][-1])
    assert "linear=true" not in pass2  # no measurements to carry
    # The defect this closes: the downgrade is OBSERVABLE, not silent.
    assert renderer.get_master_normalization_type() == "dynamic"


# --------------------------------------------------------------------------
# 2. ffmpeg silently downgrading linear -> dynamic
# --------------------------------------------------------------------------


def test_normalization_type_parsed_from_real_pass2_summary() -> None:
    assert (
        renderer._record_master_normalization_type(_pass2_stderr("Linear"), requested_linear=True)
        == "linear"
    )
    assert (
        renderer._record_master_normalization_type(_pass2_stderr("Dynamic"), requested_linear=True)
        == "dynamic"
    )


def test_normalization_type_absent_summary_is_unknown_not_downgraded() -> None:
    """No summary ⇒ ``None`` (could not observe), NOT "dynamic". Reporting a
    downgrade we did not see would be worse than reporting nothing."""
    renderer.reset_master_normalization_type()
    assert renderer._record_master_normalization_type("", requested_linear=True) is None
    assert renderer.get_master_normalization_type() is None


def test_normalization_type_not_recorded_when_mastering_is_off() -> None:
    renderer.reset_master_normalization_type()
    assert (
        renderer._record_master_normalization_type(
            _pass2_stderr("Dynamic"), requested_linear=False, master_enabled=False
        )
        is None
    )
    assert renderer.get_master_normalization_type() is None


def _measured(input_i: str, input_tp: str, input_lra: str = "6.20") -> dict[str, str]:
    return {
        "input_i": input_i,
        "input_tp": input_tp,
        "input_lra": input_lra,
        "input_thresh": "-32.00",
    }


def test_linear_refusal_is_predicted_from_pass1_readings() -> None:
    """The refusal arithmetic, on the strings pass 1 hands the filter builder, at
    the live target pair (the boundary rows are the ones measured against real
    ffmpeg in ``_loudnorm_filter``'s comment)."""
    settings = get_settings()
    assert (settings.master_lufs, settings.master_tp) == (-14.0, -1.0)
    predicted = renderer._linear_refusal_predicted
    # -3 + 8 = +5 dBTP > -1: the constant gain would breach the ceiling.
    assert predicted(settings, _measured("-22", "-3")) is True
    # -9 + 8 = -1 dBTP <= -1: exactly ON the ceiling is still linear.
    assert predicted(settings, _measured("-22", "-9")) is False
    # The production row (-4.68 + 8.51 = +3.83).
    assert predicted(settings, _measured("-22.51", "-4.68")) is True
    # The independent refusal: zero LRA, with 23 dB of headroom to spare.
    assert predicted(settings, _measured("-22.00", "-24.00", "0.00")) is True
    assert predicted(settings, _measured("-22.00", "-24.00", "0.10")) is False
    # Nothing measured / nothing parseable ⇒ NOT predicted, so the warning stays.
    assert predicted(settings, None) is False
    assert predicted(settings, {"input_i": "-22"}) is False
    assert predicted(settings, _measured("n/a", "-3")) is False


def _downgrade_log_line(capsys: pytest.CaptureFixture[str]) -> str:
    """The one ``master.normalization_downgraded`` line (structlog writes to
    stdout — console key=value locally, JSON in CI; assert substrings in both)."""
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "master.normalization_downgraded" in line
    ]
    assert len(lines) == 1, lines
    return lines[0].lower()


def test_a_predicted_downgrade_is_info_not_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """Pass 1 already said linear would be refused ⇒ the dynamic run is expected:
    INFO with ``predicted=True``. The recorded mode is unchanged."""
    renderer.reset_master_normalization_type()
    capsys.readouterr()
    assert (
        renderer._record_master_normalization_type(
            _pass2_stderr("Dynamic"), requested_linear=True, refusal_predicted=True
        )
        == "dynamic"
    )
    line = _downgrade_log_line(capsys)
    assert "info" in line and "warning" not in line, line
    assert "predicted" in line and "true" in line, line
    assert renderer.get_master_normalization_type() == "dynamic"
    assert renderer.get_master_loudnorm_evidence() == {"requested_linear": True}


def test_an_unpredicted_downgrade_is_still_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """The arithmetic said linear was reachable and ffmpeg ran dynamic anyway —
    the one case the warning is for. Also the default when no prediction is given."""
    renderer.reset_master_normalization_type()
    capsys.readouterr()
    assert (
        renderer._record_master_normalization_type(_pass2_stderr("Dynamic"), requested_linear=True)
        == "dynamic"
    )
    line = _downgrade_log_line(capsys)
    assert "warning" in line, line
    assert "predicted" in line and "false" in line, line


@pytest.mark.parametrize(
    ("pass1", "level", "predicted"),
    [
        # The production row: headroom refusal, known before pass 2 ran.
        ("refused", "info", "true"),
        # Pass 1 said linear fits, pass 2 ran dynamic regardless.
        ("granted", "warning", "false"),
    ],
)
async def test_master_audio_levels_the_downgrade_by_its_own_pass1_readings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pass1: str,
    level: str,
    predicted: str,
) -> None:
    """End to end through ``master_audio``: the prediction is computed from the
    measurements that render itself took — not passed in by the test."""
    stderr = {"refused": _PASS1_REFUSED_STDERR, "granted": _PASS1_GRANTED_STDERR}[pass1]
    _install(monkeypatch, pass1=stderr, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    capsys.readouterr()
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    line = _downgrade_log_line(capsys)
    assert level in line, line
    assert ("warning" in line) is (level == "warning"), line
    assert "predicted" in line and predicted in line, line
    assert renderer.get_master_normalization_type() == "dynamic"


async def test_master_audio_surfaces_a_silent_dynamic_downgrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE INERT-IN-PRODUCTION CASE. Pass 1 succeeded, the pass-2 filter string
    says ``linear=true``, and ffmpeg ran DYNAMIC. Without reading the mode back
    off stderr this ships with the gain riding the fix was meant to remove, and
    every string assertion above stays green."""
    captured = _install(monkeypatch, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    # We ASKED for linear...
    assert "linear=true" in " ".join(captured["encode_cmds"][-1])
    # ...and the observable truth is that we did not get it.
    assert renderer.get_master_normalization_type() == "dynamic"


async def test_master_audio_records_linear_when_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, pass2=_pass2_stderr("Linear"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    assert renderer.get_master_normalization_type() == "linear"


async def test_bed_path_also_records_the_normalization_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bed path folds the same master into its own re-encode, so it needs the
    same read-back — a film WITH music must not be the blind spot."""
    _install(monkeypatch, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m")),
        ambient=[],
        settings=get_settings(),
    )
    assert renderer.get_master_normalization_type() == "dynamic"


# --------------------------------------------------------------------------
# 3. The master_enabled=False amix had no limiter at all
# --------------------------------------------------------------------------


async def test_bed_mix_is_limited_even_with_mastering_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``normalize=0`` is right for level and means bed + narration SUM. With the
    master chain off there is no compressor, no loudnorm and nothing else between
    that sum and the AAC encoder, so a loud bed under a loud line clips.
    ``level=disabled`` matters as much as ``limit``: alimiter's ``level`` defaults
    to TRUE and re-normalises the output back to full scale."""
    captured = _install(monkeypatch)
    settings = get_settings().model_copy(update={"master_enabled": False})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m")),
        ambient=[],
        settings=settings,
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "amix=inputs=2:duration=first:normalize=0" in cmd
    # DERIVED from `master_tp` (the one configured ceiling), never a hardcoded 0.95.
    limit = renderer._ffnum(renderer._limiter_linear_limit(get_settings()))
    assert f"alimiter=limit={limit}:level=disabled[aout]" in cmd
    assert "alimiter=limit=0.95:" not in cmd
    # Still no master chain — the limiter is the ONLY thing this branch gained.
    assert "loudnorm=" not in cmd
    assert "acompressor=" not in cmd


# --------------------------------------------------------------------------
# 5. soxr on every PER-SHOT input; still NOT on the already-48 kHz buses
# --------------------------------------------------------------------------


def test_sfx_assets_resample_with_soxr() -> None:
    """Library SFX are overwhelmingly 44.1 kHz and 44.1 -> 48 is a non-integer
    ratio; swr's default cubic interpolator aliases the top octave, which on
    chimes/cymbals is exactly the artifact being chased."""
    from src.api.schemas import AccentCue, SoundCue

    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=(2, SoundCue(url="x", gain_db=-3.0)),
        accents=[(3, AccentCue(url="y", at_seconds=1.5, gain_db=-6.0))],
        settings=get_settings(),
    )
    # THREE, not two: the primary joined the assets on 2026-09-13. This count was
    # 2 while the primary was on bare `_A48` — see
    # ``test_sfx_graph_primary_narration_bus_resamples_with_soxr`` below for why
    # that was wrong. The three are [1:a] primary, [2:a] transition, [3:a] accent
    # — i.e. EVERY input to the per-shot mix, which is the actual invariant.
    assert graph.count("resampler=soxr:precision=28") == 3
    assert "[1:a]aresample=48000:resampler=soxr:precision=28," in graph
    assert "[2:a]aresample=48000:resampler=soxr:precision=28," in graph
    assert "[3:a]aresample=48000:resampler=soxr:precision=28," in graph


def test_sfx_graph_primary_narration_bus_resamples_with_soxr() -> None:
    """REPLACES ``test_narration_bus_is_not_widened_to_soxr``, whose premise was
    disproved on 2026-09-13. That test asserted ``[1:a]aresample not in graph`` on
    the grounds that "the primary/narration bus already arrives 48 kHz stereo, so
    ``_A48`` there is a genuine no-op". Both halves are false:

    * ``_audio_sfx_graph`` has exactly TWO callers (``renderer.py`` 4449 and 4459),
      both branches of ``_video_audio_map_args``, which builds the PER-SHOT segment
      command in ``render_shot``. There is no master-bus caller. The ``amix`` +
      ``alimiter`` in this graph is the per-shot limiter — the function's own
      docstring says the segment "is encoded (and stream-copied into the film)
      BEFORE the post-concat master ever sees it".
    * ``primary_label`` IS ``render_shot``'s ``audio_map``, which is one of:
      the fetched narration file (``tts_backend`` renders **24000 Hz MONO**,
      hardcoded, and ``_fetch_asset`` is a byte fetch with no transcode), the
      source clip's own audio at an arbitrary rate, or ``anullsrc`` at an explicit
      48000. So the dominant case is a real 24k -> 48k conversion, and the one
      genuinely-48k case is a byte-identical ``aresample`` pass-through (verified).

    Measured 24k mono -> 48k stereo, cubic vs soxr: identical below 11.5 kHz
    (0.00 dB), but cubic leaves the mirror image at -70.7 dBFS in 12-16 kHz against
    a -20.9 dBFS program where soxr leaves -180 dBFS.

    The scope guard this test used to perform has NOT been dropped — it moved to
    the site where the measurement actually says 0.00 dB:
    ``test_renderer_narration_resampler.py::test_master_bus_decimation_deliberately_left_on_cubic``.
    """
    from src.api.schemas import AccentCue

    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=None,
        accents=[(3, AccentCue(url="y", at_seconds=1.0))],
        settings=get_settings(),
    )
    assert "[1:a]aresample=48000:resampler=soxr:precision=28," in graph
    assert "[1:a]aformat=channel_layouts=stereo:sample_rates=48000[a_pri]" not in graph


def test_music_bed_resamples_with_soxr() -> None:
    chain = renderer._music_bed_filter(1, 30.0, -18.0, MusicBed(url="m"))
    assert chain.startswith("[1:a]aresample=48000:resampler=soxr:precision=28,")


async def test_ambient_bed_resamples_with_soxr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a", gain_db=-24.0))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "aresample=48000:resampler=soxr:precision=28,aformat=" in cmd
    # 4d: the narration split it mixes against also carries soxr now (latent-
    # proofing, not a behaviour change — byte-identical when already 48 kHz).
    # #235: the narration bus is now padded to `total` with `_pad_audio_to`
    # BEFORE the split (so the sidechain KEY branch lasts the whole film — see
    # `overlay_audio_bed`'s docstring) — hence the `apad=whole_dur=...` between
    # the soxr resample and the split, which was not there before that fix.
    assert (
        "[0:a]aresample=48000:resampler=soxr:precision=28,"
        "aformat=channel_layouts=stereo:sample_rates=48000,"
        "apad=whole_dur=10.0,asplit=2" in cmd
    )


# --------------------------------------------------------------------------
# 6. True-peak ceiling calibration
# --------------------------------------------------------------------------


def test_limiter_ceiling_lands_at_or_below_the_configured_true_peak() -> None:
    """``alimiter``'s ``limit`` is a SAMPLE-peak bound; the mastering gate measures
    TRUE peak, which is always >= it. Aiming the sample ceiling AT ``master_tp``
    delivered -0.93/-0.87 dBTP against a -1.0 gate — every master failing a check
    it missed by 0.07 dB. The ceiling must now sit far enough below ``master_tp``
    to absorb the inter-sample overshoot."""
    settings = get_settings()
    chain = renderer._master_filter(settings)
    limit = float(chain.split("alimiter=limit=")[1].split(":")[0])
    ceiling_db = 20.0 * math.log10(limit)
    # Below the configured ceiling, by enough to cover the measured 0.07-0.13 dB.
    assert ceiling_db <= settings.master_tp - 0.25
    # ...and not absurdly below it: this is a calibration offset, not an attenuator.
    assert ceiling_db >= settings.master_tp - 1.0


def test_limiter_ceiling_tracks_a_custom_true_peak() -> None:
    """The offset is applied to whatever ``master_tp`` is configured, so a per-job
    ceiling is still the value actually enforced."""
    settings = get_settings().model_copy(update={"master_tp": -2.0})
    chain = renderer._master_filter(settings)
    limit = float(chain.split("alimiter=limit=")[1].split(":")[0])
    assert 20.0 * math.log10(limit) <= -2.25


# --------------------------------------------------------------------------
# 4. The audiobook master's non-integer resample
# --------------------------------------------------------------------------


async def test_acx_master_resamples_to_44_1k_with_soxr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``loudnorm`` emits 192 kHz regardless of input rate, so this path was doing
    192000 -> 44100 (a 147:640 ratio) on swr's default cubic interpolator, on the
    one deliverable that is nothing but voice. The FILM master guards the same
    gotcha with an ``aformat=sample_rates=48000`` node; this path had none.

    ⚠️ The resampler rides as an OUTPUT OPTION, not as an ``aresample`` node in
    ``-af``. That was the first attempt and it fails the render on real ffmpeg:
    an explicit ``aresample`` downstream of ``loudnorm`` cannot negotiate a
    channel layout with the wav muxer ("Cannot select channel layout ... ->
    Conversion failed!"), on mono AND stereo. The proof that this form works and
    keeps a mono stitch mono is ``test_acx_master_lands_in_band_and_neg_control``
    in ``test_audiobook_acx.py``, which runs real ffmpeg over this exact command."""
    captured = _install(monkeypatch)
    raw = tmp_path / "book.wav"
    raw.write_bytes(b"\x00")
    await renderer._acx_program_master(
        raw, tmp_path / "book_master.wav", "wav", lufs_target=-19.0, tp=-3.0, lra=12.0
    )
    encode = captured["encode_cmds"][-1]
    assert encode[encode.index("-ar") + 1] == "44100"
    assert encode[encode.index("-resampler") + 1] == "soxr"
    assert encode[encode.index("-precision") + 1] == "28"
    # 4e: the same swr instance does loudnorm's 192 kHz FLOAT -> pcm_s16le for
    # "wav" output; swr's default dither_method is "none", which is plain
    # truncation on the one deliverable that is nothing but voice.
    assert encode[encode.index("-dither_method") + 1] == "triangular_hp"
    # The resampler options must come BEFORE the output path, i.e. be output
    # options of this file and not stray input options of the next command.
    assert encode.index("-resampler") < len(encode) - 1
    assert encode.index("-dither_method") < len(encode) - 1


async def test_acx_master_dither_measurably_breaks_up_quantization(tmp_path: Path) -> None:
    """4e end-to-end proof, real ffmpeg (NOT the module's `_run` mock): a
    -90 dBFS 1 kHz tone at 192 kHz float, run through this exact function's swr
    args, must NOT show the long runs of identical quantized samples (astats
    "Flat factor") that mark undithered truncation. Bench (wave 8): undithered
    Flat factor 23.5, dithered 0.016 — reverting the `-dither_method
    triangular_hp` argument reproduces the undithered number and this test goes
    red."""
    raw = tmp_path / "quiet.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.00003162*sin(2*PI*1000*t):s=192000:d=2",
            "-c:a",
            "pcm_f32le",
            str(raw),
        ]
    )
    out = tmp_path / "quiet_master.wav"
    # loudnorm's `I` is bounded to [-70, -5]; -70 (its floor) still leaves the
    # encode chain's own gain low enough that the quantization step (not the
    # loudness target) is what's under test.
    await renderer._acx_program_master(raw, out, "wav", lufs_target=-70.0, tp=-3.0, lra=12.0)
    stderr = await renderer._run(
        ["ffmpeg", "-i", str(out), "-af", "astats", "-f", "null", "-"], capture_stderr=True
    )
    flat = next(ln for ln in stderr.splitlines() if "Flat factor" in ln)
    flat_factor = float(flat.split(":")[1].strip())
    assert flat_factor < 5.0, f"undithered quantization staircase detected: {flat}"


# --------------------------------------------------------------------------
# 7. Fleet report (2026-09-14): negative av_gap ("picture outruns narration",
#    e.g. orchestrator wave-7 hold_seconds) correlated 4/4 with true-peak
#    clipping (+0.44..+5.42 dBTP) against a clean 10/10 on non-negative gaps.
#    INVESTIGATED, NOT REPRODUCED — see the docstring below for the full bench.
# --------------------------------------------------------------------------


async def test_master_audio_survives_video_longer_than_audio_without_clipping(
    tmp_path: Path,
) -> None:
    """Real ffmpeg, REAL `master_audio` (not mocked): reproduces the fleet's
    exact av_gap shape (video ~3.4s longer than audio -- a concat whose audio
    track is genuinely shorter, muxed with no `-shortest`) and asserts the
    delivered true peak stays under the ceiling.

    RESULT (wave 8): this does NOT reproduce the fleet clipping. Two real
    ffmpeg benches -- this one (`master_audio`, bed-less) and an equivalent run
    through `overlay_audio_bed` with a music bed -- both landed at -12.4 dBTP
    (measured two ways: `ebur128=peak=true` on the delivered file, AND the
    exact `loudnorm print_format=json` single-pass method
    `final_mux_loudness`/the fleet's own `true_peak_dbtp` uses), against a
    -1.0 dBTP ceiling. Reading the code: pass 1 (measurement) and pass 2
    (encode) are built from the IDENTICAL `parts` filtergraph list up to the
    final tail filter -- there is no window where pass 1 measures a different
    span of audio than pass 2 processes, so a duration mismatch between the
    video and audio streams cannot desynchronize what loudnorm measures from
    what the limiter bounds. `-t total_seconds` (the bed path) and the
    "no -t at all" bed-less path both behave as a MAXIMUM bound, never a pad:
    a naturally-shorter audio stream just ends early; the video stream (`-c:v
    copy`) keeps its own length, and the two tracks land in the container at
    their own natural lengths -- reproducing av_gap exactly, but the audio
    CONTENT itself is limited correctly regardless of how long the video track
    next to it is.

    Conclusion: the negative-av_gap / clipping correlation is real (10/10 vs
    0/4 in the fleet data) but is NOT explained by a measurement-window
    mismatch in this repo's master chain. The likely site is upstream
    (orchestrator's hold_seconds construction — e.g. whatever audio, if any,
    plays during the held tail, or a genuinely loud narration/SFX passage on
    those 4 specific runs) or in real narration dynamics this synthetic tone
    bench cannot exercise. This test exists as a regression guard for the
    mechanism that WAS checked here, not as a fix for the fleet clipping."""
    video = tmp_path / "video_only.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=640x360:d=32.276693:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ]
    )
    narration = tmp_path / "narration.wav"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=28.875:sample_rate=48000",
            "-af",
            "volume=-18dB",
            str(narration),
        ]
    )
    concat = tmp_path / "concat_mismatched.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(narration),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(concat),
        ]
    )
    mastered = tmp_path / "mastered.mp4"
    await renderer.master_audio(concat, mastered, get_settings())

    video_dur = float(
        (
            await renderer._run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "default=nk=1:nw=1",
                    str(mastered),
                ]
            )
        ).strip()
    )
    audio_dur = float(
        (
            await renderer._run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=duration",
                    "-of",
                    "default=nk=1:nw=1",
                    str(mastered),
                ]
            )
        ).strip()
    )
    # Confirms the bench actually reproduces the fleet's av_gap shape (video
    # meaningfully longer than audio) before trusting the peak measurement below.
    assert video_dur - audio_dur > 2.0

    stderr = await renderer._run(
        [
            "ffmpeg",
            "-i",
            str(mastered),
            "-af",
            "silencedetect=n=-50dB:d=0.3,loudnorm=I=-14:TP=-1.5:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    from src.services.loudness import parse_loudness

    measured = parse_loudness(stderr, stderr)
    true_peak = measured.get("true_peak_dbtp") if measured else None
    assert true_peak is not None
    assert true_peak < get_settings().master_tp, (
        f"master_audio delivered {true_peak} dBTP against a "
        f"{get_settings().master_tp} dBTP ceiling despite a {video_dur - audio_dur:.1f}s "
        "video/audio length mismatch -- the fleet's clipping mechanism reproduced!"
    )


# --------------------------------------------------------------------------
# 5. WHY linear was refused: `requested_linear` + pass 1's own readings
#
# `master_normalization_type="dynamic"` alone is unreadable. It means one of
# THREE different things, and the production run that first exposed the field
# (`01a0a582-2531-7145-ba7d-a1295e92b232`) gave no way to tell them apart:
#
#   a. pass 1 failed open, so we never asked for linear    (requested_linear False)
#   b. we asked, and loudnorm refused on true-peak headroom:
#        measured_TP + (target_I - measured_I) > target_TP
#   c. we asked, and loudnorm refused because measured_LRA == 0
#
# (b) and (c) are byte-identical on the record unless pass 1's readings are
# carried. These tests pin that they are. The boundary numbers come from
# measuring real ffmpeg in isolation at `I=-14:TP=-1:linear=true`:
# `measured_I=-22, measured_TP=-3` -> Dynamic; `measured_TP=-9` -> Linear.
# --------------------------------------------------------------------------

# Pass 1 on a real compressed SPEECH bus — the production shape. ~-22 LUFS wants
# ~+8 dB of constant gain, and -4.68 + 8 = +3.3 dBTP blows a -1 dBTP ceiling, so
# loudnorm REFUSES linear. Same numbers as `_PASS1_JSON_STDERR` above.
_PASS1_REFUSED_STDERR = _PASS1_JSON_STDERR

# Pass 1 on a bus with enough peak headroom that the SAME +8 dB constant gain
# lands exactly ON the ceiling (-9.50 + 8.49 = -1.01 dBTP <= -1). loudnorm grants
# linear here. Not a realistic speech master — that is the whole point.
_PASS1_GRANTED_STDERR = """\
[Parsed_loudnorm_1 @ 0x5591b4a0e400]
{
\t"input_i" : "-22.49",
\t"input_tp" : "-9.50",
\t"input_lra" : "6.20",
\t"input_thresh" : "-32.78",
\t"output_i" : "-14.00",
\t"output_tp" : "-1.01",
\t"output_lra" : "6.20",
\t"output_thresh" : "-24.28",
\t"normalization_type" : "linear",
\t"target_offset" : "0.00"
}
"""

# The SECOND, independent refusal: a bus with 23 dB of peak headroom (so the
# inequality in (b) is comfortably satisfied) but ZERO measured loudness range.
# Measured: loudnorm returns Dynamic anyway. Without `input_lra` on the record
# this row is indistinguishable from (b).
_PASS1_ZERO_LRA_STDERR = """\
[Parsed_loudnorm_1 @ 0x5591b4a0e400]
{
\t"input_i" : "-22.00",
\t"input_tp" : "-24.00",
\t"input_lra" : "0.00",
\t"input_thresh" : "-32.00",
\t"output_i" : "-14.00",
\t"output_tp" : "-16.00",
\t"output_lra" : "0.00",
\t"output_thresh" : "-24.00",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.00"
}
"""


def _linear_is_refused_on_headroom(evidence: dict[str, float | bool]) -> bool:
    """The refusal inequality, recomputed from the RECORDED evidence against the
    live target pair. If this cannot be evaluated off the report, the report does
    not answer the question the fields exist to answer."""
    settings = get_settings()
    measured_tp = evidence["input_tp"]
    measured_i = evidence["input_i"]
    assert isinstance(measured_tp, float) and isinstance(measured_i, float)
    return measured_tp + (settings.master_lufs - measured_i) > settings.master_tp


async def test_refused_linear_records_the_headroom_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE PRODUCTION ROW. We asked for linear, ffmpeg ran dynamic, and the record
    now carries enough to say WHY: the true-peak inequality is satisfied."""
    _install(monkeypatch, pass1=_PASS1_REFUSED_STDERR, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    assert renderer.get_master_normalization_type() == "dynamic"
    assert renderer.get_master_loudnorm_evidence() == {
        "requested_linear": True,
        "input_i": -22.51,
        "input_tp": -4.68,
        "input_lra": 6.20,
        "target_offset": 0.01,
    }
    # ...and the recorded numbers EXPLAIN the dynamic: -4.68 + (-14 - -22.51) = +3.83
    # dBTP against a -1.0 dBTP ceiling.
    assert _linear_is_refused_on_headroom(renderer.get_master_loudnorm_evidence())


async def test_granted_linear_records_evidence_that_does_not_refuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other side of the boundary. Same target pair, enough peak headroom, so
    the inequality is NOT satisfied and loudnorm honours linear — proving the
    recorded evidence discriminates rather than always reading "refused"."""
    _install(monkeypatch, pass1=_PASS1_GRANTED_STDERR, pass2=_pass2_stderr("Linear"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    assert renderer.get_master_normalization_type() == "linear"
    assert renderer.get_master_loudnorm_evidence() == {
        "requested_linear": True,
        "input_i": -22.49,
        "input_tp": -9.50,
        "input_lra": 6.20,
        "target_offset": 0.00,
    }
    assert not _linear_is_refused_on_headroom(renderer.get_master_loudnorm_evidence())


async def test_zero_lra_refusal_is_distinguishable_from_the_headroom_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refusal (c). 23 dB of peak headroom — the headroom inequality is FALSE — and
    loudnorm still runs dynamic, because measured_LRA is 0. Without `input_lra` on
    the record this row and the one above are the same report."""
    _install(monkeypatch, pass1=_PASS1_ZERO_LRA_STDERR, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    evidence = renderer.get_master_loudnorm_evidence()
    assert renderer.get_master_normalization_type() == "dynamic"
    assert evidence["requested_linear"] is True
    # The headroom explanation does NOT apply here...
    assert not _linear_is_refused_on_headroom(evidence)
    # ...so the ONLY thing on the record that accounts for the dynamic is the LRA.
    assert evidence["input_lra"] == 0.0


async def test_fail_open_records_requested_linear_false_and_no_readings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Case (a). Pass 1 printed no JSON, so we never asked for linear. `"dynamic"`
    here is not a refusal and must not read as one: `requested_linear` is False and
    there are no pass-1 readings to explain anything.

    It IS, however, a DEGRADATION — the measure pass did not complete and the
    fallback chain mastered a paid deliverable — and `requested_linear=False` alone
    cannot say so: a mastering-off render reports the same thing. So the run also
    carries `measure_failed` plus WHICH of the five bail paths fired."""
    captured = _install(monkeypatch, pass1="", pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    # We really did not ask: no `linear=true` in the pass-2 filter at all.
    assert "linear=true" not in " ".join(captured["encode_cmds"][-1])
    assert renderer.get_master_normalization_type() == "dynamic"
    assert renderer.get_master_loudnorm_evidence() == {
        "requested_linear": False,
        "measure_failed": True,
    }
    # The REASON is test-only state + an ERROR log; only the boolean is published.
    assert renderer._master_measure_failed.get() == "incomplete"


async def test_a_completed_measurement_reports_no_degradation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other side of the flag, which is what stops it being a signal that fires
    on everything: a render whose pass 1 measured cleanly must contribute NEITHER
    `measure_failed` nor a reason — even though loudnorm then REFUSED linear, which
    is expected on this material and is not a degradation."""
    _install(monkeypatch, pass1=_PASS1_REFUSED_STDERR, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())

    assert renderer.get_master_normalization_type() == "dynamic"
    assert renderer._master_measure_failed.get() is None
    assert "measure_failed" not in renderer.get_master_loudnorm_evidence()


async def test_bed_path_also_records_the_refusal_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A film WITH music folds the same master into its own re-encode, so it must
    not be the blind spot for the evidence either."""
    _install(monkeypatch, pass1=_PASS1_REFUSED_STDERR, pass2=_pass2_stderr("Dynamic"))
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m")),
        ambient=[],
        settings=get_settings(),
    )
    evidence = renderer.get_master_loudnorm_evidence()
    assert evidence["requested_linear"] is True
    assert evidence["input_tp"] == -4.68


def test_requested_linear_is_recorded_even_when_the_summary_is_absent() -> None:
    """ "We asked and could not observe the outcome" is a REPORTABLE state, and a
    different one from "we never asked". Recorded before the summary match."""
    renderer.reset_master_normalization_type()
    assert renderer._record_master_normalization_type("", requested_linear=True) is None
    assert renderer.get_master_normalization_type() is None
    assert renderer.get_master_loudnorm_evidence() == {"requested_linear": True}


def test_no_evidence_at_all_when_mastering_is_off() -> None:
    """A ``mastering: off`` job has no loudnorm, so it must contribute NOTHING —
    the report keeps exactly the shape it has today."""
    renderer.reset_master_normalization_type()
    renderer._record_master_normalization_type(
        _pass2_stderr("Dynamic"), requested_linear=False, master_enabled=False
    )
    assert renderer.get_master_loudnorm_evidence() == {}


def test_reset_clears_the_evidence_not_just_the_mode() -> None:
    """A stale ``input_tp`` would be read as the reason THIS render was refused
    linear — worse than reporting nothing. Every channel resets together, the
    degradation flag included: a stale `measure_failed` would blame THIS render for
    a previous one's failed measurement."""
    renderer._record_master_measure_failure("unmeasurable", key="input_i")
    renderer._record_master_loudnorm_evidence({"input_i": "-22.51", "input_tp": "-4.68"})
    renderer._record_master_normalization_type(_pass2_stderr("Dynamic"), requested_linear=True)
    assert renderer.get_master_loudnorm_evidence() != {}
    renderer.reset_master_normalization_type()
    assert renderer.get_master_loudnorm_evidence() == {}
    assert renderer.get_master_normalization_type() is None
    assert renderer._master_measure_failed.get() is None


def test_non_finite_pass1_readings_are_never_recorded() -> None:
    """A digitally silent bus measures ``-inf``. That is a NON-EMPTY string that
    slips a truthiness guard and is not JSON-serialisable, so it must not reach the
    response model. The finite ``input_lra`` on the same block still does."""
    from src.services.loudness import _extract_json_block

    data = _extract_json_block(_PASS1_SILENT_STDERR) or {}
    recorded = renderer._record_master_loudnorm_evidence(data)
    assert "input_i" not in recorded
    assert "input_tp" not in recorded
    assert "target_offset" not in recorded  # "inf"
    assert recorded == {"input_lra": 0.0}
    assert all(math.isfinite(v) for v in recorded.values())
