"""An audiobook clip with NO audio stream must fail the uniformity guard.

``_assert_uniform_audio`` is the only thing standing between a heterogeneous clip
list and the stream-copy concat demuxer (which would silently corrupt the join) or
the concat filter (which sizes its inter-chapter silence to clip 0 only).

``_probe_audio_params`` folded "ffprobe reported no audio stream at all" into the
SAME ``(24000, 1)`` value it returns for real Kokoro narration: ffprobe exits 0 with
``{"streams": []}``, so ``(json.loads(out).get("streams") or [{}])[0]`` is ``{}`` and
``int(None or 0) or 24000`` manufactures the exact value the gate wants to see.
Absence of evidence became a perfect match, the guard passed, and the audiobook
shipped ``completed`` with a corrupt join / misaligned gap silence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services import renderer
from src.services.renderer import FfmpegError

pytestmark = pytest.mark.asyncio

_REAL_NARRATION = '{"streams": [{"sample_rate": "24000", "channels": 1}]}'
# ffprobe -select_streams a:0 on a video-only / silent-container file: exit 0, no stream.
_NO_AUDIO_STREAM = '{"streams": []}'


def _stub_probe(monkeypatch: pytest.MonkeyPatch, per_path: dict[str, str]) -> None:
    async def fake_run(cmd: list[str]) -> str:
        return per_path[Path(cmd[-1]).name]

    monkeypatch.setattr(renderer, "_run", fake_run)


async def test_audioless_clip_is_not_reported_as_24k_mono(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Precondition + core: an empty stream list must be distinguishable from real audio."""
    _stub_probe(monkeypatch, {"a.wav": _REAL_NARRATION, "b.wav": _NO_AUDIO_STREAM})
    real = await renderer._probe_audio_params(tmp_path / "a.wav")
    # Precondition: the real-narration probe must genuinely resolve to 24k mono,
    # otherwise the comparison below is vacuous.
    assert real == (24000, 1)
    silent = await renderer._probe_audio_params(tmp_path / "b.wav")
    assert silent != (24000, 1), (
        "a clip with NO audio stream probes identically to real 24kHz mono narration"
    )


async def test_uniformity_guard_rejects_an_audioless_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_probe(monkeypatch, {"a.wav": _REAL_NARRATION, "b.wav": _NO_AUDIO_STREAM})
    with pytest.raises(FfmpegError):
        await renderer._assert_uniform_audio([tmp_path / "a.wav", tmp_path / "b.wav"])


async def test_uniformity_guard_still_passes_real_uniform_clips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No-op for uniform input (the prod case) — the guard must not become trigger-happy."""
    _stub_probe(monkeypatch, {"a.wav": _REAL_NARRATION, "b.wav": _REAL_NARRATION})
    await renderer._assert_uniform_audio([tmp_path / "a.wav", tmp_path / "b.wav"])


# The SIBLING gap: ffprobe answers, but the answer is unreadable. `int("N/A")` raises,
# or the key is absent, and the old code fabricated the nominal Kokoro shape for both.
_UNREADABLE_RATE = '{"streams": [{"sample_rate": "N/A", "channels": 2}]}'
_MISSING_RATE = '{"streams": [{"channels": 1}]}'
_TRUNCATED_JSON = '{"streams": [{"sample_rate": "480'


@pytest.mark.parametrize(
    "probe_output",
    [_UNREADABLE_RATE, _MISSING_RATE, _TRUNCATED_JSON],
    ids=["unreadable_field", "missing_field", "truncated_json"],
)
async def test_an_unreadable_probe_is_not_24k_mono(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probe_output: str
) -> None:
    """An unmeasured clip must not be indistinguishable from real narration.

    `int(x or 0) or 24000` produced exactly (24000, 1) for every one of these, so a
    48 kHz stereo segment whose probe was unreadable matched clip 0 and was
    stream-copied into a 24 kHz master.
    """
    _stub_probe(monkeypatch, {"x.wav": probe_output})
    with pytest.raises(FfmpegError, match="unreadable"):
        await renderer._probe_audio_params(tmp_path / "x.wav")


async def test_a_real_probe_still_reads_its_own_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: the guard must reject only what it genuinely cannot read.

    Without this, raising on everything would pass the test above while breaking
    every real compile.
    """
    _stub_probe(monkeypatch, {"hi.wav": '{"streams": [{"sample_rate": "48000", "channels": 2}]}'})
    assert await renderer._probe_audio_params(tmp_path / "hi.wav") == (48000, 2)


async def test_the_uniformity_guard_fails_rather_than_matching_a_fabricated_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end: an unreadable clip must stop the compile, not join it.

    Clip 1 is genuinely 48 kHz stereo but probes unreadably. Under the old defaults
    it reported (24000, 1), matched clip 0 exactly, and was concat-copied.
    """
    _stub_probe(monkeypatch, {"a.wav": _REAL_NARRATION, "b.wav": _UNREADABLE_RATE})
    with pytest.raises(FfmpegError, match="unreadable"):
        await renderer._assert_uniform_audio([tmp_path / "a.wav", tmp_path / "b.wav"])
