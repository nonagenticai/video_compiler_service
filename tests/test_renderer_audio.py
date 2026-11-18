"""Unit tests for the audiobook audio-concat path.

ffmpeg/ffprobe + asset fetch are mocked: these assert the command shape per
output_format / gap combination and that the duration is re-probed (never
trusted from the per-clip ``seconds``). Real muxing is proven by the live E2E.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import AudioClip
from src.config import get_settings
from src.services import renderer

pytestmark = pytest.mark.asyncio


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, out_dur: float = 9.0) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str]) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        if cmd[0] == "ffprobe":
            # _probe_audio_params parses this JSON; _ffprobe_duration calls a
            # different fake (we patch it directly below) so this is only the
            # stream-params probe.
            return '{"streams": [{"sample_rate": "24000", "channels": 1}]}'
        return ""

    async def fake_probe_duration(path: Path) -> float:
        return out_dur

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"audio")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe_duration)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


async def test_wav_no_gap_uses_concat_demuxer_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fakes(monkeypatch, out_dur=12.0)
    clips = [
        AudioClip(id="c0", audio_url="http://m/0.wav", seconds=6.0),
        AudioClip(id="c1", audio_url="http://m/1.wav", seconds=6.0),
    ]
    out_path, total, acx_report = await renderer.concat_audio(clips, get_settings(), "wav", 0)
    cmd = captured["cmds"][-1]
    joined = " ".join(cmd)
    assert "-f concat" in joined  # demuxer
    assert "-c copy" in joined  # stream copy, no re-encode
    assert "-filter_complex" not in joined
    assert out_path.suffix == ".wav"
    assert total == 12.0  # re-probed, NOT 6.0+6.0 from per-clip seconds
    assert acx_report is None  # no acx_master requested → no report


async def test_mp3_uses_filter_and_lame_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fakes(monkeypatch)
    clips = [
        AudioClip(id="c0", audio_url="http://m/0.wav"),
        AudioClip(id="c1", audio_url="http://m/1.wav"),
    ]
    out_path, _, _ = await renderer.concat_audio(clips, get_settings(), "mp3", 0)
    joined = " ".join(captured["cmds"][-1])
    assert "-filter_complex" in joined
    assert "concat=n=2:v=0:a=1" in joined  # two real inputs, no silence
    assert "libmp3lame" in joined
    assert out_path.suffix == ".mp3"


async def test_gap_inserts_silence_between_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fakes(monkeypatch)
    clips = [
        AudioClip(id="c0", audio_url="http://m/0.wav"),
        AudioClip(id="c1", audio_url="http://m/1.wav"),
        AudioClip(id="c2", audio_url="http://m/2.wav"),
    ]
    await renderer.concat_audio(clips, get_settings(), "wav", 500)
    joined = " ".join(captured["cmds"][-1])
    # 3 clips + 2 inter-clip silences → concat n=5; silence sized to 0.5s and
    # matched to the probed 24 kHz mono layout.
    assert "concat=n=5:v=0:a=1" in joined
    assert "anullsrc=r=24000:cl=mono" in joined
    assert "-t 0.5" in joined


async def test_rejects_unknown_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)
    clips = [AudioClip(id="c0", audio_url="http://m/0.wav")]
    with pytest.raises(renderer.FfmpegError):
        await renderer.concat_audio(clips, get_settings(), "flac", 0)


async def test_audio_concat_path_carries_no_cadence_retime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VC-3 touches only the picture filter chain — the audiobook concat path must
    stay byte-identical (no fps=/minterpolate/tblend leaking into audio)."""
    captured = _install_fakes(monkeypatch, out_dur=12.0)
    clips = [
        AudioClip(id="c0", audio_url="http://m/0.wav"),
        AudioClip(id="c1", audio_url="http://m/1.wav"),
    ]
    await renderer.concat_audio(clips, get_settings(), "wav", 0)
    joined = " ".join(captured["cmds"][-1])
    for token in ("fps=", "minterpolate", "tblend", "framestep"):
        assert token not in joined
