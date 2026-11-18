"""Unit tests for the sound-effects + music/ambient bed mixing.

ffmpeg/ffprobe + asset fetch are mocked: these assert the filtergraph shape
(per-shot ``amix`` of narration + SFX, and the post-concat sidechain-ducked
bed pass) and the best-effort degradation when a cue asset is unfetchable.
Real muxing is proven by the live E2E.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import AccentCue, AmbientCue, MusicBed, Shot, ShotAudio, SoundCue
from src.config import get_settings
from src.services import renderer
from tests.conftest import is_delivered_master_measure


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


async def _noop_uniform_audio(segment_paths: object) -> None:
    return None


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_dur: float = 3.0,
    unfetchable: set[str] | None = None,
    has_audio: bool = True,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}
    bad = unfetchable or set()

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        return audio_dur

    async def fake_has_audio(path: Path) -> bool:
        # A compiled concat carries an audio stream (narration / synthesised
        # silence), so the bed-less master runs. ``has_audio=False`` exercises the
        # no-audio-skip guard in ``master_audio``.
        return has_audio

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url in bad:
            raise httpx.HTTPError(f"boom: {url}")
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    # The concat audio-uniformity probe reads real ffprobe JSON; these fakes write
    # placeholder files, so it is stubbed like the other probes (proven for real in
    # tests/render_harness/test_mixed_depth_concat_render.py).
    monkeypatch.setattr(renderer, "_assert_uniform_segment_audio", _noop_uniform_audio)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


# --------------------------------------------------------------------------
# _audio_sfx_graph (pure)
# --------------------------------------------------------------------------


def test_audio_sfx_graph_transition_and_accents() -> None:
    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=(2, SoundCue(url="x", gain_db=-3.0)),
        accents=[(3, AccentCue(url="y", at_seconds=1.5, gain_db=-6.0))],
        settings=get_settings(),
    )
    # Primary normalized + split into the mix; transition at t=0; accent delayed.
    # The PRIMARY carries soxr too, as of 2026-09-13. This line used to assert the
    # bare `_A48` on the strength of a comment claiming the narration bus "already
    # arrives 48 kHz stereo" — it does not; tts_backend renders 24 kHz mono, so
    # this is a real conversion. See tests/test_renderer_narration_resampler.py.
    assert (
        "[1:a]aresample=48000:resampler=soxr:precision=28,"
        "aformat=channel_layouts=stereo:sample_rates=48000[a_pri]" in graph
    )
    # SFX are THIRD-PARTY assets at an unknown rate (library cues are almost
    # always 44.1 kHz), so they carry the explicit soxr resampler rather than the
    # bare `_A48`, whose rate change would fall to swr's aliasing cubic default.
    assert (
        "[2:a]aresample=48000:resampler=soxr:precision=28,"
        "aformat=channel_layouts=stereo:sample_rates=48000,volume=-3.0dB[a_tr]" in graph
    )
    assert "adelay=1500|1500,volume=-6.0dB[a_ac0]" in graph
    # 3 inputs, normalize off so narration isn't divided down, anchored to primary.
    # The limit is DERIVED from settings.master_tp (4a), not the old hardcoded
    # 0.95 — same formula as `_master_filter`'s bus limiter, via
    # `_limiter_linear_limit`. Default master_tp=-1.0 -> 0.860994.
    assert (
        "amix=inputs=3:duration=first:normalize=0,alimiter=limit=0.860994:level=disabled[aout]"
        in graph
    )


def test_audio_sfx_graph_accent_only() -> None:
    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=None,
        accents=[(2, AccentCue(url="y", at_seconds=0.0))],
        settings=get_settings(),
    )
    assert "adelay=0|0" in graph
    assert (
        "amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.860994:level=disabled[aout]"
        in graph
    )


def test_audio_sfx_graph_limit_derived_from_master_tp() -> None:
    """4a: the per-shot limiter ceiling must track `settings.master_tp`, not a
    hardcoded literal — this is the whole point of the fix. Mutate `master_tp`
    and confirm the emitted `limit=` moves with it; a revert to the hardcoded
    0.95 makes this FAIL because the derived value stops matching the graph."""
    settings = get_settings().model_copy(update={"master_tp": -3.0})
    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=None,
        accents=[],
        settings=settings,
    )
    expected = renderer._ffnum(renderer._limiter_linear_limit(settings))
    assert f"alimiter=limit={expected}:level=disabled[aout]" in graph
    assert expected != "0.95"


# --------------------------------------------------------------------------
# _master_filter + _sidechain_ratio (pure)
# --------------------------------------------------------------------------


def test_master_filter_enabled_has_full_chain() -> None:
    chain = renderer._master_filter(get_settings())
    assert "acompressor=" in chain
    assert "alimiter=" in chain
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11.0" in chain


def test_master_filter_disabled_is_empty() -> None:
    settings = get_settings().model_copy(update={"master_enabled": False})
    assert renderer._master_filter(settings) == ""


def test_master_filter_honours_custom_target() -> None:
    settings = get_settings().model_copy(update={"master_lufs": -11.0, "master_tp": -1.5})
    assert "loudnorm=I=-11.0:TP=-1.5:LRA=11.0" in renderer._master_filter(settings)


def test_sidechain_ratio_maps_duck_db_monotonically() -> None:
    # Schema default duck_db=-12 anchors the ratio at 4. It was 9 until the
    # pumping fix: a 9:1 slam recovering inside every inter-word gap is the
    # "breathing" the owner reported as a digital artifact.
    assert renderer._sidechain_ratio(-12.0) == 4
    # Deeper duck → higher ratio; shallower → lower; clamped to [2, 20].
    assert renderer._sidechain_ratio(-17.0) == 6
    assert renderer._sidechain_ratio(-8.0) == 3
    assert renderer._sidechain_ratio(-90.0) == 20  # clamp high
    # duck_db>=0 ("no extra attenuation") → the LIGHT floor ratio 2, NOT the heavy 9.
    # (Previously short-circuited to 9 — a bug: a bed the caller asked to barely duck
    # got slammed. The formula already yields 2 here.)
    assert renderer._sidechain_ratio(0.0) == 2
    assert renderer._sidechain_ratio(3.0) == 2


# --------------------------------------------------------------------------
# render_shot per-shot mixing
# --------------------------------------------------------------------------


async def test_render_shot_accents_build_amix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch, audio_dur=4.0)
    shot = Shot(
        id="s1",
        seconds=2.0,
        image_url="http://minio/slide.png",
        audio_url="http://minio/clip.wav",
        sfx=[AccentCue(url="http://minio/ding.wav", at_seconds=1.0, gain_db=-8.0)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    # Video folds into the same -filter_complex as the audio mix; no bare -vf.
    assert "-filter_complex" in cmd
    assert "[aout]" in cmd
    assert "amix=inputs=2" in cmd  # narration + 1 accent
    assert "adelay=1000|1000" in cmd


async def test_render_shot_transition_on_text_card_uses_filter_complex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(
        id="s1",
        seconds=3.0,
        description="A text card",
        transition_sfx=SoundCue(url="http://minio/whoosh.wav"),
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    joined = " ".join(cmd)
    # The text branch normally uses -vf; with SFX it must move to -filter_complex.
    assert "-vf" not in cmd
    assert "-filter_complex" in cmd
    assert "drawtext" in joined
    assert "[aout]" in joined


async def test_render_shot_no_sfx_keeps_direct_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s1", seconds=2.0, image_url="http://minio/slide.png")
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    assert "amix" not in cmd  # no SFX → single -map, no mixing


async def test_render_shot_unfetchable_sfx_degrades(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch, unfetchable={"http://minio/bad.wav"})
    shot = Shot(
        id="s1",
        seconds=2.0,
        image_url="http://minio/slide.png",
        audio_url="http://minio/clip.wav",
        sfx=[AccentCue(url="http://minio/bad.wav", at_seconds=1.0)],
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = " ".join(captured["cmds"][-1])
    # The single bad accent dropped → falls back to the plain single-map path.
    assert "amix" not in cmd


# --------------------------------------------------------------------------
# overlay_audio_bed
# --------------------------------------------------------------------------


async def test_overlay_audio_bed_music_and_ambient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=30.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", gain_db=-18.0, fade_out_s=2.5)),
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a", gain_db=-24.0))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    # Narration is split for the audible mix + the sidechain key.
    assert "asplit=2[nar_mix][nar_key]" in cmd
    # Music faded; bed ducked under narration; video stream-copied (audio-only).
    assert "afade=t=out:st=27.500:d=2.5" in cmd
    # The sidechain KEY is high-passed before it drives the compressor.
    assert "[nar_key]highpass=f=150[nar_hpf]" in cmd
    assert "[nar_hpf]sidechaincompress=" in cmd
    assert "-c:v copy" in cmd
    # Beds are looped to fill the timeline by CROSSFADING repetitions, not by
    # `-stream_loop`, whose butt-join clicked at every seam. The faked probe
    # reports a 3 s asset; both the music bed and the ambient cue span the 30 s
    # film ⇒ 11 reads each, 10 crossfaded seams each.
    assert "-stream_loop" not in cmd
    # 10 ramped seams each (a fade-out and a fade-in per seam), summed by amix —
    # never acrossfade, which truncates/drops audio depending on thread timing.
    assert "acrossfade" not in cmd
    # Bed-loop seams ramp with `curve=qsin` (equal POWER) — the two sides of a
    # loop seam are the SAME asset at different offsets, so equal GAIN (`tri`)
    # sums to -3 dB at the seam centre. Measured -2.77 dB (tri) vs -0.29 dB (qsin).
    assert cmd.count("afade=t=in:curve=qsin:st=0:d=0.250000") == 20
    assert "afade=t=in:curve=tri:st=0:d=0.250000" not in cmd
    assert cmd.count("amix=inputs=11:duration=longest:normalize=0") == 2


async def test_overlay_audio_bed_music_only_single_bed_no_amix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fakes(monkeypatch)
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
    cmd = " ".join(captured["cmds"][-1])
    # One bed element → no intermediate amix of beds, straight to the sidechain
    # (fed by the high-passed narration key).
    assert "[bed_music][nar_hpf]sidechaincompress" in cmd
    assert "inputs=2:duration=first:normalize=0[bed]" not in cmd


async def test_overlay_audio_bed_folds_master_into_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fakes(monkeypatch)
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
    cmd = " ".join(captured["cmds"][-1])
    # Master (default-ON) folds into the same audio re-encode via [premaster]→[aout].
    assert "[premaster]" in cmd
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11.0" in cmd
    # Only one ffmpeg pass for the bed video (no separate master pass).
    assert cmd.count("loudnorm=") == 1


async def test_overlay_audio_bed_no_master_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    settings = get_settings().model_copy(update={"master_enabled": False})
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m")),
        ambient=[],
        settings=settings,
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "loudnorm=" not in cmd
    assert "[premaster]" not in cmd
    # amix goes straight into the limiter and out — no master chain, but the
    # bed+narration sum is still bounded (`normalize=0` means they SUM).
    assert (
        "amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit={renderer._ffnum(renderer._limiter_linear_limit(get_settings()))}"
        ":level=disabled[aout]" in cmd
    )
    assert "acompressor=" not in cmd


async def test_overlay_audio_bed_deeper_duck_raises_ratio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", duck_db=-17.0)),
        ambient=[],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    # duck_db=-17 → sidechain ratio 6 (deeper than the default 4).
    assert "sidechaincompress=threshold=0.03:ratio=6:" in cmd


async def test_overlay_audio_bed_duck_defaults_are_the_breathing_duck(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller sending NO duck knobs gets the fixed duck: a key threshold low
    enough that ordinary narration opens the compressor (0.03 ≈ -30 dBFS, was a
    never-crossed 0.125 ≈ -18 dBFS) and a release short enough that the bed swells
    back inside a VO gap (300 ms, was 1000 ms)."""
    captured = _install_fakes(monkeypatch)
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
    cmd = " ".join(captured["cmds"][-1])
    assert "[nar_key]highpass=f=150[nar_hpf]" in cmd
    assert (
        "[bed_music][nar_hpf]sidechaincompress="
        "threshold=0.03:ratio=4:attack=30:release=700[ducked]" in cmd
    )
    # The old, broken values are gone for good.
    assert "threshold=0.125" not in cmd
    assert "release=1000" not in cmd
    assert "attack=200" not in cmd


async def test_overlay_audio_bed_duck_knob_overrides_reach_the_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit MusicBed duck knobs land verbatim in the sidechain filter (and
    floats are emitted as plain ffmpeg numbers, not ``30.0``)."""
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(
            tmp_path / "music.mp3",
            MusicBed(
                url="m",
                duck_attack_ms=50.0,
                duck_release_ms=800.0,
                duck_threshold=0.05,
                sidechain_hpf_hz=200.0,
            ),
        ),
        ambient=[],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "[nar_key]highpass=f=200[nar_hpf]" in cmd
    assert (
        "[bed_music][nar_hpf]sidechaincompress="
        "threshold=0.05:ratio=4:attack=50:release=800[ducked]" in cmd
    )


async def test_overlay_audio_bed_ambient_only_duck_breathes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ambient-only bed (no music) now breathes with the SAME duck shape as the
    music branch, driven by the ambient cue's own knobs. The dead legacy filter
    (threshold=0.125:ratio=9:attack=200:release=1000, no key high-pass) is gone:
    the key is HPF'd into the distinct [nar_hpf_amb] label and the shipped
    defaults are threshold 0.03 / attack 30 / release 300."""
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a"))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "[nar_key]highpass=f=150[nar_hpf_amb]" in cmd
    assert (
        "[bed_amb0][nar_hpf_amb]sidechaincompress="
        "threshold=0.03:ratio=4:attack=30:release=700[ducked]" in cmd
    )
    # The old, dead ambient values are gone for good.
    assert "threshold=0.125" not in cmd
    assert "release=1000" not in cmd
    assert "attack=200" not in cmd


async def test_overlay_audio_bed_ambient_duck_knob_overrides_reach_the_filter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit AmbientCue duck knobs land verbatim in the sidechain filter, and
    the FIRST cue's knobs drive the duck over the summed bed (the second cue's are
    ignored — the compressor is one global node)."""
    captured = _install_fakes(monkeypatch)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[
            (
                tmp_path / "amb0.mp3",
                AmbientCue(
                    url="a0",
                    duck_attack_ms=50.0,
                    duck_release_ms=800.0,
                    duck_threshold=0.05,
                    sidechain_hpf_hz=200.0,
                ),
            ),
            (tmp_path / "amb1.mp3", AmbientCue(url="a1", duck_threshold=0.5)),
        ],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "[nar_key]highpass=f=200[nar_hpf_amb]" in cmd
    assert (
        "[bed][nar_hpf_amb]sidechaincompress="
        "threshold=0.05:ratio=4:attack=50:release=800[ducked]" in cmd
    )
    # Second cue's threshold=0.5 must NOT appear — only the first cue drives the duck.
    assert "threshold=0.5:" not in cmd


# --------------------------------------------------------------------------
# compile_shots bed integration
# --------------------------------------------------------------------------


async def test_compile_shots_runs_bed_pass_when_music_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fakes(monkeypatch)
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    out, _, _, _, _ = await renderer.compile_shots(
        shots, get_settings(), music=MusicBed(url="http://m/music.mp3")
    )
    assert out.name == "final.mp4"  # bed pass produced the final file
    assert any("sidechaincompress=" in " ".join(c) for c in captured["cmds"])


async def test_compile_shots_skips_bed_when_assets_unfetchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_fakes(monkeypatch, unfetchable={"http://m/music.mp3"})
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    out, _, _, _, _ = await renderer.compile_shots(
        shots, get_settings(), music=MusicBed(url="http://m/music.mp3")
    )
    # Unfetchable music → no sidechain bed pass; the master (default-ON) still
    # runs its own audio-only pass, so the output is the mastered file.
    assert out.name == "mastered.mp4"
    assert not any("sidechaincompress=" in " ".join(c) for c in captured["cmds"])
    assert any("loudnorm=" in " ".join(c) for c in captured["cmds"])


async def test_compile_shots_reports_an_unfetchable_music_bed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost music bed must not read as a clean success. On a music-driven product
    the bed IS the soundtrack, so dropping it silently ships a mute film; the job
    result's ``music_bed`` field carries the requested/applied/fetch_failed flags."""
    _install_fakes(monkeypatch, unfetchable={"http://m/music.mp3"})
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    _, _, music_bed_report, _, _ = await renderer.compile_shots(
        shots, get_settings(), music=MusicBed(url="http://m/music.mp3")
    )
    assert music_bed_report is not None, "an unfetchable bed must not report None"
    assert music_bed_report["requested"] is True
    assert music_bed_report["applied"] is False
    assert music_bed_report["fetch_failed"] is True


async def test_compile_shots_reports_no_music_bed_when_none_was_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard on the flag above: a bed-less compile still reports ``None``, so
    the marker unambiguously means "asked for and lost"."""
    _install_fakes(monkeypatch)
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    _, _, music_bed_report, _, _ = await renderer.compile_shots(shots, get_settings())
    assert music_bed_report is None


async def test_compile_shots_masters_bedless_video(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fakes(monkeypatch)
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    out, _, _, _, _ = await renderer.compile_shots(shots, get_settings())
    # No bed → dedicated master pass (default-ON) → mastered.mp4, video copied.
    assert out.name == "mastered.mp4"
    master_cmd = " ".join(captured["cmds"][-1])
    assert "loudnorm=I=-14.0" in master_cmd
    assert "acompressor=" in master_cmd
    assert "-c:v copy" in master_cmd
    assert not any("sidechaincompress=" in " ".join(c) for c in captured["cmds"])


async def test_master_audio_skips_master_for_audioless_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bed-less video with NO audio stream skips the loudnorm master and is
    stream-copied through unchanged (byte-identical), so ``[0:a]`` can't fail
    the render. Real audio is untouched by this guard (covered elsewhere)."""
    captured = _install_fakes(monkeypatch, has_audio=False)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    cmd = " ".join(captured["cmds"][-1])
    # No loudnorm/master chain; a plain stream-copy passthrough.
    assert "loudnorm=" not in cmd
    assert "acompressor=" not in cmd
    assert "-filter_complex" not in cmd
    assert "-c copy" in cmd


async def test_master_audio_runs_master_for_audio_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A video WITH an audio stream still gets the #71 loudnorm master (guard is
    a no-op for real audio)."""
    captured = _install_fakes(monkeypatch, has_audio=True)
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.master_audio(video, tmp_path / "mastered.mp4", get_settings())
    cmd = " ".join(captured["cmds"][-1])
    assert "loudnorm=I=-14.0:TP=-1.0:LRA=11.0" in cmd
    assert "-c:v copy" in cmd


# --------------------------------------------------------------------------
# POST-master per-clip gain/mute pass (postclip) threading
# --------------------------------------------------------------------------


async def test_apply_postclip_audio_reencodes_audio_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The postclip pass applies the time-ranged volume chain to an already-mastered
    bus, re-encoding ONLY the audio. It carries NO master chain — loudnorm already
    ran upstream (and is exactly why this cannot be folded into that graph)."""
    captured = _install_fakes(monkeypatch, has_audio=True)
    video = tmp_path / "mastered.mp4"
    video.write_bytes(b"\x00")
    await renderer.apply_postclip_audio(
        video,
        tmp_path / "postclip.mp4",
        "volume=-6.0dB:enable='between(t,2.000,4.000)'",
        get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    # 4d: swapped bare `_A48` for `_A48_SOXR` — latent-proofing (byte-identical
    # at 48 kHz per the repo's own measurement note) in case this bus is ever
    # NOT already 48 kHz, which nothing here asserts.
    assert (
        "[0:a]aresample=48000:resampler=soxr:precision=28,"
        "aformat=channel_layouts=stereo:sample_rates=48000,"
        "volume=-6.0dB:enable='between(t,2.000,4.000)'[aout]" in cmd
    )
    assert "-c:v copy" in cmd
    assert "loudnorm=" not in cmd  # the master already ran in the upstream pass


async def test_apply_postclip_audio_skips_audioless_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bus with no audio stream is stream-copied through (the [0:a] map would
    otherwise fail the render) — mirrors the master_audio guard."""
    captured = _install_fakes(monkeypatch, has_audio=False)
    video = tmp_path / "mastered.mp4"
    video.write_bytes(b"\x00")
    await renderer.apply_postclip_audio(
        video, tmp_path / "postclip.mp4", "volume=0:enable='between(t,0.000,2.000)'", get_settings()
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "-c copy" in cmd
    assert "-filter_complex" not in cmd


async def test_compile_shots_gain_shot_runs_postclip_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live-bug fix end-to-end: a bed-less compile where a shot requests gain
    still runs the master pass, then a SEPARATE postclip pass carrying the
    time-ranged volume node keyed to that shot's absolute bus window."""
    captured = _install_fakes(monkeypatch)
    shots = [
        Shot(id="s0", seconds=2.0, image_url="http://m/a.png"),
        Shot(id="s1", seconds=2.0, image_url="http://m/b.png", audio=ShotAudio(gain_db=-20.0)),
    ]
    out, _, _, _, _ = await renderer.compile_shots(shots, get_settings())
    assert out.name == "postclip.mp4"
    cmds = [" ".join(c) for c in captured["cmds"]]
    # The master still ran (in its own, earlier pass) …
    assert any("loudnorm=" in c for c in cmds)
    # … and the postclip volume rides a LATER, separate pass with no master in it.
    last = cmds[-1]
    # fake probe → 3.0s segments, so shot 1's absolute window is [3.000, 6.000).
    assert "volume=-20.0dB:enable='between(t,3.000,6.000)'" in last
    assert "loudnorm=" not in last


async def test_compile_shots_no_postclip_pass_without_gain_or_mute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No shot requests gain/mute → NO postclip pass runs at all and no time-ranged
    volume node is emitted anywhere; the compile ends at the mastered bus (the
    default render is untouched). A fade-only shot does NOT trigger it either."""
    captured = _install_fakes(monkeypatch)
    shots = [
        Shot(id="s0", seconds=2.0, image_url="http://m/a.png"),
        Shot(id="s1", seconds=2.0, image_url="http://m/b.png", audio=ShotAudio(fade_in_s=0.5)),
    ]
    out, _, _, _, _ = await renderer.compile_shots(shots, get_settings())
    assert out.name == "mastered.mp4"
    cmds = [" ".join(c) for c in captured["cmds"]]
    assert not any(":enable=" in c for c in cmds)
    # The fade still rides its own segment, pre-master.
    assert any("afade=t=in:st=0:d=0.5" in c for c in cmds)


async def test_compile_shots_no_master_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fakes(monkeypatch)
    shots = [Shot(id="s1", seconds=2.0, image_url="http://m/a.png")]
    settings = get_settings().model_copy(update={"master_enabled": False})
    out, _, _, _, _ = await renderer.compile_shots(shots, settings)
    # Master disabled + no bed → NO mastering filter runs. The concat output is no
    # longer deliverable as-is (segments are lossless PCM in a MOV so the join is
    # sample-exact), so the one remaining pass is the plain delivery transcode:
    # video stream-copied, audio encoded to the AAC delivery spec exactly once.
    assert out.name == "delivery.mp4"
    cmds = [" ".join(c) for c in captured["cmds"]]
    assert not any("loudnorm=" in c for c in cmds)
    assert not any("sidechaincompress=" in c for c in cmds)
    delivery = [c for c in cmds if c.endswith("delivery.mp4")]
    assert len(delivery) == 1
    assert "-c:v copy" in delivery[0]
    assert "-c:a aac -b:a 192k -ar 48000 -ac 2" in delivery[0]
