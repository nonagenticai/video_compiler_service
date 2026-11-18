"""Real-ffmpeg proof of the per-shot render cache (C10/AC10).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
it selects/deselects with the rest of the real-render suite. It drives an ACTUAL
two-shot ``compile_shots`` twice — sharing one ``work_dir`` so the content-
addressed cache persists between calls — and proves that a re-compile in which
exactly one shot changed re-renders exactly ONE segment while the joined output
still ffprobes clean.

MinIO/storage is NOT exercised: ``compile_shots`` returns a local path (the
submit→poll→upload wrapper lives in the route, covered by the live E2E).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import Settings, get_settings
from src.services import renderer
from tests.render_harness import make_test_clip, probe_duration

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


async def test_recompile_reuses_unchanged_shot_and_rerenders_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2-shot compile, then a 2nd compile with only shot B changed, re-renders
    exactly ONE segment (shot B); shot A is served from the cache. The re-compiled
    film still probes to a valid, non-zero duration."""
    clip_a = await make_test_clip(tmp_path / "a.mp4", duration=1.0, fps=25, frequency=440)
    clip_b = await make_test_clip(tmp_path / "b.mp4", duration=1.0, fps=25, frequency=660)
    settings = get_settings().model_copy(
        update={
            "work_dir": str(tmp_path),
            "render_fps": 25,
            "render_width": 320,
            "render_height": 240,
        }
    )

    render_calls = 0
    real_render_shot = renderer.render_shot

    async def _spy(
        shot: Shot,
        out_path: Path,
        settings_: Settings,
        work_dir: Path,
        http: httpx.AsyncClient,
        clip_seek_fraction: float = 0.0,
        *,
        deterministic: bool = False,
        freeze_pad_cap: bool = False,
        # Accepted and FORWARDED, unlike source_cache below: this is a job-level
        # flag that changes the audio SAMPLES a shot renders (source-silencing on
        # an all-muted request). Dropping it would make the spy render something
        # production never renders, so the cache counts would be measured against
        # the wrong artefact.
        all_shots_muted: bool = False,
        # Accepted and deliberately DROPPED: the spy delegates to the real
        # render_shot, which would then fetch through the job's dedupe cache and
        # defeat the point of counting renders here. This test is about the
        # SHOT-RENDER cache (does an unchanged shot re-encode?), a different
        # cache entirely from the per-source fetch dedupe.
        source_cache: object | None = None,
        # Accepted and FORWARDED — real render_shot populates this out-parameter
        # with the shot's local narration audio path; dropping it here would
        # silently starve mastering.narration_presence_report of every path even
        # though no shot in this test carries audio_url.
        narration_audio_sink: list[Path | None] | None = None,
    ) -> float:
        nonlocal render_calls
        render_calls += 1
        return await real_render_shot(
            shot,
            out_path,
            settings_,
            work_dir,
            http,
            clip_seek_fraction=clip_seek_fraction,
            deterministic=deterministic,
            freeze_pad_cap=freeze_pad_cap,
            all_shots_muted=all_shots_muted,
            narration_audio_sink=narration_audio_sink,
        )

    monkeypatch.setattr(renderer, "render_shot", _spy)

    shot_a = Shot(id="a", seconds=1.0, video_url=str(clip_a))
    shots_v1 = [shot_a, Shot(id="b", seconds=1.0, video_url=str(clip_b))]

    render_calls = 0
    out1, _total1, _, _, _ = await renderer.compile_shots(shots_v1, settings)
    assert render_calls == 2  # cold cache renders both shots
    assert await probe_duration(out1) > 0.0

    # Second compile: shot A byte-identical, shot B changed (speed retime → new
    # pixels/timing → new content hash → forced re-render).
    shots_v2 = [shot_a, Shot(id="b", seconds=1.0, video_url=str(clip_b), speed=2.0)]

    render_calls = 0
    out2, _total2, _, _, _ = await renderer.compile_shots(shots_v2, settings)
    assert render_calls == 1  # only the changed shot re-renders; A is a cache hit
    assert await probe_duration(out2) > 0.0


async def test_recompile_identical_shots_renders_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-submitting an unchanged shot list re-renders zero segments — every shot
    is a cache hit — and still yields a valid film (idempotent re-compile)."""
    clip = await make_test_clip(tmp_path / "c.mp4", duration=1.0, fps=25)
    settings = get_settings().model_copy(
        update={
            "work_dir": str(tmp_path),
            "render_fps": 25,
            "render_width": 320,
            "render_height": 240,
        }
    )

    render_calls = 0
    real_render_shot = renderer.render_shot

    async def _spy(
        shot: Shot,
        out_path: Path,
        settings_: Settings,
        work_dir: Path,
        http: httpx.AsyncClient,
        clip_seek_fraction: float = 0.0,
        *,
        deterministic: bool = False,
        freeze_pad_cap: bool = False,
        # Accepted and FORWARDED — see the twin spy above: an all-muted request
        # renders different SAMPLES, so the spy must not silently render a
        # different artefact than production does.
        all_shots_muted: bool = False,
        # Accepted and deliberately DROPPED: the spy delegates to the real
        # render_shot, which would then fetch through the job's dedupe cache and
        # defeat the point of counting renders here. This test is about the
        # SHOT-RENDER cache (does an unchanged shot re-encode?), a different
        # cache entirely from the per-source fetch dedupe.
        source_cache: object | None = None,
        # Accepted and FORWARDED — real render_shot populates this out-parameter
        # with the shot's local narration audio path; dropping it here would
        # silently starve mastering.narration_presence_report of every path even
        # though no shot in this test carries audio_url.
        narration_audio_sink: list[Path | None] | None = None,
    ) -> float:
        nonlocal render_calls
        render_calls += 1
        return await real_render_shot(
            shot,
            out_path,
            settings_,
            work_dir,
            http,
            clip_seek_fraction=clip_seek_fraction,
            deterministic=deterministic,
            freeze_pad_cap=freeze_pad_cap,
            all_shots_muted=all_shots_muted,
            narration_audio_sink=narration_audio_sink,
        )

    monkeypatch.setattr(renderer, "render_shot", _spy)

    shots = [Shot(id="a", seconds=1.0, video_url=str(clip))]

    render_calls = 0
    await renderer.compile_shots(shots, settings)
    assert render_calls == 1  # cold cache

    render_calls = 0
    out2, _total, _, _, _ = await renderer.compile_shots(shots, settings)
    assert render_calls == 0  # fully warm — nothing re-rendered
    assert await probe_duration(out2) > 0.0
