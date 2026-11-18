"""An unfetchable NARRATION asset must fail the shot loudly, never degrade to
mute (video_compiler#232, Defect A).

Production evidence: a ken_burns VIDEO clip 403'd (expired 1h presigned MinIO
URL) and the compile shipped ``completed`` with that shot's narration line
ABSENT — a master silently missing its script content. Every OTHER asset kind
(video, image, title-card clip, overlay, SFX) degrades that one shot to its
next-best visual and keeps rendering, which is correct: a picture swap still
ships a watchable, narrated video. Narration is different — it is the
deliverable's content, not decoration — so ``_resolve_shot_assets`` now raises
``UnfetchableNarrationError`` instead of silently resolving to a mute shot.

These tests are fast (no real ffmpeg): ``_fetch_asset`` is monkeypatched to
return ``None`` (the same signature ``_try_fetch`` returns on any 403/expired
signature/transport error), so they pin the CONTRACT at the point of decision
rather than depending on a real HTTP 403.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.services import renderer

pytestmark = pytest.mark.asyncio


async def _unfetchable(url: str, dest: Path, http, source_cache=None) -> None:
    return None


async def test_resolve_shot_assets_raises_on_unfetchable_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(renderer, "_fetch_asset", _unfetchable)
    shot = Shot(id="s1", seconds=3.0, audio_url="http://minio/line.wav")
    async with httpx.AsyncClient() as http:
        with pytest.raises(renderer.UnfetchableNarrationError, match="s1"):
            await renderer._resolve_shot_assets(shot, tmp_path, http)


async def test_resolve_shot_assets_no_audio_url_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Precondition / non-regression: a shot that never asked for narration must
    not be swept up by the new guard — it still resolves cleanly to
    ``audio_path=None`` and falls back to ``shot.seconds``."""
    monkeypatch.setattr(renderer, "_fetch_asset", _unfetchable)
    shot = Shot(id="s2", seconds=4.0)  # no audio_url, no video_url, no image_url
    async with httpx.AsyncClient() as http:
        audio_path, _clip, _image, duration, _second_clip = await renderer._resolve_shot_assets(
            shot, tmp_path, http
        )
    assert audio_path is None
    assert duration == 4.0


async def test_resolve_shot_assets_fetched_narration_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Precondition: a shot whose narration DOES resolve renders exactly as
    before — the guard only fires on the ``None`` (unfetchable) outcome.
    ``_ffprobe_duration`` is stubbed so this stays ffmpeg-free like its siblings;
    it is not what this test is about (see the render_harness suite for real
    ffprobe/ffmpeg coverage of narration duration)."""

    async def _fake_duration(path: Path) -> float:
        return 3.0

    monkeypatch.setattr(renderer, "_ffprobe_duration", _fake_duration)
    real_audio = tmp_path / "line.wav"
    real_audio.write_bytes(b"not-really-a-wav-but-fetch-never-inspects-it")
    shot = Shot(id="s3", seconds=3.0, audio_url=str(real_audio))
    async with httpx.AsyncClient() as http:
        audio_path, *_rest = await renderer._resolve_shot_assets(shot, tmp_path, http)
    assert audio_path == real_audio


async def test_compile_shots_propagates_unfetchable_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: ``compile_shots`` must not swallow the shot-level raise into a
    ``completed`` result — the whole job fails, matching
    ``test_unfetchable_narration_fails_the_job_not_interrupted`` in
    ``test_compile_route.py`` for the HTTP-layer classification."""
    from src.config import get_settings

    monkeypatch.setattr(renderer, "_fetch_asset", _unfetchable)
    settings = get_settings().model_copy(update={"work_dir": str(tmp_path)})
    shot = Shot(id="s4", seconds=2.0, audio_url="http://minio/line.wav")
    with pytest.raises(renderer.UnfetchableNarrationError):
        await renderer.compile_shots([shot], settings)
