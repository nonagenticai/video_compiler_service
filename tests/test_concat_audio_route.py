"""Route-level tests for POST /api/concat-audio (async submit→poll).

Renderer (ffmpeg) + MinIO are mocked — these prove request validation, the
result shape the orchestrator's audiobook_stitch stage expects, the audio
bucket + extension wiring, and tag→object-key namespacing. Real stitching is
covered by tests/test_renderer_audio.py (command-shape) and the live E2E.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from httpx import AsyncClient, Response

pytestmark = pytest.mark.asyncio


async def _poll(client: AsyncClient, job_id: str, *, attempts: int = 200) -> Response:
    last: Response | None = None
    for _ in range(attempts):
        last = await client.get(f"/api/jobs/{job_id}")
        if last.status_code == 200 and last.json()["status"] != "processing":
            return last
        await asyncio.sleep(0.01)
    assert last is not None
    return last


async def test_concat_audio_returns_artifact_url_and_duration(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    fake_wav = Path("/tmp/fake-audiobook.wav")
    fake_wav.parent.mkdir(parents=True, exist_ok=True)
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    captured: dict[str, object] = {}

    async def fake_concat(clips, settings, output_format, gap_ms, deterministic=False, **kwargs):  # type: ignore[no-untyped-def]
        captured["output_format"] = output_format
        captured["gap_ms"] = gap_ms
        captured["n_clips"] = len(clips)
        captured["deterministic"] = deterministic
        return fake_wav, 12.5, None

    async def fake_upload(client, *, bucket, object_key, file_path, expiry_seconds, content_type):  # type: ignore[no-untyped-def]
        captured["bucket"] = bucket
        captured["object_key"] = object_key
        captured["content_type"] = content_type
        return "https://minio.test/w2v-audiobooks/key.wav?sig=x"

    monkeypatch.setattr("src.api.routes.compile.concat_audio", fake_concat)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {
        "clips": [
            {"id": "c1", "audio_url": "https://minio/clip0.wav", "seconds": 6.0},
            {"id": "c2", "audio_url": "https://minio/clip1.wav", "seconds": 6.5},
        ],
        "tags": {"run_id": "run-xyz"},
    }
    submit = await client.post("/api/concat-audio", json=body)
    assert submit.status_code == 202, submit.text
    assert submit.json()["status"] == "processing"

    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    assert data["artifact_url"].startswith("https://")
    assert data["duration_seconds"] == 12.5
    assert data["bucket"] == "w2v-audiobooks"
    assert data["object_key"].startswith("run-xyz/")
    assert data["object_key"].endswith(".wav")
    # Defaults: wav output, no gap, audio content-type, the audio bucket.
    assert captured["output_format"] == "wav"
    assert captured["gap_ms"] == 0
    assert captured["content_type"] == "audio/wav"
    assert captured["bucket"] == "w2v-audiobooks"
    # No deterministic tag → the flag defaults off (prod command byte-identical).
    assert captured["deterministic"] is False


async def test_concat_audio_deterministic_tag_threads_flag(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A ``tags:{"deterministic":"true"}`` submit threads ``deterministic=True``
    into ``concat_audio`` (opt-in bit-reproducible stitch)."""
    fake_wav = Path("/tmp/fake-audiobook-det.wav")
    fake_wav.parent.mkdir(parents=True, exist_ok=True)
    fake_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    captured: dict[str, object] = {}

    async def fake_concat(clips, settings, output_format, gap_ms, deterministic=False, **kwargs):  # type: ignore[no-untyped-def]
        captured["deterministic"] = deterministic
        return fake_wav, 4.0, None

    async def fake_upload(client, *, bucket, object_key, file_path, expiry_seconds, content_type):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.concat_audio", fake_concat)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/concat-audio",
        json={
            "clips": [{"id": "c1", "audio_url": "https://minio/clip0.wav"}],
            "tags": {"deterministic": "true"},
        },
    )
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    assert done.json()["status"] == "completed"
    assert captured["deterministic"] is True


async def test_concat_audio_mp3_sets_extension_and_content_type(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    fake_mp3 = Path("/tmp/fake-audiobook.mp3")
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"ID3")

    captured: dict[str, str] = {}

    async def fake_concat(clips, settings, output_format, gap_ms, deterministic=False, **kwargs):  # type: ignore[no-untyped-def]
        return fake_mp3, 3.0, None

    async def fake_upload(client, *, bucket, object_key, file_path, expiry_seconds, content_type):  # type: ignore[no-untyped-def]
        captured["object_key"] = object_key
        captured["content_type"] = content_type
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.concat_audio", fake_concat)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/concat-audio",
        json={
            "clips": [{"id": "c1", "audio_url": "https://minio/clip0.wav"}],
            "output_format": "mp3",
            "gap_ms": 400,
        },
    )
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    assert done.json()["status"] == "completed"
    assert captured["object_key"].endswith(".mp3")
    assert captured["object_key"].startswith("ad-hoc/")  # no run_id tag → ad-hoc namespace
    assert captured["content_type"] == "audio/mpeg"


async def test_concat_audio_rejects_empty_clips(client: AsyncClient) -> None:
    r = await client.post("/api/concat-audio", json={"clips": []})
    assert r.status_code == 422


async def test_concat_audio_rejects_bad_output_format(client: AsyncClient) -> None:
    r = await client.post(
        "/api/concat-audio",
        json={"clips": [{"id": "c1", "audio_url": "https://m/0.wav"}], "output_format": "flac"},
    )
    assert r.status_code == 422


async def test_concat_audio_rejects_gap_over_cap(client: AsyncClient) -> None:
    r = await client.post(
        "/api/concat-audio",
        json={"clips": [{"id": "c1", "audio_url": "https://m/0.wav"}], "gap_ms": 9000},
    )
    assert r.status_code == 422
