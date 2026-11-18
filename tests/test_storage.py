"""Unit tests for the facade-backed storage module (src/storage/minio_client).

These do NOT hit MinIO — they inject a fake StorageService to prove the caching
+ streaming-upload contract the compile/concat routes depend on:
- one initialized StorageService is cached PER bucket (session reused, no
  per-request client churn — the old ``async with make_client`` lifecycle);
- ``upload_and_presign`` streams via ``put_file`` (never ``put_bytes``) and
  presigns the same key;
- a ``None`` presign (facade degraded) surfaces as ``StorageError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, cast

import pytest

from src.config import Settings
from src.storage import minio_client


class _FakeStorage:
    instances: ClassVar[list[_FakeStorage]] = []

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        self.bucket = settings.minio_bucket
        self.initialized = 0
        self.put_file_calls: list[tuple[str, Path, str]] = []
        self.presign_url: str | None = "https://minio.test/signed?sig=x"
        _FakeStorage.instances.append(self)

    async def initialize(self) -> None:
        self.initialized += 1

    async def put_file(self, key, local_path, *, content_type="application/octet-stream"):  # type: ignore[no-untyped-def]
        self.put_file_calls.append((key, local_path, content_type))
        return key

    async def presign_get(self, key, *, expires=None, bucket=None):  # type: ignore[no-untyped-def]
        return self.presign_url


@pytest.fixture(autouse=True)
def _fresh_cache_and_fake(monkeypatch: pytest.MonkeyPatch):
    _FakeStorage.instances = []
    minio_client._STORAGE.clear()
    monkeypatch.setattr(minio_client, "StorageService", _FakeStorage)
    yield
    minio_client._STORAGE.clear()


def _settings() -> Settings:
    return Settings(
        minio_endpoint="minio.test:9000",
        minio_access_key="k",
        minio_secret_key="s",
        minio_bucket="w2v-compiled-videos",
        minio_audio_bucket="w2v-audiobooks",
    )


@pytest.mark.asyncio
async def test_get_storage_caches_and_initializes_once_per_bucket():
    settings = _settings()
    a1 = cast(_FakeStorage, await minio_client.get_storage(settings, bucket="w2v-compiled-videos"))
    a2 = cast(_FakeStorage, await minio_client.get_storage(settings, bucket="w2v-compiled-videos"))
    assert a1 is a2  # same instance reused (session not recreated per call)
    assert a1.initialized == 1  # initialize() ran exactly once

    b1 = cast(_FakeStorage, await minio_client.get_storage(settings, bucket="w2v-audiobooks"))
    assert b1 is not a1  # a distinct instance per bucket
    assert b1.bucket == "w2v-audiobooks"
    assert b1.initialized == 1


@pytest.mark.asyncio
async def test_upload_and_presign_streams_then_signs(tmp_path):
    settings = _settings()
    src = tmp_path / "out.mp4"
    src.write_bytes(b"video")
    url = await minio_client.upload_and_presign(
        settings,
        bucket="w2v-compiled-videos",
        object_key="run-x/clip.mp4",
        file_path=src,
        expiry_seconds=3600,
        content_type="video/mp4",
    )
    assert url == "https://minio.test/signed?sig=x"
    svc = _FakeStorage.instances[0]
    assert svc.put_file_calls == [("run-x/clip.mp4", src, "video/mp4")]


@pytest.mark.asyncio
async def test_upload_and_presign_raises_when_presign_none(tmp_path):
    settings = _settings()
    src = tmp_path / "out.mp4"
    src.write_bytes(b"video")

    svc = cast(_FakeStorage, await minio_client.get_storage(settings, bucket="w2v-compiled-videos"))
    svc.presign_url = None  # facade degraded → no URL

    with pytest.raises(minio_client.StorageError, match="presign returned no URL"):
        await minio_client.upload_and_presign(
            settings,
            bucket="w2v-compiled-videos",
            object_key="run-x/clip.mp4",
            file_path=src,
            expiry_seconds=3600,
        )
