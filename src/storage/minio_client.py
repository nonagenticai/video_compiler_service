"""Object-storage uploader — streams a file to MinIO via the shared
``w2v_common.storage.StorageService`` facade and returns a presigned GET URL.

The facade's ``put_file`` streams from disk (``fput_object``) so a multi-GB
compiled MP4 never lands in RAM — the reason this service could not adopt the
in-memory ``put_bytes`` earlier (video_compiler#68).

One long-lived, initialized ``StorageService`` is cached **per bucket** (this
service writes to two: ``minio_bucket`` for videos, ``minio_audio_bucket`` for
stitched audio). The miniopy aiohttp session is created once and reused across
requests — replacing the old per-request ``async with make_client(...)`` (which
existed only to close the session each call and avoid an "Unclosed client
session" leak). ``replicas: 1`` + a single event loop make a module-level cache
safe here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from miniopy_async.error import MinioException
from w2v_common.config.base import StorageSettings
from w2v_common.storage import StorageService

from src.config import Settings

log = structlog.get_logger(__name__)


class StorageError(RuntimeError):
    pass


def _storage_settings(settings: Settings, bucket: str) -> StorageSettings:
    """Map this service's flat ``Settings`` onto a ``StorageSettings`` bound to
    ``bucket``. No ``minio_external_endpoint`` (the presigned URL keeps the
    ``minio_endpoint`` host, matching the prior behaviour — no host rewrite)."""
    return StorageSettings(
        backend="minio",
        minio_endpoint=settings.minio_endpoint,
        minio_access_key=settings.minio_access_key,
        minio_secret_key=settings.minio_secret_key,
        minio_bucket=bucket,
        minio_secure=settings.minio_secure,
        presign_expiry_seconds=settings.minio_url_expiry_seconds,
    )


_STORAGE: dict[str, StorageService] = {}
_STORAGE_LOCK = asyncio.Lock()


async def get_storage(settings: Settings, *, bucket: str) -> StorageService:
    """Return a cached, initialized ``StorageService`` bound to ``bucket``.

    ``initialize()`` ensures the bucket exists (mirrors the old
    ``ensure_bucket``) and is run exactly once per bucket; subsequent calls
    reuse the same client (and its aiohttp session)."""
    svc = _STORAGE.get(bucket)
    if svc is not None:
        return svc
    async with _STORAGE_LOCK:
        svc = _STORAGE.get(bucket)
        if svc is None:
            svc = StorageService(_storage_settings(settings, bucket))
            await svc.initialize()
            _STORAGE[bucket] = svc
    return svc


async def upload_and_presign(
    settings: Settings,
    *,
    bucket: str,
    object_key: str,
    file_path: Path,
    expiry_seconds: int,
    content_type: str = "video/mp4",
) -> str:
    """Stream ``file_path`` to ``bucket/object_key`` and return a presigned GET
    URL. Raises :class:`StorageError` on any storage failure (as before)."""
    try:
        storage = await get_storage(settings, bucket=bucket)
        await storage.put_file(object_key, file_path, content_type=content_type)
        url = await storage.presign_get(object_key, expires=expiry_seconds)
    except (OSError, MinioException) as e:
        raise StorageError(f"minio upload failed: {e}") from e
    if url is None:
        raise StorageError(f"presign returned no URL for {object_key}")
    return url
