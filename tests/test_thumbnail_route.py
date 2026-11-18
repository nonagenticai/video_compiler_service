"""Route + service-clamp tests for POST /api/thumbnails (ffmpeg + MinIO mocked).

These do NOT exercise ffmpeg or MinIO — they prove the unversioned route parses
the request, forwards count/format/grade to the service, reads the caller tenant
without crashing, and returns the ThumbnailResponse shape; plus that the count
clamp lives in the SERVICE (not only the schema), so a direct caller can't drive
an unbounded number of extractions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

from src.api.schemas import SafeRect, ThumbnailCandidate
from src.config import get_settings
from src.services import thumbnail as thumbnail_service

pytestmark = pytest.mark.asyncio


async def test_thumbnails_route_returns_candidates(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, tmp_path: Path
) -> None:
    """A valid POST returns 200 + the candidate shape; count/format/grade reach
    the service. ``video_url`` is a bare local path so ``_fetch`` short-circuits to
    that path (no network), and the ffmpeg/upload work is mocked out. The bare path
    needs BOTH opt-ins now: ``allow_local_file_inputs`` (default False /
    secure-by-default) AND a path inside ``work_dir`` — the route passes ``local_root=work_dir``
    to ``_fetch``, so the old ``/tmp/does-not-need-to-exist.mp4``
    (outside work_dir) is refused there even with the flag on."""
    captured: dict[str, object] = {}

    async def fake_extract(video, count, *, fmt, grade, settings, work_dir):  # type: ignore[no-untyped-def]
        captured["count"] = count
        captured["fmt"] = fmt
        captured["grade_enabled"] = None if grade is None else grade.enabled
        return [
            ThumbnailCandidate(
                url="https://minio.test/w2v-compiled-videos/run-x-abc/thumb-0.png?sig=x",
                timestamp_seconds=1.5,
                width=1920,
                height=1080,
                action_safe=SafeRect(x=96, y=54, width=1728, height=972),
                title_safe=SafeRect(x=192, y=108, width=1536, height=864),
            )
        ]

    monkeypatch.setattr("src.api.routes.thumbnail.extract_candidates", fake_extract)
    # Bare-path video_url needs local-file inputs enabled (default is now False) on
    # BOTH gates: the route's submit check and the local_root confinement it hands to
    # _fetch. work_dir is pinned to this test's tmp_path, so a source under tmp_path is
    # inside the permitted root.
    allow_local = get_settings().model_copy(
        update={"allow_local_file_inputs": True, "work_dir": str(tmp_path)}
    )
    monkeypatch.setattr("src.api.routes.thumbnail.get_settings", lambda: allow_local)
    source = tmp_path / "hero.mp4"
    source.write_bytes(b"\x00\x00\x00\x20ftypisom")

    body = {
        "video_url": str(source),
        "count": 3,
        "format": "16x9",
        "grade": {"enabled": True, "contrast": 1.2},
        "tags": {"run_id": "run-x"},
    }
    r = await client.post("/api/thumbnails", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["candidates"]) == 1
    c = data["candidates"][0]
    assert c["url"].startswith("https://")
    assert set(c) >= {
        "url",
        "timestamp_seconds",
        "width",
        "height",
        "action_safe",
        "title_safe",
    }
    assert set(c["action_safe"]) == {"x", "y", "width", "height"}
    assert captured["count"] == 3
    assert captured["fmt"] == "16x9"
    assert captured["grade_enabled"] is True


async def test_thumbnails_route_rejects_out_of_range_count(client: AsyncClient) -> None:
    """The schema guards the count at [1, 12]: a 99 is a 422 at validation (the
    service clamp below is the belt-and-braces guard for a direct caller)."""
    r = await client.post("/api/thumbnails", json={"video_url": "/tmp/x.mp4", "count": 99})
    assert r.status_code == 422


async def test_thumbnails_route_rejects_local_file_when_disabled(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """With allow_local_file_inputs off (prod), a file:// video_url is a 400 LFI
    reject — the thumbnail route shares the compile route's local-file gate, which
    PR #136 only wired on /api/compile."""
    hardened = get_settings().model_copy(update={"allow_local_file_inputs": False})
    monkeypatch.setattr("src.api.routes.thumbnail.get_settings", lambda: hardened)
    r = await client.post("/api/thumbnails", json={"video_url": "file:///etc/passwd", "count": 2})
    assert r.status_code == 400, r.text
    assert "local-file input rejected" in r.text


async def test_thumbnails_route_maps_unfetchable_video_to_422(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """An unfetchable / SSRF-blocked / oversized video_url makes _fetch raise; the
    route maps it to a 422, not a 500 (previously it propagated uncaught)."""
    import httpx

    async def boom(url, dest, http, **kwargs):  # type: ignore[no-untyped-def]
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("src.api.routes.thumbnail._fetch", boom)
    r = await client.post(
        "/api/thumbnails", json={"video_url": "https://example.com/video.mp4", "count": 2}
    )
    assert r.status_code == 422, r.text
    assert "could not fetch video_url" in r.text


async def test_extract_candidates_clamps_count_in_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SERVICE clamps count to 12 (not only the schema): a direct call with
    99 yields exactly 12 candidates, with strictly-increasing timestamps."""

    async def fake_dur(path):  # type: ignore[no-untyped-def]
        return 10.0

    async def fake_dims(path):  # type: ignore[no-untyped-def]
        return (1920, 1080)

    async def fake_run(cmd, *, capture_stderr=False):  # type: ignore[no-untyped-def]
        # Simulate ffmpeg writing the requested output PNG (last arg).
        Path(cmd[-1]).write_bytes(b"\x89PNG\r\n")
        return ""

    async def fake_upload(  # type: ignore[no-untyped-def]
        settings, *, bucket, object_key, file_path, expiry_seconds, content_type="image/png"
    ):
        return f"https://minio.test/{bucket}/{object_key}"

    monkeypatch.setattr(thumbnail_service, "_ffprobe_duration", fake_dur)
    monkeypatch.setattr(thumbnail_service, "probe_dimensions", fake_dims)
    monkeypatch.setattr(thumbnail_service, "_run", fake_run)
    monkeypatch.setattr(thumbnail_service, "upload_and_presign", fake_upload)

    work_dir = tmp_path / "w"
    work_dir.mkdir()
    cands = await thumbnail_service.extract_candidates(
        Path("/tmp/x.mp4"),
        99,
        fmt=None,
        grade=None,
        settings=get_settings(),
        work_dir=work_dir,
    )
    assert len(cands) == 12  # clamped down from 99
    ts = [c.timestamp_seconds for c in cands]
    assert ts == sorted(ts) and len(set(ts)) == 12
    assert all(0.5 < t < 9.5 for t in ts)  # inside the middle 90% of a 10s clip


async def test_extract_candidates_rejects_match_sources_grade(tmp_path: Path) -> None:
    """A source-matched grade biases the eq midpoints toward the WHOLE FILM's
    measured signalstats — which a single extracted frame cannot reproduce — so the
    service rejects it (the route maps the ValueError to a 422) rather than shipping
    a poster whose look only APPROXIMATES the film. The correct source for a
    source-matched deliverable is the already-graded film (grade=None)."""
    from src.api.schemas import GradeSpec

    work_dir = tmp_path / "w"
    work_dir.mkdir()
    with pytest.raises(ValueError, match="match_sources"):
        await thumbnail_service.extract_candidates(
            Path("/tmp/x.mp4"),
            3,
            fmt=None,
            grade=GradeSpec(enabled=True, match_sources=True),
            settings=get_settings(),
            work_dir=work_dir,
        )


async def test_thumbnails_route_rejects_local_path_outside_workdir(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, tmp_path: Path
) -> None:
    """Flag ON gets you local inputs, not the pod's filesystem.

    The route's submit guard only checks the http(s) prefix, so with
    ``allow_local_file_inputs=True`` a ``/etc/passwd`` video_url sails past it. The
    ``local_root=work_dir`` confinement the route hands ``_fetch`` is what stops it —
    surfaced here as a 400.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    allow_local = get_settings().model_copy(
        update={"allow_local_file_inputs": True, "work_dir": str(work_dir)}
    )
    monkeypatch.setattr("src.api.routes.thumbnail.get_settings", lambda: allow_local)
    r = await client.post("/api/thumbnails", json={"video_url": "/etc/passwd", "count": 2})
    assert r.status_code == 400, r.text
    assert "local-file input" in r.text


@pytest.mark.parametrize(
    ("video_url", "why"),
    [
        ("http://:8080/x.mp4", "no extractable hostname"),
        ("http://h:99999/x.mp4", "out-of-range port"),
    ],
)
async def test_thumbnails_route_maps_malformed_video_url_to_422(
    client: AsyncClient, video_url: str, why: str
) -> None:
    """A MALFORMED http(s) video_url is a 422, driving the REAL ``_fetch``/``safe_get``
    path — nothing is monkeypatched here.

    This is the gap the sibling test above could not see: it injects an
    ``httpx.ConnectError`` into ``_fetch``, so it only ever exercises the errors the
    route's tuple already named. ``safe_get`` validates the URL BEFORE any I/O and
    raises a BARE ``ValueError`` (no hostname, out-of-range port, unsupported scheme)
    — not an ``httpx.HTTPError``, and not one of the two ``ValueError`` SUBCLASSES the
    tuple listed. The submit-time guard admits these (it only checks the http(s)
    prefix) and ``video_url`` is an unvalidated ``str`` in the schema, so before
    reusing ``_BAD_ASSET_ERRORS`` this propagated uncaught and 500'd the route.
    """
    r = await client.post("/api/thumbnails", json={"video_url": video_url, "count": 2})
    assert r.status_code == 422, f"{why}: {r.status_code} {r.text}"
    assert "could not fetch video_url" in r.text


async def test_unprobeable_dimensions_are_refused_not_served_as_zero_area_rects(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, tmp_path: Path
) -> None:
    """``probe_dimensions`` returns the (0, 0) SENTINEL meaning "unknown" on any probe
    failure. Here those dims are GEOMETRY — they size action_safe/title_safe — and
    ``SafeRect`` is ``Field(ge=0)``, so zeros VALIDATE. Before the fix this answered
    200 OK with width=0/height=0 and zero-area boxes at the origin, and a caller would
    place a title into a degenerate rect. Unknown is not a measurement: refuse it with
    422, symmetric with the ``duration <= 0`` guard directly above it.
    """

    async def fake_dur(path):  # type: ignore[no-untyped-def]
        return 10.0

    async def unprobeable_dims(path):  # type: ignore[no-untyped-def]
        return (0, 0)  # the documented failure sentinel

    async def fake_run(cmd, *, capture_stderr=False):  # type: ignore[no-untyped-def]
        Path(cmd[-1]).write_bytes(b"\x89PNG\r\n")
        return ""

    async def fake_upload(  # type: ignore[no-untyped-def]
        settings, *, bucket, object_key, file_path, expiry_seconds, content_type="image/png"
    ):
        return f"https://minio.test/{bucket}/{object_key}"

    monkeypatch.setattr(thumbnail_service, "_ffprobe_duration", fake_dur)
    monkeypatch.setattr(thumbnail_service, "probe_dimensions", unprobeable_dims)
    monkeypatch.setattr(thumbnail_service, "_run", fake_run)
    monkeypatch.setattr(thumbnail_service, "upload_and_presign", fake_upload)

    allow_local = get_settings().model_copy(
        update={"allow_local_file_inputs": True, "work_dir": str(tmp_path)}
    )
    monkeypatch.setattr("src.api.routes.thumbnail.get_settings", lambda: allow_local)
    source = tmp_path / "hero.mp4"
    source.write_bytes(b"\x00\x00\x00\x20ftypisom")

    r = await client.post("/api/thumbnails", json={"video_url": str(source), "count": 2})
    assert r.status_code == 422, r.text
    assert "positive dimensions" in r.text
    # And the caller never receives a zero-area geometry it could lay text into.
    assert '"width":0' not in r.text.replace(" ", "")
