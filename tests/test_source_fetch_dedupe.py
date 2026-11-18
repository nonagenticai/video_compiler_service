"""Per-source fetch dedupe: one download per URL per compile job.

``_fetch`` is keyed on the per-shot ``dest`` path, so nothing above it ever asked
"did another shot already pull this?". Eight shots cut from one 134 MB 4K source
therefore performed eight full downloads of identical bytes. ``SourceFetchCache``
collapses them, and the properties that make that safe under the shot
``asyncio.gather`` are asserted here:

- two concurrent shots on one URL fetch ONCE and share the result;
- a failed fetch still degrades per-shot (``None``), exactly as before;
- a fault in the dedupe machinery falls back to an undeduped fetch rather than
  failing the compile;
- the shared file is published atomically, so a crashed/cancelled fetch can never
  leave a truncated file that a later shot happily hands to ffmpeg.

No network and no ffmpeg: ``renderer._fetch`` is faked, which is the exact seam
the dedupe sits above.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from src.api.schemas import Shot
from src.services import renderer

pytestmark = pytest.mark.asyncio

_URL = "http://assets.example.com/drone-4k.mp4"
_OTHER = "http://assets.example.com/other.mp4"
_BODY = b"\x00" * 4096


def _install_counting_fetch(
    monkeypatch: pytest.MonkeyPatch, *, fail: set[str] | None = None, delay: float = 0.05
) -> list[str]:
    """Replace ``renderer._fetch`` with a slow, counting, optionally-failing fake.

    The delay is what makes the concurrency assertion meaningful: without it the
    first fetch could complete inside the same tick and a broken implementation
    would still record one call.
    """
    calls: list[str] = []
    failing = fail or set()

    async def _fake_fetch(
        url: str, dest: Path, http: httpx.AsyncClient, *, local_root: Path | None = None
    ) -> Path:
        # Mirror the REAL ``_fetch`` contract for local refs: a ``file://`` or bare
        # path is RESOLVED and returned as-is; ``dest`` is not written and not
        # returned. A stub that always wrote ``dest`` would silently override the
        # very behaviour ``test_local_refs_bypass_the_dedupe`` asserts, so the test
        # would be checking the stub rather than the code under test.
        #
        # Recorded AFTER this branch, so ``calls`` counts DOWNLOAD attempts rather
        # than every ``_fetch`` entry. A local ref does legitimately reach ``_fetch``
        # (that is where ``_resolve_local_input`` lives); what must never happen is
        # a transfer.
        if url.startswith("file://") or "://" not in url:
            return renderer._resolve_local_input(url, local_root=local_root)
        calls.append(url)
        await asyncio.sleep(delay)
        if url in failing:
            raise httpx.ConnectError("boom")
        renderer._write_atomic(dest, _BODY)
        return dest

    monkeypatch.setattr(renderer, "_fetch", _fake_fetch)
    return calls


async def test_two_concurrent_shots_on_one_url_fetch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline property: N shots, one source, one download."""
    calls = _install_counting_fetch(monkeypatch)
    cache = renderer.SourceFetchCache(tmp_path)
    shots = [Shot(id=f"s{i}", seconds=3.0, video_url=_URL) for i in range(4)]

    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(
            *(renderer._resolve_shot_assets(s, tmp_path, http, cache) for s in shots)
        )

    assert calls == [_URL], f"expected exactly one download, got {len(calls)}"
    clips = {r[1] for r in results}
    assert len(clips) == 1, "every shot must be handed the same shared path"
    clip = clips.pop()
    assert clip is not None and clip.is_file()
    assert clip.read_bytes() == _BODY


async def test_distinct_urls_are_not_collapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over-keying costs a download; under-keying renders the wrong footage."""
    calls = _install_counting_fetch(monkeypatch)
    cache = renderer.SourceFetchCache(tmp_path)

    async with httpx.AsyncClient() as http:
        a = await cache.fetch(_URL, tmp_path / "a", http)
        b = await cache.fetch(_OTHER, tmp_path / "b", http)

    assert sorted(calls) == sorted([_URL, _OTHER])
    assert a != b


async def test_a_failed_fetch_degrades_per_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad asset still returns ``None`` per shot — the degradation contract.

    The shot falls back video -> image -> text card exactly as it did undeduped;
    the compile is never failed. A sibling shot on a DIFFERENT (good) URL must be
    untouched by the neighbour's failure.
    """
    # The retry backoff sleep is not what this test is about.
    monkeypatch.setattr(renderer.asyncio, "sleep", AsyncMock(return_value=None))
    _install_counting_fetch(monkeypatch, fail={_URL})
    cache = renderer.SourceFetchCache(tmp_path)
    bad = Shot(id="bad", seconds=3.0, video_url=_URL)
    good = Shot(id="good", seconds=3.0, video_url=_OTHER)

    async with httpx.AsyncClient() as http:
        (bad_res, good_res) = await asyncio.gather(
            renderer._resolve_shot_assets(bad, tmp_path, http, cache),
            renderer._resolve_shot_assets(good, tmp_path, http, cache),
        )

    assert bad_res[1] is None, "an unfetchable source must degrade the shot, not raise"
    assert good_res[1] is not None, "one bad source must not poison an unrelated one"


async def test_a_failed_fetch_is_not_retried_for_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead URL is attempted once per JOB, not once per shot.

    Job-scoped, deliberately: re-attempting a 404 eight times only burns the
    compile's time budget, and the memo dies with the cache — the next compile of
    the same shot list retries from scratch.

    ``_try_fetch`` itself now retries a TRANSIENT failure (``httpx.ConnectError``,
    what this fixture raises) a bounded number of times BEFORE the cache memoizes
    the ``None`` — see ``renderer._TRANSIENT_FETCH_RETRIES``. That retry budget is
    per-JOB too (it lives inside the single ``cache.fetch`` call the first shot
    makes), so the property this test guards is unchanged: the SECOND shot's
    ``cache.fetch`` must trigger zero further ``_fetch`` calls, not that there is
    only ever one call in total.
    """
    # The retry backoff sleep is not what this test is about.
    monkeypatch.setattr(renderer.asyncio, "sleep", AsyncMock(return_value=None))
    calls = _install_counting_fetch(monkeypatch, fail={_URL})
    cache = renderer.SourceFetchCache(tmp_path)

    async with httpx.AsyncClient() as http:
        first = await cache.fetch(_URL, tmp_path / "a", http)
        second = await cache.fetch(_URL, tmp_path / "b", http)

    assert first is None and second is None
    # One JOB-level attempt = the full transient-retry budget, all against the
    # same URL — never one call per shot, and never MORE than the retry budget.
    assert calls == [_URL] * (renderer._TRANSIENT_FETCH_RETRIES + 1)


async def test_local_refs_bypass_the_dedupe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``file://`` / bare paths are resolved, not downloaded.

    Their ``local_root`` confinement is a per-CALL trust decision that must not be
    memoised across callers with different roots, and there is nothing to save.
    """
    calls = _install_counting_fetch(monkeypatch)
    local = tmp_path / "staged.mp4"
    local.write_bytes(_BODY)
    cache = renderer.SourceFetchCache(tmp_path)

    async with httpx.AsyncClient() as http:
        a = await cache.fetch(str(local), tmp_path / "a", http)
        b = await cache.fetch(str(local), tmp_path / "b", http)

    assert a == local and b == local
    assert calls == [], "a local ref must never be downloaded"
    # And it must not be MEMOISED either: local_root confinement is a per-call
    # trust decision, so a second caller with a different root must re-resolve
    # rather than inherit the first caller's answer.
    assert cache._tasks == {}


async def test_dedupe_fault_falls_back_to_an_undeduped_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dedupe is an optimisation and must never be able to fail a compile."""
    calls = _install_counting_fetch(monkeypatch)
    cache = renderer.SourceFetchCache(tmp_path)

    def _explode(_url: str) -> str:
        raise RuntimeError("key computation broke")

    monkeypatch.setattr(cache, "_key", _explode)

    async with httpx.AsyncClient() as http:
        result = await cache.fetch(_URL, tmp_path / "fallback.mp4", http)

    assert result == tmp_path / "fallback.mp4"
    assert calls == [_URL]


async def test_aclose_cancels_an_in_flight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown must drain the shared downloads before the work_root is removed.

    They are shielded from the per-shot ``t.cancel()`` fan-out by design, so
    nothing else would stop them writing into a directory being deleted.
    """
    _install_counting_fetch(monkeypatch, delay=5.0)
    cache = renderer.SourceFetchCache(tmp_path)

    async with httpx.AsyncClient() as http:
        waiter = asyncio.ensure_future(cache.fetch(_URL, tmp_path / "a", http))
        await asyncio.sleep(0.05)  # let the shared task start
        await cache.aclose()
        with pytest.raises(asyncio.CancelledError):
            await waiter


# ---------------------------------------------------------------------------
# Atomic publication of the shared file
# ---------------------------------------------------------------------------
async def test_write_atomic_publishes_the_whole_file(tmp_path: Path) -> None:
    dest = tmp_path / "src.mp4"
    renderer._write_atomic(dest, _BODY)
    assert dest.read_bytes() == _BODY
    assert not list(tmp_path.glob("*.part-*")), "temp file must not survive a successful write"


async def test_a_failed_write_never_exposes_a_truncated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the dedupe depends on.

    ``dest.write_bytes`` creates/truncates FIRST and fills afterwards, so a crash
    mid-write leaves a short file at the shared path — which a later shot would
    hand to ffmpeg, decoding to a shorter clip and silently dropping footage from
    the delivered film. With the temp-file + ``os.replace`` publication, a failed
    write leaves the previous content (or nothing) and no partial behind.
    """
    dest = tmp_path / "src.mp4"
    dest.write_bytes(_BODY)  # a good earlier fetch

    def _boom(_self: Path, _target: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        renderer._write_atomic(dest, b"TRUNC")

    assert dest.read_bytes() == _BODY, "a failed write must not replace a good file"
    assert not list(tmp_path.glob("*.part-*")), "a failed write must clean up its partial"


async def test_cleanup_unlink_failure_does_not_mask_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``tmp.unlink()`` cleanup in the ``except`` handler is itself capable of
    raising (permission error, a race, a disk-full condition that also breaks the
    unlink). If it does, the ORIGINAL exception being handled (here, the write
    failure) must still be what propagates out of ``_write_atomic`` — not the
    cleanup's own OSError. Mutation check: removing the ``contextlib.suppress``
    guard around the unlink makes this test see ``PermissionError`` instead of
    the expected ``OSError('disk full')``, i.e. it goes red.
    """
    dest = tmp_path / "src.mp4"

    def _boom_write(_self: Path, _body: bytes) -> int:
        raise OSError("disk full")

    def _boom_unlink(_self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(Path, "write_bytes", _boom_write)
    monkeypatch.setattr(Path, "unlink", _boom_unlink)

    with pytest.raises(OSError, match="disk full") as excinfo:
        renderer._write_atomic(dest, b"TRUNC")

    assert excinfo.type is OSError, "the write failure must survive, not the cleanup failure"


async def test_deband_cleanup_unlink_failure_does_not_fail_the_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_deband_segment`` documents that it never raises: deband is best-effort
    picture polish, and ``_apply_conditional_deband`` gathers these with
    ``return_exceptions=True`` precisely so a deband failure degrades instead of
    killing the compile. Its ffmpeg-failure branch cleans up the ``-deband.mp4``
    tmp file with ``tmp_path.unlink(missing_ok=True)`` — the same shape as
    ``_write_atomic``'s cleanup, and the same defect: if the unlink itself raises
    OSError (permission, a race), that must not replace/propagate past the
    already-handled ffmpeg failure. Mutation check: removing the
    ``contextlib.suppress`` guard around the unlink makes this test see
    ``PermissionError`` propagate instead of the clean segment being kept.
    """
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"clean-single-pass-bytes")

    async def _boom_run(_cmd: list[str]) -> None:
        raise RuntimeError("ffmpeg exploded")

    def _boom_unlink(_self: Path, missing_ok: bool = False) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(renderer, "_run", _boom_run)
    monkeypatch.setattr(Path, "unlink", _boom_unlink)

    from src.config import Settings

    # Must not raise: the compile must survive an ffmpeg failure AND a cleanup
    # failure, and the clean single-pass segment must be left untouched.
    await renderer._deband_segment(seg, settings=Settings())

    assert seg.exists(), "the clean segment must be kept when deband fails"
    assert seg.read_bytes() == b"clean-single-pass-bytes", "the clean segment must be untouched"


async def test_shared_path_is_only_ever_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second shot arriving mid-download sees the full file or waits — never a
    partially-written one.

    Asserted by sampling the shared path while the download is in flight: it must
    be ABSENT (the bytes are still in the ``.part-`` sibling), never present and
    short.
    """
    _install_counting_fetch(monkeypatch, delay=0.2)
    cache = renderer.SourceFetchCache(tmp_path)

    async with httpx.AsyncClient() as http:
        task = asyncio.ensure_future(cache.fetch(_URL, tmp_path / "a.mp4", http))
        await asyncio.sleep(0.05)
        sources = tmp_path / renderer._SOURCE_DIR_NAME
        mid_flight = [p for p in sources.glob("src-*") if ".part-" not in p.name]
        assert mid_flight == [], f"shared path published before the write completed: {mid_flight}"
        shared = await task

    assert shared is not None
    assert shared.read_bytes() == _BODY
    assert not list((tmp_path / renderer._SOURCE_DIR_NAME).glob("*.part-*"))
