"""``_try_fetch`` must degrade EVERY unfetchable shot asset, not just some.

Its contract (``renderer._try_fetch`` docstring) is that a bad asset "must not take
the entire video down with it" — the shot degrades to its next-best visual and the
compile keeps going. The except-tuple caught ``httpx.HTTPError`` / ``OSError`` /
``SSRFError`` / ``ResponseTooLargeError``, but ``w2v_common.net.safe_get`` validates
the URL BEFORE any I/O and raises a BARE ``ValueError`` for a malformed one, and
``httpx.InvalidURL`` is not an ``httpx.HTTPError``. Neither was caught, so a single
malformed ``video_url`` propagated out of ``render_shot`` through the shot
``asyncio.gather`` and failed the whole paid compile.

The submit-time guard does not stop these: ``_guard_local_file_inputs`` only checks
``url.startswith(("http://", "https://"))``, which both inputs below satisfy.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.services import renderer
from src.services.durable_dispatch import inputs_reconstructable

pytestmark = pytest.mark.asyncio

# Both are accepted by the submit-time LFI/reconstructability guard, and both make
# safe_get raise a bare ValueError before it opens a socket.
_MALFORMED_BUT_ACCEPTED = [
    "http://:8080/shot.mp4",  # no hostname -> ValueError("Cannot extract hostname ...")
    "http://example.com:99999/shot.mp4",  # urlparse .port -> ValueError("Port out of range ...")
]


@pytest.mark.parametrize("url", _MALFORMED_BUT_ACCEPTED)
async def test_submit_guard_accepts_these_urls(url: str) -> None:
    """Precondition: these really do reach the renderer (not rejected at submit)."""
    ok, reason = inputs_reconstructable({"video_url": url})
    assert ok, f"precondition broken — submit guard already rejects {url}: {reason}"


@pytest.mark.parametrize("url", _MALFORMED_BUT_ACCEPTED)
async def test_try_fetch_degrades_a_malformed_url(url: str, tmp_path: Path) -> None:
    """A malformed asset URL must degrade that one shot (None), never raise."""
    async with httpx.AsyncClient() as http:
        result = await renderer._try_fetch(url, tmp_path / "asset", http)
    assert result is None, "a malformed shot-asset URL must degrade to None, not a path"
