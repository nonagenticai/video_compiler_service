"""``renderer._stream_raw_stdout`` must kill + reap its child and cancel its
stderr reader on EVERY exit path — not only cancellation/timeout. A consumer
that raises mid-stream (a malformed PGM header, an exception out of the
spatial measure) used to leave ffmpeg alive, blocked on a full stdout pipe,
with a leaked stderr task, while the caller returned ``{}`` and looked clean.

Uses a real child process (an endless stdout writer), not a mock: the
property under test is the OS process's lifetime.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from src.services import renderer

# Writes 64 KiB chunks forever: once the consumer stops reading, the pipe fills
# and the child blocks — exactly the leaked-ffmpeg shape.
_ENDLESS_WRITER = [
    sys.executable,
    "-c",
    "import sys\nwhile True:\n    sys.stdout.buffer.write(b'x' * 65536)\n    sys.stdout.flush()",
]


def _capture_procs(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.subprocess.Process]:
    procs: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real(*args, **kwargs)
        procs.append(proc)
        return proc

    monkeypatch.setattr(renderer.asyncio, "create_subprocess_exec", spy)
    return procs


def _pending_tasks() -> list[asyncio.Task[object]]:
    me = asyncio.current_task()
    return [t for t in asyncio.all_tasks() if t is not me and not t.done()]


async def test_consumer_error_kills_and_reaps_child(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _capture_procs(monkeypatch)

    async def consume(reader: asyncio.StreamReader) -> None:
        await reader.readexactly(65536)
        raise ValueError("unexpected PGM header")

    with pytest.raises(ValueError, match="unexpected PGM header"):
        await renderer._stream_raw_stdout(_ENDLESS_WRITER, consume, budget_seconds=30)
    assert len(procs) == 1
    assert procs[0].returncode is not None  # killed AND reaped, not left blocked
    await asyncio.sleep(0)
    assert _pending_tasks() == []  # the stderr reader did not leak


async def test_budget_expiry_kills_child_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    procs = _capture_procs(monkeypatch)

    async def consume(reader: asyncio.StreamReader) -> None:
        while True:
            await reader.read(65536)

    assert await renderer._stream_raw_stdout(_ENDLESS_WRITER, consume, budget_seconds=0.5) is False
    assert procs[0].returncode is not None
    await asyncio.sleep(0)
    assert _pending_tasks() == []


async def test_clean_run_returns_true() -> None:
    cmd = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'abc')"]
    seen: list[bytes] = []

    async def consume(reader: asyncio.StreamReader) -> None:
        seen.append(await reader.read())

    assert await renderer._stream_raw_stdout(cmd, consume, budget_seconds=30) is True
    assert seen == [b"abc"]
