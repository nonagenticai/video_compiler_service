"""The per-pod ffmpeg concurrency gate — the exit=-9 OOM fix.

`renderer._run` is the single chokepoint for every ffmpeg/ffprobe subprocess.
Without a cap, concurrent /v1/compile requests interleave on the event loop and
their combined full-resolution ffmpeg encodes OOM-kill the process (exit=-9) on
the 2Gi pod. These tests prove the gate bounds in-flight subprocesses to the
configured size — no real ffmpeg needed; `asyncio.create_subprocess_exec` is
faked with a coroutine that observes peak concurrency.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services import renderer


@pytest.fixture(autouse=True)
def _reset_ffmpeg_gate() -> Any:
    """Each test sets its own cap; restore the default afterwards so module
    global state never leaks between tests."""
    yield
    renderer.configure_ffmpeg_concurrency(renderer._DEFAULT_MAX_CONCURRENT_FFMPEG)


def _install_concurrency_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch `asyncio.create_subprocess_exec` with a fake whose `communicate()`
    sleeps while it's "running", recording how many run simultaneously."""
    state = {"current": 0, "peak": 0}

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            try:
                # Yield so sibling tasks that hold a slot overlap here; a single
                # event-loop tick is enough for peak to reflect the real cap.
                await asyncio.sleep(0.02)
            finally:
                state["current"] -= 1
            return b"out", b""

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return state


@pytest.mark.parametrize("cap", [1, 2, 3])
async def test_run_concurrency_never_exceeds_configured_cap(
    monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    """With far more concurrent `_run` calls than the cap, peak in-flight
    subprocesses equals the cap exactly — proving the gate both bounds AND is
    sized from `configure_ffmpeg_concurrency` (i.e. from settings)."""
    renderer.configure_ffmpeg_concurrency(cap)
    state = _install_concurrency_probe(monkeypatch)

    await asyncio.gather(*[renderer._run(["ffmpeg", "-version"]) for _ in range(12)])

    assert state["peak"] <= cap, f"peak {state['peak']} exceeded cap {cap}"
    # 12 tasks > cap, so the gate should be fully saturated at the cap.
    assert state["peak"] == cap, f"peak {state['peak']} never reached the cap {cap}"
    assert state["current"] == 0, "every slot must be released after _run returns"


async def test_run_releases_slot_on_ffmpeg_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero exit raises FfmpegError, but the slot must still be released
    (the `async with` exits on the raise) so one bad render can't starve the
    pod. Verified by running more failing calls than the cap and confirming
    they all complete (no deadlock) and the counter returns to zero."""
    renderer.configure_ffmpeg_concurrency(2)
    state = {"current": 0, "peak": 0}

    class _FailingProc:
        returncode = 137  # 128 + SIGKILL(9): the OOM exit we're guarding against

        async def communicate(self) -> tuple[bytes, bytes]:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            try:
                await asyncio.sleep(0.01)
            finally:
                state["current"] -= 1
            return b"", b"out of memory"

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _FailingProc:
        return _FailingProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    results = await asyncio.gather(
        *[renderer._run(["ffmpeg", "-version"]) for _ in range(6)],
        return_exceptions=True,
    )

    assert all(isinstance(r, renderer.FfmpegError) for r in results)
    assert "exit=137" in str(results[0])
    assert state["peak"] <= 2, "gate must hold even when every call fails"
    assert state["current"] == 0, "a failed _run must still release its slot"


async def test_run_kills_child_on_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `_run` is cancelled (or its subprocess deadline fires) while
    `communicate()` is still blocked, the child must be reaped — `kill()` then
    an awaited `wait()` — before the cancellation propagates, so a wedged/orphan
    ffmpeg can't linger holding a semaphore slot + memory on the pod."""
    events: dict[str, Any] = {"killed": False, "waited": False}

    class _WedgedProc:
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            # Never completes on its own — only cancellation ends this.
            await asyncio.Event().wait()
            return b"", b""  # pragma: no cover — unreachable

        def kill(self) -> None:
            events["killed"] = True
            self.returncode = -9

        async def wait(self) -> int:
            events["waited"] = True
            return -9

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _WedgedProc:
        return _WedgedProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
        await asyncio.wait_for(renderer._run(["ffmpeg", "-version"]), 0.05)

    assert events["killed"], "a cancelled/timed-out _run must kill() the child"
    assert events["waited"], "and await wait() to reap it before propagating"


async def test_run_success_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path is untouched: a proc that returns promptly is never
    killed and `_run` returns its stdout verbatim."""
    events = {"killed": False}

    class _FastProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"OK", b""

        def kill(self) -> None:
            events["killed"] = True  # pragma: no cover — must not fire

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _FastProc:
        return _FastProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    out = await renderer._run(["ffmpeg", "-version"])
    assert out == "OK"
    assert events["killed"] is False, "a successful render must never kill the child"


async def test_run_classifies_signal_kill_as_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NEGATIVE returncode is `_run`'s own signal-kill classification, and it
    must be RETRYABLE — this is the exit=-9 OOM case the gate above exists to
    survive.

    Distinct from `test_run_releases_slot_on_ffmpeg_failure`, which uses a
    POSITIVE 137 (a shell-reported 128+SIGKILL) and correctly expects a terminal
    `FfmpegError`. asyncio's `Process.returncode` reports a signal death as
    `-signum` directly, so `-9` is the shape `_run` actually sees. Nothing else
    in the suite drives `_run` with rc < 0: the one other "killed by signal"
    test (test_render_lifecycle.py) HAND-CONSTRUCTS a RetryableRenderError and
    only asserts the lifecycle's handling of an already-classified error. So a
    regression that reclassified this branch as `FfmpegError` would fail the job
    PERMANENTLY instead of retrying, with the whole suite green.
    """

    class _SignalledProc:
        returncode = -9  # SIGKILL — the OOM killer, as asyncio reports it

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"ffmpeg: Killed"

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _SignalledProc:
        return _SignalledProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises(renderer.RetryableRenderError) as excinfo:
        await renderer._run(["ffmpeg", "-i", "in.mp4", "out.mp4"])

    # RetryableRenderError must not be a *subclass* of FfmpegError, or the
    # assertion above would pass for a terminal error too.
    assert not isinstance(excinfo.value, renderer.FfmpegError)
    assert "killed by signal 9" in str(excinfo.value)
    # The stderr tail must survive so the OOM is diagnosable from the message.
    assert "Killed" in str(excinfo.value)


@pytest.mark.parametrize("signum", [9, 11, 15])
async def test_run_signal_kill_retryable_for_every_signal(
    monkeypatch: pytest.MonkeyPatch, signum: int
) -> None:
    """Every signal death is a transient RESOURCE failure, not just SIGKILL:
    SIGSEGV(11) and SIGTERM(15) must classify identically to SIGKILL(9)."""

    class _SignalledProc:
        returncode = -signum

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"stderr tail"

    async def _fake_exec(*_args: Any, **_kwargs: Any) -> _SignalledProc:
        return _SignalledProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with pytest.raises(renderer.RetryableRenderError, match=f"killed by signal {signum}"):
        await renderer._run(["ffmpeg", "-version"])
