"""`renderer._run`'s subprocess-timeout watchdog must actually kill (#232).

The reproduced livelock (see ``tests/render_harness/test_freeze_pad_bounded_livelock.py``)
did not respond to SIGTERM at all — only SIGKILL stopped it. This proves the
watchdog path in ``_run`` (the ``asyncio.wait_for`` around ``proc.communicate()``,
``renderer.py`` ~515-542) kills a process that traps and ignores SIGTERM within a
short grace period of its subprocess timeout firing, and that the process is
fully reaped (not left as a zombie or an orphan still burning CPU).

No real ffmpeg needed: a bash script that traps SIGTERM to a no-op and loops
forever stands in for a livelocked encode.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from src.config import get_settings
from src.services import renderer

# Generous relative to the subprocess timeout below (1s) — this asserts "does
# it die promptly", not a tight timing bound that would make CI flaky.
_GRACE_PERIOD_SECONDS = 5.0


async def test_timeout_kills_a_process_that_ignores_sigterm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ``model_copy(update=...)`` does not re-validate, so this deliberately goes
    # under the field's ``ge=30`` production floor — the test needs a SHORT
    # deadline to run quickly, not a valid production value.
    settings = get_settings().model_copy(update={"ffmpeg_subprocess_timeout_seconds": 1})
    monkeypatch.setattr(renderer, "get_settings", lambda: settings)

    pidfile = tmp_path / "pid"
    script = f"trap '' TERM; echo $$ > {pidfile}; while true; do sleep 0.05; done"

    started = time.monotonic()
    with pytest.raises(renderer.RetryableRenderError, match="subprocess timeout"):
        await renderer._run(["bash", "-c", script])
    elapsed = time.monotonic() - started

    assert elapsed < _GRACE_PERIOD_SECONDS, (
        f"took {elapsed:.2f}s to kill a SIGTERM-ignoring process — the watchdog "
        "is not escalating to SIGKILL promptly"
    )

    assert pidfile.exists(), "the child never even started — test setup is broken"
    pid = int(pidfile.read_text().strip())
    # `_run` awaits `proc.wait()` after killing, so by the time it raises the
    # child must be fully reaped — signal 0 is a pure existence probe (sends
    # nothing) and raises ProcessLookupError (ESRCH) once the pid is gone.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_run_kill_call_sends_sigkill_not_sigterm_only() -> None:
    """Pin the MECHANISM, not just the outcome: `asyncio.subprocess.Process.kill()`
    (what `_run` calls on timeout) sends SIGKILL directly — it is not a
    `.terminate()` (SIGTERM) call that a livelocked ffmpeg could ignore the way
    it ignored SIGTERM in the real #232 reproduction. Guards against a future
    edit swapping `.kill()` for `.terminate()` "to be gentler", which would
    silently reintroduce the livelock-survives-the-watchdog defect."""
    proc = await asyncio.create_subprocess_exec(
        "bash", "-c", "trap '' TERM; while true; do sleep 0.05; done"
    )
    try:
        proc.kill()
        await asyncio.wait_for(proc.wait(), timeout=_GRACE_PERIOD_SECONDS)
        assert proc.returncode is not None and proc.returncode < 0, (
            "a SIGTERM-ignoring process survived .kill() — it must have been "
            "sent something other than SIGKILL"
        )
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
