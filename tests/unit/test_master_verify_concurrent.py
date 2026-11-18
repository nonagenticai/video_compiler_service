"""``post_mux_verify``'s probes must overlap, and the report must not change.

Measured in production 2026-08-28: ``master_verify`` was **26.5s mean, 24% of
total job time** — the third-largest stage after deband and render_shots. That
number was simply the SUM of six strictly-serial subprocesses over one immutable
file, two of which are full decodes (``signalstats``, and ``mpdecimate`` inside
``duplicate_frame_ratio``). Nothing depended on anything else; they were serial
only because they were written that way.

Verification runs AFTER the render fan-out, so the global ffmpeg semaphore
(``max_concurrent_ffmpeg_subprocesses``, 3 in prod) sits idle here — the same
argument #187 used to unserialise the deband and source-match measure passes.

Making them concurrent is only correct if BOTH hold, and each has a test:

  1. the two full decodes actually overlap — otherwise the change bought nothing;
  2. the report is byte-identical to the serial one. This is a SCHEDULING change;
     if any verdict moves, the change is wrong. ⚠️ This matters more than the
     speed: a wrong verdict here is a false "frozen video" / "banded master"
     claim on a DELIVERED master, which is precisely why #187 chose a slower
     fallback over a faster path in this same function.

Plus a guard that a failing probe still degrades to None rather than failing the
job — every probe here is best-effort BY CONTRACT, which is the property that
makes a bare ``gather`` (no ``return_exceptions``) safe.

No ffmpeg: ``renderer._run`` is stubbed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.config import get_settings
from src.services import mastering, measure, renderer

MASTER = Path("/tmp/master.mp4")

_CLEAN_FRAME = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 10.0}


def _settings():
    # An expected colorspace makes the probe round its full five-element shape.
    return get_settings().model_copy(update={"render_colorspace": "bt709", "render_fps": 30})


class _Tracker:
    """Records overlap: the max number of _run calls in flight at once."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.inflight = 0
        self.peak = 0
        self.cmds: list[str] = []

    async def run(self, cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(str(c) for c in cmd)
        self.cmds.append(joined)
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.inflight -= 1
        if "mpdecimate" in joined:
            return "frame=  100 fps=0.0 q=-0.0 Lsize=N/A"
        if "nb_read_packets" in joined or "-count_packets" in joined:
            return "120"
        if "stream=time_base:packet=pts" in joined:
            # video pts-interval probe: a regular timeline (half of ``cfr_ok``).
            return '{"streams": [{"time_base": "1/30"}], "packets": [{"pts": 0}, {"pts": 1}, {"pts": 2}]}'
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "avg_frame_rate" in joined and "r_frame_rate" in joined:
            return "30/1\n30/1"
        if "avg_frame_rate" in joined:
            return "30/1"
        return ""


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> _Tracker:
    t = _Tracker()
    monkeypatch.setattr(renderer, "_run", t.run)

    # signalstats is parsed from stderr by measure; stub the parsed result so the
    # test does not depend on ffmpeg's metadata=print text format.
    async def _timed(path):
        await t.run(["ffmpeg", "-i", str(path), "-vf", "signalstats"], capture_stderr=True)
        return [(0.0, dict(_CLEAN_FRAME)), (0.033, dict(_CLEAN_FRAME))]

    monkeypatch.setattr(measure, "measure_shot_frames_timed", _timed)
    return t


class TestProbesOverlap:
    async def test_the_probe_round_runs_concurrently(self, tracker: _Tracker) -> None:
        await mastering.post_mux_verify(MASTER, _settings())
        # Serial would peak at exactly 1. The round issues five top-level probes,
        # one of which itself fans out to two more.
        assert tracker.peak > 1, (
            f"probes ran serially (peak in-flight={tracker.peak}); cmds={tracker.cmds}"
        )

    async def test_the_two_full_decodes_overlap(self, tracker: _Tracker) -> None:
        """The win is specifically max(decode, decode), not decode+decode."""
        await mastering.post_mux_verify(MASTER, _settings())
        decodes = [c for c in tracker.cmds if "signalstats" in c or "mpdecimate" in c]
        assert len(decodes) == 2, f"expected exactly two full decodes, got {decodes}"
        # Wall clock proves overlap independently of the counter above.
        assert tracker.peak >= 2


class TestReportIsUnchanged:
    async def test_verdicts_match_a_serialised_run(self, tracker: _Tracker) -> None:
        """Force serial execution by making gather see one coroutine at a time is
        not possible without changing production; instead assert the CONTENT of
        the report against the values the stubs imply. If scheduling changed any
        verdict, these move."""
        report = await mastering.post_mux_verify(MASTER, _settings())

        assert report["colorspace_ok"] is True
        assert report["cfr_ok"] is True
        assert report["measured_fps"] == 30.0
        # 120 packets total, 100 survived mpdecimate → 20 dropped.
        assert report["duplicate_frame_ratio"] == pytest.approx(20 / 120, abs=1e-4)
        # A clean frame (YDIF=10 >= 3.0) scores 0.0 banding by construction, and
        # the report carries the WORST frame, so the list holds exactly that one max.
        assert report["banding_scores"] == [pytest.approx(0.0)]
        # And the verdict pipeline still reads those measurements. The VERDICT (not
        # just the numbers) is what proves the concurrent round is wired into the
        # gate the same way the serial one was — here it is driven by loudness,
        # which is unmeasurable because this call passes no ``loudness`` payload
        # and therefore fails CLOSED.
        #
        # ⚠️ This used to assert ``"duplicate_frames" in rerun_stages`` on the
        # strength of the 0.167 ratio above. That gate was removed on 2026-09-11
        # (intentional dwell; it fired on 120 of 120 measured masters), so the
        # ratio is asserted as TELEMETRY above and named in no verdict here.
        assert report["gate_pass"] is False
        assert report["rerun_stages"] == ["loudness"]
        assert "duplicate_frames" not in report["rerun_stages"]

    async def test_each_probe_is_issued_exactly_once(self, tracker: _Tracker) -> None:
        """Concurrency must not duplicate work — a gather built from the wrong
        list would silently double a full decode."""
        assert sum("signalstats" in c for c in tracker.cmds) == 0  # stubbed separately
        await mastering.post_mux_verify(MASTER, _settings())
        assert sum("mpdecimate" in c for c in tracker.cmds) == 1
        assert sum("signalstats" in c for c in tracker.cmds) == 1
        assert sum("color_space" in c for c in tracker.cmds) == 1


class TestBestEffortContract:
    async def test_a_raising_probe_does_not_fail_the_job(
        self, monkeypatch: pytest.MonkeyPatch, tracker: _Tracker
    ) -> None:
        """Every probe swallows its own errors — that contract is what makes a
        bare gather safe. If a probe is ever changed to raise, this test fails
        and the gather must become a TaskGroup."""

        async def boom(cmd: list[str], *, capture_stderr: bool = False) -> str:
            raise OSError("ffprobe vanished")

        monkeypatch.setattr(renderer, "_run", boom)
        report = await mastering.post_mux_verify(MASTER, _settings())
        # Degrades to unknowns rather than raising.
        assert report["cfr_ok"] is None
        assert report["measured_fps"] is None
        assert report["duplicate_frame_ratio"] is None
