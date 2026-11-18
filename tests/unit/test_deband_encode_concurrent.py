"""The deband ENCODE pass must be concurrent, isolated and still cancellable.

``_apply_conditional_deband`` already gathers its MEASURE pass (see
``test_renderer_concurrent_measure``). The ENCODE stayed serial, justified in-code
by "the re-encode is believed rare — the measure pass is the dominant cost".

Production disagrees: **129 applied / 2 skipped = 98.5%** of segments re-encode.
That is the trigger working as designed — ``banding_risk_frames`` takes a MAX over
frames, so one smooth frame trips a whole segment, deliberately, to keep the
mastering gate satisfiable. So the "rare" branch is the common one, and it held ONE
ffmpeg slot while the rest of ``max_concurrent_ffmpeg_subprocesses`` (2 by default,
3 in production) sat idle. ``deband`` measures ~38s of a ~102s job.

Overlapping the encodes is only CORRECT if four properties hold, one test each:

  1. the encodes actually overlap (otherwise the change bought nothing);
  2. the applied/skipped/failed counters are UNCHANGED — ``applied`` still counts
     the DECISION, exactly as it did when the encode was inline, so the production
     fire-rate series stays comparable across this change;
  3. one segment's encode blowing up does NOT fail the job, and does not stop its
     siblings. Deband is documented best-effort polish that "never fails a
     compile"; a bare gather would propagate and turn picture polish into a
     job-killer. This is the property that matters most;
  4. a cancel is still a cancel — best-effort must not mean un-cancellable.

Plus: with nothing banded, no encode is attempted at all.

No ffmpeg: ``measure`` and ``_deband_segment`` are stubbed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

from src.config import Settings, get_settings
from src.services import measure, renderer

# Same frame shapes the trigger/gate tests use.
_BANDED = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 0.5}
_CLEAN = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 10.0}


def _segs(n: int = 4) -> list[Path]:
    return [Path(f"/tmp/seg-{i:04d}.mp4") for i in range(n)]


def _deband_settings() -> Settings:
    return get_settings().model_copy(update={"render_deband": "on"})


def _deband_count(result: str) -> float:
    value = REGISTRY.get_sample_value("video_compile_deband_segments_total", {"result": result})
    return float(value or 0.0)


def _all_banded(banded: set[Path]):
    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        return [_BANDED] if seg in banded else [_CLEAN]

    return fake_measure


class _EncodeProbe:
    """Records the peak number of deband ENCODES in flight simultaneously."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self.seen: list[Path] = []

    async def __call__(self, seg: Path, settings: Settings, deterministic: bool = False) -> None:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.seen.append(seg)
        # Yield twice so a SERIAL caller cannot overlap by accident and a
        # concurrent one certainly does.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.inflight -= 1


# ---------------------------------------------------------------------------
# 1. concurrency — the point of the change
# ---------------------------------------------------------------------------


async def test_deband_encode_pass_runs_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    segs = _segs(4)
    probe = _EncodeProbe()
    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded(set(segs)))
    monkeypatch.setattr(renderer, "_deband_segment", probe)

    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert probe.peak == 4, f"encodes must overlap; peak in flight was {probe.peak}"
    assert sorted(probe.seen) == sorted(segs)


async def test_only_the_banded_segments_are_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean segment must stay a single-pass, byte-identical encode."""
    segs = _segs(4)
    probe = _EncodeProbe()
    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded({segs[1], segs[3]}))
    monkeypatch.setattr(renderer, "_deband_segment", probe)

    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert sorted(probe.seen) == sorted([segs[1], segs[3]])


async def test_nothing_banded_attempts_no_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unreachable(*args: object, **kwargs: object) -> None:
        raise AssertionError("clean segments must not be debanded")

    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded(set()))
    monkeypatch.setattr(renderer, "_deband_segment", unreachable)

    before = _deband_count("skipped")
    await renderer._apply_conditional_deband(_segs(3), _deband_settings())
    assert _deband_count("skipped") == before + 3


# ---------------------------------------------------------------------------
# 2. the counters must not move — the production series stays comparable
# ---------------------------------------------------------------------------


async def test_counters_are_unchanged_by_concurrent_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = _segs(3)

    async def fake_measure(seg: Path) -> list[dict[str, float]]:
        if seg == segs[0]:
            return [_BANDED]
        if seg == segs[1]:
            return [_CLEAN]
        raise RuntimeError("ffprobe blew up")

    monkeypatch.setattr(measure, "measure_shot_frames", fake_measure)
    monkeypatch.setattr(renderer, "_deband_segment", _EncodeProbe())

    before = {r: _deband_count(r) for r in ("applied", "skipped", "failed")}
    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert _deband_count("applied") == before["applied"] + 1
    assert _deband_count("skipped") == before["skipped"] + 1
    assert _deband_count("failed") == before["failed"] + 1


async def test_a_failing_encode_does_not_move_the_applied_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``applied`` counts the DECISION, as it did when the encode was inline.

    Pinning this makes the change a pure scheduling change on the metric side too:
    a failed encode was never counted as ``failed`` before (that label is the
    MEASURE failure), and must not start being counted as one now.
    """
    segs = _segs(2)

    async def boom(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded(set(segs)))
    monkeypatch.setattr(renderer, "_deband_segment", boom)

    before = {r: _deband_count(r) for r in ("applied", "skipped", "failed")}
    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert _deband_count("applied") == before["applied"] + 2
    assert _deband_count("failed") == before["failed"]
    assert _deband_count("skipped") == before["skipped"]


# ---------------------------------------------------------------------------
# 3. per-segment failure isolation — the one that must never regress
# ---------------------------------------------------------------------------


async def test_one_failing_encode_does_not_fail_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``gather`` would propagate seg-0001's error and kill the compile."""
    segs = _segs(4)
    done: list[Path] = []

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        if seg == segs[1]:
            raise OSError("replace() failed: cross-device link")
        await asyncio.sleep(0)
        done.append(seg)

    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded(set(segs)))
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)

    # Must NOT raise.
    await renderer._apply_conditional_deband(segs, _deband_settings())

    assert sorted(done) == sorted([segs[0], segs[2], segs[3]]), (
        "a sibling encode failing must not stop the others"
    )


# ---------------------------------------------------------------------------
# 4. cancellation still cancels
# ---------------------------------------------------------------------------


async def test_a_cancellation_during_encode_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort must not mean un-cancellable.

    ``CancelledError`` is a BaseException, not an Exception, so the encode round
    re-raises it rather than logging it as a best-effort failure — the same split
    the measure round above it makes.
    """
    segs = _segs(2)

    async def fake_deband(seg: Path, settings: Settings, deterministic: bool = False) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(measure, "measure_shot_frames", _all_banded(set(segs)))
    monkeypatch.setattr(renderer, "_deband_segment", fake_deband)

    with pytest.raises(asyncio.CancelledError):
        await renderer._apply_conditional_deband(segs, _deband_settings())
