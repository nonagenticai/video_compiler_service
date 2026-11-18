"""Smoke test for the real-ffmpeg render harness (WS-0).

Gated at MODULE level: skipped whenever ffmpeg/ffprobe aren't on the runner,
and tagged ``render`` so the real-render suite can be selected/deselected as a
group. ``asyncio_mode = "auto"`` runs the async test without an explicit
per-function marker.

This proves the harness building blocks work end-to-end against a real encode
so the forthcoming speed/xfade tests can trust ``probe_*`` on actual output.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.render_harness import (
    make_test_clip,
    probe_duration,
    probe_has_audio,
    probe_video_frame_count,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


async def test_harness_generates_and_probes_real_clip(tmp_path: Path) -> None:
    """A 1s lavfi clip probes to ~1s, ~25 frames, and reports an audio stream."""
    clip = await make_test_clip(tmp_path / "smoke.mp4", duration=1.0, fps=25)

    assert clip.exists()
    assert await probe_duration(clip) == pytest.approx(1.0, abs=0.15)
    assert await probe_video_frame_count(clip) == pytest.approx(25, abs=2)
    assert await probe_has_audio(clip) is True
