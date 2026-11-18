"""Real-ffmpeg render harness package.

Re-exports the harness helpers so later test modules can import them directly:
``from tests.render_harness import make_test_clip, probe_duration,
probe_video_frame_count, probe_has_audio``.
"""

from __future__ import annotations

from tests.render_harness.harness import (
    make_bed_tone,
    make_premastered_narration_clip,
    make_test_clip,
    probe_band_mean_volume_window,
    probe_duration,
    probe_frame_md5,
    probe_has_audio,
    probe_mean_volume,
    probe_mean_volume_window,
    probe_video_frame_count,
)

__all__ = [
    "make_bed_tone",
    "make_premastered_narration_clip",
    "make_test_clip",
    "probe_band_mean_volume_window",
    "probe_duration",
    "probe_frame_md5",
    "probe_has_audio",
    "probe_mean_volume",
    "probe_mean_volume_window",
    "probe_video_frame_count",
]
