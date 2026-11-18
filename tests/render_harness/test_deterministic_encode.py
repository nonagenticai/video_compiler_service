"""Real-ffmpeg proof that the opt-in ``deterministic`` encode is reproducible.

The ``deterministic`` render tag makes ``compile_shots`` bit-reproducible across
runs (``-threads 1`` + ``+bitexact`` video/audio encodes + ``+bitexact`` container
mux at every point + ``-fps_mode:v cfr``). This drives an ACTUAL two-shot compile
TWICE with ``deterministic=True`` and asserts the delivered MP4s are identical
both at the container level (whole-file md5) and frame-exactly (framemd5 body).

Each compile runs in its OWN ``work_dir`` so both actually re-encode from cold —
a shared cache would make the second a byte-copy of the first and prove nothing.

CRITICAL: both renders run on the SAME local ffmpeg; the assertion is WITHIN-
binary equality of the two runs, never a comparison against a pre-baked hash
(framemd5/container bytes differ across ffmpeg builds). Gated at MODULE level
(skipped without ffmpeg/ffprobe) and tagged ``render`` like the rest of the
real-render suite.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.api.schemas import Shot
from src.config import Settings, get_settings
from src.services import renderer
from tests.render_harness import make_test_clip, probe_duration, probe_frame_md5

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _file_md5(path: Path) -> str:
    """Whole-file MD5 of the delivered MP4 (read in chunks)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _det_settings(work_dir: Path) -> Settings:
    """Small, fast render settings pinned to a fresh ``work_dir`` (cold cache)."""
    return get_settings().model_copy(
        update={
            "work_dir": str(work_dir),
            "render_fps": 25,
            "render_width": 320,
            "render_height": 240,
        }
    )


async def test_deterministic_encode_is_byte_and_frame_reproducible(tmp_path: Path) -> None:
    """Two ``deterministic=True`` compiles of the SAME fixed shot set produce a
    byte-identical MP4 (whole-file md5) whose frames are also identical (framemd5).

    Both compiles use distinct ``work_dir``s so each genuinely re-encodes rather
    than replaying a cached segment — the encode itself is what's under test."""
    # One fixed synthetic source set, generated once and reused by both compiles.
    clip_a = await make_test_clip(tmp_path / "a.mp4", duration=1.0, fps=25, frequency=440)
    clip_b = await make_test_clip(tmp_path / "b.mp4", duration=1.0, fps=25, frequency=660)
    shots = [
        Shot(id="a", seconds=1.0, video_url=str(clip_a)),
        Shot(id="b", seconds=1.0, video_url=str(clip_b)),
    ]

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()

    out1, _t1, _, _, _ = await renderer.compile_shots(
        shots, _det_settings(run1), deterministic=True
    )
    out2, _t2, _, _, _ = await renderer.compile_shots(
        shots, _det_settings(run2), deterministic=True
    )

    assert await probe_duration(out1) > 0.0
    assert await probe_duration(out2) > 0.0

    # (a) container bytes identical, and (b) frame-exact identical.
    assert _file_md5(out1) == _file_md5(out2)
    assert await probe_frame_md5(out1) == await probe_frame_md5(out2)
