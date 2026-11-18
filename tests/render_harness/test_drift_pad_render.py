"""Real-ffmpeg render-harness test for T7 — drift instead of freeze (AC#5).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_directed_footage_render.py``. This drives an ACTUAL encode and compares
delivered pixels: a per-line interaction clip that UNDER-RUNS its narration is
frame-extended by ``tpad`` (last frame cloned) AND drifted by a gentle zoompan,
so two frames sampled INSIDE the padded tail DIFFER — no identical-frame run,
the dead-freeze the T7 drift replaces. A control clip that already fills its
narration is NOT padded, so its tail stays static — proving the drift fires
ONLY on under-run.

MinIO/storage is NOT exercised: ``render_shot`` operates on local paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import get_settings
from src.services import renderer
from tests.render_harness import make_test_clip

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}


async def _frame_rgb_at(path: Path, t: float, out: Path) -> bytes:
    """Decode the frame at time ``t`` of ``path`` to raw rgb24 bytes."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(out),
        ]
    )
    return out.read_bytes()


def _diff_pixels(a: bytes, b: bytes) -> int:
    """Number of rgb24 pixels that differ between two equal-length frames."""
    n = min(len(a), len(b))
    return sum(1 for i in range(0, n - 2, 3) if a[i : i + 3] != b[i : i + 3])


async def _static_pattern_clip(path: Path, *, duration: float) -> Path:
    """A STATIC (no motion) but detailed clip: a grey field with three filled,
    off-centre boxes, held for ``duration``. Sharp box edges give the drift
    zoompan something to move — a flat field would zoom to identical pixels — and
    being static every native frame is identical, so an un-padded tail holds."""
    boxes = ",".join(
        [
            "drawbox=x=40:y=30:w=70:h=50:color=red:t=fill",
            "drawbox=x=180:y=60:w=60:h=90:color=blue:t=fill",
            "drawbox=x=110:y=150:w=90:h=40:color=green:t=fill",
        ]
    )
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s={_W}x{_H}:d={duration}:r={_FPS}",
            "-vf",
            boxes,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


async def test_short_clip_drifts_through_padded_tail(tmp_path: Path) -> None:
    """AC#5: a ~4s interaction clip under-running a ~9s narration is frame-
    extended (tpad clones the last frame) and drifted (zoompan eases the zoom
    over ONLY the padded tail). Two frames sampled INSIDE the padded region
    (t=6.0, t=7.5) come from the SAME cloned source frame yet DIFFER — the
    padded tail keeps moving, so there is no dead identical-frame run. A control
    clip that fills its narration is NOT padded, so its tail is static."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=9.0, size=f"{_W}x{_H}", fps=_FPS
    )
    async with httpx.AsyncClient() as http:
        # Under-run: 4s video vs 9s narration → freeze-pad + drift fires.
        short_src = await _static_pattern_clip(tmp_path / "short.mp4", duration=4.0)
        drift = tmp_path / "drift.mp4"
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(short_src), audio_url=str(narration)),
            drift,
            settings,
            tmp_path,
            http,
        )
        # Control: 10s video vs 9s narration → trimmed, never padded (no drift).
        long_src = await _static_pattern_clip(tmp_path / "long.mp4", duration=10.0)
        held = tmp_path / "held.mp4"
        await renderer.render_shot(
            Shot(id="b", seconds=1.0, video_url=str(long_src), audio_url=str(narration)),
            held,
            settings,
            tmp_path,
            http,
        )

    # Both timestamps sit in the padded tail of the under-run render (clip ends at
    # ~4s, narration runs to 9s), and inside the held control's live region.
    d6 = await _frame_rgb_at(drift, 6.0, tmp_path / "d6.rgb")
    d75 = await _frame_rgb_at(drift, 7.5, tmp_path / "d75.rgb")
    h6 = await _frame_rgb_at(held, 6.0, tmp_path / "h6.rgb")
    h75 = await _frame_rgb_at(held, 7.5, tmp_path / "h75.rgb")

    drift_diff = _diff_pixels(d6, d75)
    held_diff = _diff_pixels(h6, h75)
    # AC#5: the padded tail MOVES (drift) — the two frames are not identical.
    assert d6 != d75
    assert drift_diff > 0
    # The drift fires ONLY on under-run: the un-padded static control barely moves.
    assert held_diff < drift_diff


async def test_freeze_pad_cap_settles_the_drift_at_the_ceiling(tmp_path: Path) -> None:
    """Step B: with ``freeze_pad_cap`` on and a shot ``max_seconds`` ceiling, the
    drift eases over ``clip_dur → max_seconds`` and then SETTLES for the tail, while
    the ``tpad`` still fills the full narration — the video covers the audio (no
    black, no truncation), only the motion is bounded at the ceiling.

    A 4s clip under-runs a 9s narration with ``max_seconds=6.0`` (clip 4 + a 2s
    ceiling). Two frames INSIDE the drift window (t=4.5, t=5.5) differ (motion still
    easing); two frames BEYOND the ceiling (t=7.0, t=8.5) are identical (settled).
    The same render with the cap OFF keeps easing, so its beyond-6s frames differ —
    the contrast proves the cap, not just a slow drift."""
    settings = get_settings().model_copy(update=_SMALL)
    narration = await make_test_clip(
        tmp_path / "narration.mp4", duration=9.0, size=f"{_W}x{_H}", fps=_FPS
    )
    async with httpx.AsyncClient() as http:
        src = await _static_pattern_clip(tmp_path / "short.mp4", duration=4.0)
        capped = tmp_path / "capped.mp4"
        await renderer.render_shot(
            Shot(
                id="a", seconds=1.0, video_url=str(src), audio_url=str(narration), max_seconds=6.0
            ),
            capped,
            settings,
            tmp_path,
            http,
            freeze_pad_cap=True,
        )
        uncapped = tmp_path / "uncapped.mp4"
        await renderer.render_shot(
            Shot(
                id="b", seconds=1.0, video_url=str(src), audio_url=str(narration), max_seconds=6.0
            ),
            uncapped,
            settings,
            tmp_path,
            http,
            freeze_pad_cap=False,  # max_seconds present but the flag gates it off
        )

    # Capped: the drift still eases below the ceiling (4.5 vs 5.5 differ — motion).
    c45 = await _frame_rgb_at(capped, 4.5, tmp_path / "c45.rgb")
    c55 = await _frame_rgb_at(capped, 5.5, tmp_path / "c55.rgb")
    capped_below = _diff_pixels(c45, c55)
    assert capped_below > 0
    # Beyond the ceiling the motion SETTLES: two frames the pad filled (they exist ⇒
    # the audio is covered, no black/truncation) barely change relative to the eased
    # region. Not asserted == 0 because libx264 is lossy — identical input frames
    # still differ by a little compression noise (same reason the T7 test above uses
    # a relative bound, not equality).
    c70 = await _frame_rgb_at(capped, 7.0, tmp_path / "c70.rgb")
    c85 = await _frame_rgb_at(capped, 8.5, tmp_path / "c85.rgb")
    capped_tail = _diff_pixels(c70, c85)
    assert capped_tail < capped_below  # the tail moved far less than the easing region
    # With the cap OFF the drift keeps easing past 6s, so the SAME two beyond-ceiling
    # frames move much more — the cap is what settled the capped tail, not a flat clip.
    u70 = await _frame_rgb_at(uncapped, 7.0, tmp_path / "u70.rgb")
    u85 = await _frame_rgb_at(uncapped, 8.5, tmp_path / "u85.rgb")
    assert capped_tail < _diff_pixels(u70, u85)
