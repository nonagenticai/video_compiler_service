"""The seam-crossfaded music-bed loop must deliver ALL of its samples, every time
(real ffmpeg, no mocks).

The loop (``_bed_loop_filter``) used to chain ``acrossfade`` over N reads of the
asset. ``acrossfade``'s output depends on the arrival order of its inputs'
frames/EOF — thread timing — and under CPU pressure it truncated or dropped the
audio (measured on the prod image, 1 CPU: 0..21056-sample stubs of a 72000-sample
crossfade). The loop is now built by ``_timeline_crossfade_mix``.

The check is on DECODED samples, repeated, because the defect was a race: the
exact sample count of the looped bed, and a level that never collapses (a
truncated or dropped loop decodes as silence where the bed should be).
"""

from __future__ import annotations

import array
import math
import shutil
from pathlib import Path

import pytest

from src.services import renderer
from tests.render_harness import make_bed_tone

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="decoded bed-loop round trip requires ffmpeg + ffprobe",
    ),
]

_SR = 48000
_ASSET_SECONDS = 1.0
_NEEDED_SECONDS = 5.0
_WINDOW = 0.05


async def _decode(path: Path) -> array.array[int]:
    raw = path.with_name(path.name + ".raw")
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a?",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(_SR),
            str(raw),
        ]
    )
    samples: array.array[int] = array.array("h")
    if raw.exists():
        samples.frombytes(raw.read_bytes())
        raw.unlink()
    return samples


def _window_rms_db(samples: array.array[int]) -> list[float]:
    n = round(_WINDOW * _SR)
    out: list[float] = []
    for i in range(0, len(samples) - n + 1, n):
        chunk = samples[i : i + n]
        mean_sq = sum(s * s for s in chunk) / n
        out.append(-120.0 if mean_sq == 0 else 10 * math.log10(mean_sq / 32768.0**2))
    return out


async def test_bed_loop_delivers_every_sample_on_every_render(tmp_path: Path) -> None:
    # 440 Hz x 1.0 s is a whole number of cycles, so the copies are phase-aligned
    # at every seam and a linear crossfade of the asset with itself is flat.
    asset = await make_bed_tone(
        tmp_path / "bed.wav", duration=_ASSET_SECONDS, frequency=440, gain_db=-6.0
    )
    asset_seconds = await renderer._ffprobe_duration(asset)
    copies, xfade = renderer._bed_loop_plan(asset_seconds, _NEEDED_SECONDS)
    assert copies >= 2, "fixture must actually exercise the loop"
    expected = round((copies * asset_seconds - (copies - 1) * xfade) * _SR)
    graph = renderer._bed_loop_filter(0, copies, xfade, "bed", asset_seconds)
    assert "acrossfade" not in graph

    failures: list[str] = []
    for n in range(10):
        out = tmp_path / f"loop-{n}.wav"
        await renderer._run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *(["-i", str(asset)] * copies),
                "-filter_complex",
                graph,
                "-map",
                "[bed]",
                "-c:a",
                "pcm_s16le",
                str(out),
            ]
        )
        samples = await _decode(out)
        levels = _window_rms_db(samples)
        floor = min(levels) if levels else -120.0
        ceiling = max(levels) if levels else -120.0
        if abs(len(samples) - expected) > 48 or ceiling - floor > 3.0:
            failures.append(
                f"render {n}: {len(samples)} samples (expected {expected}), "
                f"window RMS {floor:.1f}..{ceiling:.1f} dBFS"
            )
    assert not failures, f"{len(failures)}/10 bed loops lost audio: {failures}"
