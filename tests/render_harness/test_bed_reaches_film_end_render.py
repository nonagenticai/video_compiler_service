"""Real-ffmpeg render-harness test: the music bed must reach the END of the film (#235).

Production master ``01a0b00d`` (website_review, ``jazz_lounge``, ffmpeg 7.1.5) carried
the bed at about -28 dB up to 13.75s and then digital silence for the last ~1.05s of a
15.05s film: the outro title card played over nothing. The bed itself was fine: it was
``atrim``-ed and faded against the full ``total``.

Mechanism: ``overlay_audio_bed`` keys ``sidechaincompress`` on the concat's own audio.
``sidechaincompress`` ends ``[ducked]`` when the KEY reaches EOF, without draining the bed
frames still queued behind it, and ``amix=duration=first`` then carries on with the
narration alone. On the production assets the ``[ducked]`` branch came back
14.06-14.57s against a 15.050s bed (a scheduling race). A key that is simply SHORTER
than ``total`` truncates the bed every time, which is the shape a title card gives when
its AAC leg ends before its picture. That deterministic shape is what this test renders.

MUTATION-PROVED 2026-09-17 on ffmpeg 7.1.5: removing the ``_pad_audio_to(total)`` on the
narration bus turns both cases RED (tail band level ``-inf``); restoring it turns them
GREEN.

Method follows ``test_vo_relative_duck_render.py``: the bed is an 8 kHz tone, outside the
speech band, so ``probe_band_mean_volume_window`` measures the BED inside the mix.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from src.api.schemas import MusicBed
from src.config import get_settings
from src.services import renderer
from tests.render_harness import (
    make_premastered_narration_clip,
    probe_band_mean_volume_window,
)

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_TOTAL = 6.0
_AUDIO_SECONDS = 4.5  # the narration bus ends 1.5 s before the picture does
_FADE_OUT = 0.5
# Bed with no narration over it, well inside the narration's audio.
_MID_WINDOW = (2.0, 4.0)
# After the narration audio has ended, and before the bed's own fade-out starts.
_TAIL_WINDOW = (4.8, _TOTAL - _FADE_OUT - 0.1)
_TAIL_FLOOR_DB = -40.0
_MAX_TAIL_DROP_DB = 6.0


async def _film_with_short_audio(work: Path) -> Path:
    """A ``_TOTAL``-second picture whose audio stream ends at ``_AUDIO_SECONDS``."""
    narration = await make_premastered_narration_clip(
        work / "narration.mp4", duration=_AUDIO_SECONDS, gap=(1.5, _AUDIO_SECONDS)
    )
    film = work / "compiled.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={_TOTAL}:size=320x240:rate=25",
            "-i",
            str(narration),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(film),
        ]
    )
    return film


async def _long_bed(path: Path) -> Path:
    """A compressed stereo 44.1 kHz bed, much longer than the film (like a library track)."""
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=8000:duration=30:sample_rate=44100",
            "-af",
            "volume=-6dB",
            "-ac",
            "2",
            "-c:a",
            "aac",
            str(path),
        ]
    )
    return path


@pytest.mark.parametrize("master_enabled", [False, True], ids=["master_off", "master_on"])
async def test_music_bed_reaches_the_end_of_the_film(tmp_path: Path, master_enabled: bool) -> None:
    work = tmp_path / ("on" if master_enabled else "off")
    work.mkdir(parents=True, exist_ok=True)
    film = await _film_with_short_audio(work)
    bed_path = await _long_bed(work / "bed.m4a")
    settings = get_settings().model_copy(
        update={"master_enabled": master_enabled, "work_dir": str(work)}
    )
    bed = MusicBed(url="bed", gain_db=0.0, fade_in_s=0.1, fade_out_s=_FADE_OUT)
    final, _report = await renderer.overlay_audio_bed(
        film, work / "final.mp4", _TOTAL, (bed_path, bed), [], settings=settings
    )

    mid = await probe_band_mean_volume_window(final, *_MID_WINDOW)
    tail = await probe_band_mean_volume_window(final, *_TAIL_WINDOW)
    assert math.isfinite(mid) and mid > _TAIL_FLOOR_DB, f"bed not audible mid-film: {mid} dB"
    assert math.isfinite(tail) and tail > _TAIL_FLOOR_DB, (
        f"music bed is gone before the film ends: {tail} dB over {_TAIL_WINDOW} "
        f"(mid-film {mid} dB). The bed branch ended with the narration bus."
    )
    assert mid - tail <= _MAX_TAIL_DROP_DB, (
        f"bed dropped {mid - tail:.1f} dB between mid-film ({mid}) and the tail ({tail})"
    )
