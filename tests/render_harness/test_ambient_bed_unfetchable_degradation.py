"""Real-ffmpeg proof for the ambient-bed-loss defect (Defect B, video_compiler#232).

Production evidence: a compile requesting BOTH ``music`` and ``ambient`` delivered a
master with digital silence where the bed should have been, and the response's
``music_bed`` field was entirely absent — indistinguishable from a compile that never
asked for a bed at all. ``_flag_unapplied_music_bed`` (``src/services/renderer.py``)
closes that for BOTH bed types; this module proves it end-to-end through
``compile_shots`` (not just the unit-level ``_flag_unapplied_music_bed`` call), on a
real render, for the ambient half specifically — the half that had NO reporting
mechanism at all before this fix (music at least had ``music_bed_unfetchable``).

Two proofs, same shot list (a single silent, narration-free video shot — the
sidechain key is silence throughout, so the bed's own gain is never ducked and its
absence/presence is unambiguous over the WHOLE window):

1. ``test_compile_shots_carries_a_fetched_ambient_bed`` — the ambient cue resolves to
   a real local WAV; the delivered master must carry audible bed energy, proving the
   fix didn't accidentally break the working (fetch-succeeds) path.
2. ``test_compile_shots_flags_an_unfetchable_ambient_bed`` — the ambient cue's URL is
   patched to fail (``_fetch_asset`` returns ``None`` for it, same as a 403/expired
   presign in production); the delivered master must be silent AND
   ``compile_shots``'s returned ``music_bed`` report must name the loss
   (``ambient_fetch_failed``/``ambient_requested``/``ambient_lost``), never ``None``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import AmbientCue, Shot
from src.config import Settings, get_settings
from src.services import renderer
from tests.render_harness.harness import make_bed_tone, probe_mean_volume_window

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_DURATION = 6.0
# Well inside the clip, away from any encode-boundary artefact at the very edges.
_WINDOW = (1.0, 5.0)
# A bed tone at 0 dB gain, undamaged, reads far louder than this; digital silence
# (no bed at all) reads at or below the AAC encoder's noise floor, well under it.
# Midpoint chosen the same way the sibling render_harness tests calibrate: a floor
# that only has to separate "bed present" from "bed silent", not pin an exact dB.
_BED_PRESENT_FLOOR_DBFS = -40.0
_BED_ABSENT_CEILING_DBFS = -60.0


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={"render_fps": 25, "master_enabled": False, "work_dir": str(tmp_path)}
    )


async def test_compile_shots_carries_a_fetched_ambient_bed(tmp_path: Path) -> None:
    bed = await make_bed_tone(tmp_path / "bed.wav", duration=_DURATION, frequency=1000, gain_db=0.0)
    # A bare text-card shot (no video/image/audio url at all) renders SYNTHESISED
    # SILENCE for its own primary audio track — unlike a real clip (which would
    # carry its own baked tone), this isolates the bed as the ONLY source of
    # audible energy in the delivered mix.
    shot = Shot(id="s0", seconds=_DURATION)

    final_path, _, music_bed_report, _, _ = await renderer.compile_shots(
        [shot],
        _settings(tmp_path),
        ambient=[AmbientCue(url=str(bed), start_seconds=0.0, end_seconds=_DURATION, gain_db=0.0)],
    )

    level = await probe_mean_volume_window(final_path, *_WINDOW)
    assert level > _BED_PRESENT_FLOOR_DBFS, f"expected an audible bed, measured {level} dBFS"
    # A successfully-applied ambient bed with no music must not be flagged as lost.
    assert music_bed_report is None or not music_bed_report.get("ambient_fetch_failed")


async def test_compile_shots_flags_an_unfetchable_ambient_bed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shot = Shot(id="s0", seconds=_DURATION)
    bad_url = "http://bed.invalid.example/gone.wav"

    orig_fetch_asset = renderer._fetch_asset

    async def _selective_fetch_asset(url, dest, http, source_cache=None):
        # Only the AMBIENT cue's url is made unfetchable — the shot's own
        # video_url (a real local path) still resolves via the real function, so
        # this isolates the ambient-fetch failure from every other asset.
        if url == bad_url:
            return None
        return await orig_fetch_asset(url, dest, http, source_cache)

    monkeypatch.setattr(renderer, "_fetch_asset", _selective_fetch_asset)

    final_path, _, music_bed_report, _, _ = await renderer.compile_shots(
        [shot],
        _settings(tmp_path),
        ambient=[AmbientCue(url=bad_url, start_seconds=0.0, end_seconds=_DURATION, gain_db=0.0)],
    )

    # The structured degradation: NEVER None for a compile that asked for a bed.
    assert music_bed_report is not None, "a requested-but-lost ambient bed must not report None"
    assert music_bed_report["ambient_fetch_failed"] is True
    assert music_bed_report["ambient_requested"] == 1
    assert music_bed_report["ambient_lost"] == 1

    # And the delivered master really is silent where the bed should have been —
    # the report is not merely bookkeeping detached from what was actually shipped.
    level = await probe_mean_volume_window(final_path, *_WINDOW)
    assert level < _BED_ABSENT_CEILING_DBFS, f"expected digital silence, measured {level} dBFS"
