"""Real-ffmpeg render-harness tests for VO-relative music-bed leveling (T1).

Acceptance (great-video spec 06-sound-music.md §8, criterion #2): "VO-relative
leveling: measured music level sits within +/-1.5 LU of the requested offset
below VO integrated loudness across >=3 different casts/voices (proves T1
kills drift)."

Contract under test (``MusicBed.music_offset_lu`` + ``overlay_audio_bed`` in
``src/services/renderer.py``): when ``music_offset_lu`` is set, the bed's gain
is derived by MEASURING the concat's VO bus and the raw music asset (both via
``final_mux_loudness``) and solving for the ``volume=`` gain that lands the
bed ``music_offset_lu`` LU below the measured VO — instead of the old literal
``gain_db`` knob, which sits at a FIXED level regardless of how loud the
narration actually is. That fixed-level behaviour is exactly the "drift" this
feature kills: two casts recorded/TTS'd at different loudness get the same raw
bed gain today, so the music-to-VO balance silently drifts from mix to mix.

``tests/render_harness/test_duck_breathing_render.py`` proves criterion #3
(the duck breathes in gaps); this proves criterion #2 (the bed's RESTING level
tracks VO loudness), on real decoded samples, not the filtergraph string.

Method, and why:

* Narration is speech-band pink noise premastered via ``loudnorm`` to a
  REQUESTED integrated LUFS (``make_premastered_narration_clip``), with a long
  silent gap near the end. The gap is where the duck has fully released, so
  the bed sits at its own RESTING level there, uncontaminated by the sidechain
  key currently being open. Three casts at -12 / -18 / -24 LUFS stand in for
  ">=3 different casts/voices" at materially different loudness.
* The "measured VO integrated loudness" ground truth is NOT the nominal
  ``lufs=`` argument passed to the builder — it is ``final_mux_loudness`` run
  against the actual encoded narration file, exactly the measurement the
  production code path is contracted to perform. loudnorm's own linear pass
  does not land exactly on the target, so re-measuring is the only way to get
  a number the implementation itself would agree with.
* The delivered bed's level is measured the same way: the gap window is
  ``atrim``-extracted to its own file and run back through
  ``final_mux_loudness`` — an INTEGRATED LUFS measurement of the delivered
  samples, not a quick ``volumedetect`` dBFS mean, because the acceptance
  criterion is stated in LU (a loudness unit, not a peak/RMS unit) and the two
  are not interchangeable at the +/-1.5 LU tolerance this test holds to.
* Per the calibration trap: this suite NEVER asserts on an absolute dBFS/LUFS
  number read off a synthetic source. Every assertion is a DELTA — measured
  VO LUFS minus measured bed LUFS, or bed-A LUFS minus bed-B LUFS — which
  cancels out whatever offset ffmpeg 4.4's ``sine``/``anoisesrc`` generators
  happen to run at.

On today's (pre-T1) code, ``music_offset_lu`` is not a field ``MusicBed``
declares; pydantic silently drops the unknown kwarg (default ``extra`` is
"ignore"), so the bed renders on its literal ``gain_db`` regardless of VO
level. Test 1 and Test 2 are therefore EXPECTED TO FAIL until the sibling
lands the T1 gain-solving code in ``overlay_audio_bed``; Test 3 (back-compat,
no offset requested) is expected to PASS unconditionally because it exercises
only the pre-existing literal-gain path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.api.schemas import MusicBed
from src.config import get_settings
from src.services import renderer
from src.services.loudness import final_mux_loudness
from tests.render_harness import make_bed_tone, make_premastered_narration_clip

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_DURATION = 12.0
_GAP = (7.0, 9.7)  # long silent VO gap near the tail — plenty for a gated LUFS read
# Sample well inside the gap: >= 300 ms past the gap start (default duck release
# has fully swelled the bed back to resting level) and comfortably before the
# gap ends / narration resumes (duck attack) or the bed's own fade-out.
_WINDOW = (7.5, 9.5)

_TOLERANCE_LU = 1.5


async def _measure_lufs(path: Path) -> float:
    """Integrated LUFS via the real production measurer; fails loudly if unmeasurable."""
    result = await final_mux_loudness(str(path))
    assert result is not None, f"final_mux_loudness could not measure {path}"
    lufs = result["integrated_lufs"]
    assert lufs is not None, f"final_mux_loudness returned no integrated_lufs for {path}"
    return lufs


async def _extract_window(src: Path, dest: Path, start: float, end: float) -> Path:
    """Cut ``[start, end)`` out of ``src`` into a standalone file so it can be
    re-measured on its own via ``final_mux_loudness`` (which measures a whole
    file, not a window).

    Stays STEREO. EBU R128 sums the channel energies, so downmixing this window
    to mono would read exactly 3 LU quieter than the same signal measured as
    stereo — while the VO it is compared against is measured on the stereo
    narration file. That mismatch would inject a constant +3 LU into every
    offset and make a correct bed look 3 LU light.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(dest),
        ]
    )
    return dest


async def _render(tmp_path: Path, tag: str, narration: Path, music: Path, bed: MusicBed) -> Path:
    """Render ``narration`` (as the already-"concatenated" video bus) with
    ``music`` under it through the REAL public bed path, master OFF (T1 solves
    for a pre-master bed level; the master's own loudnorm/compressor would
    smear exactly what's being measured)."""
    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(update={"master_enabled": False, "work_dir": str(work)})
    out = work / "final.mp4"
    final, _report = await renderer.overlay_audio_bed(
        narration, out, _DURATION, (music, bed), [], settings=settings
    )
    return final


async def _delivered_bed_lufs(tmp_path: Path, tag: str, final: Path) -> float:
    window = await _extract_window(final, tmp_path / f"{tag}_window.wav", *_WINDOW)
    return await _measure_lufs(window)


async def test_bed_level_tracks_vo_across_three_casts(tmp_path: Path) -> None:
    """Headline drift-kill assertion: three casts at materially different
    loudness (-12 / -18 / -24 LUFS), same music asset, same requested offset.
    The delivered bed's offset from ITS OWN cast's measured VO loudness must
    land within +/-1.5 LU of the requested offset for ALL THREE — i.e. the
    bed-to-VO balance does not drift with cast loudness.

    On the literal-``gain_db`` code path this must FAIL: the bed sits at a
    level independent of VO, so the -12 LUFS cast (quietest offset needed) and
    the -24 LUFS cast (biggest offset needed) get the SAME delivered bed
    level, and only one (if any) of the three casts can land inside the
    tolerance band.
    """
    requested_offset_lu = 10.0
    cast_lufs = [-12.0, -18.0, -24.0]
    music = await make_bed_tone(tmp_path / "music.wav", duration=_DURATION, frequency=8000)

    offsets: list[tuple[float, float]] = []
    for i, target_lufs in enumerate(cast_lufs):
        tag = f"cast{i}"
        narration = await make_premastered_narration_clip(
            tmp_path / f"{tag}_narration.mp4", duration=_DURATION, gap=_GAP, lufs=target_lufs
        )
        vo_lufs = await _measure_lufs(narration)
        bed = MusicBed(
            url="music",
            music_offset_lu=requested_offset_lu,
            fade_in_s=0.1,
            fade_out_s=0.1,
        )
        final = await _render(tmp_path, tag, narration, music, bed)
        bed_lufs = await _delivered_bed_lufs(tmp_path, tag, final)
        delivered_offset = vo_lufs - bed_lufs
        offsets.append((target_lufs, delivered_offset))

    bad = [
        (cast, delivered)
        for cast, delivered in offsets
        if abs(delivered - requested_offset_lu) > _TOLERANCE_LU
    ]
    assert not bad, (
        f"requested {requested_offset_lu} LU offset below VO; measured offsets "
        f"{offsets} — casts outside +/-{_TOLERANCE_LU} LU: {bad}"
    )


async def test_bed_level_normalizes_across_track_loudness(tmp_path: Path) -> None:
    """Two music assets ~14 dB apart in INHERENT loudness (baked into the WAV
    via ``gain_db`` at generation time, standing in for two real tracks that
    were mastered differently), same VO, same requested ``music_offset_lu``.
    Delivered bed levels must land within ~1.5 dB of each other — i.e. T1
    measures the ASSET's own loudness and compensates, rather than applying
    the same raw gain to whatever the track happens to be.

    On the literal-``gain_db`` code path this must FAIL: the same
    ``gain_db`` is applied to both assets, so the ~14 dB baked-in difference
    survives straight through to the delivered mix.
    """
    narration = await make_premastered_narration_clip(
        tmp_path / "narration.mp4", duration=_DURATION, gap=_GAP, lufs=-18.0
    )
    quiet_track = await make_bed_tone(
        tmp_path / "quiet_track.wav", duration=_DURATION, frequency=8000, gain_db=0.0
    )
    loud_track = await make_bed_tone(
        tmp_path / "loud_track.wav", duration=_DURATION, frequency=8000, gain_db=14.0
    )

    delivered = {}
    for tag, track in (("quiet", quiet_track), ("loud", loud_track)):
        bed = MusicBed(url="music", music_offset_lu=10.0, fade_in_s=0.1, fade_out_s=0.1)
        final = await _render(tmp_path, tag, narration, track, bed)
        delivered[tag] = await _delivered_bed_lufs(tmp_path, tag, final)

    spread = abs(delivered["quiet"] - delivered["loud"])
    assert spread <= _TOLERANCE_LU, (
        f"two music assets ~14 dB apart in inherent loudness delivered bed levels "
        f"{delivered} — spread {spread:.2f} LU exceeds {_TOLERANCE_LU} LU; asset "
        "loudness is leaking through instead of being normalized out"
    )


async def test_no_offset_requested_keeps_literal_gain_behaviour(tmp_path: Path) -> None:
    """Back-compat: a bed with NO offset fields (``music_offset_lu=None``, the
    default) must keep the OLD behaviour — a fixed delivered level independent
    of VO loudness, driven purely by ``gain_db``. This is the regression guard
    that T1 does not change the no-offset path: two casts at very different
    loudness, same literal ``gain_db``, must deliver (approximately) the SAME
    bed level, because the literal path never looks at VO loudness at all.
    """
    gain_db = -6.0
    music = await make_bed_tone(tmp_path / "music.wav", duration=_DURATION, frequency=8000)

    delivered = {}
    for tag, target_lufs in (("quiet_vo", -24.0), ("loud_vo", -12.0)):
        narration = await make_premastered_narration_clip(
            tmp_path / f"{tag}_narration.mp4", duration=_DURATION, gap=_GAP, lufs=target_lufs
        )
        bed = MusicBed(url="music", gain_db=gain_db, fade_in_s=0.1, fade_out_s=0.1)
        final = await _render(tmp_path, tag, narration, music, bed)
        delivered[tag] = await _delivered_bed_lufs(tmp_path, tag, final)

    spread = abs(delivered["quiet_vo"] - delivered["loud_vo"])
    assert spread <= _TOLERANCE_LU, (
        f"no music_offset_lu requested, but delivered bed level moved with VO loudness: "
        f"{delivered} (spread {spread:.2f} LU) — the literal gain_db path regressed"
    )
