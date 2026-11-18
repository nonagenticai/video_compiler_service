"""A delivered master's audio track must have NO timestamp holes (real ffmpeg, no mocks).

THE DEFECT, measured on production master ``01a10e19`` (showcase_v1, ffmpeg 7.1.5):
the AAC track had no samples for 23.2107-23.2160 s (5.4 ms) and 34.7360-34.8252 s
(89.1 ms), and the picture carried two stretched frames (``avg_frame_rate`` 29.987
against ``r_frame_rate`` 30). Three symptoms, two causes:

* PER CUT. A shot's ``-t`` was off the output frame grid, so its audio (cut to the
  sample) and its video (cut to the nearest FRAME) ended at different instants.
  ``concat_shots`` stream-copies and starts each file at the previous file's longest
  stream: audio shorter than video leaves a hole in the audio timestamps; audio
  longer than video stretches the last frame before the cut.
* PER FILM. ``loudnorm`` in DYNAMIC mode consumes whole 100 ms frames and drops the
  missing part of a partial last one, jumping its output timestamps ~2.9 s before
  the end of the film.

⛔ A LOCAL PASS ON FFMPEG 4.4 PROVES NOTHING. 4.4's filters restamp their output, so
neither hole reaches the encoder there; both reproduce on 7.1.x, which is what CI and
production run. MUTATION-PROVED 2026-10-06 on ffmpeg 7.1.1: with the frame-grid snap
and the loudnorm frame alignment both reverted, the bed and no-bed cases go RED on the
hole count AND on the frame cadence; restored, GREEN.

⛔ WHY THE HOLE IS READ FROM PACKET INTERVALS. An MP4 cannot store a gap — the muxer
stretches the duration of the packet before it — so ``pts[n] + duration[n] ==
pts[n+1]`` holds across a hole and reports a clean stream. That check is kept below
because it is the obvious one, but ``mastering.audio_pts_gaps`` (interval against the
codec's frame size) is the one that can fail.

NON-VACUITY. Hole 2 only exists when loudnorm runs DYNAMIC, and ffmpeg decides that
from the measured bus. The narration fixture is built to force it (quiet body, loud
transients: the linear gain would overshoot the true-peak target), and every case
ASSERTS the run was dynamic — a fixture that drifted into linear mode fails loudly
instead of passing for the wrong reason.
"""

from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import MusicBed, Shot, Transition
from src.config import Settings, get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="audio timestamp round trip requires ffmpeg + ffprobe",
    ),
]

_SR = 48000
_AAC_FRAME = 1024
_W, _H = 320, 240
# 25 fps, not the default 30, and deliberately: a 25 fps frame is 40 ms, so every
# on-grid segment length is a whole number of milliseconds. ffmpeg 4.4's concat
# demuxer advances by the MP4 header duration, which is stored in MILLISECONDS, and
# so stretches the frame before a cut by up to 0.67 ms whenever a segment is N/30 s
# with N not a multiple of 3. That is a 4.4 artifact (7.1 advances by the stream
# durations; production master 01a10e19 has no such interval after its 380-frame
# shot) and it would make the cadence assertion below fail locally for a reason
# that has nothing to do with the defect.
_FPS = 25
# Narration lengths, all OFF the 25 fps grid, on both sides of the nearest frame:
#   2.509570 s = 62.74 frames -> 63 video frames: audio SHORTER (audio hole, 10.4 ms)
#   3.661305 s = 91.53 frames -> 92 video frames: audio SHORTER (audio hole, 18.7 ms)
#   1.450000 s = 36.25 frames -> 36 video frames: audio LONGER  (stretched frame)
# The first two are the production run's own shots 1 and 2, minus a whole 8 s / 9 s.
# Snapped they are 63 + 92 + 37 frames, + the 50-frame card = 9.68 s: NOT a multiple
# of 100 ms, so loudnorm still sees a partial last frame once the cuts are clean and
# the frame alignment is exercised on its own.
_NARRATION_SECONDS = (2.50957, 3.661305, 1.45)
# The un-narrated outro title card: only the bed plays under it.
_CARD_SECONDS = 2.0


def _settings(work: Path) -> Settings:
    return get_settings().model_copy(
        update={
            "render_width": _W,
            "render_height": _H,
            "render_fps": _FPS,
            "master_enabled": True,
            "work_dir": str(work),
        }
    )


async def _still(path: Path) -> Path:
    await renderer._run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=blue:s={_W}x{_H}"),
            *("-frames:v", "1", str(path)),
        ]
    )
    return path


async def _narration(path: Path, seconds: float) -> Path:
    """Quiet 440 Hz body with a 20 ms near-full-scale transient every 0.7 s.

    The crest factor is the point: integrated loudness sits near -30 LUFS while the
    true peak sits near -3 dBTP, so the constant gain that reaches ``master_lufs``
    would overshoot ``master_tp`` and ffmpeg demotes ``linear=true`` to DYNAMIC —
    the production master's mode (``master_normalization_type: dynamic``)."""
    expr = "0.03*sin(2*PI*440*t)+0.7*sin(2*PI*1320*t)*lt(mod(t\\,0.7)\\,0.02)"
    await renderer._run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi"),
            *("-i", f"aevalsrc={expr}:s={_SR}:d={seconds}", "-c:a", "pcm_s16le", str(path)),
        ]
    )
    return path


async def _bed(path: Path) -> Path:
    await renderer._run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi"),
            *("-i", "sine=frequency=220:duration=30:sample_rate=44100", "-ac", "2", str(path)),
        ]
    )
    return path


async def _render_and_concat(work: Path, settings: Settings) -> tuple[Path, float]:
    """The REAL ``render_shot`` for three off-grid narrated shots and an un-narrated
    card, then the REAL ``concat_shots``. Returns the film and the summed durations
    ``render_shot`` reported (what ``compile_shots`` passes on as ``total_seconds``)."""
    still = await _still(work / "still.png")
    shots = [
        Shot(
            id=f"s{i}",
            seconds=seconds,
            image_url=str(still),
            audio_url=str(await _narration(work / f"vo{i}.wav", seconds)),
        )
        for i, seconds in enumerate(_NARRATION_SECONDS)
    ]
    shots.append(Shot(id="outro", seconds=_CARD_SECONDS, description="The End"))
    segments: list[Path] = []
    total = 0.0
    async with httpx.AsyncClient() as http:
        for i, shot in enumerate(shots):
            seg = work / f"seg-{i:04d}{renderer._SEGMENT_EXT}"
            total += await renderer.render_shot(shot, seg, settings, work, http)
            segments.append(seg)
    film = work / f"compiled{renderer._SEGMENT_EXT}"
    await renderer.concat_shots(segments, film)
    return film, total


async def _probe(path: Path) -> dict[str, list[dict[str, str]]]:
    out = await renderer._run(
        [
            *("ffprobe", "-v", "error", "-show_entries"),
            "stream=codec_type,duration,r_frame_rate,avg_frame_rate,time_base"
            ":packet=stream_index,pts,duration",
            *("-of", "json", str(path)),
        ]
    )
    return json.loads(out)


def _stream(probe: dict[str, list[dict[str, str]]], codec_type: str) -> tuple[int, dict[str, str]]:
    for index, stream in enumerate(probe["streams"]):
        if stream["codec_type"] == codec_type:
            return index, stream
    raise AssertionError(f"no {codec_type} stream")


def _packets(probe: dict[str, list[dict[str, str]]], index: int) -> list[tuple[int, int]]:
    return sorted(
        (int(p["pts"]), int(p["duration"]))
        for p in probe["packets"]
        if int(p["stream_index"]) == index
    )


async def _decoded_samples(path: Path) -> int:
    raw = path.with_name(path.name + ".raw")
    await renderer._run(
        [
            *("ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", "0:a:0"),
            *("-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(_SR), str(raw)),
        ]
    )
    count = raw.stat().st_size // 2
    raw.unlink(missing_ok=True)
    return count


async def _assert_master_has_no_holes(master: Path) -> None:
    probe = await _probe(master)

    # --- audio: no holes in the packet timeline -----------------------------------
    a_index, a_stream = _stream(probe, "audio")
    assert a_stream["time_base"] == f"1/{_SR}", a_stream
    a_packets = _packets(probe, a_index)
    count, total_ms = mastering.audio_pts_gaps([pts for pts, _ in a_packets], _SR)
    assert count == 0, (
        f"{count} hole(s) in the audio packet timeline, {total_ms} ms with no samples: "
        + ", ".join(
            f"{(b - a - _AAC_FRAME) / _SR * 1000:.3f} ms at {(a + _AAC_FRAME) / _SR:.4f}s"
            for (a, _), (b, _) in itertools.pairwise(a_packets)
            if b - a - _AAC_FRAME > 1
        )
    )
    # The obvious check. It CANNOT see an MP4 hole (see the module docstring) and is
    # here so that a real pts jump — a different container, a future muxer — is not
    # missed by the interval reading alone.
    for (pts, duration), (next_pts, _) in itertools.pairwise(a_packets):
        assert abs(next_pts - (pts + duration)) <= 1, (pts, duration, next_pts)
    # The production measurement agrees with the one taken here.
    assert await mastering.audio_pts_gap_report(master) == (0, 0.0)

    # --- audio: what DECODES is what the stream claims -------------------------------
    # A hole is time the header counts and the decoder never produces: the production
    # master decoded 94.5 ms (4.4 AAC frames) short of its own stream duration.
    decoded = await _decoded_samples(master)
    claimed = round(float(a_stream["duration"]) * _SR)
    assert abs(decoded - claimed) <= _AAC_FRAME, (
        f"decoded {decoded} samples against a stream duration of {claimed} "
        f"({(claimed - decoded) / _SR * 1000:.2f} ms missing)"
    )

    # --- video: every frame interval equal ------------------------------------------
    v_index, v_stream = _stream(probe, "video")
    v_pts = [pts for pts, _ in _packets(probe, v_index)]
    intervals = {b - a for a, b in itertools.pairwise(v_pts)}
    assert len(intervals) == 1, (
        f"frame intervals are not uniform: {sorted(intervals)} (time_base "
        f"{v_stream['time_base']}) — a segment's audio outlasted its video and the "
        "concat stretched the last frame before the cut"
    )
    assert v_stream["avg_frame_rate"] == v_stream["r_frame_rate"] == f"{_FPS}/1", v_stream


def _assert_ran_dynamic() -> None:
    assert renderer.get_master_normalization_type() == "dynamic", (
        "the fixture no longer drives loudnorm into DYNAMIC mode "
        f"(got {renderer.get_master_normalization_type()!r}); the partial-last-frame "
        "hole only exists there, so this run could not have detected it"
    )


async def test_segments_end_with_audio_and_video_the_same_length(tmp_path: Path) -> None:
    """Every rendered segment's audio is exactly as long as its picture, and
    ``render_shot`` reports that length — the property the concat relies on."""
    settings = _settings(tmp_path)
    await _render_and_concat(tmp_path, settings)
    for i, asked in enumerate((*_NARRATION_SECONDS, _CARD_SECONDS)):
        probe = await _probe(tmp_path / f"seg-{i:04d}{renderer._SEGMENT_EXT}")
        video = float(_stream(probe, "video")[1]["duration"])
        audio = float(_stream(probe, "audio")[1]["duration"])
        assert abs(audio - video) <= 1.5 / _SR, f"segment {i}: audio {audio}s, video {video}s"
        # Never shorter than asked (narration is not cut), never a whole frame longer.
        assert -1e-6 <= video - asked < 1 / _FPS, f"segment {i}: asked {asked}s, got {video}s"


async def test_bed_master_has_no_audio_holes_and_a_uniform_cadence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film, total = await _render_and_concat(tmp_path, settings)
    bed = MusicBed(url="bed", gain_db=-18.0, fade_in_s=0.2, fade_out_s=1.0)
    renderer.reset_master_normalization_type()
    master, _report = await renderer.overlay_audio_bed(
        film, tmp_path / "final.mp4", total, (await _bed(tmp_path / "bed.wav"), bed), [], settings
    )
    _assert_ran_dynamic()
    await _assert_master_has_no_holes(master)
    # No film gets longer or shorter: the aligned pad is trimmed straight back off.
    audio = float(_stream(await _probe(master), "audio")[1]["duration"])
    assert abs(audio - total) <= 0.001, f"mastered audio {audio}s for a {total}s film"


async def test_bedless_master_has_no_audio_holes_and_a_uniform_cadence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    film, total = await _render_and_concat(tmp_path, settings)
    renderer.reset_master_normalization_type()
    master = await renderer.master_audio(film, tmp_path / "mastered.mp4", settings)
    _assert_ran_dynamic()
    await _assert_master_has_no_holes(master)
    audio = float(_stream(await _probe(master), "audio")[1]["duration"])
    assert abs(audio - total) <= 0.001, f"mastered audio {audio}s for a {total}s film"


# --- the two paths the cases above do not take ---------------------------------------
#
# Both are measured on the FILM the join produces (lossless ALAC, before any master),
# so nothing downstream can restamp a hole away, and on each joined FILE's two stream
# lengths — the cause itself, which does not depend on the ffmpeg version.

# (source fps, source frames, own-audio seconds): recordings whose picture and sound
# are different lengths and neither is on the 25 fps output grid.
#   30 fps x 77 = 2.566667 s = 64.17 output frames, audio 2.59 s  (audio LONGER)
#   24 fps x 50 = 2.083333 s = 52.08 output frames, audio 2.05 s  (audio SHORTER)
#   30 fps x 43 = 1.433333 s = 35.83 output frames, audio 1.40 s  (audio SHORTER)
_OWN_AUDIO_CLIPS = ((30, 77, 2.59), (24, 50, 2.05), (30, 43, 1.40))
# Off the 25 fps grid: 8.25 frames.
_XFADE_SECONDS = 0.33


async def _recording(path: Path, fps: int, frames: int, audio_seconds: float) -> Path:
    """A clip WITH its own audio whose two streams are deliberately different lengths."""
    await renderer._run(
        [
            *("ffmpeg", "-y", "-v", "error"),
            *(
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={_W}x{_H}:rate={fps}:duration={frames / fps:.6f}",
            ),
            *(
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=330:sample_rate={_SR}:duration={audio_seconds}",
            ),
            *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", str(path)),
        ]
    )
    # The fixture IS the premise: a recording whose streams disagree, off the grid.
    video, audio = await _stream_seconds(path)
    assert abs(video - frames / fps) < 1e-3, (video, frames / fps)
    assert abs(audio - audio_seconds) < 1e-3, (audio, audio_seconds)
    return path


async def _stream_seconds(path: Path) -> tuple[float, float]:
    probe = await _probe(path)
    return (
        float(_stream(probe, "video")[1]["duration"]),
        float(_stream(probe, "audio")[1]["duration"]),
    )


async def _assert_film_has_no_holes(film: Path) -> None:
    """The stream-copied film: no interval of the audio packet timeline longer than
    a codec frame, every picture interval equal, both streams the same length."""
    assert await mastering.audio_pts_gap_report(film) == (0, 0.0)
    probe = await _probe(film)
    v_index, v_stream = _stream(probe, "video")
    intervals = {b - a for a, b in itertools.pairwise(pts for pts, _ in _packets(probe, v_index))}
    assert len(intervals) == 1, f"frame intervals are not uniform: {sorted(intervals)}"
    assert v_stream["avg_frame_rate"] == v_stream["r_frame_rate"] == f"{_FPS}/1", v_stream
    video, audio = await _stream_seconds(film)
    assert abs(audio - video) < 1 / _FPS, f"film: audio {audio}s, video {video}s"


async def test_own_audio_clips_join_without_audio_holes(tmp_path: Path) -> None:
    """Clips that play their OWN audio (no narration): each segment's two streams end
    together on the frame grid, and the film they are stream-copied into has no hole."""
    settings = _settings(tmp_path)
    segments: list[Path] = []
    total = 0.0
    async with httpx.AsyncClient() as http:
        for i, (fps, frames, audio_seconds) in enumerate(_OWN_AUDIO_CLIPS):
            clip = await _recording(tmp_path / f"rec{i}.mp4", fps, frames, audio_seconds)
            seg = tmp_path / f"own-{i:04d}{renderer._SEGMENT_EXT}"
            # ``seconds`` is the placeholder such a shot carries; it must not cut the clip.
            reported = await renderer.render_shot(
                Shot(id=f"rec{i}", seconds=1.0, video_url=str(clip)), seg, settings, tmp_path, http
            )
            video, audio = await _stream_seconds(seg)
            assert abs(audio - video) <= 1.5 / _SR, f"segment {i}: audio {audio}s, video {video}s"
            assert abs(reported - video) <= 1.5 / _SR, f"segment {i}: reported {reported}s"
            on_grid = video * _FPS
            assert abs(on_grid - round(on_grid)) < 1e-3, f"segment {i}: {video}s is off the grid"
            # The recording is played whole: less than one output frame is given up.
            assert 0 <= frames / fps - video < 1 / _FPS + 1e-6, (frames / fps, video)
            segments.append(seg)
            total += reported
    film = tmp_path / f"own-film{renderer._SEGMENT_EXT}"
    await renderer.concat_shots(segments, film)
    await _assert_film_has_no_holes(film)
    assert abs((await _stream_seconds(film))[0] - total) <= 0.001


async def _narrated_segments(work: Path, settings: Settings) -> list[Path]:
    await _render_and_concat(work, settings)
    return [work / f"seg-{i:04d}{renderer._SEGMENT_EXT}" for i in range(4)]


async def test_crossfaded_groups_join_without_audio_holes(tmp_path: Path) -> None:
    """Two crossfaded pairs with a hard cut between them: each group file is
    re-encoded by ``_render_xfade_group`` and the two are then stream-copied together.
    An overlap off the frame grid used to leave each group's picture longer than its
    audio, which is a hole at the hard cut."""
    settings = _settings(tmp_path)
    segments = await _narrated_segments(tmp_path, settings)
    fade = Transition(type="fade", duration=_XFADE_SECONDS)
    film = tmp_path / f"xf-film{renderer._SEGMENT_EXT}"
    total = await renderer.xfade_concat_shots(segments, [None, fade, None, fade], film, settings)
    for gi in (0, 1):
        video, audio = await _stream_seconds(
            tmp_path / f"xf-film-xf{gi:03d}{renderer._SEGMENT_EXT}"
        )
        assert abs(audio - video) <= 1.5 / _SR, f"group {gi}: audio {audio}s, video {video}s"
    await _assert_film_has_no_holes(film)
    assert abs((await _stream_seconds(film))[0] - total) <= 0.001


async def test_a_single_crossfaded_run_keeps_audio_and_video_the_same_length(
    tmp_path: Path,
) -> None:
    """One group covering the whole film renders straight to the output (no concat
    after it): its two streams must agree with each other and with what is reported."""
    settings = _settings(tmp_path)
    segments = await _narrated_segments(tmp_path, settings)
    fade = Transition(type="fade", duration=_XFADE_SECONDS)
    film = tmp_path / f"xf-single{renderer._SEGMENT_EXT}"
    total = await renderer.xfade_concat_shots(segments, [None, fade, fade, fade], film, settings)
    video, audio = await _stream_seconds(film)
    assert abs(audio - video) <= 1.5 / _SR, f"audio {audio}s, video {video}s"
    assert abs(video - total) <= 0.001, f"reported {total}s, delivered {video}s"
    # Three overlaps of ~0.33 s come off the 9.68 s the four segments add up to.
    assert abs(total - (9.68 - 3 * _XFADE_SECONDS)) < 3 / _FPS
