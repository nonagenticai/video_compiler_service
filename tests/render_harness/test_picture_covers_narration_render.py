"""The picture must cover the narration even when the PROBE is wrong (real ffmpeg).

THE DEFECT, reproduced here rather than asserted: ``_freeze_pad_suffix`` decides
whether to clone the last frame from ``clip_dur``, a PROBED number
(``_playable_clip_duration`` → ``_ffprobe_duration``). PR #202/#211 made that
number as picture-specific as a probe can be, but a probe still reads a HEADER,
and a header can outlast the frames the file actually holds — a truncated
download, a Chromium/Matroska capture, a fragmented writer. On that shape
``clip_dur >= duration``, the clip reads as "already fills its narration", NO
clone is emitted, and the output-side ``-t`` keeps the segment at the planned
length — so the delivered container duration AGREES with the plan and the picture
simply stops. Measured below: a clip that PROBES 12.0s and DECODES 2.84s
delivered 2.84s of picture under 12.0s of audio, a 9.16s gap, which is the
``av_coverage`` failure shape of orchestrator run
``01a08b19-e72e-71aa-8dfe-8d05e8d42a41`` (lesson-motion-slides, 7.549s).

The fix removes the probe from the COVERAGE decision: a ``-t``-bounded segment
carries a terminal ``tpad=stop_mode=clone:stop_duration=<computed>`` (see
``renderer._bounded_cover_pad``) that holds the last decoded frame until the
``-t`` terminates the stream. It cannot lengthen a segment (the bound is
unchanged) and cannot add a frame to a picture that already reaches that bound —
``test_a_picture_that_already_covers_is_not_lengthened`` is the false-positive
guard that pins exactly that, and it is what protects every master that works
today.

⚠️ The clone's ``stop_duration`` used to be unconditional ``stop=-1`` (infinite,
relying on the ``-t`` alone to terminate it). That livelocked a segment whose
video under-runs its target by less than one output frame AND carries a
second, finite input (narration audio) in the same filtergraph: see
``tests/render_harness/test_freeze_pad_bounded_livelock.py``. The clone is now
always a FINITE, computed duration — never ``-1`` — so it can only ever close a
bounded gap, correctly even when the gap itself was computed from the lying
header this file exercises.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from src.api.schemas import Shot
from src.config import Settings, get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="real-ffmpeg picture-coverage round trip requires ffmpeg + ffprobe",
    ),
]

_W, _H, _FPS = 320, 240, 25
_SMALL = {"render_width": _W, "render_height": _H, "render_fps": _FPS}


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update={**_SMALL, **overrides})


async def _stream_seconds(path: Path) -> tuple[float, float]:
    """The delivered ``(video_seconds, audio_seconds)`` — PER STREAM.

    Deliberately the production measurement (``mastering.av_stream_seconds``),
    not a hand-rolled ffprobe: the gate this test defends reads exactly these two
    numbers, so a test that measured them differently could pass while the gate
    still failed the master.
    """
    video, audio = await mastering.av_stream_seconds(path)
    assert video is not None, f"no measurable video stream in {path}"
    assert audio is not None, f"no measurable audio stream in {path}"
    return video, audio


async def _probe_json(path: Path, entries: str) -> dict:
    out = await renderer._run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)]
    )
    return json.loads(out)


async def _decoded_video_seconds(path: Path) -> float:
    """How much picture the file really HANDS OVER, measured by decoding it.

    A full video-only transcode to a fresh container, whose duration is then the
    length of what actually came out of the decoder. Deliberately independent of
    every header the renderer's own probe reads — that independence is the whole
    point: it is what distinguishes "the clip CLAIMS 12s" from "the clip HANDS
    YOU 2.8s", and a header-derived measure could not tell those apart.
    """
    decoded = path.with_name(path.stem + "-decoded.mp4")
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(decoded),
        ]
    )
    payload = await _probe_json(decoded, "stream=duration:format=duration")
    return renderer._duration_field(payload["streams"][0]) or renderer._duration_field(
        payload.get("format")
    )


async def _narration(tmp_path: Path, seconds: float, name: str = "nar.wav") -> Path:
    """A real narration-shaped WAV (48k stereo), the shape Kokoro delivers."""
    dest = tmp_path / name
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(dest),
        ]
    )
    return dest


async def _clip(tmp_path: Path, seconds: float, name: str, *, rate: int = _FPS) -> Path:
    dest = tmp_path / name
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_W}x{_H}:rate={rate}:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dest),
        ]
    )
    return dest


async def _truncated_clip(tmp_path: Path, *, claims: float, keep_fraction: float) -> Path:
    """A clip whose HEADER claims ``claims`` seconds and whose DATA stops early.

    ``+faststart`` puts the moov (with the full duration) at the front, so the
    truncated file probes as the whole thing and decodes a fraction of it — the
    partially-delivered-asset shape, and the same shape a Matroska capture whose
    header outlasts its frames presents.
    """
    whole = await _clip(tmp_path, claims, "whole.mp4")
    data = whole.read_bytes()
    dest = tmp_path / "truncated.mp4"
    dest.write_bytes(data[: int(len(data) * keep_fraction)])
    return dest


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


async def test_a_clip_that_probes_longer_than_it_decodes_still_covers_narration(
    tmp_path: Path,
) -> None:
    """A 12s-probing / ~3s-decoding clip under a 12s narration delivers 12s of
    PICTURE, not 3s of picture under 12s of audio."""
    narration = await _narration(tmp_path, 12.0)
    clip = await _truncated_clip(tmp_path, claims=12.0, keep_fraction=1 / 3)

    # Preconditions, asserted rather than assumed — without BOTH of these the test
    # would pass for the wrong reason (an honest short clip freeze-pads already).
    probed = await renderer._ffprobe_duration(clip)
    decoded = await _decoded_video_seconds(clip)
    assert probed >= 11.9, f"the header must still claim ~12s, got {probed}"
    assert decoded < 6.0, f"the file must decode far less than it claims, got {decoded}"

    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(clip), audio_url=str(narration)),
            out,
            _settings(),
            tmp_path,
            http,
        )

    video, audio = await _stream_seconds(out)
    assert audio == pytest.approx(12.0, abs=0.25)
    assert video >= audio - mastering.AV_COVERAGE_TOLERANCE_SECONDS, (
        f"picture {video:.3f}s under {audio:.3f}s of audio"
    )


async def test_the_gate_no_longer_fires_but_would_have_on_the_decoded_length(
    tmp_path: Path,
) -> None:
    """The 0.5s tolerance boundary, anchored at both ends on ONE real render.

    The delivered master clears ``_av_coverage_violation``; the SAME master's
    audio measured against the clip's DECODED picture length (what shipped before
    the terminal clone) does not. So the gate is still able to fire — this is not
    a test that passes because the check was loosened.
    """
    narration = await _narration(tmp_path, 12.0)
    clip = await _truncated_clip(tmp_path, claims=12.0, keep_fraction=1 / 3)
    decoded = await _decoded_video_seconds(clip)

    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="a", seconds=1.0, video_url=str(clip), audio_url=str(narration)),
            out,
            _settings(),
            tmp_path,
            http,
        )
    video, audio = await _stream_seconds(out)

    assert mastering._av_coverage_violation(mastering._av_gap_seconds(video, audio)) is None
    would_have = mastering._av_coverage_violation(mastering._av_gap_seconds(decoded, audio))
    assert would_have is not None, "the un-padded picture must still be a violation"
    assert would_have.stage == "av_coverage"
    assert would_have.value is not None
    assert would_have.value > mastering.AV_COVERAGE_TOLERANCE_SECONDS


async def test_a_vfr_source_covers_its_narration(tmp_path: Path) -> None:
    """The same defect on a VFR source, and the frame COUNT is no help here.

    A dense burst then a sparse dwell — the capture shape — whose data is then cut
    short. Its header claims 4.03s, it decodes ~0.5s, and the narration is 3.0s:
    long enough that the probed 4.03s reads as "already covers" (so the planned
    freeze pad is not emitted at all) and short enough that the real picture does
    not reach it.

    The precondition asserts ``nb_frames / r_frame_rate`` is wrong by several
    seconds on this file, so nothing here can be satisfied by deriving a duration
    from the frame count."""
    narration = await _narration(tmp_path, 3.0)
    vfr = tmp_path / "vfr.mp4"
    # A DENSE burst (60 fps for 1s) followed by a SPARSE dwell (1 fps for 5s),
    # stream-copied into one container so the timestamps survive intact. That is a
    # capture's real shape — motion, then a held slide — and it makes the frame
    # COUNT say ~1.1s while the timeline says 6s.
    dense = await _clip(tmp_path, 1.0, "dense.mp4", rate=60)
    sparse = await _clip(tmp_path, 5.0, "sparse.mp4", rate=1)
    listing = tmp_path / "vfr.concat.txt"
    listing.write_text(f"file '{dense}'\nfile '{sparse}'\n")
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(vfr),
        ]
    )
    # Precondition: nb_frames/r_frame_rate must be badly wrong for this file, or
    # the "do not derive from the frame count" claim is untested here.
    meta = await _probe_json(vfr, "stream=nb_frames,r_frame_rate:format=duration")
    stream = meta["streams"][0]
    frames = float(stream["nb_frames"])
    num, den = (float(x) for x in str(stream["r_frame_rate"]).split("/"))
    naive = frames / (num / den)
    real = renderer._duration_field(meta.get("format"))
    assert abs(naive - real) > 1.0, f"nb_frames/fps={naive} vs real={real} — not VFR enough"

    cut = tmp_path / "vfr-cut.mp4"
    cut.write_bytes(vfr.read_bytes()[: int(vfr.stat().st_size * 0.4)])
    # Preconditions: the header must still out-claim the narration (so the planned
    # pad is NOT emitted) and the data must stop well before it.
    probed = await renderer._ffprobe_duration(cut)
    decoded = await _decoded_video_seconds(cut)
    assert probed > 3.0, f"the header must out-claim the 3.0s narration, got {probed}"
    assert decoded < 2.5, f"the file must decode well short of the narration, got {decoded}"

    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="v", seconds=1.0, video_url=str(cut), audio_url=str(narration)),
            out,
            _settings(),
            tmp_path,
            http,
        )
    video, audio = await _stream_seconds(out)
    assert video >= audio - mastering.AV_COVERAGE_TOLERANCE_SECONDS, (
        f"picture {video:.3f}s under {audio:.3f}s of audio"
    )


async def test_a_still_slide_covers_a_narration_longer_than_its_planned_seconds(
    tmp_path: Path,
) -> None:
    """The still/slide path: a shot planned at 4s whose narration runs 12s must
    hold the still for the NARRATION, not for the plan."""
    narration = await _narration(tmp_path, 12.0)
    still = tmp_path / "slide.png"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_W}x{_H}:rate=1:duration=1",
            "-frames:v",
            "1",
            str(still),
        ]
    )

    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="s", seconds=4.0, image_url=str(still), audio_url=str(narration)),
            out,
            _settings(),
            tmp_path,
            http,
        )
    video, audio = await _stream_seconds(out)
    assert audio == pytest.approx(12.0, abs=0.25)
    assert video >= audio - mastering.AV_COVERAGE_TOLERANCE_SECONDS


# ---------------------------------------------------------------------------
# The false-positive guard — this is what protects every working master
# ---------------------------------------------------------------------------


async def test_a_picture_that_already_covers_is_not_lengthened(tmp_path: Path) -> None:
    """A clip that outlasts its narration is delivered at the NARRATION length,
    frame for frame — the terminal clone must not add a single frame to a segment
    whose picture already reaches the bound."""
    narration = await _narration(tmp_path, 6.0)
    clip = await _clip(tmp_path, 10.0, "long.mp4")

    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        nominal = await renderer.render_shot(
            Shot(id="c", seconds=1.0, video_url=str(clip), audio_url=str(narration)),
            out,
            _settings(),
            tmp_path,
            http,
        )
    assert nominal == pytest.approx(6.0, abs=0.05)
    video, audio = await _stream_seconds(out)
    assert video == pytest.approx(6.0, abs=0.05)
    assert audio == pytest.approx(6.0, abs=0.25)
    # Frame-exact: 6.0s at 25 fps is 150 frames, and a clone that leaked past the
    # bound would show up here before it showed up in the duration.
    meta = await _probe_json(out, "stream=nb_frames")
    frames = int(meta["streams"][0]["nb_frames"])
    assert frames == round(6.0 * _FPS)


async def test_an_unbounded_own_audio_clip_gets_no_terminal_clone(tmp_path: Path) -> None:
    """The own-audio clip branch carries NO output ``-t`` (it plays at its natural
    remaining length), so nothing would ever terminate a ``stop=-1`` clone there —
    it must not be emitted, and the segment must still deliver the clip's own
    length rather than running away.

    Both halves are asserted: the pure gate (no clone without a bound) and the
    delivered file (which is what would hang if the gate were wrong).
    """
    settings = _settings()
    assert _freeze_pad_no_clone(settings), "an unbounded segment must emit no terminal clone"

    clip = tmp_path / "own.mp4"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={_W}x{_H}:rate={_FPS}:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=200:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(clip),
        ]
    )
    out = tmp_path / "out.mp4"
    async with httpx.AsyncClient() as http:
        await renderer.render_shot(
            Shot(id="o", seconds=1.0, video_url=str(clip)),
            out,
            settings,
            tmp_path,
            http,
        )
    video, audio = await _stream_seconds(out)
    assert video == pytest.approx(5.0, abs=0.2)
    assert video >= audio - mastering.AV_COVERAGE_TOLERANCE_SECONDS


def _freeze_pad_no_clone(settings: Settings) -> bool:
    """True when the UNBOUNDED suffix carries no EXTRA terminal clone, for both
    the already-covers case (no clone at all: the "covers" case is out of the
    narration-driven range and short-circuits to ``""``) and the known-under-run
    case (exactly one ``tpad`` — the drift's own sized clone that fills the
    picture to ``duration`` — never a second, terminal one appended after it;
    only a ``bounded=True`` call appends that second clone)."""
    covers = renderer._freeze_pad_suffix(True, 10.0, 6.0, settings, bounded=False)
    under = renderer._freeze_pad_suffix(True, 4.0, 9.0, settings, bounded=False)
    return covers == "" and under.count("tpad=stop_mode=clone") == 1
