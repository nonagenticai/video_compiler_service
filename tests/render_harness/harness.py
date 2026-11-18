"""Real-ffmpeg render harness — probe helpers + a tiny lavfi clip generator.

WS-0 of the compiler-verification plan: future compiler changes (per-clip
speed, xfade transitions) must be checked against ACTUAL rendered output, not
just the filtergraph command string. These helpers let a test synthesise a
throwaway real clip and then interrogate the encoded container with ffprobe.

Every ffmpeg/ffprobe invocation reuses ``renderer._run`` — the same
semaphore-gated ``asyncio.create_subprocess_exec`` wrapper the production
renderer uses — so the harness exercises the exact subprocess path shipped in
the pod (and honours the per-loop ffmpeg concurrency gate). All helpers are
``async`` and must be awaited inside the running test event loop
(``asyncio_mode = "auto"`` in pyproject makes the test functions async).

These functions are deliberately kept in a NON-test module so later test
modules (speed / xfade) can import them: ``from tests.render_harness import
make_test_clip, probe_duration, probe_video_frame_count, probe_has_audio``.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services import renderer

# ---------------------------------------------------------------------------
# Runner capability probe
# ---------------------------------------------------------------------------
#
# The render-harness modules guard themselves with
# ``skipif(shutil.which("ffmpeg") is None)``. That checks the binary EXISTS; it
# does not check the binary can do the work, and the two are not the same thing.
#
# Measured 2026-08-26 on a macOS dev box: Homebrew's ffmpeg is present and on
# PATH, but is built without libass/libfreetype —
#
#     ffmpeg -filters | grep -E '^ . (ass|subtitles|drawtext) '   ->  nothing
#     ffmpeg -version | grep -oE -- '--enable-lib(ass|freetype)'  ->  nothing
#
# so every caption/callout test failed with a bare ``ffmpeg exit=8: ... Filter
# not found``, which reads like a code defect and is not one. Seven tests failed
# that way, and because the pre-push gate runs the suite with ``-x`` it blocked
# ALL pushes from that machine regardless of the diff.
#
# ``requires_filters`` closes the presence-vs-capability gap: a runner missing a
# filter now SKIPS with a reason that names the filter, instead of failing with
# an ffmpeg exit code. A runner that HAS the filter is unaffected and still runs
# the test — this weakens no coverage in CI.


@functools.cache
def available_filters() -> frozenset[str]:
    """Filter names this runner's ffmpeg actually exposes (empty if absent)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return frozenset()
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell, resolved exe
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except OSError, subprocess.SubprocessError:  # pragma: no cover — defensive
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Rows look like: " TSC film_grain      V->V       Add film grain."
        if len(parts) >= 2 and not line.startswith("Filters:"):
            names.add(parts[1])
    return frozenset(names)


def requires_filters(*names: str) -> pytest.MarkDecorator:
    """Skip when this runner's ffmpeg lacks any of ``names`` — but NEVER on CI.

    ⚠️ THE SKIP IS DELIBERATELY DISABLED UNDER ``CI``. A guard that silently
    skips is how real coverage quietly disappears: CI installs ffmpeg via the
    archetype's ``extra_apt: ffmpeg`` (Debian's build, which carries libass and
    libfreetype), so these filters MUST be present there. If they ever are not,
    that is a runner-image regression and it should go RED — loudly, with the
    ffmpeg error — instead of turning green with 20 quiet skips that nobody
    reads. The skip exists only to stop a dev box's capability gap from being
    misread as a code defect (and, since the pre-push gate runs with ``-x``,
    from blocking every push off that machine).
    """
    missing = sorted(n for n in names if n not in available_filters())
    on_ci = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
    return pytest.mark.skipif(
        bool(missing) and not on_ci,
        reason=(
            f"ffmpeg on this runner lacks filter(s) {', '.join(missing)} "
            "(built without libass/libfreetype?) — capability, not code. "
            "Not skipped under CI: there a missing filter is a real regression."
        ),
    )


async def make_test_clip(
    path: Path,
    *,
    duration: float = 1.0,
    size: str = "320x240",
    fps: int = 25,
    frequency: int = 440,
) -> Path:
    """Synthesise a tiny real MP4 (H.264 video + AAC audio) via ffmpeg lavfi.

    Uses the ``testsrc`` video source and a ``sine`` audio tone so no external
    asset is needed. ``pix_fmt yuv420p`` keeps the output broadly decodable.
    Writes to ``path`` (must live under a pytest ``tmp_path`` — never the repo)
    and returns it for chaining.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={size}:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


async def make_premastered_narration_clip(
    path: Path,
    *,
    duration: float = 5.0,
    gap: tuple[float, float] = (2.0, 2.8),
    size: str = "320x240",
    fps: int = 25,
    lufs: float = -18.0,
    band: tuple[int, int] = (300, 3400),
    sample_rate: int = 48000,
) -> Path:
    """Synthesise a real MP4 whose audio stands in for PRODUCTION VOICEOVER.

    A sine tone is NOT a usable stand-in for narration when the thing under test
    is a sidechain key: a tone's detected level bears no relation to speech's, so
    the compressor opens (or doesn't) for reasons the production mix never sees.
    This builder mirrors the actual production audio invariant instead —

    * **speech-band pink noise** (``anoisesrc=c=pink`` band-limited to
      ``band``, 300-3400 Hz), whose spectrum and crest factor read like speech to
      the sidechain detector, and
    * **premastered to ``lufs`` (-18 LUFS)** with ``loudnorm``, which is what
      EVERY voiceover on the bus already is: TTS premasters to -18 LUFS and
      uploaded VO is normalised to the same target.

    A timeline-enabled ``volume=0`` punches a silent ``gap`` (default 800 ms at
    t=2.0-2.8) into the noise BEFORE the loudnorm pass, so the delivered clip
    reads ≈ -17.7 dBFS under speech and ≈ -91 dBFS in the gap — a real VO pause.
    ``-ar`` pins the mux back to ``sample_rate`` (loudnorm runs its own internal
    rate).
    """
    g0, g1 = gap
    lo, hi = band
    # The lowpass is CASCADED: one pole leaves ~-39 dB of noise above 6 kHz, which is
    # louder than the ducked bed and would make the out-of-band probe measure narration
    # instead of bed (it reads a *negative* rise). Three poles push the leak to ~-52 dB,
    # well under the bed, so the probe band belongs to the bed alone.
    lowpass = ",".join([f"lowpass=f={hi}"] * 3)
    chain = (
        f"[1:a]highpass=f={lo},{lowpass},"
        f"volume=volume=0:enable='between(t,{g0:.3f},{g1:.3f})',"
        f"loudnorm=I={lufs}:TP=-1.5:linear=true[a]"
    )
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={size}:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=c=pink:d={duration}:r={sample_rate}",
            "-filter_complex",
            chain,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(sample_rate),
            "-shortest",
            str(path),
        ]
    )
    return path


async def make_bed_tone(
    path: Path,
    *,
    duration: float = 5.0,
    frequency: int = 8000,
    gain_db: float = -6.0,
    sample_rate: int = 48000,
) -> Path:
    """Synthesise a bare WAV holding a continuous ``sine`` — the music bed.

    ``frequency`` deliberately sits ABOVE the speech band the narration occupies
    (8 kHz vs 300-3400 Hz). That out-of-band placement is the whole trick: the
    duck delivers a MIX (ducked bed summed back under the narration that keys
    it), so the bed's own level is only measurable inside the mix if a band probe
    can isolate it from the narration (see ``probe_band_mean_volume_window``). A
    bed inside the speech band would just be measured as narration.
    """
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}:sample_rate={sample_rate}",
            "-af",
            f"volume={gain_db}dB",
            str(path),
        ]
    )
    return path


async def probe_duration(path: Path) -> float:
    """Container duration in seconds (reuses the renderer's ffprobe wrapper)."""
    return await renderer._ffprobe_duration(path)


async def probe_video_frame_count(path: Path) -> int:
    """Exact decoded frame count of the primary video stream.

    Uses ``-count_frames`` so the number reflects frames actually present in the
    container (what a speed/xfade change would move), not a rate*duration
    estimate. Returns 0 when the value can't be parsed.
    """
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return int(out.strip())
    except ValueError:
        return 0


async def probe_frame_md5(path: Path) -> str:
    """Per-frame MD5 body of ``path`` via ffmpeg's ``framemd5`` muxer.

    Emits one MD5 line per decoded frame (video AND audio) to stdout; the leading
    ``#``-prefixed lines are a version/stream-layout banner, so they are stripped
    and only the per-frame hash rows are returned. Two encodes that decode to the
    same frames produce an identical body — the frame-exact reproducibility check.

    CRITICAL: only compare this WITHIN one ffmpeg build (two renders on the SAME
    binary). The hashes are build-specific, so it must never be asserted against a
    hardcoded / pre-baked value.
    """
    out = await renderer._run(["ffmpeg", "-i", str(path), "-f", "framemd5", "-"])
    return "\n".join(line for line in out.splitlines() if line and not line.startswith("#"))


async def probe_mean_volume(path: Path) -> float:
    """Mean audio volume (dBFS) of the container, via ffmpeg ``volumedetect``.

    Returns ``-inf`` when no ``mean_volume`` line is parsed (e.g. no audio). Used
    by the per-clip audio-level render tests to prove a gain/mute actually moved
    the delivered loudness, not just the command string.
    """
    out = await renderer._run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_stderr=True,
    )
    import re

    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    return float(m.group(1)) if m else float("-inf")


async def probe_mean_volume_window(path: Path, start: float, end: float) -> float:
    """Mean audio volume (dBFS) over the ``[start, end)`` window of the container.

    ``atrim`` slices the bus to the window (``asetpts`` re-bases the timestamps)
    before ``volumedetect`` measures it, so the per-clip POST-master gain/mute
    pass can be verified over exactly the edited shot's absolute bus offset.
    Returns ``-inf`` when no ``mean_volume`` line is parsed."""
    out = await renderer._run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    import re

    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    return float(m.group(1)) if m else float("-inf")


async def probe_band_mean_volume_window(
    path: Path, start: float, end: float, *, highpass_hz: float = 6000.0, stages: int = 2
) -> float:
    """Mean volume (dBFS) over ``[start, end)`` of ONLY the content above
    ``highpass_hz``.

    The duck lives in a MIX: the ducked bed is summed back under the narration
    that keys it, so a plain ``volumedetect`` over the mix mostly measures the
    narration. With the bed placed out of band (an 8 kHz tone) and the narration
    inside the speech band (300-3400 Hz), the two cascaded 2-pole high-passes at
    6 kHz put the narration ~40 dB under the bed inside this window, so what is
    measured really is the BED — during narration vs inside a VO gap. Probing the
    mix through a cutoff that OVERLAPS the speech band (the earlier 700 Hz
    version) measures narration and reports a duck that isn't there.

    Returns ``-inf`` when no ``mean_volume`` line is parsed."""
    hpf = ",".join([f"highpass=f={highpass_hz:g}"] * stages)
    out = await renderer._run(
        [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            str(path),
            "-af",
            f"atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,{hpf},volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_stderr=True,
    )
    import re

    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*) dB", out)
    return float(m.group(1)) if m else float("-inf")


async def probe_has_audio(path: Path) -> bool:
    """True when the container carries at least one audio stream."""
    out = await renderer._run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(out.strip())
