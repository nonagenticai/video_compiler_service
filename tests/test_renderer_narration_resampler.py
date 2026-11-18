"""The per-shot PRIMARY (narration) bus must carry an explicit soxr resampler.

Why this file exists: ``_A48`` names a rate but no resampler, so a rate change
falls to swr's default cubic interpolator. A comment in ``renderer.py`` used to
assert that this did not matter on the narration bus because it "already arrives
48 kHz stereo" — that is FALSE. ``tts_backend`` renders narration at 24000 Hz
MONO (hardcoded), and the orchestrator's premaster pins its output back to the
probed input rate, so the per-shot primary is a real 24k -> 48k conversion.

Measured 2026-09-13 on a 2 s clip, 24k mono -> 48k stereo, cubic vs soxr:
identical below 11.5 kHz (0.00 dB), but cubic leaves the source's mirror image at
-70.7 dBFS in 12-16 kHz against a -20.9 dBFS program, where soxr leaves
-180 dBFS. soxr costs nothing in the passband.

These assertions pin the filtergraph STRING, because the resampler is invisible
in any output-format check: both chains emit 48 kHz stereo, identical duration,
identical loudness. Nothing else in the suite can tell them apart.
"""

from __future__ import annotations

from src.api.schemas import AccentCue, SoundCue
from src.config import get_settings
from src.services import renderer

# The exact head the narration bus must carry. Spelled out literally rather than
# interpolated from `renderer._A48_SOXR` so that redefining that constant to the
# bare cubic form fails this test instead of silently redefining "correct".
_SOXR_HEAD = (
    "aresample=48000:resampler=soxr:precision=28,aformat=channel_layouts=stereo:sample_rates=48000"
)
# The bare form, which is what these two sites carried before 2026-09-13.
_BARE = "aformat=channel_layouts=stereo:sample_rates=48000"


def test_sfx_graph_primary_narration_bus_resamples_with_soxr() -> None:
    """`_audio_sfx_graph`'s primary IS the narration (24 kHz mono from
    tts_backend). The SFX inputs already used soxr; the narration they are mixed
    against did not."""
    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=(2, SoundCue(url="x", gain_db=-3.0)),
        accents=[(3, AccentCue(url="y", at_seconds=1.5, gain_db=-6.0))],
        settings=get_settings(),
    )
    assert f"[1:a]{_SOXR_HEAD}[a_pri]" in graph
    # And specifically NOT the bare cubic head it used to carry.
    assert f"[1:a]{_BARE}[a_pri]" not in graph


def test_sfx_graph_primary_keeps_soxr_ahead_of_the_fade_fragment() -> None:
    """`audio_frag` (the per-shot afade envelope) chains AFTER the resample, so a
    fade must not displace the resampler."""
    graph = renderer._audio_sfx_graph(
        "1:a",
        transition=None,
        accents=[(2, AccentCue(url="y", at_seconds=0.0))],
        settings=get_settings(),
        audio_frag=",afade=t=in:st=0:d=0.008",
        shot_seconds=4.0,
    )
    assert f"[1:a]{_SOXR_HEAD},afade=t=in:st=0:d=0.008[a_pri]" in graph


def test_no_sfx_retime_path_resamples_with_soxr() -> None:
    """The no-SFX branch of `_video_audio_map_args` (atempo and/or a fade, no SFX
    inputs) carries the same narration stream and must resample it the same way."""
    args = renderer._video_audio_map_args(
        has_sfx=False,
        is_text=False,
        video_vf="scale=1920:1080",
        audio_map="0:a",
        next_idx=1,
        transition_sfx=None,
        accent_sfx=[],
        trim_args=["-t", "4.0"],
        settings=get_settings(),
        atempo="atempo=1.25",
        audio_frag="",
    )
    fc = args[args.index("-filter_complex") + 1]
    assert f"[0:a]{_SOXR_HEAD},atempo=1.25[aout]" in fc
    assert f"[0:a]{_BARE},atempo=1.25[aout]" not in fc


def test_master_bus_decimation_deliberately_left_on_cubic() -> None:
    """MEASURED AND LEFT ALONE, pinned so a later sweep does not "also fix" it
    without measuring.

    `_master_filter` decimates loudnorm's 192 kHz bus to 48 kHz ahead of
    `alimiter`. That was the intuitive suspect, but cubic vs soxr there measured
    0.00 dB across every band from 0 to 20 kHz: loudnorm's 192 kHz output carries
    nothing above 30 kHz (-121 dBFS) for cubic to alias down. Only 20-24 kHz moves
    (~0.5 dB), above audibility and discarded by the AAC delivery encode.
    """
    from src.config import get_settings

    chain = renderer._master_filter(get_settings())
    assert "aformat=sample_rates=48000:channel_layouts=stereo,alimiter=" in chain
    assert "resampler=soxr" not in chain
