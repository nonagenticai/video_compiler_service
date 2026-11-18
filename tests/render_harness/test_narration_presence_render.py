"""Real-ffmpeg proof that ``narration_presence_report`` catches a narration clip
the compiler was GIVEN but that never made it into the delivered master.

THE DEFECT: a production master shipped with one of its four narration clips
completely ABSENT and 68% silence, and nothing noticed — two loudness-statistic
approaches (#231's ``integrated_lufs - noise_floor_db``, and a reverted RMS-spread
heuristic) both failed on real audio: digital silence and digital-zero word gaps
both manufacture a ``-inf`` floor that reads as "speech present". A single number
over the whole file cannot separate "every clip present" from "one clip missing,
three present" — both shapes contain plenty of real speech.

THE MEASURE: cross-correlate each shot's SOURCE narration clip against the
delivered master's own audio at that shot's [start, end) window, via ffmpeg's
``axcorrelate`` + ``astats`` (see ``mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``
for the full provenance, including real production numbers).

These tests build real, distinct narration-shaped tones with ffmpeg, concatenate
them into a real AAC-in-MP4 master (the container this pipeline delivers, exactly
like ``test_speech_presence_render.py`` and ``test_av_coverage_render.py``), and
drive the real ``narration_presence_report`` function — no mocks, no hand-written
expectation of what ffmpeg "should" report.

⛔ NEVER RE-FETCHES (fixed on this branch): the check used to re-download each
shot's ``audio_url`` fresh, at the very end of a compile — exactly when a job's
presigned narration-clip URLs are most likely EXPIRED (a proven production
failure: a compile retried past 60 minutes got 403 on its 1-hour presigned
inputs). It now reuses the LOCAL copy ``compile_shots`` already fetched while
rendering, threaded in as ``narration_audio_paths`` (index-aligned to
``shots``/``shot_timeline``) — ``audio_url`` survives only as the "is this shot
narrated at all" signal. ``test_the_check_never_refetches_even_with_an_unreachable_url``
proves this directly: ``audio_url`` points at a closed local port and the check
still succeeds, because it never touches the network. A shot whose local copy did
not survive to verification time (``None`` in ``narration_audio_paths`` — e.g. the
content-addressed render cache served that shot, so the raw source was never
fetched this compile) is reported UNMEASURED, not missing — see
``test_a_shot_with_no_local_copy_is_reported_unmeasured_not_missing``.

MUTATION-PROVED (performed by hand during development, not re-encoded as a test):
commenting out the ``correlation_db >= NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``
comparison in ``_narration_presence_entry`` (forcing ``present`` to always be
``True`` for any measured clip) turns ``test_a_missing_clip_is_flagged_naming_it``
and ``test_a_missing_clip_is_still_flagged_with_a_music_bed`` RED — no violation is
raised even though the wrong clip was fed in — confirming the comparison is what
these tests actually exercise, not just a return-shape assertion.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import get_settings
from src.services import mastering, renderer

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

_CLIP_SECONDS = 3.0


def _tone(freq: float, duration: float = _CLIP_SECONDS) -> str:
    return f"sine=frequency={freq}:duration={duration}:sample_rate=48000"


def _gate(period: float, duty: float, duration: float = _CLIP_SECONDS) -> str:
    """A narration-shaped envelope: on for ``duty`` of every ``period`` seconds —
    the property that matters is PAUSES, not that it is a voice (same idiom as
    ``test_speech_presence_render.py``'s ``_NARRATION_GATE``)."""
    on = period * duty
    return f"aevalsrc='0.4*lt(mod(t\\,{period})\\,{on})':d={duration}:s=48000"


#: Three DISTINCT narration-shaped signals so a wrong pairing cannot accidentally
#: correlate: different tone frequency AND different gate rhythm each.
_CLIP_A = (_tone(220), _gate(0.6, 0.6))
_CLIP_B = (_tone(440), _gate(0.5, 0.55))
#: Stands in for "a different clip entirely" at the position the missing clip
#: should have occupied — never fetched as anyone's source, only ever muxed into
#: the master, exactly like the real defect (some other/no content in that slot).
_OTHER = (_tone(880), _gate(0.4, 0.65))


async def _make_clip(dest: Path, spec: tuple[str, str]) -> Path:
    """One shot's SOURCE narration clip — what ``Shot.audio_url`` would fetch."""
    tone, gate = spec
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            tone,
            "-f",
            "lavfi",
            "-i",
            gate,
            "-filter_complex",
            "[0:a][1:a]amultiply",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    return dest


async def _make_master(dest: Path, segments: list[tuple[str, str]], *, bed: bool = False) -> Path:
    """Concatenate ``segments`` (each ``_CLIP_SECONDS`` long) into one real
    AAC-in-MP4 master — the container this pipeline delivers — optionally under a
    continuous pink-noise bed spanning the whole thing (the music-bed case)."""
    inputs: list[str] = []
    parts: list[str] = []
    for i, spec in enumerate(segments):
        tone, gate = spec
        inputs += ["-f", "lavfi", "-i", tone, "-f", "lavfi", "-i", gate]
        parts.append(f"[{2 * i}:a][{2 * i + 1}:a]amultiply[seg{i}]")
    next_idx = 2 * len(segments)
    concat_labels = "".join(f"[seg{i}]" for i in range(len(segments)))
    parts.append(f"{concat_labels}concat=n={len(segments)}:v=0:a=1[narr]")
    out_label = "[narr]"
    if bed:
        total = _CLIP_SECONDS * len(segments)
        inputs += [
            "-f",
            "lavfi",
            "-i",
            f"anoisesrc=color=pink:amplitude=0.12:duration={total}:sample_rate=48000",
        ]
        parts.append(f"[narr][{next_idx}:a]amix=inputs=2:duration=first:weights='1 0.5'[out]")
        out_label = "[out]"
    await renderer._run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            *inputs,
            "-filter_complex",
            ";".join(parts),
            "-map",
            out_label,
            "-c:a",
            "aac",
            str(dest),
        ]
    )
    return dest


def _shot_timeline(n: int) -> list[dict[str, object]]:
    return [
        {"line_id": f"shot{i}", "start": i * _CLIP_SECONDS, "end": (i + 1) * _CLIP_SECONDS}
        for i in range(n)
    ]


#: A URL that can never be re-fetched: nothing listens on localhost port 1 (a
#: privileged, unassigned port), so a connection attempt is refused immediately.
#: ``audio_url`` no longer drives any fetch (see the module docstring) — every
#: test below uses this by default specifically to PROVE that, short of the one
#: test that deliberately supplies a real reachable value for contrast.
_UNREACHABLE_URL = "http://127.0.0.1:1/closed-port-never-answers"


def _shots(*clip_paths: Path, audio_url: str = _UNREACHABLE_URL) -> list[SimpleNamespace]:
    """Shots carrying only the "this is narrated" signal (``audio_url``, never
    fetched by ``narration_presence_report`` any more — see the module
    docstring). Defaults to an UNREACHABLE url so every test that does not
    override it is itself proof the check never re-fetches."""
    return [SimpleNamespace(audio_url=audio_url) for _ in clip_paths]


def _local_paths(*clip_paths: Path) -> list[Path]:
    """The ``narration_audio_paths`` a real ``compile_shots`` would have handed
    to ``post_mux_verify`` — the local copies it fetched while rendering,
    index-aligned to ``shots``/``shot_timeline``."""
    return list(clip_paths)


@pytest.mark.asyncio
async def test_a_missing_clip_is_flagged_naming_it(tmp_path: Path) -> None:
    """(a) — a master carrying clip A and something else (not clip B) at shot1's
    slot must flag shot1 by name, and shot0 must clear."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _OTHER])

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b),
        get_settings(),
        _local_paths(clip_a, clip_b),
    )

    assert violation is not None, f"gate did not fire on a missing clip: {per_clip}"
    assert violation.stage == "narration_presence"
    assert violation.tier == "structural"
    assert "shot1" in violation.detail, violation.detail
    assert "shot0" not in violation.detail.split("shot(s)")[-1], violation.detail

    by_id = {c["line_id"]: c for c in per_clip}
    assert by_id["shot0"]["present"] is True, by_id["shot0"]
    assert by_id["shot1"]["present"] is False, by_id["shot1"]
    shot0_db = by_id["shot0"]["correlation_db"]
    shot1_db = by_id["shot1"]["correlation_db"]
    assert isinstance(shot0_db, float) and isinstance(shot1_db, float)
    assert shot0_db > mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB
    assert shot1_db < mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB


@pytest.mark.asyncio
async def test_a_complete_master_flags_nothing(tmp_path: Path) -> None:
    """(b) — every shot's real clip present at its real slot: no violation."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _CLIP_B])

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b),
        get_settings(),
        _local_paths(clip_a, clip_b),
    )

    assert violation is None, f"gate fired on a complete master: {per_clip}"
    assert all(c["present"] is True for c in per_clip), per_clip


#: A LONG (~5s, vs the 3s siblings above) narration-shaped clip, reproducing
#: the real production shape that false-negatived under the OLD (padded)
#: statistic: two of four genuinely-present clips on a clean production master
#: were the two LONGER (~5s) clips, both reading below the old -20.0 dB
#: threshold (-21.03 dB, -20.23 dB) despite being fully present. See
#: ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB`` for the full real-audio table.
_CLIP_LONG_SECONDS = 5.0
_CLIP_LONG = (
    _tone(330, duration=_CLIP_LONG_SECONDS),
    _gate(0.55, 0.6, duration=_CLIP_LONG_SECONDS),
)


@pytest.mark.asyncio
async def test_a_long_present_clip_is_not_flagged(tmp_path: Path) -> None:
    """(a) — this test's synthetic render-harness fixture does NOT pin the
    margin=0.0 decision: measured 2026-09-17, setting
    ``NARRATION_PRESENCE_WINDOW_MARGIN_SECONDS`` to 0.5, 1.0, and 2.0 against
    this fixture leaves this test GREEN at every value — its padded-window RMS
    never drops far enough to cross ``NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB``.
    An earlier version of this docstring claimed reverting to 0.5 turned it RED;
    that was never reproduced and has now been checked false.

    The margin=0.0 decision is pinned by real production measurements instead:
    ``tests/unit/test_mastering.py::test_margin_mutation_proof_against_the_current_threshold``
    monkeypatches the real (measured) padded/unpadded RMS pair for a genuinely
    present clip ("broken line3": -32.94 dB padded, -12.48 dB unpadded) and
    proves margin=0.5 crosses below the CURRENT -25.0 dB threshold while
    margin=0.0 does not. (Its sibling,
    ``test_long_clip_length_defect_mutation_proof``, proves the same shape
    against the OLD, now-retired -20.0 dB threshold using a different real pair
    — good clip3's -21.03/-7.92 dB — which stays clear of the current -25.0 dB
    threshold and so cannot on its own prove today's gate goes red.)

    This test still stands as a real-ffmpeg, real-audio integration check that a
    long present clip is NOT flagged missing under the CURRENT (fixed)
    statistic — it just is not the margin regression proof."""
    clip_short = await _make_clip(tmp_path / "short.wav", _CLIP_A)
    clip_long = await _make_clip(tmp_path / "long.wav", _CLIP_LONG)
    master = await _make_master(
        tmp_path / "master.mp4",
        [(_CLIP_A[0], _CLIP_A[1]), (_CLIP_LONG[0], _CLIP_LONG[1])],
    )
    shot_timeline = [
        {"line_id": "shot0", "start": 0.0, "end": _CLIP_SECONDS},
        {"line_id": "shot1", "start": _CLIP_SECONDS, "end": _CLIP_SECONDS + _CLIP_LONG_SECONDS},
    ]

    per_clip, violation = await mastering.narration_presence_report(
        master,
        shot_timeline,
        _shots(clip_short, clip_long),
        get_settings(),
        _local_paths(clip_short, clip_long),
    )

    assert violation is None, f"a long-but-present clip was flagged missing: {per_clip}"
    by_id = {c["line_id"]: c for c in per_clip}
    assert by_id["shot0"]["present"] is True, by_id["shot0"]
    assert by_id["shot1"]["present"] is True, by_id["shot1"]
    long_db = by_id["shot1"]["correlation_db"]
    assert isinstance(long_db, float)
    assert long_db >= mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB


@pytest.mark.asyncio
async def test_a_missing_clip_is_still_flagged_with_a_music_bed(tmp_path: Path) -> None:
    """(c) — a continuous bed under a missing clip must not make an energy-only
    measure falsely pass it. ``axcorrelate`` correlates SHAPE, not loudness, so
    the bed (non-silent everywhere) must not manufacture a false "present"."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _OTHER], bed=True)

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b),
        get_settings(),
        _local_paths(clip_a, clip_b),
    )

    assert violation is not None, f"a bed hid the missing clip: {per_clip}"
    assert "shot1" in violation.detail, violation.detail
    by_id = {c["line_id"]: c for c in per_clip}
    assert by_id["shot0"]["present"] is True, by_id["shot0"]
    assert by_id["shot1"]["present"] is False, by_id["shot1"]


@pytest.mark.asyncio
async def test_a_complete_master_with_a_bed_flags_nothing(tmp_path: Path) -> None:
    """(d) — the bed-robustness counterpart to (b): a real bed under a COMPLETE
    master must not itself trip the gate."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _CLIP_B], bed=True)

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b),
        get_settings(),
        _local_paths(clip_a, clip_b),
    )

    assert violation is None, f"gate fired on a complete master with a bed: {per_clip}"
    assert all(c["present"] is True for c in per_clip), per_clip


@pytest.mark.asyncio
async def test_it_is_not_armed_by_default(tmp_path: Path) -> None:
    """The gate's own kill switch defaults off (FLEET n=1 — see
    ``_GATE_ARMED_DEFAULTS``): ``post_mux_verify`` must still REPORT the missing
    clip (gate_pass=False, a violations entry) without raising."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _OTHER])
    settings = get_settings()
    assert mastering.gate_armed(settings, "narration_presence") is False

    report = await mastering.post_mux_verify(
        master,
        settings,
        loudness=None,
        shot_timeline=_shot_timeline(2),
        shots=_shots(clip_a, clip_b),
        narration_audio_paths=_local_paths(clip_a, clip_b),
    )

    assert report["gate_pass"] is False
    stages = {v["stage"] for v in report["violations"]}
    assert "narration_presence" in stages
    by_id = {c["line_id"]: c for c in report["narration_presence"]}
    assert by_id["shot1"]["present"] is False


@pytest.mark.asyncio
async def test_arming_it_raises_on_the_same_missing_master(tmp_path: Path) -> None:
    """Arming the kill switch escalates the SAME finding to a hard failure —
    proves the switch is real and wired, mirroring
    ``test_speech_presence_gate.py::test_arming_it_makes_the_same_hiss_master_raise``."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _OTHER])
    settings = get_settings().model_copy(update={"mastering_gate_narration_presence_enabled": True})

    with pytest.raises(mastering.MasteringGateError) as exc_info:
        await mastering.post_mux_verify(
            master,
            settings,
            loudness=None,
            shot_timeline=_shot_timeline(2),
            shots=_shots(clip_a, clip_b),
            narration_audio_paths=_local_paths(clip_a, clip_b),
        )
    assert any(v.stage == "narration_presence" for v in exc_info.value.violations)


@pytest.mark.asyncio
async def test_the_check_never_refetches_even_with_an_unreachable_url(tmp_path: Path) -> None:
    """MUTATION-PROOF (a): every ``audio_url`` above is already unreachable
    (``_UNREACHABLE_URL``, a closed local port) and every test above already
    passes — this test makes that fact EXPLICIT rather than incidental, on a
    COMPLETE master where success is unambiguous. Before this fix, a fetch
    attempt against this URL would raise/degrade to ``None`` and every clip
    would read UNMEASURED; the real correlation succeeding here proves the
    check reads local disk, never the network."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _CLIP_B])

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b, audio_url=_UNREACHABLE_URL),
        get_settings(),
        _local_paths(clip_a, clip_b),
    )

    assert violation is None, f"gate fired on a complete master: {per_clip}"
    by_id = {c["line_id"]: c for c in per_clip}
    assert by_id["shot0"]["present"] is True, by_id["shot0"]
    assert by_id["shot1"]["present"] is True, by_id["shot1"]
    assert isinstance(by_id["shot0"]["correlation_db"], float)
    assert isinstance(by_id["shot1"]["correlation_db"], float)


@pytest.mark.asyncio
async def test_a_shot_with_no_local_copy_is_reported_unmeasured_not_missing(
    tmp_path: Path,
) -> None:
    """MUTATION-PROOF (b): a shot whose narration source was never fetched THIS
    compile (e.g. its segment was served by the content-addressed render cache —
    ``narration_audio_paths[i] is None``) must read UNMEASURED
    (``correlation_db=None, present=None``), never as "missing" (``present=False``,
    which would name it in ``violation.detail`` and fail the gate under
    enforce). shot1's real clip is genuinely present in the master — the point
    is that this function has no way to know that, and must not guess "absent"."""
    clip_a = await _make_clip(tmp_path / "a.wav", _CLIP_A)
    clip_b = await _make_clip(tmp_path / "b.wav", _CLIP_B)
    master = await _make_master(tmp_path / "master.mp4", [_CLIP_A, _CLIP_B])

    per_clip, violation = await mastering.narration_presence_report(
        master,
        _shot_timeline(2),
        _shots(clip_a, clip_b),
        get_settings(),
        [clip_a, None],  # shot1's local copy did not survive to verification time
    )

    assert violation is None, f"an unmeasured shot must never gate: {per_clip}"
    by_id = {c["line_id"]: c for c in per_clip}
    assert by_id["shot0"]["present"] is True, by_id["shot0"]
    assert by_id["shot1"]["present"] is None, by_id["shot1"]
    assert by_id["shot1"]["correlation_db"] is None, by_id["shot1"]
