"""Music/ambient bed filtergraph correctness (captured ffmpeg cmd, no real render).

- a multi-cue AMBIENT-only bed must span the LONGEST cue, not be truncated to the first
  cue's window (``amix=duration=longest``);
- a music bed with a 0-second fade must NOT emit a degenerate ``afade=d=0`` (which can
  abort the pass) — the fade is simply omitted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import AmbientCue, MusicBed
from src.config import get_settings
from src.services import renderer
from tests.conftest import is_delivered_master_measure


async def _capture_bed_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    music: tuple[Path, MusicBed] | None,
    ambient: list[tuple[Path, AmbientCue]],
) -> str:
    captured: dict[str, list[str]] = {}

    async def _fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        # `capture_stderr` is keyword-only on the real `_run` and the bed path now
        # passes it (it reads loudnorm's normalization type back off pass-2 stderr),
        # so the fake has to accept it or every bed test dies with a TypeError that
        # the fail-open master swallows into a `measure_failed` warning.
        if is_delivered_master_measure(cmd):
            return ""
        captured["cmd"] = cmd
        return ""

    monkeypatch.setattr(renderer, "_run", _fake_run)
    await renderer.overlay_audio_bed(
        tmp_path / "v.mp4",
        tmp_path / "o.mp4",
        10.0,
        music,
        ambient,
        get_settings(),
    )
    cmd = captured["cmd"]
    return cmd[cmd.index("-filter_complex") + 1]


async def test_ambient_multi_cue_bed_spans_longest_not_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # cue[0] ends at 2s, cue[1] runs to 10s: the bed must not be cut at 2s.
    fc = await _capture_bed_filter(
        monkeypatch,
        tmp_path,
        music=None,
        ambient=[
            (tmp_path / "a.wav", AmbientCue(url="a", start_seconds=0, end_seconds=2, gain_db=-20)),
            (tmp_path / "b.wav", AmbientCue(url="b", start_seconds=0, end_seconds=10, gain_db=-20)),
        ],
    )
    # the BED mix (2 ambient cues) spans the longest cue. (The separate narration+bed
    # premaster mix legitimately still uses duration=first — both its inputs are full-length.)
    assert "amix=inputs=2:duration=longest" in fc


async def test_music_bed_zero_fade_emits_no_degenerate_afade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fc = await _capture_bed_filter(
        monkeypatch,
        tmp_path,
        music=(tmp_path / "m.mp3", MusicBed(url="m", gain_db=-20, fade_in_s=0, fade_out_s=0)),
        ambient=[],
    )
    # both fades omitted (0-duration = no fade); no degenerate afade emitted. (Can't assert
    # "d=0" absent — it substring-matches "threshold=0.03" in the sidechain duck.)
    assert "afade" not in fc


async def test_music_bed_nonzero_fades_still_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fc = await _capture_bed_filter(
        monkeypatch,
        tmp_path,
        music=(tmp_path / "m.mp3", MusicBed(url="m", gain_db=-20, fade_in_s=1.5, fade_out_s=2.5)),
        ambient=[],
    )
    assert "afade=t=in:d=1.5" in fc
    assert "afade=t=out" in fc


def test_sidechain_ratio_duck_db_zero_is_light_not_heavy() -> None:
    """duck_db=0 ("no extra attenuation") maps to the LIGHT floor ratio 2, not the
    heaviest one.

    REVERT-PROOF: restore `if duck_db >= 0: return 9` and ratio(0) becomes 9 (the bug).
    """
    assert renderer._sidechain_ratio(0.0) == 2  # no extra duck → lightest
    assert renderer._sidechain_ratio(5.0) == 2  # positive → floor, not heavy
    assert renderer._sidechain_ratio(-0.001) == 2  # "almost no duck" → light (was discontinuous)
    assert renderer._sidechain_ratio(-12.0) == 4  # schema default anchor (was 9: it pumped)
    assert renderer._sidechain_ratio(-24.0) == 8  # deeper → heavier
