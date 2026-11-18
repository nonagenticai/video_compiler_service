"""Unit tests for VO-relative leveling (T1, great-video spec 06-sound-music.md
§3 T1, acceptance #2).

Two pieces under test:

* ``_offset_gain_db`` / ``_vo_relative_threshold`` — the PURE helpers
  (``src/services/renderer.py``) that turn a measured VO/asset loudness pair
  into a gain / sidechain threshold, with a literal fallback. No I/O.
* ``overlay_audio_bed`` wiring — with ffmpeg (``renderer._run``) and the
  measurement pass (``src.services.loudness.final_mux_loudness``) faked, the
  emitted filter STRING must carry the computed value when an offset field is
  set, and the untouched literal when it isn't (T1 is additive/opt-in).

The real render-harness proof that measured levels land within tolerance on
decoded audio lives in ``tests/render_harness/test_vo_relative_bed_render.py``
(owned separately) — these are the fast, mocked, filtergraph-shape tests.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from src.api.schemas import AmbientCue, MusicBed
from src.config import get_settings
from src.services import loudness as loudness_module
from src.services import renderer
from tests.conftest import is_delivered_master_measure

# --------------------------------------------------------------------------
# _offset_gain_db (pure)
# --------------------------------------------------------------------------


def test_offset_gain_db_none_offset_falls_back() -> None:
    assert renderer._offset_gain_db(None, -18.0, -14.0, -20.0) == (-18.0, False)


def test_offset_gain_db_none_vo_lufs_falls_back() -> None:
    assert renderer._offset_gain_db(10.0, -18.0, None, -20.0) == (-18.0, False)


def test_offset_gain_db_none_asset_lufs_falls_back() -> None:
    assert renderer._offset_gain_db(10.0, -18.0, -14.0, None) == (-18.0, False)


@pytest.mark.parametrize(
    "vo_lufs,asset_lufs",
    [
        (math.nan, -20.0),
        (-14.0, math.nan),
        (math.inf, -20.0),
        (-14.0, -math.inf),
        (math.nan, math.nan),
    ],
)
def test_offset_gain_db_non_finite_measurement_falls_back(
    vo_lufs: float, asset_lufs: float
) -> None:
    assert renderer._offset_gain_db(10.0, -18.0, vo_lufs, asset_lufs) == (-18.0, False)


def test_offset_gain_db_computes_target_minus_asset() -> None:
    # target = vo_lufs - offset_lu = -14 - 10 = -24; gain = target - asset = -24 - (-20) = -4
    gain, measured = renderer._offset_gain_db(10.0, -18.0, -14.0, -20.0)
    assert measured is True
    assert gain == pytest.approx(-4.0)


def test_offset_gain_db_clamped_at_positive_rail() -> None:
    # A very quiet asset needs a huge boost to hit the target — clamped at +12.
    gain, measured = renderer._offset_gain_db(0.0, -18.0, -6.0, -60.0)
    assert measured is True
    assert gain == 12.0


def test_offset_gain_db_clamped_at_negative_rail() -> None:
    # A very loud asset needs a huge cut to hit the target — clamped at -40.
    gain, measured = renderer._offset_gain_db(40.0, -18.0, -6.0, 0.0)
    assert measured is True
    assert gain == -40.0


def test_offset_gain_db_loud_and_quiet_asset_converge_to_same_target() -> None:
    """Two music assets ~26 dB apart in inherent loudness, same VO + same
    requested offset, must deliver the SAME target level (vo_lufs -
    offset_lu) — the gain compensates for whatever the asset happens to be."""
    vo_lufs, offset_lu = -18.0, 10.0
    target = vo_lufs - offset_lu
    quiet_gain, quiet_measured = renderer._offset_gain_db(offset_lu, -18.0, vo_lufs, -40.0)
    loud_gain, loud_measured = renderer._offset_gain_db(offset_lu, -18.0, vo_lufs, -14.0)
    assert quiet_measured is True
    assert loud_measured is True
    assert (quiet_gain + -40.0) == pytest.approx(target)
    assert (loud_gain + -14.0) == pytest.approx(target)


# --------------------------------------------------------------------------
# _vo_relative_threshold (pure)
# --------------------------------------------------------------------------


def test_vo_relative_threshold_none_open_lu_falls_back() -> None:
    assert renderer._vo_relative_threshold(None, -14.0, 0.03) == 0.03


def test_vo_relative_threshold_none_vo_lufs_falls_back() -> None:
    assert renderer._vo_relative_threshold(10.0, None, 0.03) == 0.03


@pytest.mark.parametrize("vo_lufs", [math.nan, math.inf, -math.inf])
def test_vo_relative_threshold_non_finite_vo_lufs_falls_back(vo_lufs: float) -> None:
    assert renderer._vo_relative_threshold(10.0, vo_lufs, 0.03) == 0.03


def test_vo_relative_threshold_computes_linear_from_dbfs() -> None:
    # dbfs = vo_lufs - open_lu_below_vo = -14 - 10 = -24 -> linear = 10**(-24/20)
    threshold = renderer._vo_relative_threshold(10.0, -14.0, 0.03)
    assert threshold == pytest.approx(10 ** (-24.0 / 20.0))


def test_vo_relative_threshold_clamped_at_upper_rail() -> None:
    assert renderer._vo_relative_threshold(0.0, 0.0, 0.03) == 1.0  # dbfs=0 -> linear=1.0
    assert renderer._vo_relative_threshold(-10.0, 0.0, 0.03) == 1.0  # dbfs=+10 -> >1, clamped


def test_vo_relative_threshold_clamped_at_lower_rail() -> None:
    assert renderer._vo_relative_threshold(80.0, 0.0, 0.03) == 0.001  # dbfs=-80 -> tiny, clamped


# --------------------------------------------------------------------------
# overlay_audio_bed wiring: filter string carries computed vs. literal values
# --------------------------------------------------------------------------


def _install_fake_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        if is_delivered_master_measure(cmd):
            return ""
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)
    return captured


def _install_fake_measure(
    monkeypatch: pytest.MonkeyPatch, by_substring: dict[str, float | None]
) -> list[str]:
    """Fake ``final_mux_loudness``: the measured ``integrated_lufs`` is looked
    up by the first matching substring key in ``path``; a ``None`` value
    mirrors a failed best-effort measurement (the real function's contract:
    never raises, returns ``None`` on any failure)."""
    calls: list[str] = []

    async def fake_measure(path: str) -> dict[str, float | None] | None:
        calls.append(path)
        for key, lufs in by_substring.items():
            if key in path:
                return None if lufs is None else {"integrated_lufs": lufs}
        return None

    monkeypatch.setattr(loudness_module, "final_mux_loudness", fake_measure)
    return calls


async def test_overlay_audio_bed_no_offset_fields_stays_literal_and_measures_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Byte-identical guard: with no offset/threshold field set anywhere, the
    filter string keeps the exact literal values and ZERO measurement passes
    run — today's cost, unchanged."""
    captured = _install_fake_run(monkeypatch)
    calls = _install_fake_measure(monkeypatch, {})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", gain_db=-18.0)),
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a", gain_db=-24.0))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "volume=-18.0dB" in cmd
    assert "volume=-24.0dB" in cmd
    assert "threshold=0.03:" in cmd
    assert calls == []
    assert report is not None
    assert report["music_was_measured"] is False
    assert report["vo_lufs"] is None
    assert report["music_gain_db"] == -18.0


async def test_overlay_audio_bed_music_offset_lu_computes_gain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fake_run(monkeypatch)
    _install_fake_measure(monkeypatch, {"compiled": -14.0, "music": -30.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", gain_db=-18.0, music_offset_lu=10.0)),
        ambient=[],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    # target = -14 - 10 = -24; gain = -24 - (-30) = 6.0
    assert "volume=6.0dB" in cmd
    assert "volume=-18.0dB" not in cmd  # literal fallback NOT used
    assert report is not None
    assert report["music_was_measured"] is True
    assert report["vo_lufs"] == pytest.approx(-14.0)
    assert report["music_lufs"] == pytest.approx(-30.0)
    assert report["music_gain_db"] == pytest.approx(6.0)


async def test_overlay_audio_bed_duck_open_lu_below_vo_computes_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fake_run(monkeypatch)
    _install_fake_measure(monkeypatch, {"compiled": -14.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", duck_open_lu_below_vo=10.0)),
        ambient=[],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    expected = 10 ** ((-14.0 - 10.0) / 20.0)
    assert f"threshold={expected:g}:" in cmd
    assert "threshold=0.03:" not in cmd
    assert report is not None
    assert report["duck_threshold"] == pytest.approx(expected)


async def test_overlay_audio_bed_ambient_offset_lu_computes_gain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_fake_run(monkeypatch)
    _install_fake_measure(monkeypatch, {"compiled": -14.0, "amb": -40.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[
            (tmp_path / "amb.mp3", AmbientCue(url="a", gain_db=-24.0, ambient_offset_lu=20.0))
        ],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    # target = -14 - 20 = -34; gain = -34 - (-40) = 6.0
    assert "volume=6.0dB" in cmd
    assert "volume=-24.0dB" not in cmd
    assert report is None  # music_bed report is music-specific; no music here


async def test_overlay_audio_bed_ambient_duck_open_lu_below_vo_computes_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ambient-only branch, mirroring the music-branch test above: with
    ``AmbientCue.duck_open_lu_below_vo`` set, the sidechain key is derived from
    the MEASURED VO level, not the absolute ``duck_threshold`` literal.

    Guards the ``wants_measurement`` gate too — if that gate did not also fire
    on an ambient cue's ``duck_open_lu_below_vo``, ``vo_lufs`` would stay
    ``None``, ``_vo_relative_threshold`` would silently return the fallback,
    and the field would be INERT while every other assertion still passed."""
    captured = _install_fake_run(monkeypatch)
    calls = _install_fake_measure(monkeypatch, {"compiled": -14.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a", duck_open_lu_below_vo=10.0))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    expected = 10 ** ((-14.0 - 10.0) / 20.0)
    assert f"threshold={expected:g}:" in cmd
    assert "threshold=0.03:" not in cmd
    assert calls, "the VO measurement pass never ran — wants_measurement did not fire"
    assert report is None  # music_bed report is music-specific; no music here


async def test_overlay_audio_bed_ambient_without_duck_open_lu_stays_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Additive guard: an ambient-only bed with no offset/threshold field keeps
    the literal ``duck_threshold`` and runs ZERO measurement passes."""
    captured = _install_fake_run(monkeypatch)
    calls = _install_fake_measure(monkeypatch, {"compiled": -14.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[(tmp_path / "amb.mp3", AmbientCue(url="a"))],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "threshold=0.03:" in cmd
    assert calls == []


async def test_overlay_audio_bed_ambient_duck_open_lu_falls_back_when_vo_unmeasurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The VO measurement fails (``final_mux_loudness`` -> ``None``): the
    ambient duck key falls back to the literal ``duck_threshold`` rather than
    failing the render."""
    captured = _install_fake_run(monkeypatch)
    calls = _install_fake_measure(monkeypatch, {"compiled": None})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=None,
        ambient=[
            (
                tmp_path / "amb.mp3",
                AmbientCue(url="a", duck_threshold=0.05, duck_open_lu_below_vo=10.0),
            )
        ],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "threshold=0.05:" in cmd
    assert calls, "the VO measurement pass never ran — wants_measurement did not fire"


async def test_overlay_audio_bed_measurement_failure_falls_back_to_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The VO measurement pass fails (``final_mux_loudness`` -> ``None``, its
    documented best-effort failure mode) -> the offset silently falls back to
    the literal gain, never fails the render."""
    captured = _install_fake_run(monkeypatch)
    _install_fake_measure(monkeypatch, {"compiled": None, "music": -30.0})
    video = tmp_path / "compiled.mp4"
    video.write_bytes(b"\x00")
    _, report = await renderer.overlay_audio_bed(
        video,
        tmp_path / "final.mp4",
        total_seconds=10.0,
        music=(tmp_path / "music.mp3", MusicBed(url="m", gain_db=-18.0, music_offset_lu=10.0)),
        ambient=[],
        settings=get_settings(),
    )
    cmd = " ".join(captured["cmds"][-1])
    assert "volume=-18.0dB" in cmd  # fell back to the literal
    assert report is not None
    assert report["music_was_measured"] is False
    assert report["vo_lufs"] is None
