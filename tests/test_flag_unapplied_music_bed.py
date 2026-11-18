"""Unit coverage for ``_flag_unapplied_music_bed`` — the single channel that must
NEVER report ``music_bed: null`` for a compile that asked for a continuous bed
(music OR ambient) and didn't get one applied (video_compiler#232, Defect B).

Production evidence: a compile requesting a ``music`` bed (jazz_lounge.mp3, with
``duck_db``/``music_offset_lu`` set) shipped a ``completed`` master with digital
silence where the bed should have been, and the response carried no ``music_bed``
key at all — indistinguishable from a compile that never asked for one. These
tests pin every branch that must now report something instead of ``None``, plus
the one case that must stay untouched (nothing requested).
"""

from __future__ import annotations

from src.services.renderer import _flag_unapplied_music_bed

_REPORT: dict[str, float | bool | None] = {"duck_db": -12.0, "vo_lufs": -18.0}


def test_nothing_requested_stays_none() -> None:
    assert _flag_unapplied_music_bed(None, requested=False, unfetchable=False) is None


def test_nothing_requested_leaves_a_populated_report_untouched() -> None:
    # Defensive: requested=False should never fire even if a caller somehow had
    # a report (it shouldn't, but the contract is "untouched", not "cleared").
    assert _flag_unapplied_music_bed(_REPORT, requested=False, unfetchable=False) == _REPORT


def test_music_unfetchable_flags_requested_and_failed() -> None:
    out = _flag_unapplied_music_bed(None, requested=True, unfetchable=True, url="http://m")
    assert out == {"requested": True, "applied": False, "fetch_failed": True}


def test_music_requested_fetched_but_report_lost_is_flagged_not_none() -> None:
    """The state the production defect actually hit: fetch succeeded (or was
    never flagged as failed), yet no report ever came back from
    ``overlay_audio_bed`` — this must NOT silently return ``None``."""
    out = _flag_unapplied_music_bed(None, requested=True, unfetchable=False, url="http://m")
    assert out == {"requested": True, "applied": False, "fetch_failed": False}


def test_music_requested_and_report_present_is_returned_unchanged() -> None:
    out = _flag_unapplied_music_bed(_REPORT, requested=True, unfetchable=False, url="http://m")
    assert out == _REPORT


def test_ambient_only_totally_lost_is_flagged_even_with_no_music() -> None:
    """An ambient-only compile (the common case) whose bed fails to fetch must be
    visible even though `requested` (music) is False."""
    out = _flag_unapplied_music_bed(
        None, requested=False, unfetchable=False, ambient_requested=1, ambient_lost=1
    )
    assert out == {"ambient_requested": 1, "ambient_lost": 1, "ambient_fetch_failed": True}


def test_ambient_partial_loss_reports_the_exact_counts() -> None:
    out = _flag_unapplied_music_bed(
        None, requested=False, unfetchable=False, ambient_requested=3, ambient_lost=1
    )
    assert out is not None
    assert out["ambient_requested"] == 3
    assert out["ambient_lost"] == 1
    assert out["ambient_fetch_failed"] is True


def test_ambient_fully_fetched_does_not_flag() -> None:
    assert (
        _flag_unapplied_music_bed(
            None, requested=False, unfetchable=False, ambient_requested=2, ambient_lost=0
        )
        is None
    )


def test_music_applied_and_ambient_lost_merges_both_without_clobbering() -> None:
    """A mixed request (music succeeded, ambient partially lost) must report
    BOTH facts — this is the shape a real production request carried (music +
    ambient in the same compile)."""
    out = _flag_unapplied_music_bed(
        _REPORT, requested=True, unfetchable=False, ambient_requested=1, ambient_lost=1
    )
    assert out is not None
    assert out["duck_db"] == -12.0  # the music report's own fields survive
    assert out["vo_lufs"] == -18.0
    assert out["ambient_requested"] == 1
    assert out["ambient_lost"] == 1
    assert out["ambient_fetch_failed"] is True
