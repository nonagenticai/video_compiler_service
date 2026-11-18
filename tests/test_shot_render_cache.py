"""Unit tests for the content-addressed per-shot render cache (C10/AC10).

These do NOT touch ffmpeg: they exercise the cache KEY (stability + sensitivity)
and the get/put file materialization directly. The real-render proof — that a
2-shot re-compile with one changed shot re-renders exactly one segment — lives in
``tests/render_harness/test_shot_cache_render.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.api.schemas import Callout, CursorKey, FocusKey, Shot
from src.config import Settings, get_settings
from src.services import renderer


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"work_dir": str(tmp_path)})


def test_shot_cache_key_is_stable_across_repeated_calls(tmp_path: Path) -> None:
    """The same shot + settings hashes to the same key every time (no clock /
    dict-ordering / object-identity leakage)."""
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", caption="hi")
    k1 = renderer._shot_cache_key(shot, settings, 0.25)
    k2 = renderer._shot_cache_key(shot, settings, 0.25)
    assert k1 == k2
    # Reconstructing an equal shot must also collide (value-, not identity-based).
    shot_again = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", caption="hi")
    assert renderer._shot_cache_key(shot_again, settings, 0.25) == k1


def test_shot_cache_key_ignores_pixel_irrelevant_line_id(tmp_path: Path) -> None:
    """``line_id`` is a pure identifier render_shot never reads, so it must not
    move the key (else the loop would needlessly re-render on a relabel)."""
    settings = _settings(tmp_path)
    base = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", line_id="L1")
    relabelled = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", line_id="L2")
    assert renderer._shot_cache_key(base, settings, 0.0) == renderer._shot_cache_key(
        relabelled, settings, 0.0
    )


def test_shot_cache_key_changes_on_pixel_affecting_shot_fields(tmp_path: Path) -> None:
    """Every field that reaches a segment's pixels/timing must shift the key."""
    settings = _settings(tmp_path)
    base = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4")
    base_key = renderer._shot_cache_key(base, settings, 0.0)

    variants = [
        base.model_copy(update={"seconds": 4.0}),
        base.model_copy(update={"max_seconds": 5.5}),
        base.model_copy(update={"speed": 2.0}),
        base.model_copy(update={"video_url": "http://x/other.mp4"}),
        base.model_copy(update={"image_url": "http://x/img.png"}),
        base.model_copy(update={"audio_url": "http://x/vo.wav"}),
        base.model_copy(update={"caption": "label"}),
        base.model_copy(update={"fit": "cover"}),
        base.model_copy(update={"footage_viewport": (100, 200)}),
        base.model_copy(update={"focus_track": [FocusKey(t=0.0, bbox=(0.0, 0.0, 10.0, 10.0))]}),
        base.model_copy(update={"cursor_track": [CursorKey(t=0.0, x=5.0, y=6.0)]}),
        base.model_copy(
            update={
                "callouts": [Callout(at=0.0, type="badge", bbox=(0.0, 0.0, 5.0, 5.0), label="1")]
            }
        ),
    ]
    keys = {renderer._shot_cache_key(v, settings, 0.0) for v in variants}
    assert base_key not in keys
    # Each variant must be distinct from every other (no accidental collisions).
    assert len(keys) == len(variants)


def test_shot_cache_key_changes_on_clip_seek_fraction(tmp_path: Path) -> None:
    """Reused footage panned to a different in-point is different pixels."""
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4")
    assert renderer._shot_cache_key(shot, settings, 0.0) != renderer._shot_cache_key(
        shot, settings, 0.5
    )


def test_shot_cache_key_changes_on_freeze_pad_cap_flag(tmp_path: Path) -> None:
    """The job-level ``freeze_pad_cap`` enforcement flag changes an over-narrated
    footage shot's drift pixels, so a capped render must not collide with an
    uncapped one — even though the shot (incl. ``max_seconds``) is identical. The
    default (flag off) key is UNCHANGED, so the existing pod cache stays valid."""
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", max_seconds=4.5)
    off = renderer._shot_cache_key(shot, settings, 0.0)
    on = renderer._shot_cache_key(shot, settings, 0.0, freeze_pad_cap=True)
    assert off != on
    # Flag off == the pre-existing default key (no argument), so no cache churn.
    assert off == renderer._shot_cache_key(shot, settings, 0.0, deterministic=False)


def test_shot_cache_key_changes_on_global_render_setting(tmp_path: Path) -> None:
    """A resolved render dimension (per-job format/tier override) must shift the
    key so a 9x16 segment can't be served for a 16x9 request."""
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4")
    s_16x9 = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "render_width": 1920, "render_height": 1080}
    )
    s_9x16 = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "render_width": 1080, "render_height": 1920}
    )
    assert renderer._shot_cache_key(shot, s_16x9, 0.0) != renderer._shot_cache_key(
        shot, s_9x16, 0.0
    )
    # fps is also segment-affecting.
    s_60 = s_16x9.model_copy(update={"render_fps": 60})
    assert renderer._shot_cache_key(shot, s_16x9, 0.0) != renderer._shot_cache_key(shot, s_60, 0.0)


def test_intermediate_crf_delta_is_not_segment_affecting(tmp_path: Path) -> None:
    """``render_intermediate_crf_delta`` is deliberately EXCLUDED from the key.

    It is only ever read through ``mastering.encode_ladder(..., intermediate=True)``,
    and the only three callers that pass that flag (``_deband_segment`` /
    ``_regrade_segment`` / ``_polish_segment``) run from ``_apply_picture_master``
    — i.e. AFTER ``_shot_cache_put``, against the caller's own copy of the segment.
    The encode that produces the CACHED bytes never passes the flag, so the delta
    cannot reach them.

    Two assertions, because either half alone would be vacuous:
      1. the key does not move (else every delta change is a full, pointless
         cache invalidation), and
      2. the cached render path's argv is genuinely delta-independent — which is
         what MAKES (1) safe. If a future change ever routes the delta into the
         segment encode itself, (2) fails and the exclusion must be revisited.
    """
    base = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "render_crf": 18, "render_intermediate_crf_delta": 0}
    )
    bumped = base.model_copy(update={"render_intermediate_crf_delta": 6})
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4")

    assert renderer._shot_cache_key(shot, base, 0.0) == renderer._shot_cache_key(shot, bumped, 0.0)
    # The segment encode (what render_shot emits into the cache) ignores the delta.
    assert renderer._video_codec_args(base) == renderer._video_codec_args(bumped)
    # ...whereas the post-cache polish encode does NOT — proving the knob is live
    # and that the equality above is a real property, not a dead comparison.
    assert renderer._video_codec_args(base, intermediate=True) != renderer._video_codec_args(
        bumped, intermediate=True
    )


async def test_shot_cache_get_is_a_cold_miss_before_put(tmp_path: Path) -> None:
    """No entry → ``None`` and the destination is left untouched (caller renders)."""
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=2.0, video_url="http://x/clip.mp4")
    key = renderer._shot_cache_key(shot, settings, 0.0)
    dest = tmp_path / "seg.mp4"
    assert await renderer._shot_cache_get(key, dest, settings) is None
    assert not dest.exists()


async def test_shot_cache_put_then_get_materializes_without_render(tmp_path: Path) -> None:
    """Put a rendered segment, then a hit copies the exact bytes back and returns
    the stored duration — no render invocation involved."""
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=2.0, video_url="http://x/clip.mp4")
    key = renderer._shot_cache_key(shot, settings, 0.0)

    produced = tmp_path / "produced.mp4"
    produced.write_bytes(b"SEGMENT-PIXELS")
    await renderer._shot_cache_put(key, produced, 1.75, settings)

    dest = tmp_path / "materialized.mp4"
    duration = await renderer._shot_cache_get(key, dest, settings)
    assert duration == 1.75
    assert dest.read_bytes() == b"SEGMENT-PIXELS"
    # It is a real copy, not a symlink — deband/cleanup on ``dest`` must not reach
    # back into the shared cache entry.
    assert not dest.is_symlink()


async def test_shot_cache_get_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    """A segment present without its duration sidecar is a half-entry and must
    read as a miss, never a hit on unknown duration."""
    settings = _settings(tmp_path)
    key = "deadbeef"
    cache_dir = renderer._shot_cache_dir(settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.mp4").write_bytes(b"orphan")
    dest = tmp_path / "seg.mp4"
    assert await renderer._shot_cache_get(key, dest, settings) is None
    assert not dest.exists()


async def test_shot_cache_disabled_when_max_bytes_zero(tmp_path: Path) -> None:
    """``shot_cache_max_bytes=0`` disables the cache: put writes nothing and get
    is always a miss, so the compiler falls back to a normal render."""
    settings = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "shot_cache_max_bytes": 0}
    )
    shot = Shot(id="a", seconds=2.0, video_url="http://x/clip.mp4")
    key = renderer._shot_cache_key(shot, settings, 0.0)
    produced = tmp_path / "produced.mp4"
    produced.write_bytes(b"SEGMENT")
    await renderer._shot_cache_put(key, produced, 1.0, settings)
    assert not renderer._shot_cache_dir(settings).exists()
    dest = tmp_path / "seg.mp4"
    assert await renderer._shot_cache_get(key, dest, settings) is None


async def test_shot_cache_evicts_oldest_over_cap(tmp_path: Path) -> None:
    """Once the cache exceeds ``shot_cache_max_bytes``, publishing a new entry
    evicts the oldest one so the cache stays bounded on the long-lived pod."""
    # Cap that holds ~2 small segments but not 3.
    settings = get_settings().model_copy(
        update={"work_dir": str(tmp_path), "shot_cache_max_bytes": 4096}
    )
    blob = b"x" * 1500  # entry ≈ segment + tiny sidecar

    async def _put(key: str) -> None:
        src = tmp_path / f"{key}.src"
        src.write_bytes(blob)
        await renderer._shot_cache_put(key, src, 1.0, settings)

    cache_dir = renderer._shot_cache_dir(settings)
    await _put("oldest")
    # Age the first entry so mtime ordering is unambiguous.
    old_seg = cache_dir / "oldest.mp4"
    os.utime(old_seg, (1.0, 1.0))
    await _put("mid")
    await _put("newest")

    remaining = {p.stem for p in cache_dir.glob("*.mp4")}
    assert "oldest" not in remaining  # evicted first
    assert "newest" in remaining
    total = sum(p.stat().st_size for p in cache_dir.iterdir())
    assert total <= 4096


def test_every_settings_field_is_classified() -> None:
    """Determinism guard: every Settings field is EITHER in the shot-cache hashed
    view OR explicitly excluded — never silently unclassified. A new
    segment-affecting field forgotten from the hashed view would let a stale
    segment be served across a settings change; a new downstream/infra field
    forgotten from the excluded set trips this test so the author must classify
    it (and bump _SHOT_CACHE_KEY_VERSION if it belongs in the view)."""
    hashed = set(renderer._shot_cache_settings_view(get_settings()))
    classified = hashed | renderer._SHOT_CACHE_EXCLUDED_SETTINGS
    missing = set(Settings.model_fields) - classified
    assert not missing, (
        f"Settings field(s) {sorted(missing)} are neither in the shot-cache hashed view "
        f"nor in _SHOT_CACHE_EXCLUDED_SETTINGS. Add to the view (and bump "
        f"_SHOT_CACHE_KEY_VERSION) if segment-affecting, else to the excluded set."
    )


def test_shot_cache_key_version_is_folded_into_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bumping ``_SHOT_CACHE_KEY_VERSION`` must invalidate every existing entry.

    The version is the only lever that can retire cache entries whose *pixels*
    are still correct but whose persisted **metadata** is not. That is not
    hypothetical: the seeked-clip duration fix corrected the arithmetic without
    touching the key, so entries written before it kept hashing the same and
    ``_shot_cache_get`` replayed the pre-fix (overstated) duration out of the
    ``{key}.json`` sidecar — the bad number outlived the bug.

    This locks the lever in place: if a refactor ever drops ``"v"`` from the
    hashed payload, the version stops invalidating anything and that failure is
    silent (a stale hit is indistinguishable from a fresh one at the call site).
    """
    settings = _settings(tmp_path)
    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", caption="hi")
    baseline = renderer._shot_cache_key(shot, settings, 0.25)

    monkeypatch.setattr(renderer, "_SHOT_CACHE_KEY_VERSION", renderer._SHOT_CACHE_KEY_VERSION + 1)
    bumped = renderer._shot_cache_key(shot, settings, 0.25)
    monkeypatch.undo()

    assert bumped != baseline, (
        "_SHOT_CACHE_KEY_VERSION is not folded into the cache key, so bumping it "
        "cannot invalidate stale entries"
    )
    assert renderer._shot_cache_key(shot, settings, 0.25) == baseline


# ---------------------------------------------------------------------------
# Segment codec change → the pre-change (AAC) generation must be unreachable
# ---------------------------------------------------------------------------


def test_the_key_version_covers_the_segment_codec_change() -> None:
    """``_SHOT_CACHE_KEY_VERSION`` was bumped 7→8 WITH the segment audio codec
    change (AAC → lossless ALAC), and that bump is the ENTIRE barrier.

    It is the first entry in that log that is about samples rather than pixels, and
    skipping it would not merely serve stale pixels: a cached AAC segment replayed
    for one shot beside a freshly rendered ALAC one gets stream-COPIED into the
    same film by ``concat_shots`` — a heterogeneous join that reintroduces the
    24 ms-per-cut hole the codec change removes, intermittently, on whichever
    shots happen to hit the cache. Reverting the bump alone makes today's key
    collide with a pre-change entry again, which is what this pins.
    """
    assert renderer._SHOT_CACHE_KEY_VERSION >= 8
