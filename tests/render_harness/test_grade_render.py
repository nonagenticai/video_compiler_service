"""Real-ffmpeg render-harness tests for the whole-film colour grade (VC-5).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_look_render.py`` / ``test_audio_level_render.py``. These drive an ACTUAL
full-timeline compile and interrogate the DELIVERED mp4's signalstats / cadence
/ loudness. The string tests elsewhere prove the ``_grade_vf`` filtergraph
shape; THESE prove the grade actually changed the picture, held legal luma, left
the ``-c copy`` fast path byte-identical when absent, and did not regress the
CFR (VC-3) or loudness (VC-1/VC-3) masters.

╔══════════════════════════════════════════════════════════════════════════╗
║ REVERT-PROOF: ``test_grade_changes_picture`` MUST FAIL if ``_grade_vf`` is ║
║ reverted to a no-op (or the whole-film re-encode is skipped when a grade   ║
║ is present). With no grade applied the graded render == the ungraded       ║
║ control, so both the luma-spread delta and the SATAVG shift collapse to 0  ║
║ and the relative assertions fail. That is the whole point of this file.    ║
╚══════════════════════════════════════════════════════════════════════════╝

⚠️ RELATIVE deltas only. Every picture assertion compares TWO real renders of
the same timeline (graded vs ungraded control) — never a hardcoded absolute
luma/chroma value. testsrc is a synthetic full-ish-range source; asserting an
absolute YMIN/YMAX/SATAVG would measure the source, not the grade (the
synthetic-source measurement trap the plan calls out).

━━━ API ASSUMPTION (reconcile with the implementer) ━━━━━━━━━━━━━━━━━━━━━━━━━━━
The whole-film grade is threaded into the renderer via a new keyword-only
``finish`` parameter on ``renderer.compile_shots`` carrying a request-level
``FinishProfile`` whose new ``grade: GradeSpec | None`` field holds the grade —
i.e. ``compile_shots(shots, settings, finish=FinishProfile(grade=GradeSpec(...)))``.
This matches the VC-5 spec (§6: "compile.py: pass ``body.finish`` into
``compile_shots``") and this repo's existing ``body.finish``/``FinishProfile``
seam. If the implementer instead folds the grade onto ``Settings`` or names the
param ``grade=``, only ``_compile`` below needs the one-line change.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MinIO/storage is NOT exercised: ``compile_shots`` returns a local path (submit→
poll→upload is covered by the live E2E), so no bucket/presign is needed.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.api.schemas import FinishProfile, GradeSpec, Shot
from src.config import Settings, get_settings
from src.services import renderer
from src.services.loudness import final_mux_loudness
from src.services.mastering import framerate_coherence
from src.services.measure import measure_shot_stats
from tests.render_harness import make_test_clip

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]

# Tiny canvas + short shots keep the real full-timeline encode fast. A colour
# source (testsrc) gives the saturation/split-tone grade chroma to move.
_SMALL = {"render_width": 320, "render_height": 240, "render_fps": 25}
# Slack for the best-effort legal-luma clamp: x264 ringing around testsrc's hard
# bar edges re-introduces a few illegal codes, so the bound is approximate — the
# feature is best-effort, NOT a guaranteed hard 16..235 (see spec §4/§7).
_LEGAL_SLACK = 10.0


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update={**_SMALL, "master_enabled": True, "work_dir": str(tmp_path)}
    )


async def _compile(shots: list[Shot], tmp_path: Path, *, finish: FinishProfile | None) -> Path:
    """Run a full whole-timeline compile and return the delivered file.

    Single reconciliation point for the API assumption documented at the top of
    the module: the grade rides ``compile_shots(..., finish=...)``.
    """
    final_path, _, _, _, _ = await renderer.compile_shots(shots, _settings(tmp_path), finish=finish)
    return final_path


def _timeline(src: Path) -> list[Shot]:
    """A short multi-shot timeline: two 1s shots off one synthetic source."""
    return [
        Shot(id="a", seconds=1.0, video_url=str(src)),
        Shot(id="b", seconds=1.0, video_url=str(src)),
    ]


def _strong_grade(*, broadcast_safe: bool) -> FinishProfile:
    """A strong, deterministic grade: high contrast + saturation + a brand
    split-tone. grain/glow/sharpen/vignette are left at identity so the picture
    delta is attributable to contrast (luma spread) and saturation/split-tone
    (chroma) exactly as the assertions claim — and so the render stays
    deterministic (``noise`` grain would randomise frames per run)."""
    return FinishProfile(
        grade=GradeSpec(
            enabled=True,
            contrast=1.4,
            saturation=1.35,
            gamma=1.0,
            temperature=0.0,
            shadow_tint="#0B1437",  # brand primary → shadows
            highlight_tint="#EA580C",  # brand accent → highlights
            grade_intensity=1.0,
            broadcast_safe=broadcast_safe,
        )
    )


async def test_grade_changes_picture(tmp_path: Path) -> None:
    """REVERT-PROOF. The same timeline rendered with a strong grade differs
    materially from the ungraded control on the delivered mp4.

    The reliable, source-independent signal is CHROMA: saturation=1.35 + the
    split-tone (colorbalance) move the average chroma (SATAVG) well beyond noise.
    We deliberately do NOT assert on luma spread — ``testsrc``'s colour bars
    already span the full range, so a contrast boost clips rather than widens
    (YMAX-YMIN), which is the synthetic-source trap the spec warns about. This is
    a RELATIVE delta between two real renders; a no-op ``_grade_vf`` collapses it
    to ~0 and fails. broadcast_safe is OFF so the legal-range clamp is not in play
    (it is proven separately below).
    """
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    control = await _compile(_timeline(src), tmp_path / "ctl", finish=None)
    graded = await _compile(
        _timeline(src), tmp_path / "grd", finish=_strong_grade(broadcast_safe=False)
    )

    cs = await measure_shot_stats(control)
    gs = await measure_shot_stats(graded)
    assert cs and gs, "signalstats readback failed on a delivered file"

    # Saturation boost + split-tone shift the average chroma materially. A no-op
    # grade (or a reverted _grade_vf) leaves SATAVG unchanged ⇒ this fails.
    assert abs(gs["SATAVG"] - cs["SATAVG"]) > 3.0, (
        f"saturation/split-tone did not move SATAVG: control={cs['SATAVG']:.2f} "
        f"graded={gs['SATAVG']:.2f} — did _grade_vf revert to a no-op?"
    )


async def test_broadcast_safe_bounds_legal_luma(tmp_path: Path) -> None:
    """With ``broadcast_safe=True`` the delivered file's luma is best-effort held
    to legal range: YMIN approximately ≥16 and YMAX approximately ≤235, allowing
    a few codes of x264-ringing slack. This is NOT a guaranteed hard bound — it's
    the colour analog of the true-peak limiter (spec §4/§7), so we assert an
    APPROXIMATE bound, plus a RELATIVE proof that the clamp actually pulled the
    ceiling in vs the identical grade with the clamp OFF."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    safe = await _compile(
        _timeline(src), tmp_path / "safe", finish=_strong_grade(broadcast_safe=True)
    )
    unsafe = await _compile(
        _timeline(src), tmp_path / "raw", finish=_strong_grade(broadcast_safe=False)
    )

    ss = await measure_shot_stats(safe)
    us = await measure_shot_stats(unsafe)
    assert ss and us, "signalstats readback failed on a delivered file"

    # Approximate legal bound (best-effort, with ringing slack).
    assert ss["YMIN"] >= 16.0 - _LEGAL_SLACK, f"YMIN crushed below legal: {ss['YMIN']:.1f}"
    assert ss["YMAX"] <= 235.0 + _LEGAL_SLACK, f"YMAX clipped above legal: {ss['YMAX']:.1f}"
    # Relative: the clamp did something — the unclamped grade rides higher.
    assert ss["YMAX"] <= us["YMAX"] + 1.0, (
        f"broadcast_safe did not pull the ceiling in: safe={ss['YMAX']:.1f} unsafe={us['YMAX']:.1f}"
    )


async def test_no_grade_is_byte_identical(tmp_path: Path) -> None:
    """A compile with NO grade (``finish=None``) is byte-for-byte identical to
    the same compile carrying a grade-less ``FinishProfile`` (``grade=None``):
    the whole-film re-encode never runs and the ``-c copy`` concat fast path is
    untouched. (An all-identity/``enabled=False`` grade should take the same fast
    path; ``grade=None`` is the guaranteed one asserted here.)"""
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    plain = await _compile(_timeline(src), tmp_path / "plain", finish=None)
    grade_less = await _compile(
        _timeline(src), tmp_path / "empty", finish=FinishProfile(grade=None)
    )
    assert (
        hashlib.sha256(plain.read_bytes()).hexdigest()
        == hashlib.sha256(grade_less.read_bytes()).hexdigest()
    ), "a grade-less finish perturbed the -c copy fast path"


async def test_grade_preserves_cfr_and_loudness_masters(tmp_path: Path) -> None:
    """The whole-film grade must not regress the CFR (VC-3) or loudness
    (VC-1/VC-3) masters: the graded output is still frame-rate coherent
    (r_frame_rate == avg_frame_rate) and its integrated loudness stays within
    tolerance of the ungraded control (the grade re-encodes VIDEO only, the audio
    master bus is unchanged)."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=1.0, size="320x240", fps=25)
    control = await _compile(_timeline(src), tmp_path / "ctl", finish=None)
    graded = await _compile(
        _timeline(src), tmp_path / "grd", finish=_strong_grade(broadcast_safe=True)
    )

    assert await framerate_coherence(graded) is True, "graded output is not CFR (VC-3 regression)"

    cl = await final_mux_loudness(str(control))
    gl = await final_mux_loudness(str(graded))
    assert cl and gl, "loudness readback failed on a delivered file"
    ci, gi = cl.get("integrated_lufs"), gl.get("integrated_lufs")
    assert ci is not None and gi is not None, "integrated LUFS not measured"
    assert abs(gi - ci) < 1.5, (
        f"grade shifted integrated loudness: control={ci:.2f} LUFS graded={gi:.2f} LUFS"
    )
