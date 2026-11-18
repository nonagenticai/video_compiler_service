"""Real-ffmpeg render-harness tests for thumbnail / poster GRADE PARITY (spec
07-color-finish §E, AC:147).

Gated at MODULE level (skipped without ffmpeg/ffprobe) and tagged ``render`` so
the real-render suite selects/deselects as a group — same pattern as
``test_grade_render.py``. MinIO is NOT exercised: ``upload_and_presign`` is
stubbed so the extracted PNG stays on disk for pixel/dimension assertions.

╔══════════════════════════════════════════════════════════════════════════╗
║ PARITY PROOF: ``test_thumbnail_grade_parity`` proves the poster frame is   ║
║ graded through the IDENTICAL ``_grade_vf`` chain — the graded candidate    ║
║ differs from the ungraded one AND is byte-identical to a frame produced by ║
║ applying ``-vf {_grade_vf(grade)}`` by hand at the same seek. If the       ║
║ service stopped reusing ``_grade_vf`` (or applied a different chain) the   ║
║ identity assertion fails.                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from src.api.schemas import GradeSpec
from src.config import Settings, get_settings
from src.services import thumbnail as thumbnail_service
from src.services.renderer import _ffprobe_duration, _grade_vf, _run, probe_dimensions
from src.services.thumbnail import _timestamps, extract_candidates
from tests.render_harness import make_test_clip

pytestmark = [
    pytest.mark.render,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not on runner",
    ),
]


def _settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(update={"work_dir": str(tmp_path)})


def _stub_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the MinIO uploader with a stub that leaves the PNG on disk (so the
    test can read it back) and returns a deterministic fake presigned URL."""

    async def fake_upload(  # type: ignore[no-untyped-def]
        settings, *, bucket, object_key, file_path, expiry_seconds, content_type="image/png"
    ):
        assert Path(file_path).exists(), "service uploaded a missing PNG"
        assert content_type == "image/png"
        return f"https://minio.test/{bucket}/{object_key}?sig=x"

    monkeypatch.setattr(thumbnail_service, "upload_and_presign", fake_upload)


async def test_thumbnail_extracts_n_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extract 3 real frames from a synthetic clip: 3 candidates, timestamps
    strictly increasing and inside (0, dur), and every PNG has the source dims."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=4.0, size="320x240", fps=25)
    _stub_upload(monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    cands = await extract_candidates(
        src, 3, fmt=None, grade=None, settings=_settings(tmp_path), work_dir=work_dir
    )

    assert len(cands) == 3
    dur = await _ffprobe_duration(src)
    ts = [c.timestamp_seconds for c in cands]
    assert ts == sorted(ts) and len(set(ts)) == 3, f"timestamps not strictly increasing: {ts}"
    assert all(0.0 < t < dur for t in ts), f"timestamps outside (0,{dur}): {ts}"

    for i, c in enumerate(cands):
        assert c.width == 320 and c.height == 240, (c.width, c.height)
        png = work_dir / f"thumb-{i}.png"
        assert png.exists(), f"missing extracted frame {png}"
        assert await probe_dimensions(png) == (320, 240)
        # Safe rects: 90% action-safe, 80% title-safe, centered.
        assert c.action_safe.width == round(320 * 0.90)
        assert c.title_safe.width == round(320 * 0.80)
        assert c.action_safe.x == (320 - c.action_safe.width) // 2


async def test_thumbnail_grade_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame extracted with a grade DIFFERS from the ungraded one and MATCHES a
    reference produced by applying ``-vf {_grade_vf(grade)}`` by hand at the same
    seek — i.e. the poster is graded through the identical whole-film chain."""
    src = await make_test_clip(tmp_path / "src.mp4", duration=4.0, size="320x240", fps=25)
    _stub_upload(monkeypatch)
    settings = _settings(tmp_path)
    grade = GradeSpec(
        enabled=True,
        contrast=1.4,
        saturation=1.35,
        shadow_tint="#0B1437",
        highlight_tint="#EA580C",
    )
    assert _grade_vf(grade), "test grade must produce a non-empty filter chain"

    graded_dir = tmp_path / "graded"
    graded_dir.mkdir()
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()

    g = await extract_candidates(
        src, 1, fmt=None, grade=grade, settings=settings, work_dir=graded_dir
    )
    p = await extract_candidates(
        src, 1, fmt=None, grade=None, settings=settings, work_dir=plain_dir
    )
    assert len(g) == 1 and len(p) == 1

    graded_png = (graded_dir / "thumb-0.png").read_bytes()
    plain_png = (plain_dir / "thumb-0.png").read_bytes()

    # Reference: apply the SAME grade chain by hand at the SAME deterministic seek.
    dur = await _ffprobe_duration(src)
    t = _timestamps(dur, 1)[0]
    ref_png_path = tmp_path / "ref.png"
    await _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-vf",
            _grade_vf(grade),
            str(ref_png_path),
        ]
    )
    ref_png = ref_png_path.read_bytes()

    graded_hash = hashlib.sha256(graded_png).hexdigest()
    plain_hash = hashlib.sha256(plain_png).hexdigest()
    ref_hash = hashlib.sha256(ref_png).hexdigest()

    assert graded_hash != plain_hash, "grade did not change the extracted frame"
    assert graded_hash == ref_hash, (
        "graded frame != identical-chain reference — the service is not reusing _grade_vf"
    )
