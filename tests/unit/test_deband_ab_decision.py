"""Unit tests for ``scripts/deband_ab.py``'s pure ``decide()`` — the branch
classifier over an A/B measurement batch. NOT integration-marked: these run
in pytest's default selection (no ``@pytest.mark.integration``), unlike the
harness script itself, which needs real ffmpeg + real material and is invoked
by hand.

One case per branch (destroys / no_op / helps_selectively / helps_broadly)
plus the empty-batch edge, all against synthetic ``SegmentRecord``s — no
filesystem, no ffmpeg, no network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_deband_ab_module() -> ModuleType:
    """Import ``scripts/deband_ab.py`` by path (mirrors
    ``tests/test_contract_artifact.py``'s ``_load_regen_module``): ``scripts/``
    is not a distributed package, so it may not be importable by name
    depending on how pytest was invoked.

    Registered in ``sys.modules`` before ``exec_module`` — ``deband_ab.py``'s
    ``from __future__ import annotations`` dataclasses need their defining
    module resolvable via ``sys.modules[cls.__module__]`` while they build
    (``dataclasses._is_type``), which only exists once the module is
    registered."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "deband_ab.py"
    spec = importlib.util.spec_from_file_location("deband_ab", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_deband_ab = _load_deband_ab_module()
SegmentRecord = _deband_ab.SegmentRecord
decide = _deband_ab.decide


def _record(
    segment: str,
    *,
    before: float,
    after: float,
    ssim: float | None = 0.999,
) -> SegmentRecord:
    """Build a SegmentRecord with only the fields ``decide()`` reads varying;
    everything else is a plausible filler value."""
    return SegmentRecord(
        segment=segment,
        max_banding_before=before,
        max_banding_after=after,
        p95_banding_before=before,
        p95_banding_after=after,
        mean_banding_before=before,
        mean_banding_after=after,
        frac_over_ceiling_before=1.0 if before >= 0.15 else 0.0,
        frac_over_ceiling_after=1.0 if after >= 0.15 else 0.0,
        ssim=ssim,
        vmaf=None,
        measure_before_seconds=0.1,
        deband_seconds=1.0,
        measure_after_seconds=0.1,
        compare_seconds=0.2,
    )


def test_decide_no_data_on_empty_batch() -> None:
    decision = decide([])
    assert decision.verdict == "no_data"
    assert decision.n_segments == 0


def test_decide_no_op_when_reduction_and_ssim_both_flat() -> None:
    # Banding barely moves (0.01 < the 0.02 floor) and SSIM stays essentially
    # perfect (> 0.995) on every segment — deband is not doing anything visible.
    records = [
        _record("seg-0001.mp4", before=0.20, after=0.19, ssim=0.999),
        _record("seg-0002.mp4", before=0.18, after=0.175, ssim=0.998),
        _record("seg-0003.mp4", before=0.16, after=0.15, ssim=0.9995),
    ]
    decision = decide(records)
    assert decision.verdict == "no_op"
    assert decision.median_max_banding_reduction is not None
    assert decision.median_max_banding_reduction < 0.02
    assert decision.median_ssim is not None
    assert decision.median_ssim > 0.995


def test_decide_helps_selectively_when_minority_cross_the_floor() -> None:
    # 11 segments so a bare *majority* (>50%, enough to carry the median past
    # the 0.02 floor) still lands well under the 60% broad-help bar: 6 of 11
    # (54.5%) reduce max banding by >= 0.02, the other 5 barely move, and SSIM
    # never drops below the destruction ceiling. This is the case that
    # distinguishes "median reduction crossed the floor" from "most segments
    # crossed the floor" — the two are NOT the same test at n=3 (any median
    # >= floor there forces exactly the 60% bar), so it takes this shape to
    # separate the helps_selectively branch from helps_broadly.
    records = [_record(f"seg-{i:04d}.mp4", before=0.30, after=0.275, ssim=0.99) for i in range(6)]
    records += [
        _record(f"seg-1{i:03d}.mp4", before=0.20, after=0.199, ssim=0.999) for i in range(5)
    ]
    decision = decide(records)
    assert decision.verdict == "helps_selectively"
    assert decision.median_max_banding_reduction is not None
    assert decision.median_max_banding_reduction >= 0.02
    assert decision.helped_fraction is not None
    assert decision.helped_fraction < 0.60


def test_decide_helps_broadly_when_majority_cross_the_floor() -> None:
    # 2 of 3 segments (67%, >= the 60% bar) reduce max banding by >= 0.02.
    records = [
        _record("seg-0001.mp4", before=0.30, after=0.10, ssim=0.99),  # reduction 0.20
        _record("seg-0002.mp4", before=0.25, after=0.05, ssim=0.99),  # reduction 0.20
        _record("seg-0003.mp4", before=0.18, after=0.178, ssim=0.999),  # ~0
    ]
    decision = decide(records)
    assert decision.verdict == "helps_broadly"
    assert decision.helped_fraction is not None
    assert decision.helped_fraction >= 0.60


def test_decide_destroys_detail_outranks_a_broad_help_reading() -> None:
    # Banding reduction alone would read as "helps broadly" (2/3 segments
    # cross the floor by a wide margin) but one segment's SSIM crashed below
    # the 0.98 destruction ceiling — that must win regardless of the banding
    # numbers.
    records = [
        _record("seg-0001.mp4", before=0.30, after=0.10, ssim=0.99),
        _record("seg-0002.mp4", before=0.25, after=0.05, ssim=0.90),  # destroyed
        _record("seg-0003.mp4", before=0.18, after=0.178, ssim=0.999),
    ]
    decision = decide(records)
    assert decision.verdict == "destroys_detail"
    assert decision.median_ssim is not None  # the banding read is still reported


def test_decide_verdict_is_stable_regardless_of_input_order() -> None:
    records = [
        _record("seg-0001.mp4", before=0.30, after=0.10, ssim=0.99),
        _record("seg-0002.mp4", before=0.25, after=0.05, ssim=0.99),
        _record("seg-0003.mp4", before=0.18, after=0.178, ssim=0.999),
    ]
    forward = decide(records)
    backward = decide(list(reversed(records)))
    assert forward.verdict == backward.verdict == "helps_broadly"
