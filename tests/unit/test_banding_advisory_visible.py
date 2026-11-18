"""A master whose shots flag ``needs_deband`` must not report a silent, empty verdict.

MEASURED on delivered run ``01a0ab23-6c47-76a2-9d53-f16220e4792d``: ``needs_deband``
true on ALL THREE shots (banding 0.262 / 0.259 / 0.5935), ``gate_pass: true``,
``violations: []``. Nothing at the top level of the report said banding had been
over the trigger and deliberately excluded from the verdict.

The residual banding score is NOT a gate (see ``mastering._banding_violation``: its
calibration is unmeasured on the current formula), so the fix is VISIBILITY, not
arming: the breach is published in ``advisories`` with the same five-key shape as a
violation. It is kept OUT of ``violations`` on purpose — both this repo and the
orchestrator consumer define every ``violations`` entry as a check that CLEARED
``gate_pass``, and under ``mastering_gate_mode=enforce`` every violation stage is
armed, so an entry there would fail the compile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import MasteringReport
from src.config import Settings, get_settings
from src.services import mastering, renderer

CLEAN_FRAME = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 9.0}
BANDED_FRAME = {"YMIN": 16.0, "YMAX": 235.0, "YDIF": 0.5}

_SHOTS: list[dict[str, object]] = [
    {"line_id": "a", "start": 0.0, "end": 0.03},
    {"line_id": "b", "start": 0.03, "end": 1.0},
]


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=dict(overrides))


def _fake_run(dump: str):
    async def run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "mpdecimate" in joined:
            return "frame= 3\n"
        return dump

    return run


def _frames_dump(frames: list[dict[str, float]]) -> str:
    out = []
    for i, f in enumerate(frames):
        out.append(f"frame:{i} pts:{i * 512} pts_time:{i * 0.017:.3f}\n")
        out.append(
            f"lavfi.signalstats.YMIN={f['YMIN']}\n"
            f"lavfi.signalstats.YMAX={f['YMAX']}\n"
            f"lavfi.signalstats.YDIF={f['YDIF']}\n"
        )
    return "".join(out)


async def _verify(
    monkeypatch: pytest.MonkeyPatch, frames: list[dict[str, float]], settings: Settings
) -> dict[str, object]:
    monkeypatch.setattr(renderer, "_run", _fake_run(_frames_dump(frames)))
    return await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        settings,
        loudness={"integrated_lufs": float(settings.master_lufs), "true_peak_dbtp": -1.1},
        shot_timeline=_SHOTS,
    )


@pytest.mark.asyncio
async def test_shots_flagging_needs_deband_publish_a_structured_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = await _verify(monkeypatch, [BANDED_FRAME] * 3, _settings())

    per_shot = report["per_shot"]
    assert isinstance(per_shot, list)
    # Not vacuous: the shots really do carry the flag that used to be the only signal.
    assert any(shot["needs_deband"] for shot in per_shot), per_shot
    scores = report["banding_scores"]
    assert isinstance(scores, list)
    worst = max(scores)
    assert worst >= mastering.BANDING_GATE_CEILING

    assert report["advisories"] == [
        {
            "stage": "banding_residual",
            "metric": "banding_scores",
            "value": worst,
            "limit": mastering.BANDING_GATE_CEILING,
            "tier": "tunable",
        }
    ]
    # Visibility only — the verdict is exactly what it was before this channel existed.
    assert report["gate_pass"] is True, report
    assert report["violations"] == [], report
    rerun_stages = report["rerun_stages"]
    assert isinstance(rerun_stages, list)
    assert "banding_residual" not in rerun_stages


@pytest.mark.asyncio
async def test_a_clean_master_publishes_no_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    report = await _verify(monkeypatch, [CLEAN_FRAME] * 3, _settings())

    per_shot = report["per_shot"]
    assert isinstance(per_shot, list)
    assert per_shot, "the clean case must still bucket shots, or the assertion below is vacuous"
    assert not any(shot["needs_deband"] for shot in per_shot)
    assert report["advisories"] == []


@pytest.mark.asyncio
async def test_the_advisory_never_fails_a_compile_even_under_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mastering_gate_mode=enforce`` arms EVERY stage that reaches ``violations``.
    The advisory must not be one of them, or a banded master would stop shipping."""
    report = await _verify(
        monkeypatch, [BANDED_FRAME] * 3, _settings(mastering_gate_mode="enforce")
    )

    assert report["gate_pass"] is True
    assert report["advisories"], report


@pytest.mark.asyncio
async def test_the_advisory_survives_the_response_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """pydantic drops undeclared keys silently — the way ``dead_frame_ratio`` and
    ``violations`` were both once computed and seen by nobody."""
    report = await _verify(monkeypatch, [BANDED_FRAME] * 3, _settings())

    dumped = MasteringReport.model_validate(report).model_dump(mode="json")
    assert dumped["advisories"] == report["advisories"]
    assert dumped["advisories"][0]["stage"] == "banding_residual"


def test_a_degraded_report_carries_no_advisory() -> None:
    """Nothing was measured, so nothing can be over the trigger."""
    assert mastering.degraded_report()["advisories"] == []
