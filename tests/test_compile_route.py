"""Route-level tests with renderer + storage mocked.

These do NOT exercise ffmpeg or MinIO — they prove the request shape is
validated, the async submit→poll contract matches the orchestrator's
``VideoCompileClient`` expectations, and tags propagate as the object-key
namespace. The render runs in a background task, so each test submits then polls
``/api/jobs/{job_id}`` until the job reaches a terminal status.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest
from httpx import AsyncClient, Response

from src.config import get_settings
from src.services import mastering, renderer
from src.services.captions_ass import CaptionLook

# Bound at import — BEFORE the autouse `_stub_loudness` fixture replaces the module
# attribute — so the enforce-mode tests below can run the REAL gate end to end.
from src.services.mastering import post_mux_verify as _real_post_mux_verify

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_loudness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep route tests ffmpeg-free: the real ``final_mux_loudness`` /
    ``post_mux_verify`` would exec ffmpeg over the fake mp4 (returning best-effort
    degraded values). Stub both to fixed results so the loudness- and
    mastering-surfacing assertions are deterministic. Tests that need a different
    value re-patch after this autouse fixture runs."""

    async def fake_loudness(path: str) -> dict[str, float | None]:
        return {
            "integrated_lufs": -12.4,
            "true_peak_dbtp": -1.1,
            "lra": 6.8,
            "silence_ratio": 0.05,
        }

    async def fake_verify(
        path, settings, *, loudness=None, shot_timeline=None, shots=None, narration_audio_paths=None
    ):  # type: ignore[no-untyped-def]
        return {
            "colorspace_ok": True,
            "integrated_lufs": (loudness or {}).get("integrated_lufs"),
            "true_peak_dbtp": (loudness or {}).get("true_peak_dbtp"),
            "banding_scores": [0.0],
            "cfr_ok": True,
            "gate_pass": True,
            "rerun_stages": [],
        }

    monkeypatch.setattr("src.api.routes.compile.final_mux_loudness", fake_loudness)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", fake_verify)


async def _poll(client: AsyncClient, job_id: str, *, attempts: int = 200) -> Response:
    """Poll the job until it leaves ``processing`` (the background task is fast
    here — the renderer/upload are mocked)."""
    last: Response | None = None
    for _ in range(attempts):
        last = await client.get(f"/api/jobs/{job_id}")
        if last.status_code == 200 and last.json()["status"] != "processing":
            return last
        await asyncio.sleep(0.01)
    assert last is not None
    return last


async def test_compile_returns_artifact_url_and_duration(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    fake_mp4 = Path("/tmp/fake-compiled.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="overlay",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {
        "shots": [
            {"id": "s1", "line_id": "l1", "description": "First shot", "seconds": 2.5},
            {"id": "s2", "description": "Second shot", "seconds": 3.0},
        ],
        "tags": {"run_id": "run-abc"},
    }
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    accepted = submit.json()
    assert accepted["status"] == "processing"
    assert accepted["poll_url"] == f"/api/jobs/{accepted['job_id']}"

    done = await _poll(client, accepted["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    assert data["artifact_url"].startswith("https://")
    assert data["duration_seconds"] == 5.5
    assert data["bucket"] == "w2v-compiled-videos"
    assert data["object_key"].startswith("run-abc/")
    assert data["object_key"].endswith(".mp4")


async def test_compile_surfaces_shot_timeline_when_requested(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """With ``emit_shot_timeline=True`` the flag reaches ``compile_shots`` and the
    per-shot timeline it returns (4th tuple element) surfaces on the poll result."""
    fake_mp4 = Path("/tmp/fake-compiled-timeline.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    captured: dict[str, object] = {}
    timeline = [
        {"line_id": "l1", "start": 0.0, "end": 2.5},
        {"line_id": None, "start": 2.5, "end": 5.5},
    ]

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        captured["emit_shot_timeline"] = emit_shot_timeline
        return fake_mp4, sum(s.seconds for s in shots), None, timeline, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {
        "shots": [
            {"id": "s1", "line_id": "l1", "description": "First", "seconds": 2.5},
            {"id": "s2", "description": "Second", "seconds": 3.0},
        ],
        "emit_shot_timeline": True,
    }
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    assert captured["emit_shot_timeline"] is True
    assert data["shot_timeline"] == timeline


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({}, None),
        (
            {
                "caption_font_family": "Inter",
                "caption_text_colour": "#F8FAFC",
                "caption_box_colour": "#0B1F3A",
                "caption_box_opacity": 0.85,
            },
            CaptionLook("Inter", "#F8FAFC", "#0B1F3A", 0.85),
        ),
    ],
)
async def test_caption_typography_reaches_compile_shots(
    monkeypatch: pytest.MonkeyPatch,
    client: AsyncClient,
    extra: dict[str, object],
    expected: CaptionLook | None,
) -> None:
    """The request's caption typography fields arrive at ``compile_shots`` as ONE
    ``caption_look``; a request without them passes ``None`` (today's look)."""
    fake_mp4 = Path("/tmp/fake-compiled-caption-look.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    captured: dict[str, object] = {}

    async def fake_compile(shots, settings, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {"shots": [{"id": "s1", "description": "First", "seconds": 2.5}], **extra}
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    assert done.json()["status"] == "completed", done.json()
    assert captured["caption_look"] == expected


async def test_compile_deterministic_tag_reaches_compile_shots(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A truthy ``tags.deterministic`` string parses to ``deterministic=True`` and
    reaches ``compile_shots``; an absent tag leaves the prod default ``False``. This
    is the ONLY coverage of the route-level tag→bool wiring (the real-ffmpeg harness
    test calls ``compile_shots`` directly and bypasses the route parser)."""
    fake_mp4 = Path("/tmp/fake-compiled-determ.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    captured: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        captured["deterministic"] = deterministic
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    shots = [{"id": "s1", "description": "First", "seconds": 2.0}]

    on = await client.post("/api/compile", json={"shots": shots, "tags": {"deterministic": "true"}})
    assert on.status_code == 202, on.text
    assert (await _poll(client, on.json()["job_id"])).json()["status"] == "completed"
    assert captured["deterministic"] is True

    captured.clear()
    off = await client.post("/api/compile", json={"shots": shots, "tags": {"run_id": "r1"}})
    assert off.status_code == 202, off.text
    assert (await _poll(client, off.json()["job_id"])).json()["status"] == "completed"
    assert captured["deterministic"] is False


async def test_compile_freeze_pad_cap_tag_reaches_compile_shots(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A truthy ``tags.freeze_pad_cap`` string parses to ``freeze_pad_cap=True`` and
    reaches ``compile_shots``; an absent tag leaves the prod default ``False``. Route-
    level tag→bool wiring only (the real-ffmpeg harness exercises the pixel effect)."""
    fake_mp4 = Path("/tmp/fake-compiled-freezepad.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    captured: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        captured["freeze_pad_cap"] = freeze_pad_cap
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    shots = [{"id": "s1", "description": "First", "seconds": 2.0}]

    on = await client.post("/api/compile", json={"shots": shots, "tags": {"freeze_pad_cap": "on"}})
    assert on.status_code == 202, on.text
    assert (await _poll(client, on.json()["job_id"])).json()["status"] == "completed"
    assert captured["freeze_pad_cap"] is True

    captured.clear()
    off = await client.post("/api/compile", json={"shots": shots, "tags": {"run_id": "r1"}})
    assert off.status_code == 202, off.text
    assert (await _poll(client, off.json()["job_id"])).json()["status"] == "completed"
    assert captured["freeze_pad_cap"] is False


async def test_parse_truthy_tag_accepts_spellings_case_insensitively() -> None:
    """The shared ``_parse_truthy_tag`` helper (single source of truth for the
    ``deterministic`` / ``freeze_pad_cap`` opt-in flags) accepts {1,true,on,yes}
    case-insensitively with surrounding whitespace, and is None-safe on a missing
    key. Behaviour must match the three inlined literals it replaced exactly."""
    from src.api.routes.compile import _parse_truthy_tag

    for spelling in ("1", "true", "TRUE", " True ", "on", "ON", "yes", "  yes"):
        assert _parse_truthy_tag({"k": spelling}, "k") is True, spelling
    for spelling in ("0", "false", "FALSE", "", "  ", "maybe", "2", "off", "no"):
        assert _parse_truthy_tag({"k": spelling}, "k") is False, spelling
    # None-safe: a missing key reads falsey, never raises.
    assert _parse_truthy_tag({}, "absent") is False
    assert _parse_truthy_tag({"other": "true"}, "absent") is False


async def test_compile_shot_timeline_none_by_default(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """Without the request flag, the poll result's ``shot_timeline`` is null — the
    RESPONSE contract is gated on ``body.emit_shot_timeline`` only. Internally the
    compiler still computes boundaries when deband measurement is active (default
    ``deband=auto``) so per-shot banding can be surfaced; that is a read-only ffprobe
    pass, byte-identical to the render, and never leaks into the response."""
    fake_mp4 = Path("/tmp/fake-compiled-notimeline.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    captured: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        captured["emit_shot_timeline"] = emit_shot_timeline
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {"shots": [{"id": "s1", "description": "First", "seconds": 2.5}]}
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    # deband=auto (default) → boundaries computed internally to drive per-shot banding.
    assert captured["emit_shot_timeline"] is True
    # ...but the response contract is unchanged: shot_timeline is null unless requested.
    assert data["shot_timeline"] is None


async def test_compile_surfaces_output_sha256(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The job result carries a sha256 content hash of the DELIVERED file."""
    payload = b"\x00\x00\x00\x20ftypisom-delivered-bytes"
    fake_mp4 = Path("/tmp/fake-compiled-sha.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    async def fake_probe(path):  # type: ignore[no-untyped-def]
        return 1920, 1080

    async def fake_dur(path):  # type: ignore[no-untyped-def]
        return 2.0

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.probe_dimensions", fake_probe)
    monkeypatch.setattr("src.api.routes.compile.probe_output_duration", fake_dur)

    body = {
        "shots": [{"id": "s1", "line_id": "l1", "description": "Shot", "seconds": 2.0}],
        "tags": {"run_id": "run-sha"},
    }
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    assert data["output_sha256"] == expected
    assert len(data["output_sha256"]) == 64


async def test_compile_forwards_captions_render_mode(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The route parses `captions_render` off the request and forwards it to
    `compile_shots` — defaulting to 'overlay' when omitted (subtitles are never
    burned in by default) and passing an explicit 'burned_in' through verbatim. This is the seam the orchestrator drives to pick soft-mux vs
    burn; a route that dropped the field would silently burn every overlay run."""
    fake_mp4 = Path("/tmp/fake-compiled-mode.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    seen: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        seen["captions_render"] = captions_render
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    base_body = {"shots": [{"id": "s1", "description": "A shot", "seconds": 2.0}]}

    # Omitted → default 'overlay' (never burned in by default).
    submit = await client.post("/api/compile", json=base_body)
    assert submit.status_code == 202, submit.text
    await _poll(client, submit.json()["job_id"])
    assert seen["captions_render"] == "overlay"

    # Explicit 'burned_in' → forwarded verbatim.
    submit = await client.post("/api/compile", json={**base_body, "captions_render": "burned_in"})
    assert submit.status_code == 202, submit.text
    await _poll(client, submit.json()["job_id"])
    assert seen["captions_render"] == "burned_in"


async def test_compile_surfaces_output_dimensions(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The job result carries the ACTUAL probed output resolution so the e2e
    post-promote check can assert the 2K/4K tier landed in the encoded file."""
    fake_mp4 = Path("/tmp/fake-compiled-4k.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    async def fake_probe(path):  # type: ignore[no-untyped-def]
        return 3840, 2160

    async def fake_dur(path):  # type: ignore[no-untyped-def]
        return 3.0

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.probe_dimensions", fake_probe)
    monkeypatch.setattr("src.api.routes.compile.probe_output_duration", fake_dur)

    body = {
        "shots": [{"id": "s1", "description": "Shot", "seconds": 3.0}],
        "tags": {"run_id": "run-4k", "render_width": "3840", "render_height": "2160"},
    }
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "completed", data
    assert data["width"] == 3840
    assert data["height"] == 2160


async def test_compile_surfaces_actual_duration(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The job result carries the ACTUAL probed duration alongside the declared
    one, so the orchestrator can flag a truncated / short-rendered file."""
    fake_mp4 = Path("/tmp/fake-compiled-dur.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, 263.5, None, None, 30  # declared/nominal duration

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/fake-key.mp4?sig=x"

    async def fake_probe(path):  # type: ignore[no-untyped-def]
        return 1920, 1080

    async def fake_dur(path):  # type: ignore[no-untyped-def]
        return 234.0  # a short render — 234s of frames under a 263.5s declaration

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.probe_dimensions", fake_probe)
    monkeypatch.setattr("src.api.routes.compile.probe_output_duration", fake_dur)

    body = {"shots": [{"id": "s1", "description": "Shot", "seconds": 263.5}]}
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202, submit.text
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    assert data["duration_seconds"] == 263.5
    assert data["actual_duration_seconds"] == 234.0


async def test_compile_surfaces_loudness(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The read-only loudness measurement is surfaced on the job status."""
    fake_mp4 = Path("/tmp/fake-compiled-loud.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    assert data["loudness"] == {
        "integrated_lufs": -12.4,
        "true_peak_dbtp": -1.1,
        "lra": 6.8,
        "silence_ratio": 0.05,
    }


async def test_compile_request_master_lufs_flows_into_settings_clamp(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A top-level ``master_lufs`` on the request populates ``tags['master_lufs']``
    and flows through the existing [-30, -5] clamp in ``_settings_for_tags``; an
    out-of-range value is ignored; an explicit tag wins over the field."""
    fake_mp4 = Path("/tmp/fake-compiled-lufs.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    seen: dict[str, float] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        seen["master_lufs"] = settings.master_lufs
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    # In-range field → applied.
    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}], "master_lufs": -16.0},
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["master_lufs"] == -16.0

    # Out-of-range field → ignored, default kept.
    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}], "master_lufs": -2.0},
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["master_lufs"] == -14.0  # unchanged default

    # Explicit tag wins over the field.
    submit = await client.post(
        "/api/compile",
        json={
            "shots": [{"id": "s1", "description": "x", "seconds": 1.0}],
            "master_lufs": -16.0,
            "tags": {"master_lufs": "-9.0"},
        },
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["master_lufs"] == -9.0


async def test_compile_surfaces_mastering_report(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The encode/delivery master soft-gate report (VC-2) is surfaced on the job
    status, reusing the loudness measured on the delivered mux."""
    fake_mp4 = Path("/tmp/fake-compiled-master.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    report = data["mastering"]
    assert report is not None
    assert report["colorspace_ok"] is True
    assert report["cfr_ok"] is True
    assert report["gate_pass"] is True
    # integrated_lufs is REUSED from the loudness measure (not re-measured).
    assert report["integrated_lufs"] == -12.4


async def test_compile_surfaces_measured_cadence(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The VC-3 measured-cadence metrics (measured_fps, duplicate_frame_ratio) on
    the mastering report are surfaced on the job status; cfr_ok is still preferred
    as the coherence signal."""
    fake_mp4 = Path("/tmp/fake-compiled-cadence.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    async def fake_verify(
        path, settings, *, loudness=None, shot_timeline=None, shots=None, narration_audio_paths=None
    ):  # type: ignore[no-untyped-def]
        return {
            "colorspace_ok": True,
            "integrated_lufs": (loudness or {}).get("integrated_lufs"),
            "true_peak_dbtp": (loudness or {}).get("true_peak_dbtp"),
            "banding_scores": [0.0],
            "cfr_ok": True,
            "measured_fps": 30.0,
            "duplicate_frame_ratio": 0.02,
            "gate_pass": True,
            "rerun_stages": [],
        }

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", fake_verify)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    report = data["mastering"]
    assert report["cfr_ok"] is True
    assert report["measured_fps"] == 30.0
    assert report["duplicate_frame_ratio"] == 0.02


async def test_compile_cadence_override_reaches_settings(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A request-level ``cadence`` pins the whole compile onto the snapped legal
    rate and switches off the auto selector (mode=fixed)."""
    fake_mp4 = Path("/tmp/fake-compiled-cad-override.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    seen: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        seen["render_fps"] = settings.render_fps
        seen["mode"] = settings.render_cadence_mode
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}], "cadence": 48},
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["render_fps"] == 50  # 48 snaps to nearest legal rate
    assert seen["mode"] == "fixed"


async def test_mastering_profile_overrides_reach_settings(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A structured MasteringProfile on the request retargets the render settings
    via the shared tag-clamp path (colorspace / deband / bitrate / CRF)."""
    fake_mp4 = Path("/tmp/fake-compiled-profile.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    seen: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        seen["colorspace"] = settings.render_colorspace
        seen["deband"] = settings.render_deband
        seen["bitrate_mode"] = settings.render_bitrate_mode
        seen["maxrate"] = settings.render_vbv_maxrate
        seen["crf"] = settings.render_crf
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile",
        json={
            "shots": [{"id": "s1", "description": "x", "seconds": 1.0}],
            "mastering": {
                "colorspace": "bt2020",
                "deband": "off",
                "bitrate_mode": "vbv",
                "maxrate": "12M",
                "crf": 16,
            },
        },
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["colorspace"] == "bt2020"
    assert seen["deband"] == "off"
    assert seen["bitrate_mode"] == "vbv"
    assert seen["maxrate"] == "12M"
    assert seen["crf"] == 16

    # An out-of-range / garbage profile value degrades to the env default (no 5xx).
    seen.clear()
    submit = await client.post(
        "/api/compile",
        json={
            "shots": [{"id": "s1", "description": "x", "seconds": 1.0}],
            "mastering": {"colorspace": "rec2100", "crf": 18},
        },
    )
    assert (await _poll(client, submit.json()["job_id"])).json()["status"] == "completed"
    assert seen["colorspace"] == "bt709"  # unknown → default kept
    assert seen["crf"] == 18


async def test_post_mux_verify_failure_still_completes_with_gate_false(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """If post-mux verification itself blows up the job FAILS — it does not
    complete with a degraded report and it is still never a 5xx.

    ⚠️ INVERTED IN WAVE-13, BY OWNER DECISION. This test used to pin "still
    completes with gate_pass=False", i.e. an unverified master uploaded and
    billed exactly like a verified one, with nothing at all known about it. An
    absent report is a STRUCTURAL finding (it reports ABSENCE, not a degree), so
    ``verify`` is armed by default. The historical fail-open is one env var away
    (``MASTERING_GATE_VERIFY_ENABLED=false``) and is asserted below, because a
    kill switch nobody exercises is not a back-out path."""
    fake_mp4 = Path("/tmp/fake-compiled-gatefail.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    async def boom_verify(
        path, settings, *, loudness=None, shot_timeline=None, shots=None, narration_audio_paths=None
    ):  # type: ignore[no-untyped-def]
        raise RuntimeError("ffprobe wedged")

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", boom_verify)

    from src.config import get_settings

    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}

    # ARMED (the default): the job is stamped terminal `failed` and its error
    # names the verification that could not run, rather than uploading.
    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "failed", data  # NOT a 5xx, and NOT delivered
    assert "post_mux_verify could not run" in data["error"]
    assert "ffprobe wedged" in data["error"]

    # DISARMED via the kill switch: the historical degraded-report behaviour,
    # byte-for-byte. Proves the switch is the real back-out path.
    monkeypatch.setenv("MASTERING_GATE_VERIFY_ENABLED", "false")
    get_settings.cache_clear()

    submit = await client.post("/api/compile", json=body)
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    assert data["mastering"]["gate_pass"] is False
    assert data["mastering"]["rerun_stages"] == ["verify"]
    # ...and the degraded report names the finding in the contracted shape, so a
    # consumer on the fail-open path is not left parsing prose either.
    assert [v["stage"] for v in data["mastering"]["violations"]] == ["verify"]
    assert data["mastering"]["violations"][0]["tier"] == "structural"


async def test_master_measure_failure_reaches_the_response_body(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """THE CHAIN, END TO END, because every link of it existed and the signal
    still never arrived: renderer records the bail ->
    ``get_master_loudnorm_evidence()`` returns ``measure_failed`` -> the route
    prefixes it to ``master_measure_failed`` -> MasteringReport must DECLARE it or
    pydantic drops it at the response boundary.

    A unit test on the model alone cannot catch a break in the prefixing or the
    fold-in, and the model was the link that was missing, so the assertion belongs
    on the delivered JSON."""
    fake_mp4 = Path("/tmp/fake-compiled-measurefail.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(shots, settings, **kwargs):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    async def fake_verify(
        path, settings, *, loudness=None, shot_timeline=None, shots=None, narration_audio_paths=None
    ):  # type: ignore[no-untyped-def]
        return mastering.degraded_report()

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", fake_verify)
    monkeypatch.setenv("MASTERING_GATE_VERIFY_ENABLED", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "src.api.routes.compile.get_master_loudnorm_evidence",
        lambda: {"measure_failed": True, "requested_linear": True},
    )

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    assert data["mastering"]["master_measure_failed"] is True, data["mastering"]
    # VC-W14/V4 — AND IT NOW DECIDES SOMETHING. The flag above arrived on the
    # wire from wave 13 onward and had NO gating consumer: this same response
    # carried ``gate_pass=True`` and no note, so a master that silently fell back
    # to DYNAMIC loudnorm was indistinguishable from a clean one. The gate is
    # deliberately DISARMED (a dynamically-normalised film still ships; the
    # pass-1 failure rate is n=0), so the job must still COMPLETE — but the
    # degradation is now visible in the report the caller already reads.
    assert data["mastering"]["gate_pass"] is False, data["mastering"]
    assert "measure" in data["mastering"]["rerun_stages"], data["mastering"]
    (violation,) = [v for v in data["mastering"]["violations"] if v["stage"] == "measure"]
    assert violation["metric"] == "master_measure_failed"
    assert violation["tier"] == "structural"


async def test_compile_rejects_empty_shots(client: AsyncClient) -> None:
    r = await client.post("/api/compile", json={"shots": []})
    assert r.status_code == 422


async def test_compile_rejects_zero_duration(client: AsyncClient) -> None:
    r = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 0}]},
    )
    assert r.status_code == 422


async def test_compile_uses_ad_hoc_namespace_when_run_id_missing(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    fake_mp4 = Path("/tmp/fake-compiled-2.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    captured: dict[str, str] = {}

    async def fake_upload(client, *, bucket, object_key, file_path, expiry_seconds):  # type: ignore[no-untyped-def]
        captured["object_key"] = object_key
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]},
    )
    assert submit.status_code == 202
    done = await _poll(client, submit.json()["job_id"])
    assert done.json()["status"] == "completed"
    assert captured["object_key"].startswith("ad-hoc/")


async def test_failed_render_is_reported_as_failed_job(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A render error is surfaced as a terminal ``failed`` status + message, not
    a synchronous 500 — the caller's poll_job turns that into a RuntimeError."""

    async def boom(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("src.api.routes.compile.compile_shots", boom)

    submit = await client.post(
        "/api/compile",
        json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]},
    )
    assert submit.status_code == 202
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "failed"
    assert "ffmpeg exploded" in (data["error"] or "")


async def test_unfetchable_narration_fails_the_job_not_interrupted(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """An unfetchable ``audio_url`` (video_compiler#232, Defect A) must end the job
    ``failed`` — never ``interrupted`` (the reaper-retryable status). The reaper
    redispatches a reclaimed job from its ORIGINAL persisted ``job_options`` — the
    SAME already-dead presigned URL — so marking this ``interrupted`` would just
    burn the poison-attempt budget re-trying a fetch that can never succeed, while
    the job sits ``processing``/``interrupted`` (never ``completed``) and the
    orchestrator never sees the terminal signal it needs to resume/regenerate with
    a freshly presigned URL. ``renderer.UnfetchableNarrationError`` is a bare
    ``Exception`` (deliberately not ``RetryableRenderError``), so it must fall
    into ``_run``'s generic ``except Exception`` branch and stamp ``failed``.

    Mirrors ``test_failed_render_is_reported_as_failed_job`` above but pins the
    STATUS DISTINCTION (failed vs interrupted) that a plain ``RuntimeError``
    doesn't exercise — a naive fix could raise the right exception TYPE but the
    wrong classification would still leave a narration-less job stuck retrying
    forever against a dead URL.
    """

    async def boom(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        raise renderer.UnfetchableNarrationError(
            "shot s1: narration audio unfetchable: http://minio/line.wav"
        )

    monkeypatch.setattr("src.api.routes.compile.compile_shots", boom)

    submit = await client.post(
        "/api/compile",
        json={
            "shots": [
                {
                    "id": "s1",
                    "description": "x",
                    "seconds": 1.0,
                    "audio_url": "http://minio/line.wav",
                }
            ]
        },
    )
    assert submit.status_code == 202
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "failed"
    assert data["status"] != "interrupted"
    assert "narration audio unfetchable" in (data["error"] or "")


async def test_poll_unknown_job_is_404(client: AsyncClient) -> None:
    r = await client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


async def test_health_endpoints(client: AsyncClient) -> None:
    # Liveness paths are static 200. Readiness (/health/ready, /readyz) now gates
    # on MinIO reachability and is covered in test_health.py.
    for path in ("/healthz", "/health", "/health/live"):
        r = await client.get(path)
        assert r.status_code == 200


def _settings_no_local_files() -> object:
    """A Settings with the LFI guard armed (allow_local_file_inputs=False)."""
    from src.config import Settings

    s = Settings()
    object.__setattr__(s, "allow_local_file_inputs", False)
    return s


def _settings_allow_local_files() -> object:
    """A Settings that opts INTO file:// inputs (allow_local_file_inputs=True).

    The default is now False (secure-by-default), so a test that submits a
    ``file://`` fixture through the guarded route must opt in explicitly — this
    mirrors ``ALLOW_LOCAL_FILE_INPUTS=true`` for local dev.
    """
    from src.config import Settings

    s = Settings()
    object.__setattr__(s, "allow_local_file_inputs", True)
    return s


async def test_compile_rejects_local_file_input_when_guard_armed(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """With allow_local_file_inputs=False, a file:// shot input is rejected 400 at submit."""
    monkeypatch.setattr("src.api.routes.compile.get_settings", _settings_no_local_files)
    body = {
        "shots": [
            {
                "id": "s1",
                "description": "hostile",
                "seconds": 1.0,
                "image_url": "file:///etc/passwd",
            }
        ]
    }
    resp = await client.post("/api/compile", json=body)
    assert resp.status_code == 400, resp.text
    assert "local-file input rejected" in resp.text


async def test_concat_audio_rejects_local_file_input_when_guard_armed(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The same guard covers the concat-audio submit route."""
    monkeypatch.setattr("src.api.routes.compile.get_settings", _settings_no_local_files)
    body = {
        "clips": [{"id": "c1", "audio_url": "file:///etc/shadow", "seconds": 1.0}],
        "output_format": "wav",
    }
    resp = await client.post("/api/concat-audio", json=body)
    assert resp.status_code == 400, resp.text
    assert "local-file input rejected" in resp.text


async def test_compile_rejects_local_file_input_by_default(
    client: AsyncClient,
) -> None:
    """With NO settings override, the default (secure-by-default) rejects file:// at submit.

    This is the mutation-proof of the config flip: revert config.py's default back
    to True and this test fails. It uses the real cached ``get_settings`` (no
    monkeypatch), so it asserts prod's default behaviour, not an explicit override.
    """
    body = {
        "shots": [
            {
                "id": "s1",
                "description": "hostile",
                "seconds": 1.0,
                "image_url": "file:///etc/passwd",
            }
        ]
    }
    resp = await client.post("/api/compile", json=body)
    assert resp.status_code == 400, resp.text
    assert "local-file input rejected" in resp.text


async def test_compile_allows_local_file_input_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """With allow_local_file_inputs=True (opt-in), file:// inputs are accepted."""
    monkeypatch.setattr("src.api.routes.compile.get_settings", _settings_allow_local_files)
    fake_mp4 = Path("/tmp/fake-lfi-default.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(shots, settings, **kwargs):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/w2v-compiled-videos/x.mp4?sig=y"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {
        "shots": [
            {"id": "s1", "description": "local", "seconds": 1.0, "image_url": "file:///tmp/x.png"}
        ]
    }
    resp = await client.post("/api/compile", json=body)
    assert resp.status_code == 202, resp.text
    await _poll(client, resp.json()["job_id"])


# ---------------------------------------------------------------------------
# mastering_gate_mode=enforce (T0-1). Advisory (default) behaviour is pinned by
# `test_post_mux_verify_failure_still_completes_with_gate_false` above.
# ---------------------------------------------------------------------------


def _fake_compile_to(fake_mp4: Path):  # type: ignore[no-untyped-def]
    """A `compile_shots` stand-in that yields `fake_mp4` and the declared duration."""

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    return fake_compile


async def _fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
    return "https://minio.test/x"


def _enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip the deployment-level gate mode for one test (settings are read per request)."""
    from src.config import get_settings

    monkeypatch.setenv("MASTERING_GATE_MODE", "enforce")
    get_settings.cache_clear()


async def test_enforce_mode_fails_the_job_naming_the_offending_metric(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """Under enforce a gated master goes terminal `failed` and the error names the
    metric, its value and its ceiling.

    ⚠️ The defect driving this is now a DEAD (uniform/blank) master. It used to be
    a 92%-duplicate one, until the duplicate-frame gate was removed on 2026-09-11
    (intentional dwell; it fired on 120 of 120 measured masters). The 0.9197 ratio
    is deliberately KEPT in the fixture and asserted ABSENT from the error, so this
    test now also pins that duplicates cannot fail a job."""
    fake_mp4 = Path("/tmp/fake-compiled-enforce-metric.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "signalstats" in joined:  # uniform frame: YMAX == YMIN → dead master
            return (
                "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=16\nlavfi.signalstats.YDIF=0.0\n"
            )
        return ""

    async def frozen_master(path) -> float:  # type: ignore[no-untyped-def]
        return 0.9197

    monkeypatch.setattr("src.api.routes.compile.compile_shots", _fake_compile_to(fake_mp4))
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", _fake_upload)
    # Un-stub the gate (the autouse fixture replaced it with an always-pass fake) so
    # the REAL gating decision runs, over mocked probes.
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", _real_post_mux_verify)
    monkeypatch.setattr("src.services.renderer._run", fake_run)
    monkeypatch.setattr("src.services.mastering.duplicate_frame_ratio", frozen_master)
    _enforce(monkeypatch)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "failed", data
    assert "dead_frame" in data["error"]
    assert "1.0000" in data["error"]  # the measured value (every frame is blank)
    assert "0.1000" in data["error"]  # the ceiling it breached
    # The 92%-duplicate ratio measured on the same master gates NOTHING.
    assert "duplicate_frames" not in data["error"]
    assert "0.9197" not in data["error"]


async def test_enforce_mode_fails_closed_when_verification_itself_crashes(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The degrade path is the real hole: a CRASHED verification returned
    `degraded_report()` and uploaded anyway, indistinguishable from a check that
    ran. Under enforce an unverified master is a FAILED render."""
    fake_mp4 = Path("/tmp/fake-compiled-enforce-crash.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    uploaded: list[str] = []

    async def tracking_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        uploaded.append(kwargs.get("object_key", ""))
        return "https://minio.test/x"

    async def boom_verify(
        path, settings, *, loudness=None, shot_timeline=None, shots=None, narration_audio_paths=None
    ):  # type: ignore[no-untyped-def]
        raise RuntimeError("ffprobe wedged")

    monkeypatch.setattr("src.api.routes.compile.compile_shots", _fake_compile_to(fake_mp4))
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", tracking_upload)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", boom_verify)
    _enforce(monkeypatch)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "failed", data
    assert "post_mux_verify could not run" in data["error"]
    assert "ffprobe wedged" in data["error"]
    assert uploaded == []  # the unverified master never reached MinIO


async def test_cfr_ok_is_verified_against_the_resolved_master_cadence(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A master the VC-3 auto selector RETIMED must be verified at the rate it was
    actually encoded at — not at the request's pre-resolution ``render_fps``.

    ``compile_shots`` resolves one master cadence and rebinds ``render_fps`` on its
    OWN copy of the settings; the route's copy still holds the original 30. Feeding
    that stale 30 to ``post_mux_verify`` makes ``framerate_coherence`` compare a
    perfectly coherent 24fps master against an expected 30 under a +/-0.5 tolerance
    and report ``cfr_ok=False`` forever — a false red on the one advisory field the
    mastering module tells consumers to read directly to act on cadence, and one
    that ``unmeasured`` cannot catch because ``False`` is not ``None``.

    The trigger is the DEFAULT shape: ``render_cadence_mode='auto'``,
    ``render_fps=30``, no request ``cadence``, shots declaring ``target_fps=24``.
    The ``compile_shots`` stand-in resolves the cadence through the REAL selector
    and reports it back the way the renderer now does, and every ffprobe is
    answered as a genuine 24fps CFR file.
    """
    fake_mp4 = Path("/tmp/fake-compiled-auto-cadence.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")
    seen: dict[str, object] = {}

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        # Exactly what compile_shots does: resolve ONE master cadence through the
        # real selector, encode at it, and hand it back to the caller.
        chosen = renderer._resolve_master_cadence(list(shots), settings)
        seen["requested_fps"] = settings.render_fps
        seen["chosen_fps"] = chosen
        return fake_mp4, sum(s.seconds for s in shots), None, None, chosen

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "24/1\n24/1\n"  # a genuinely coherent 24fps master
        if "stream=time_base:packet=pts" in joined:
            # ...whose pts timeline is regular too (the other half of ``cfr_ok``).
            return '{"streams": [{"time_base": "1/24"}], "packets": [{"pts": 0}, {"pts": 1}, {"pts": 2}]}'
        if "stream=avg_frame_rate" in joined:
            return "24/1\n"
        if "signalstats" in joined:
            return "lavfi.signalstats.YMIN=16\nlavfi.signalstats.YMAX=235\nlavfi.signalstats.YDIF=9.0\n"
        return ""

    async def clean_master(path) -> float:  # type: ignore[no-untyped-def]
        return 0.02

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", _fake_upload)
    monkeypatch.setattr("src.api.routes.compile.mastering.post_mux_verify", _real_post_mux_verify)
    monkeypatch.setattr("src.services.renderer._run", fake_run)
    monkeypatch.setattr("src.services.mastering.duplicate_frame_ratio", clean_master)

    submit = await client.post(
        "/api/compile",
        json={
            "shots": [
                {"id": "s1", "description": "x", "seconds": 1.0, "target_fps": 24.0},
                {"id": "s2", "description": "y", "seconds": 1.0, "target_fps": 24.0},
            ]
        },
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    # The fixture must actually EXERCISE the mismatch, or the assertion below is
    # vacuous: the request asked for 30, the selector moved the master to 24.
    assert seen["requested_fps"] == 30, seen
    assert seen["chosen_fps"] == 24, seen
    report = data["mastering"]
    assert report["measured_fps"] == 24.0, report
    assert report["cfr_ok"] is True, report


async def test_compile_surfaces_the_linear_refusal_evidence(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """``master_normalization_type="dynamic"`` reaches the wire WITH the evidence
    that makes it readable: whether linear was requested, and pass 1's own
    readings, which are the inputs to loudnorm's refusal decision.

    ⚠️ THIS IS THE END-TO-END GATE FOR BOTH HALVES OF THE FIX. The report is
    validated through ``MasteringReport``, whose pydantic default
    ``extra="ignore"`` DROPS any key with no declared field — so a field missing
    from the schema, or a key the route folds in under the wrong name, fails here
    rather than shipping as a no-op that looks like a fix.
    """
    fake_mp4 = Path("/tmp/fake-compiled-linear-evidence.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    pass2_dynamic = (
        "[Parsed_loudnorm_1 @ 0x5591b4a0e400]\n"
        "Input Integrated:    -22.5 LUFS\n"
        "Normalization Type:   Dynamic\n"
        "Target Offset:        +0.0 LU\n"
    )

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        # Stand in for the real two-pass master: pass 1 measured a compressed
        # speech bus, pass 2 asked for linear and ffmpeg ran dynamic anyway. Uses
        # the REAL recorders so the ContextVar channel the route reads is the one
        # under test, not a hand-built dict.
        renderer.reset_master_normalization_type()
        renderer._record_master_loudnorm_evidence(
            {
                "input_i": "-22.51",
                "input_tp": "-4.68",
                "input_lra": "6.20",
                "input_thresh": "-32.78",
                "target_offset": "0.01",
            }
        )
        renderer._record_master_normalization_type(pass2_dynamic, requested_linear=True)
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    report = data["mastering"]
    assert report["master_normalization_type"] == "dynamic"
    # Without this, "dynamic" cannot be told from "pass 1 failed open".
    assert report["master_requested_linear"] is True
    # The readings that decide the refusal, `master_`-prefixed by the route so they
    # cannot be confused with the gate's own DELIVERED-mux loudness fields.
    assert report["master_input_i"] == -22.51
    assert report["master_input_tp"] == -4.68
    assert report["master_input_lra"] == 6.20
    assert report["master_target_offset"] == 0.01
    # The recorded numbers must actually explain the dynamic: -4.68 + (-14 - -22.51)
    # = +3.83 dBTP against the -1.0 dBTP ceiling, so linear was unreachable.
    assert report["master_input_tp"] + (-14.0 - report["master_input_i"]) > -1.0


async def test_mastering_off_job_carries_no_linear_evidence_keys(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """The shape guarantee. A render that never ran a master pass must report the
    evidence fields as None — not as a stale reading from a previous job, and not
    by changing the report shape every existing consumer already accepts."""
    fake_mp4 = Path("/tmp/fake-compiled-no-evidence.mp4")
    fake_mp4.parent.mkdir(parents=True, exist_ok=True)
    fake_mp4.write_bytes(b"\x00\x00\x00\x20ftypisom")

    async def fake_compile(
        shots,
        settings,
        music=None,
        ambient=None,
        finish=None,
        captions=None,
        captions_render="burned_in",
        caption_mood=None,
        captions_karaoke=False,
        caption_look=None,
        emit_shot_timeline=False,
        deterministic=False,
        freeze_pad_cap=False,
        narration_audio_sink=None,
    ):  # type: ignore[no-untyped-def]
        renderer.reset_master_normalization_type()
        return fake_mp4, sum(s.seconds for s in shots), None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile",
        json={
            "shots": [{"id": "s1", "description": "x", "seconds": 1.0}],
            "tags": {"mastering": "off"},
        },
    )
    assert submit.status_code == 202
    data = (await _poll(client, submit.json()["job_id"])).json()
    assert data["status"] == "completed", data
    report = data["mastering"]
    assert report["master_normalization_type"] is None
    assert report["master_requested_linear"] is None
    assert report["master_input_i"] is None
    assert report["master_input_tp"] is None
    assert report["master_input_lra"] is None
    assert report["master_target_offset"] is None
