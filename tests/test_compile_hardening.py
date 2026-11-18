"""Production-safety guards on the async compile surface.

Covers the hardening added on top of the submit→poll contract:
- H3 backpressure (429 when the per-pod in-flight cap is reached),
- M1 per-job work timeout (a wedged render force-fails, not immortal),
- M2 shutdown drain (cancelling in-flight tasks stamps a terminal status),
- M3 tenant-scoped polls (cross-tenant poll 404s),
- L3 run_id sanitization (no path traversal into the MinIO key),
- M4 Idempotency-Key dedup (a replayed submit returns the same job, one render).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from src.api.routes import compile as compile_routes
from src.config import Settings
from tests.conftest import TENANT_A, TENANT_B


def _settings(**overrides: object) -> Settings:
    """A Settings whose tunables can be pushed below their Field bounds for tests.

    ``job_work_timeout_seconds`` has a ``ge=60`` floor that a fast test can't
    wait out, so it is set via ``object.__setattr__`` (validate_assignment is
    off) to bypass the bound rather than sleeping a real minute.
    """
    s = Settings()
    for key, value in overrides.items():
        object.__setattr__(s, key, value)
    return s


def test_settings_for_tags_applies_valid_resolution_and_crf() -> None:
    """The orchestrator threads 2K/4K + CRF via compile_tags; valid values
    override the env default so a single run can request higher resolution
    without a redeploy."""
    base = Settings()
    out = compile_routes._settings_for_tags(
        base, {"render_width": "3840", "render_height": "2160", "crf": "18"}
    )
    assert (out.render_width, out.render_height, out.render_crf) == (3840, 2160, 18)
    # original untouched
    assert (base.render_width, base.render_height) == (1920, 1080)


def test_settings_for_tags_forces_even_dims() -> None:
    """yuv420p requires even W/H — an odd dimension tag is rounded down, never
    passed through to ffmpeg (odd → the whole render fails)."""
    out = compile_routes._settings_for_tags(Settings(), {"render_width": "2561"})
    assert out.render_width == 2560


def test_settings_for_tags_rejects_garbage_and_out_of_range() -> None:
    """A non-numeric or out-of-bounds tag degrades to the env default, not a 500."""
    base = Settings()
    out = compile_routes._settings_for_tags(
        base, {"render_width": "../etc", "render_height": "999999", "crf": "nope"}
    )
    assert out is base  # nothing valid → same object, env default


def test_settings_for_tags_no_tags_is_identity() -> None:
    base = Settings()
    assert compile_routes._settings_for_tags(base, None) is base
    assert compile_routes._settings_for_tags(base, {}) is base


def test_settings_for_tags_mastering_kill_switch() -> None:
    """A per-job 'mastering' tag can disable the default-ON loudness master
    (the escape hatch for a film that must not be re-mastered) or re-enable it."""
    base = Settings()
    assert base.master_enabled is True
    for off in ("off", "false", "0", "no"):
        assert compile_routes._settings_for_tags(base, {"mastering": off}).master_enabled is False
    for on in ("on", "true", "1", "yes"):
        disabled = _settings(master_enabled=False)
        assert compile_routes._settings_for_tags(disabled, {"mastering": on}).master_enabled is True


def test_settings_for_tags_master_lufs_override_and_clamp() -> None:
    """A per-job 'master_lufs' retargets the loudness master within [-30, -5];
    out-of-range or garbage is ignored (keeps the env default), never a 500."""
    base = Settings()
    assert compile_routes._settings_for_tags(base, {"master_lufs": "-11"}).master_lufs == -11.0
    # out of range (too hot / too quiet) and garbage → default retained
    assert (
        compile_routes._settings_for_tags(base, {"master_lufs": "-2"}).master_lufs
        == base.master_lufs
    )
    assert (
        compile_routes._settings_for_tags(base, {"master_lufs": "-40"}).master_lufs
        == base.master_lufs
    )
    assert (
        compile_routes._settings_for_tags(base, {"master_lufs": "loud"}).master_lufs
        == base.master_lufs
    )


async def _poll(client: AsyncClient, job_id: str, *, attempts: int = 200, **headers: str):
    last = None
    for _ in range(attempts):
        last = await client.get(f"/api/jobs/{job_id}", headers=headers or None)
        if last.status_code != 200 or last.json()["status"] != "processing":
            return last
        await asyncio.sleep(0.01)
    assert last is not None
    return last


def test_cleanup_job_dir_never_deletes_outside_work_dir(tmp_path: Path) -> None:
    """Regression: the route used to ``rmtree(out_path.parent)`` unguarded, so a
    compile returning ``/tmp/fake.mp4`` deleted the WHOLE of ``/tmp`` — including
    pytest's basetemp, which wiped 51 tests on Linux CI. Cleanup must only ever
    remove a strict subdirectory of ``work_dir``.
    """
    work = tmp_path / "work"
    (work / "job-1").mkdir(parents=True)
    artifact = work / "job-1" / "out.mp4"
    artifact.write_bytes(b"\x00")
    settings = _settings(work_dir=str(work))

    # A real per-job scratch dir IS cleaned.
    compile_routes._cleanup_job_dir(artifact, settings)
    assert not (work / "job-1").exists()
    assert work.exists()

    # work_dir itself is refused (parent == work_dir, not a subdir of it).
    stray = work / "loose.mp4"
    stray.write_bytes(b"\x00")
    compile_routes._cleanup_job_dir(stray, settings)
    assert work.exists()

    # A path entirely outside work_dir is refused — the /tmp/fake.mp4 case.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "fake.mp4").write_bytes(b"\x00")
    compile_routes._cleanup_job_dir(outside / "fake.mp4", settings)
    assert outside.exists(), "cleanup must never delete a directory outside work_dir"

    # And a None out_path is a no-op.
    compile_routes._cleanup_job_dir(None, settings)


async def test_backpressure_returns_429_at_capacity(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, tmp_path: Path
) -> None:
    """With the in-flight cap at 1, a second submit while the first still renders
    gets a 429 + Retry-After instead of spawning an unbounded render."""
    gate = asyncio.Event()
    # The route cleans up ``out_path.parent`` when the job finishes, so the stub
    # MUST return a file inside a per-job scratch dir under work_dir. Returning
    # ``/tmp/fake.mp4`` (as this once did) made the parent ``/tmp`` and the
    # cleanup deleted the whole of /tmp — including pytest's basetemp.
    job_dir = tmp_path / "job-fake"
    job_dir.mkdir()
    artifact = job_dir / "fake.mp4"

    async def blocking_compile(
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
        await gate.wait()
        return artifact, 1.0, None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", blocking_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)
    monkeypatch.setattr(
        "src.api.routes.compile.get_settings",
        lambda: _settings(max_concurrent_compile_jobs=1, work_dir=str(tmp_path)),
    )

    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    first = await client.post("/api/compile", json=body)
    assert first.status_code == 202

    second = await client.post("/api/compile", json=body)
    assert second.status_code == 429, second.text
    assert second.headers.get("Retry-After") == "5"

    # Let the first job finish so the test doesn't leak a pending task.
    artifact.write_bytes(b"\x00")
    gate.set()
    done = await _poll(client, first.json()["job_id"])
    assert done.json()["status"] == "completed"


async def test_wedged_render_force_fails_on_work_timeout(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient, tmp_path: Path
) -> None:
    """A render that never returns is killed by the per-job work timeout and the
    job reports ``failed`` rather than staying ``processing`` forever."""

    async def never_returns(
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
        await asyncio.sleep(5)
        # Never actually reached (the work timeout cancels this), but keep it
        # inside a scratch dir anyway — a bare /tmp path here would make the
        # route's cleanup rmtree /tmp if the timing ever changed.
        return tmp_path / "job-never" / "never.mp4", 1.0, None, None, 30

    monkeypatch.setattr("src.api.routes.compile.compile_shots", never_returns)
    monkeypatch.setattr(
        "src.api.routes.compile.get_settings",
        lambda: _settings(job_work_timeout_seconds=0.05),
    )

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    done = await _poll(client, submit.json()["job_id"])
    data = done.json()
    assert data["status"] == "failed", data
    assert "work timeout" in (data["error"] or "")


async def test_run_id_is_sanitized_into_object_key(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """A path-traversal run_id can't escape its MinIO namespace."""
    fake_mp4 = Path("/tmp/fake-sanitize.mp4")
    fake_mp4.write_bytes(b"\x00")
    captured: dict[str, str] = {}

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
        return fake_mp4, 1.0, None, None, 30

    async def fake_upload(client, *, bucket, object_key, file_path, expiry_seconds):  # type: ignore[no-untyped-def]
        captured["object_key"] = object_key
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    submit = await client.post(
        "/api/compile",
        json={
            "shots": [{"id": "s1", "description": "x", "seconds": 1.0}],
            "tags": {"run_id": "../evil/../x"},
        },
    )
    assert submit.status_code == 202
    done = await _poll(client, submit.json()["job_id"])
    assert done.json()["status"] == "completed"
    key = captured["object_key"]
    prefix = key.split("/")[0]
    assert re.fullmatch(r"[A-Za-z0-9_-]+", prefix), prefix
    assert key.count("/") == 1  # exactly the namespace separator, no traversal


async def test_cross_tenant_poll_is_404() -> None:
    """A job submitted by tenant A is invisible (404) to tenant B's poll."""
    app = FastAPI()
    app.state.instance_id = "test-xtenant"

    @app.middleware("http")
    async def _stamp_tenant(request: Request, call_next):  # type: ignore[no-untyped-def]
        tenant = request.headers.get("X-Test-Tenant")
        if tenant:
            request.state.tenant_id = tenant
        return await call_next(request)

    app.include_router(compile_routes.router)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
        submit = await client.post(
            "/api/compile", json=body, headers={"X-Test-Tenant": str(TENANT_A)}
        )
        assert submit.status_code == 202
        job_id = submit.json()["job_id"]

        # tenant B must not see it (404, not 403 — no existence leak).
        other = await client.get(f"/api/jobs/{job_id}", headers={"X-Test-Tenant": str(TENANT_B)})
        assert other.status_code == 404

        # the owner still sees it.
        owner = await client.get(f"/api/jobs/{job_id}", headers={"X-Test-Tenant": str(TENANT_A)})
        assert owner.status_code == 200

        # A caller forwarding NO tenant at all gets 400 — it does not fall back to a
        # sentinel scope and read the row. Third leg of the boundary: the old code
        # coalesced an absent tenant to ``''``, so every unattributed caller shared one
        # readable bucket. Both the poll and the submit must refuse.
        anonymous_poll = await client.get(f"/api/jobs/{job_id}")
        assert anonymous_poll.status_code == 400, anonymous_poll.text
        anonymous_submit = await client.post("/api/compile", json=body)
        assert anonymous_submit.status_code == 400, anonymous_submit.text


async def test_shutdown_drain_marks_inflight_job_interrupted(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """drain_background_tasks() cancels an in-flight render and the durable row is
    stamped ``interrupted`` — reported as ``processing`` on the wire so a resend
    re-runs it. A mid-render deploy must NOT fail the orchestrator's compile stage
    (the whole point of the durable-idempotency work)."""
    started = asyncio.Event()

    async def blocks_forever(
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
        started.set()
        await asyncio.sleep(3600)
        return Path("/tmp/never2.mp4"), 1.0, None, None, 30

    monkeypatch.setattr("src.api.routes.compile.compile_shots", blocks_forever)

    submit = await client.post(
        "/api/compile", json={"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    )
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]
    await asyncio.wait_for(started.wait(), timeout=2)

    await compile_routes.drain_background_tasks()

    # Wire status: 'processing' (a resend re-runs it), NOT a terminal 'failed'.
    done = await client.get(f"/api/jobs/{job_id}")
    data = done.json()
    assert data["status"] == "processing", data
    assert data["error"] is None
    assert not compile_routes._background_tasks

    # Durable row: 'interrupted' — proving drain stamped it, not left it processing.
    from src.database.session import get_db_session
    from src.services import compile_job_store

    async with get_db_session() as session:
        row = await compile_job_store.get(session, job_id, TENANT_A)
    assert row is not None
    assert row.status == "interrupted"
    # ...and stamped WHY, as 'drain' specifically. This is the ONLY end-to-end link
    # between the real SIGTERM handler and the literal the reclaim claim exempts from
    # the poison-attempt increment. MUTATION-PROOF: flip the CancelledError branch in
    # src/api/routes/compile.py from INTERRUPT_REASON_DRAIN to any other reason and this
    # fails — without it the whole no-refund mechanism can be unwired while CI stays green.
    assert row.interrupt_reason == compile_job_store.INTERRUPT_REASON_DRAIN


async def test_idempotency_key_dedupes_replayed_submit(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    """Two submits with the same Idempotency-Key (a conn-reset replay) yield the
    same job and exactly one render — not a duplicate non-idempotent compile."""
    fake_mp4 = Path("/tmp/fake-idem.mp4")
    fake_mp4.write_bytes(b"\x00")
    renders = {"n": 0}

    async def counting_compile(
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
        renders["n"] += 1
        return fake_mp4, 1.0, None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", counting_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    headers = {"Idempotency-Key": "replay-key-1"}
    first = await client.post("/api/compile", json=body, headers=headers)
    second = await client.post("/api/compile", json=body, headers=headers)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]

    done = await _poll(client, first.json()["job_id"])
    assert done.json()["status"] == "completed"
    assert renders["n"] == 1  # the replay did not start a second render


async def test_distinct_idempotency_keys_make_distinct_jobs(
    monkeypatch: pytest.MonkeyPatch, client: AsyncClient
) -> None:
    fake_mp4 = Path("/tmp/fake-idem2.mp4")
    fake_mp4.write_bytes(b"\x00")

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
        return fake_mp4, 1.0, None, None, 30

    async def fake_upload(*args, **kwargs):  # type: ignore[no-untyped-def]
        return "https://minio.test/x"

    monkeypatch.setattr("src.api.routes.compile.compile_shots", fake_compile)
    monkeypatch.setattr("src.api.routes.compile.upload_and_presign", fake_upload)

    body = {"shots": [{"id": "s1", "description": "x", "seconds": 1.0}]}
    first = await client.post("/api/compile", json=body, headers={"Idempotency-Key": "k1"})
    second = await client.post("/api/compile", json=body, headers={"Idempotency-Key": "k2"})
    assert first.json()["job_id"] != second.json()["job_id"]
