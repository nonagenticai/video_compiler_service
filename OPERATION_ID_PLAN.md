# Operation ID Plan: video_compiler_service

Canonical registry of `operation_id` values for every route in
`video_compiler_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service is a single-domain sync compiler: a video-compile POST plus an
audio-concat POST (audiobook stitching) and a health surface. No
ambiguity-prefixing is required.

## Files to edit

- `src/api/routes/health.py`
- `src/api/routes/compile.py`

## Mapping Tables

### `src/api/routes/health.py` (tag: `health`)

| Method | Function       | operation_id     | Rationale                                                  |
|--------|----------------|------------------|------------------------------------------------------------|
| GET    | `healthz`      | `healthz`        | Kubernetes-idiom liveness path; op_id matches path verb.   |
| GET    | `health`       | `health_check`   | Primary health probe; matches the fleet standard verb.     |
| GET    | `health_live`  | `health_live`    | Liveness sub-probe; one-to-one with path.                  |
| GET    | `health_ready` | `health_ready`   | Readiness sub-probe; one-to-one with path.                 |

### `src/api/routes/compile.py` (tag: `compile`)

| Method | Function           | operation_id    | Rationale                                                                                  |
|--------|--------------------|-----------------|--------------------------------------------------------------------------------------------|
| POST   | `compile_endpoint`      | `compile_video` | `compile_endpoint` is implementation-leaking; `compile_video` describes the operation.     |
| POST   | `concat_audio_endpoint` | `concat_audio`  | Stitches narration clips into one audiobook WAV/MP3; `concat_audio` names the operation.   |
| GET    | `get_job`               | `get_compile_job` | Terminal-status poll for a submitted compile/concat job.                                 |
| POST   | `cancel_jobs`           | `cancel_compile_jobs` | Cancels the caller's live jobs under an Idempotency-Key prefix; plural because one run owns several. |

### `src/api/routes/thumbnail.py` (tag: `thumbnail`)

| Method | Function             | operation_id         | Rationale                                                                                       |
|--------|----------------------|----------------------|-------------------------------------------------------------------------------------------------|
| POST   | `extract_thumbnails` | `extract_thumbnails` | Synchronous poster/thumbnail extraction with grade parity; names the operation one-to-one.      |

## Verification

```bash
cd video_compiler_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the table above. Drift (rename, missed
decorator, fallback) fails the test loudly.

## Future routes

When adding a route in this service, **always** set `operation_id="…"`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
