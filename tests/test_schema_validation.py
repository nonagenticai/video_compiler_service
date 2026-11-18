"""Input-validation guards on the compile/concat request schemas.

These are security + robustness bounds: ``Shot.id``/``AudioClip.id`` become scratch-file
paths AND inline ffmpeg filtergraph tokens, so an unsafe id could traverse the filesystem
or break out of the filter graph; and unbounded ``shots``/``clips``/``seconds`` let one
request exhaust the pod's /tmp (a fetch-driven DoS). A rejected id/size fails fast with a
422 instead of a cryptic ffmpeg error or an injection.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import AudioClip, AudioConcatRequest, CompileRequest, Shot


def _shot(**kw: object) -> dict[str, object]:
    return {"id": "s1", "seconds": 2.0, "image_url": "https://minio/x.png", **kw}


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",  # path traversal
        "a/b",  # path separator
        "a:reload=1",  # ffmpeg filtergraph option breakout
        "a,b",  # filtergraph arg separator
        "a'b",  # quote
        "a b",  # whitespace
        "",  # empty
        "x" * 129,  # over max_length
    ],
)
def test_shot_id_rejects_unsafe(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Shot.model_validate(_shot(id=bad_id))


@pytest.mark.parametrize("ok_id", ["s1", "clip-3", "disclaimer-card", "kb-abc123", "line_7", "a.b"])
def test_shot_id_accepts_real_ids(ok_id: str) -> None:
    assert Shot.model_validate(_shot(id=ok_id)).id == ok_id


def test_audio_clip_id_rejects_unsafe() -> None:
    with pytest.raises(ValidationError):
        AudioClip.model_validate({"id": "a:b", "audio_url": "https://minio/a.wav"})


def test_shot_seconds_upper_bounded() -> None:
    # a multi-year single shot is refused (would be a /tmp + billing DoS)
    with pytest.raises(ValidationError):
        Shot.model_validate(_shot(seconds=100_000_000.0))
    assert Shot.model_validate(_shot(seconds=3600.0)).seconds == 3600.0


def test_compile_request_shots_capped() -> None:
    with pytest.raises(ValidationError):
        CompileRequest.model_validate({"shots": [_shot(id=f"s{i}") for i in range(2001)]})
    # a normal-sized request is fine
    CompileRequest.model_validate({"shots": [_shot()]})


def test_concat_request_clips_capped() -> None:
    clip = {"id": "c1", "audio_url": "https://minio/a.wav"}
    with pytest.raises(ValidationError):
        AudioConcatRequest.model_validate({"clips": [dict(clip, id=f"c{i}") for i in range(10001)]})
    AudioConcatRequest.model_validate({"clips": [clip]})
