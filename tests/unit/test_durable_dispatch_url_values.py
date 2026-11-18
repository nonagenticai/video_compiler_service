"""``_url_values`` / ``inputs_reconstructable``: the crash-redispatch input scan.

These two functions decide whether a crashed render may be re-dispatched to a DIFFERENT
pod. A URL the scan MISSES is the dangerous direction: a pod-affine ``file://`` input
would pass the check and the reaper would re-run the job on a pod where the file does not
exist — a broken render instead of a clean fail-fast. So the scan must be greedy, and
these tests pin every payload shape a request can put a URL key into.
"""

from __future__ import annotations

from src.services.durable_dispatch import _url_values, inputs_reconstructable


def test_scalar_url_key_collected() -> None:
    assert _url_values({"video_url": "https://cdn/a.mp4"}) == ["https://cdn/a.mp4"]


def test_url_key_nested_in_mapping_collected() -> None:
    payload: dict[str, object] = {"bed": {"track": {"audio_url": "https://cdn/bed.wav"}}}
    assert _url_values(payload) == ["https://cdn/bed.wav"]


def test_list_of_mappings_each_url_collected() -> None:
    payload: dict[str, object] = {
        "shots": [
            {"line_id": "l1", "image_url": "https://cdn/1.png"},
            {"line_id": "l2", "video_url": "https://cdn/2.mp4", "audio_url": "https://cdn/2.wav"},
        ],
    }
    assert _url_values(payload) == [
        "https://cdn/1.png",
        "https://cdn/2.mp4",
        "https://cdn/2.wav",
    ]


def test_url_key_holding_list_of_strings_collects_every_item() -> None:
    """RED on the pre-fix code: bare strings inside a list had no key of their own."""
    payload: dict[str, object] = {"url": ["file:///tmp/pod-local.png", "https://cdn/b.png"]}
    assert _url_values(payload) == ["file:///tmp/pod-local.png", "https://cdn/b.png"]


def test_url_key_holding_mixed_list_collects_strings_and_walks_objects() -> None:
    payload: dict[str, object] = {
        "second_source_url": ["https://cdn/a.mp4", {"video_url": "https://cdn/nested.mp4"}],
    }
    assert _url_values(payload) == ["https://cdn/a.mp4", "https://cdn/nested.mp4"]


def test_empty_string_under_url_key_not_collected() -> None:
    assert _url_values({"image_url": "", "url": [""]}) == []


def test_non_url_key_holding_url_string_not_collected() -> None:
    payload: dict[str, object] = {"lut_url": "file:///tmp/look.cube", "caption": "https://x/y"}
    assert _url_values(payload) == []


def test_none_under_url_key_is_ignored() -> None:
    assert _url_values({"audio_url": None}) == []


def test_inputs_reconstructable_true_for_http_only_payload() -> None:
    payload: dict[str, object] = {"shots": [{"image_url": "https://cdn/1.png"}]}
    assert inputs_reconstructable(payload) == (True, "")


def test_inputs_reconstructable_rejects_file_url_inside_a_string_list() -> None:
    """End-to-end: the list-shaped pod-affine input must fail-fast, not silently pass."""
    payload: dict[str, object] = {
        "shots": [{"url": ["https://cdn/ok.png", "file:///tmp/pod-local.png"]}],
    }
    ok, reason = inputs_reconstructable(payload)
    assert ok is False
    assert "file:///tmp/pod-local.png" in reason
