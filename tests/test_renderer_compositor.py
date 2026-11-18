"""Unit tests for the image/clip + audio compositor.

ffmpeg/ffprobe are mocked: these assert the command shape per visual/audio
combination and the per-shot duration reconciliation. Real muxing is proven
by the live E2E ffprobe gate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.api.schemas import Shot, ShotAudio
from src.config import get_settings
from src.services import renderer

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    audio_dur: float = 3.0,
    clip_dur: float = 4.0,
    clip_has_audio: bool = True,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"cmds": []}

    async def fake_run(cmd: list[str]) -> str:
        captured["cmds"].append(cmd)
        if cmd[0] == "ffmpeg" and cmd[-1] != "-":  # a "-" sink is stdout, not a file
            Path(cmd[-1]).write_bytes(b"\x00\x00\x00\x20ftypisom")
        return ""

    async def fake_probe(path: Path) -> float:
        # Audio temp files are named "<shot-id>-audio"; everything else
        # (clips) reports the clip duration.
        return audio_dur if Path(path).name.endswith("-audio") else clip_dur

    async def fake_has_audio(path: Path) -> bool:
        return clip_has_audio

    async def fake_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_run", fake_run)
    monkeypatch.setattr(renderer, "_ffprobe_duration", fake_probe)
    monkeypatch.setattr(renderer, "_has_audio_stream", fake_has_audio)
    monkeypatch.setattr(renderer, "_fetch", fake_fetch)
    return captured


async def test_image_plus_audio_uses_narration_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch, audio_dur=5.5)
    shot = Shot(
        id="s1",
        seconds=2.0,
        image_url="http://minio/slide0.png",
        audio_url="http://minio/clip0.wav",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    assert "-loop" in cmd and "1" in cmd  # still image looped
    joined = " ".join(cmd)
    assert "anullsrc" not in joined  # real narration, not silence
    # Every shot now carries a baked per-shot edge fade (3b8d1e7), so the audio
    # is routed through -filter_complex rather than a bare -map; the input index
    # shows up as the filtergraph's source label, not a standalone argv token.
    assert "[1:a]" in joined  # audio mapped from the second input
    assert cmd[cmd.index("-t") + 1] == "5.5"  # stretched to narration
    assert dur == 5.5


async def test_image_only_is_silent_and_uses_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s2", seconds=4.0, image_url="http://minio/slide1.png")
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    assert "anullsrc=channel_layout=stereo:sample_rate=48000" in " ".join(cmd)
    assert cmd[cmd.index("-t") + 1] == "4.0"
    assert dur == 4.0


async def test_text_card_fallback_when_no_visual(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s3", seconds=3.0, description="Hello world")
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    # text now arrives via `textfile=` (drawtext reads the file raw,
    # zero escape interpretation). Asserts the filter wired the textfile
    # param + the file actually exists with the description content.
    assert "drawtext" in joined
    assert "textfile=" in joined
    text_path = tmp_path / "s3-text.txt"
    assert text_path.exists()
    assert text_path.read_text(encoding="utf-8") == "Hello world"
    assert "color=c=" in joined  # solid background card
    assert dur == 3.0


def _install_failing_fetch(monkeypatch: pytest.MonkeyPatch, *, fail_substr: str) -> None:
    """Replace ``_fetch`` with one that 403s for any URL containing *fail_substr*
    (an expired-presign / deleted-asset stand-in) and otherwise behaves normally."""

    async def failing_fetch(url: str, dest: Path, client: Any, **_kw: object) -> Path:
        if fail_substr in url:
            raise httpx.HTTPStatusError(
                "expired", request=httpx.Request("GET", url), response=httpx.Response(403)
            )
        if url.startswith("file://"):
            return Path(url[len("file://") :])
        if "://" not in url:
            return Path(url)
        Path(dest).write_bytes(b"asset")
        return Path(dest)

    monkeypatch.setattr(renderer, "_fetch", failing_fetch)


async def test_unfetchable_image_degrades_to_text_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    _install_failing_fetch(monkeypatch, fail_substr="slide")
    shot = Shot(
        id="s9",
        seconds=3.0,
        image_url="http://minio/slide-dead.png",
        description="Fallback line",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    joined = " ".join(cmd)
    # A single 403 on the image must NOT 500 the compile — the shot degrades to
    # its text card (the line text), not the image branch.
    assert "drawtext" in joined and "textfile=" in joined
    assert "-loop" not in cmd
    assert (tmp_path / "s9-text.txt").read_text(encoding="utf-8") == "Fallback line"
    assert dur == 3.0


async def test_unfetchable_video_degrades_to_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch)
    _install_failing_fetch(monkeypatch, fail_substr="clip")
    shot = Shot(
        id="s10",
        seconds=4.0,
        video_url="http://minio/clip-dead.mp4",
        image_url="http://minio/slide.png",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    # Video 403s → fall through to the still image, not the text card.
    assert "-loop" in cmd
    assert "drawtext" not in " ".join(cmd)
    assert dur == 4.0


async def test_unfetchable_audio_fails_loud_not_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """video_compiler#232 (Defect A): a shot's unfetchable NARRATION must not
    silently render mute — see ``UnfetchableNarrationError``. This replaces the
    prior contract (assert below, now the bug) that a 403'd ``audio_url``
    rendered the visual silently at the shot's own ``seconds``: that is exactly
    how a paid master shipped `completed` missing a narration line."""
    _install_fakes(monkeypatch)
    _install_failing_fetch(monkeypatch, fail_substr="voice")
    shot = Shot(
        id="s11",
        seconds=4.0,
        image_url="http://minio/slide.png",
        audio_url="http://minio/voice.wav",
    )
    with pytest.raises(renderer.UnfetchableNarrationError, match="s11"):
        await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)


async def test_text_card_writes_raw_multiline_into_textfile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Regression: multi-paragraph LLM markdown with `:`, `,`, `'`, `*` was
    breaking the inline `text=` path even with escape gymnastics (live
    failure 2026-05-28 on website_review run 47fde1d9, ffmpeg exit=234).
    textfile= reads raw bytes — no escape concerns. With 2026-05-31 wrap
    fix the file holds the input with WORD-WRAPPED newlines (drawtext
    renders each `\\n`-separated chunk on its own baseline so the long
    LLM critique fits inside the canvas safe area) — earlier raw bytes
    were single-line and overflowed off both edges.
    """
    captured = _install_fakes(monkeypatch)
    raw = (
        "From a technical standpoint, the site is highly vulnerable: missing CSP and HSTS headers.\n\n"
        "* **Security & SEO:** add headers; add a meta description.\n"
        "* **Navigation & UX:** build a navigation menu and a search bar.\n"
        "* **Conversion:** replace vague headlines with the site's value proposition."
    )
    shot = Shot(id="s4", seconds=2.0, description=raw)
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    assert "textfile=" in joined
    # the previous inline `text='…'` shape is gone — must not regress
    assert "text='" not in joined
    text_path = tmp_path / "s4-text.txt"
    content = text_path.read_text(encoding="utf-8")
    # Whitespace-collapse + wrap means the file now has `\n` between
    # word-wrapped chunks (instead of the prior single line). The original
    # markdown bullets/headings flatten into a continuous paragraph that
    # the wrapper then breaks into canvas-safe lines — so the substring
    # checks need to flatten the wrap-introduced newlines back out.
    flat = content.replace("\n", " ")
    assert "Security & SEO:" in flat
    assert "site's value proposition." in flat
    assert "\n" in content  # wrap actually produced multiple lines
    # No line exceeds the wrap budget — verifies safe area honored.
    for line in content.split("\n"):
        assert len(line) <= 80  # generous upper bound for fs=56 at 1920px


async def test_caption_is_burned_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """Captions now flow via textfile= (same shape as the body card) so the
    on-frame text gets the same word-wrap treatment instead of overflowing
    when the script line is longer than a single visual line.
    """
    captured = _install_fakes(monkeypatch)
    shot = Shot(id="s4", seconds=2.0, image_url="http://minio/s.png", caption="Lower third")
    await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    # The caption is no longer inline (the old `text='Lower third'`); it lives
    # in a sibling textfile that the drawtext filter loads at render time.
    assert "textfile=" in joined and "s4-caption.txt" in joined
    cap_path = tmp_path / "s4-caption.txt"
    assert cap_path.exists()
    assert cap_path.read_text(encoding="utf-8") == "Lower third"


async def test_video_clip_keeps_own_audio_and_natural_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch, clip_dur=12.0, clip_has_audio=True)
    shot = Shot(id="lesson1", seconds=1.0, video_url="http://minio/lesson1.mp4")
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    # Routed through -filter_complex for the mandatory per-shot edge fade
    # (3b8d1e7); "0:a" is the filtergraph source label, not a bare -map token.
    assert "[0:a]" in " ".join(cmd)  # clip's own audio
    # Natural length: bounded at the CLIP's probed 12.0 s (on the frame grid, so its
    # audio and picture end together), never at the placeholder ``seconds=1.0``.
    assert cmd.count("-t") == 1
    assert cmd[cmd.index("-t") + 1] == "12.0"
    assert dur == 12.0  # reported as the clip's real duration


async def test_video_clip_without_audio_gets_silence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    captured = _install_fakes(monkeypatch, clip_dur=7.0, clip_has_audio=False)
    shot = Shot(id="clip2", seconds=1.0, video_url="http://minio/silent.mp4")
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    assert "anullsrc" in " ".join(cmd)
    assert cmd[cmd.index("-t") + 1] == "7.0"
    assert dur == 7.0


async def test_scroll_fit_image_pans_over_narration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A tall full-page screenshot with fit='scroll' is scaled to width and
    panned top→bottom across the narration (no pillarbox sliver)."""
    captured = _install_fakes(monkeypatch, audio_dur=9.0)
    shot = Shot(
        id="page1",
        seconds=1.0,
        image_url="http://minio/fullpage.png",
        audio_url="http://minio/line.wav",
        fit="scroll",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    assert "scale=1920:-2" in joined  # fit to canvas width
    assert "crop=1920:1080:0:" in joined  # full-width pan window
    assert "t/9.000" in joined  # pan spans the narration duration
    assert "force_original_aspect_ratio" not in joined  # not contain/cover
    assert dur == 9.0


async def test_short_clip_with_narration_drifts_last_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A per-line interaction clip shorter than its narration is frame-extended
    (tpad clones the last frame) AND drifted (a gentle zoompan eases the zoom
    over ONLY the padded tail) so the cloned frames never dead-freeze — T7/AC#5."""
    captured = _install_fakes(monkeypatch, audio_dur=9.0, clip_dur=4.0)
    shot = Shot(
        id="reveal1",
        seconds=1.0,
        video_url="http://minio/reveal1.mp4",
        audio_url="http://minio/line1.wav",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    settings = get_settings()
    joined = " ".join(captured["cmds"][-1])
    # tpad still extends the stream (zoompan can't synthesize frames).
    assert "tpad=stop_mode=clone:stop_duration=5" in joined  # 9.0 - 4.0
    # ...followed by the drift zoompan that keeps the tail moving.
    assert "zoompan=" in joined
    # Ramp fires over ONLY the padded tail: onset clip_dur*fps=120, window
    # (duration-clip_dur)*fps=150 (fps=30). Assert the exact smoothstep fragment
    # (not a bare "120"/"150" that could match a coincidental substring).
    onset = renderer._ffnum(4.0 * settings.render_fps)  # "120"
    window = renderer._ffnum((9.0 - 4.0) * settings.render_fps)  # "150"
    assert f"(on-{onset})/{window}" in joined
    assert f"s={settings.render_width}x{settings.render_height}" in joined  # canvas
    assert f"fps={renderer._ffnum(settings.render_fps)}" in joined  # cadence
    # Filtergraph source label (see the mandatory per-shot edge fade, 3b8d1e7),
    # not a bare -map token.
    assert "[1:a]" in joined  # narration, not the clip's track
    assert dur == 9.0  # segment runs the full narration length


async def test_long_clip_with_narration_is_not_padded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A clip at least as long as its narration is trimmed by -t, never DRIFTED.

    "Not padded" here means no DRIFT freeze-pad: no drift-sized ``stop_duration``
    clone tied to ``duration - clip_dur`` and no drift ``zoompan``, because the
    probe says the picture already fills the narration. A terminal, FINITE
    cover clone (``renderer._bounded_cover_pad``) IS still emitted — it is what
    makes coverage correct EVEN WHEN that probe is a LYING container header
    (video_compiler #212: it can claim ``clip_dur >= duration`` while decoding
    far less), which is why the clone's ceiling is the whole segment
    ``duration`` rather than the (probe-derived) delta — see
    ``tests/render_harness/test_picture_covers_narration_render.py`` for the
    end-to-end proof of that shape. Asserting the clone's ABSENCE here would
    pin the very defect (a probe that over-claims delivers a short picture) as
    the contract."""
    captured = _install_fakes(monkeypatch, audio_dur=3.0, clip_dur=12.0)
    shot = Shot(
        id="reveal2",
        seconds=1.0,
        video_url="http://minio/reveal2.mp4",
        audio_url="http://minio/line2.wav",
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    joined = " ".join(cmd)
    assert "zoompan" not in joined  # no drift when the clip already fills its narration
    # The terminal cover clone is finite (never "stop=-1") and ceiling'd at the
    # whole segment duration (3.0s) + margin, not at "duration - clip_dur"
    # (which a lying header can make read as zero or negative) — computed here
    # rather than pinned as a literal so a margin tune does not touch this test.
    ceiling = 3.0 + renderer._COVER_PAD_MARGIN_SECONDS
    assert f"tpad=stop_mode=clone:stop_duration={renderer._ffnum(ceiling)}" in joined
    assert "stop=-1" not in joined  # never the infinite clone that livelocked #232
    assert cmd[cmd.index("-t") + 1] == "3.0"  # trimmed to narration
    assert dur == 3.0


# ---------------------------------------------------------------------------
# VC-3 byte-identity: default master cadence → the exact pre-VC-3 filter string
# ---------------------------------------------------------------------------


async def test_scale_pad_filter_default_cadence_is_byte_identical() -> None:
    """With default settings the master cadence resolves to render_fps=30, so the
    contain filter is the pre-VC-3 `setsar=1,fps=30,…,format=yuv420p`.

    The expected string carries the frame-exactness rebase
    (`setpts=PTS-STARTPTS`, appended by `_cadence_fragment`) — deliberately
    UPDATED, not relaxed: it pinned the pre-fix chain, which delivered one frame
    short on any mid-frame `-ss` (685 frames against a 690-frame plan). The
    assertion is still a full byte-for-byte equality on the whole chain."""
    settings = get_settings()
    frag = renderer._scale_pad_filter(settings, None)
    # swscale flags are pinned deliberately (lanczos: +9.8% retained detail on
    # a real 1920->2560 upscale vs the bicubic default); this guard still pins
    # the EXACT string, it just pins the new one.
    assert frag == (
        "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos+accurate_rnd+full_chroma_int"
        ":in_color_matrix=bt709:out_color_matrix=bt709,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x111827,"
        "setsar=1,fps=30,setpts=PTS-STARTPTS,format=yuv420p"
    )


async def test_scale_pad_filter_cover_default_cadence_is_byte_identical() -> None:
    settings = get_settings()
    frag = renderer._scale_pad_filter(settings, None, fit="cover")
    assert frag.endswith("setsar=1,fps=30,setpts=PTS-STARTPTS,format=yuv420p")


async def test_scroll_filter_default_cadence_tail_is_byte_identical() -> None:
    settings = get_settings()
    frag = renderer._scroll_filter(settings, duration=5.0)
    assert frag.endswith("setsar=1,fps=30,setpts=PTS-STARTPTS,format=yuv420p")


async def test_cadence_fragment_default_is_plain_fps() -> None:
    assert renderer._cadence_fragment(get_settings()) == "fps=30,setpts=PTS-STARTPTS"


# ---------------------------------------------------------------------------
# VC-3 retime wiring: the SOURCE rate must actually reach retime_plan.
#
# The regression this class of test exists for: _cadence_fragment used to pass
# settings.render_fps as BOTH source and target, so the ratio was always exactly
# 1.0, _is_integer_ratio always True, and the minterpolate/tblend branches were
# structurally unreachable from production code — render_motion_blur was a dead
# config surface and Shot.retime was never read at all.
# ---------------------------------------------------------------------------


async def test_cadence_fragment_non_integer_source_with_motion_blur_interpolates() -> None:
    """24 fps source → 30 fps master is a 1.25 (non-integer) ratio: with
    render_motion_blur ON that MUST be motion-compensated, not a frame-dup."""
    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    assert renderer._cadence_fragment(settings, source_fps=24.0) == (
        "minterpolate=fps=30:mi_mode=mci,setpts=PTS-STARTPTS"
    )


async def test_cadence_fragment_integer_source_stays_clean_even_with_motion_blur() -> None:
    """30→60 is an exact 2x: a clean drop/dup, never worth interpolating."""
    settings = get_settings().model_copy(update={"render_fps": 60, "render_motion_blur": True})
    assert renderer._cadence_fragment(settings, source_fps=30.0) == "fps=60,setpts=PTS-STARTPTS"


async def test_cadence_fragment_undeclared_source_is_byte_identical() -> None:
    """A shot that declares no source rate keeps the pre-VC-3 fragment even with
    motion blur on — an unknown ratio must never trigger an unwanted retime."""
    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    assert renderer._cadence_fragment(settings, source_fps=None) == "fps=30,setpts=PTS-STARTPTS"


async def test_cadence_fragment_per_shot_retime_hints_override_the_service_default() -> None:
    """Shot.retime is honoured: 'motion'/'budget' opt IN with the service flag
    OFF, and 'clean' opts OUT with it ON."""
    off = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": False})
    assert (
        renderer._cadence_fragment(off, 24.0, "motion")
        == "minterpolate=fps=30:mi_mode=mci,setpts=PTS-STARTPTS"
    )
    assert "tblend" in renderer._cadence_fragment(off, 24.0, "budget")
    assert renderer._cadence_fragment(off, 24.0, "clean") == "fps=30,setpts=PTS-STARTPTS"
    on = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    assert renderer._cadence_fragment(on, 24.0, "clean") == "fps=30,setpts=PTS-STARTPTS"
    # An unknown hint degrades to the service default (advisory field).
    assert (
        renderer._cadence_fragment(on, 24.0, "sideways")
        == "minterpolate=fps=30:mi_mode=mci,setpts=PTS-STARTPTS"
    )


async def test_scale_pad_filter_threads_the_source_rate_into_the_retime() -> None:
    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    frag = renderer._scale_pad_filter(settings, None, source_fps=24.0)
    assert "minterpolate=fps=30:mi_mode=mci" in frag
    assert "setsar=1,minterpolate=fps=30:mi_mode=mci,setpts=PTS-STARTPTS,format=yuv420p" in frag


async def test_split_filter_threads_the_source_rate_into_the_retime() -> None:
    from src.api.schemas import SplitScreen

    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    graph = renderer._split_filter(
        SplitScreen(layout="h", second_source_url="http://minio/b.mp4"),
        2,
        settings,
        None,
        None,
        None,
        2.0,
        source_fps=24.0,
    )
    assert "minterpolate=fps=30:mi_mode=mci" in graph


async def test_render_shot_footage_uses_the_shots_declared_source_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """END OF THE WIRE: a footage shot declaring target_fps=24 against a 30 fps
    master renders with minterpolate — the whole point of VC-3 retime. Before the
    fix this command contained a bare ``fps=30`` no matter what the shot declared."""
    captured = _install_fakes(monkeypatch)
    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    shot = Shot(
        id="footage24",
        seconds=2.0,
        video_url="http://minio/clip24.mp4",
        target_fps=24.0,
    )
    await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    assert "minterpolate=fps=30:mi_mode=mci" in joined, joined


async def test_render_shot_footage_without_declared_rate_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The default compile (no declared rates) keeps the plain fps= fragment
    (now followed by the frame-exactness `setpts=PTS-STARTPTS` rebase)."""
    captured = _install_fakes(monkeypatch)
    settings = get_settings().model_copy(update={"render_fps": 30, "render_motion_blur": True})
    shot = Shot(id="footage", seconds=2.0, video_url="http://minio/clip.mp4")
    await renderer.render_shot(shot, tmp_path / "seg.mp4", settings, tmp_path, http)
    joined = " ".join(captured["cmds"][-1])
    assert "minterpolate" not in joined
    assert "setsar=1,fps=30,setpts=PTS-STARTPTS,format=yuv420p" in joined


# --------------------------------------------------------------------------
# Beat-cut segments (action_reel): a shot cut OUT of a long source is bounded by
# its OUT-point, on the INPUT side. An IN-point alone is NOT a cut length — that
# shot keeps the natural-length contract asserted above.
# --------------------------------------------------------------------------


async def test_in_point_without_out_point_still_delivers_the_whole_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """An IN-point alone does NOT make ``seconds`` a cut length. Reading it as one
    (an earlier revision did) truncates every "play this recording from here" shot,
    whose ``seconds`` is routinely a placeholder — the contract the render harness
    pins in ``test_clip_out_point.py::test_no_out_point_still_delivers_the_whole_tail``,
    restated here at argv level. A real beat cut states its out-point instead, and
    that is bounded on the INPUT side (see the two tests below).
    """
    captured = _install_fakes(monkeypatch, clip_dur=45.0, clip_has_audio=True)
    shot = Shot(
        id="beat1",
        seconds=2.0,
        video_url="http://minio/drone-4k.mp4",
        clip_lead_seconds=2.0,
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    # Filtergraph source label (see the mandatory per-shot edge fade, 3b8d1e7),
    # not a bare -map token.
    assert "[0:a]" in " ".join(cmd)  # still the clip's own audio
    # Natural length: the only bound is the whole post-seek tail itself (on the
    # frame grid), never ``seconds=2.0``.
    assert cmd.count("-t") == 1
    assert cmd[cmd.index("-t") + 1] == "43.0"
    assert dur == 43.0


async def test_windowed_silent_clip_is_bounded_to_the_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """A 2.0s beat cut out of a 45s drone take, silent source. The window bounds the
    READ on the input side; the synthesised ``anullsrc`` is infinite, so this branch
    ALSO needs its own output-side ``-t`` — and that one must be the window (2.0s),
    not the whole ~43s post-seek tail.
    """
    captured = _install_fakes(monkeypatch, clip_dur=45.0, clip_has_audio=False)
    shot = Shot(
        id="beat2",
        seconds=9.0,  # a placeholder: the WINDOW, not this, defines the segment
        video_url="http://minio/drone-4k.mp4",
        clip_lead_seconds=2.0,
        clip_out_seconds=4.0,
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    assert "anullsrc" in " ".join(cmd)
    # Input-side bound on the read (before the -i it qualifies) ...
    first_t = cmd.index("-t")
    assert first_t < cmd.index("-i")
    assert cmd[first_t + 1] == "2.000"
    # ... and the output-side bound that stops the infinite silence.
    last_t = len(cmd) - 1 - cmd[::-1].index("-t")
    assert last_t > first_t
    assert cmd[last_t + 1] == "2.0"
    assert dur == 2.0


async def test_unwindowed_clip_keeps_its_natural_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """The guard on the fix: with NO stated in/out point, ``seconds`` is a
    placeholder (a 12s lesson recording is routinely sent as ``seconds=1.0``), so
    the clip still plays to its natural end: the only ``-t`` is the clip's OWN
    probed length (on the frame grid), never the placeholder."""
    captured = _install_fakes(monkeypatch, clip_dur=12.0, clip_has_audio=True)
    shot = Shot(id="lesson1", seconds=1.0, video_url="http://minio/lesson1.mp4")
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    assert cmd.count("-t") == 1
    assert cmd[cmd.index("-t") + 1] == "12.0"
    assert dur == 12.0


async def test_out_point_window_keeps_its_input_side_bound_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """An explicit OUT-point already bounds the READ with an INPUT-side ``-t``
    (before ``-i``), and that window — not ``shot.seconds`` — defines the segment.
    So the own-audio branch must NOT gain a second, SHORTER output-side bound: the
    output-side ``-t`` it carries is that same window on the frame grid."""
    captured = _install_fakes(monkeypatch, clip_dur=45.0, clip_has_audio=True)
    shot = Shot(
        id="beat3",
        seconds=1.0,
        video_url="http://minio/drone-4k.mp4",
        clip_lead_seconds=2.0,
        clip_out_seconds=5.5,
    )
    dur = await renderer.render_shot(shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http)
    cmd = captured["cmds"][-1]
    # The window is the INPUT-side -t, before the -i it qualifies ...
    assert cmd.index("-t") < cmd.index("-i")
    assert cmd[cmd.index("-t") + 1] == "3.500"
    # ... and the output-side one is the same 3.5 s, not ``seconds=1.0``.
    assert cmd.count("-t") == 2
    last_t = len(cmd) - 1 - cmd[::-1].index("-t")
    assert cmd[last_t + 1] == "3.5"
    assert dur == 3.5


async def test_muted_clip_renders_silence_instead_of_mapping_its_own_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http: httpx.AsyncClient
) -> None:
    """``audio.mute`` on a footage shot of an ALL-MUTED request must drop the source
    audio from the SHOT, not merely attenuate it post-master: the post-master pass is
    time-ranged to the shot's solo window (nothing inside a crossfade), and the mapped
    source would still drive ``[nar_key]``, ducking the music bed for the whole film.
    (The mixed-request half of this rule lives in ``test_renderer_mute_placement.py``.)
    """
    captured = _install_fakes(monkeypatch, clip_dur=45.0, clip_has_audio=True)
    shot = Shot(
        id="beat4",
        seconds=2.0,
        video_url="http://minio/drone-4k.mp4",
        clip_lead_seconds=2.0,
        clip_out_seconds=4.0,
        audio=ShotAudio(mute=True),
    )
    dur = await renderer.render_shot(
        shot, tmp_path / "seg.mp4", get_settings(), tmp_path, http, all_shots_muted=True
    )
    cmd = captured["cmds"][-1]
    joined = " ".join(cmd)
    assert "anullsrc" in joined
    assert "0:a" not in joined  # the drone's motor/wind never enters the mix
    # Silencing does not change the length: the OUT-point still defines the segment
    # (input-side "-t 2.000"), and the infinite anullsrc gets the matching
    # output-side bound.
    last_t = len(cmd) - 1 - cmd[::-1].index("-t")
    assert cmd[last_t + 1] == "2.0"
    assert dur == 2.0
