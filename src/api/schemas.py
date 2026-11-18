"""Request / response schemas for the compile API.

The shape mirrors w2v_orchestrator's `VideoCompileClient` contract — the
sync entrypoint that PR-2's `video_render` stage calls.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Safe charset for caller-supplied ids that become scratch-file paths + inline ffmpeg
# filtergraph tokens: alphanumerics, dash, underscore, dot. No ``/`` (path traversal),
# no ``:`` ``,`` ``'`` ``\`` (filtergraph breakout), no whitespace/control chars.
_SAFE_ID_PATTERN = r"^[A-Za-z0-9_.-]+$"
# Per-shot / per-clip duration ceiling (seconds). Far above any real shot (typically
# 1-30s); bounds a single request from declaring a multi-year render (a /tmp + billing DoS).
_MAX_SHOT_SECONDS = 3600.0
# Per-request collection ceilings. Generous (far above any real video/audiobook) but finite,
# so one request can't carry millions of shots/clips and exhaust the pod's disk on fetch.
_MAX_SHOTS = 2000
# A caption colour as sent on the wire: "#RRGGBB".
_HEX_RGB_PATTERN = r"^#[0-9A-Fa-f]{6}$"
_MAX_CLIPS = 10000
# Safe charset for caller-supplied ffmpeg OVERLAY POSITION EXPRESSIONS
# (``OverlayClip.x`` / ``.y``). These land INLINE in a ``-filter_complex`` as
# ``overlay=x=<expr>:y=<expr>``, so the same breakout surface as ``_SAFE_ID_PATTERN``
# applies — only wider, because a position genuinely is an expression: ``(W-w)/2``,
# ``H-h-40``, ``w*0.05``. Permitted: alphanumerics (the ``W w H h t n main_w
# overlay_h`` variable names), space, and ``_ . ( ) + - * /``.
# Deliberately EXCLUDED: ``:`` and ``,`` (would inject another filter option or a
# whole new filter into the chain), ``;`` ``[`` ``]`` (would inject a new graph
# chain / link label), and ``' " \`` (quote/escape breakout). Excluding ``,`` also
# rules out multi-arg expression functions (``if(gte(t\,1)\,…)``) — those need
# filtergraph comma-escaping that is not worth the breakout risk here; a timed
# appearance is expressed with ``start_seconds``/``duration_seconds`` instead.
_SAFE_OVERLAY_EXPR_PATTERN = r"^[A-Za-z0-9_ .()+*/-]*$"
# Per-shot ceiling on composited alpha overlay clips. Two is the real product
# shape (a lower-third + a persistent logo bug); each one costs an extra decoded
# VP9 input and a full-canvas ``overlay`` pass per frame, so the ceiling is a
# render-cost bound as much as a design one. Mirrored by
# ``renderer._OVERLAY_CLIP_MAX_PER_SHOT`` so the renderer is safe on its own.
_MAX_OVERLAY_CLIPS = 2


class SoundCue(BaseModel):
    """A one-shot SFX played at the start of a shot (transition whoosh/pop).

    ``url`` is the same reference shape as ``audio_url`` (presigned MinIO
    http(s), ``file://``, or a bare local path). ``gain_db`` is a level trim
    fed straight to ffmpeg ``volume=<gain_db>dB`` (0 = unchanged).
    """

    url: str
    gain_db: float = Field(default=0.0, description="Level trim (dB) applied to this cue")


class AccentCue(BaseModel):
    """A one-shot SFX placed at an offset WITHIN a shot's local timeline."""

    url: str
    at_seconds: float = Field(ge=0, description="Offset from the shot start, seconds")
    gain_db: float = Field(default=0.0, description="Level trim (dB) applied to this cue")


class MusicBed(BaseModel):
    """Continuous music under the whole video, ducked beneath narration.

    Request-level (not per-shot) because it spans the entire timeline — a
    per-shot music input would restart at every cut. Looped to fill the video,
    faded in/out, and sidechain-ducked under the concatenated narration.

    The duck is a ``sidechaincompress`` keyed off the narration bus. ``duck_db``
    sets its DEPTH (mapped to the compression ratio); the four knobs below set
    its TIMING and SENSITIVITY. Their defaults are the shipped behaviour — a
    caller that sends none of them still gets a duck that opens on ordinary
    narration and breathes back in the gaps.
    """

    url: str
    gain_db: float = Field(default=-18.0, description="Resting (un-ducked) music level, dB")
    fade_in_s: float = Field(default=1.5, ge=0)
    fade_out_s: float = Field(default=2.5, ge=0)
    duck_db: float = Field(
        default=-12.0,
        description="Advisory extra attenuation under narration; tunes the sidechain compressor",
    )
    duck_attack_ms: float = Field(
        default=30.0,
        ge=1,
        le=1000,
        description="Sidechain attack, ms — how fast the bed dips when narration starts",
    )
    duck_release_ms: float = Field(
        default=700.0,
        ge=10,
        le=5000,
        description=(
            "Sidechain release, ms — how fast the bed swells back once narration stops. "
            "700 ms is long enough that inter-WORD gaps do NOT recover (which is what "
            "made a shorter release pump once per syllable) and short enough that a real "
            "inter-SENTENCE pause still breathes"
        ),
    )
    duck_threshold: float = Field(
        default=0.03,
        ge=0.001,
        le=1.0,
        description=(
            "Sidechain open threshold as an ABSOLUTE linear key level (0.03 ~= -30 dBFS), "
            "fed straight to ffmpeg sidechaincompress. Ordinary narration must cross it or "
            "the compressor never opens and the bed never ducks. A measured / VO-relative "
            "threshold is a documented follow-up, not this knob"
        ),
    )
    sidechain_hpf_hz: float = Field(
        default=150.0,
        ge=20,
        le=2000,
        description=(
            "High-pass applied to the sidechain KEY (not the audible narration) so the duck "
            "triggers on speech presence rather than rumble / plosives / room tone"
        ),
    )
    music_offset_lu: float | None = Field(
        default=None,
        ge=0,
        le=40,
        description=(
            "How many LU BELOW the measured narration the bed should sit. When set, the "
            "compiler MEASURES the VO bus and the music asset and computes the gain; "
            "``gain_db`` is then the fallback used only if measurement fails."
        ),
    )
    duck_open_lu_below_vo: float | None = Field(
        default=None,
        ge=0,
        le=40,
        description=(
            "When set, the sidechain open threshold is derived from the MEASURED VO level "
            "instead of the absolute ``duck_threshold``."
        ),
    )


class AmbientCue(BaseModel):
    """A continuous ambient bed segment on the GLOBAL video timeline.

    ``end_seconds=None`` runs the bed to the end of the video. Multiple cues
    let different sections carry different ambience; today the orchestrator
    emits a single full-span cue, but the list shape leaves room to grow
    without a schema change.

    The four ``duck_*``/``sidechain_hpf_hz`` knobs below shape the sidechain
    duck the SUMMED ambient bus gets under narration. They mirror ``MusicBed``'s
    breathing-duck defaults and bounds EXACTLY (identical names so the renderer
    reads music and ambient uniformly) because the old ambient-only duck was a
    DEAD filter: it hardcoded ``threshold=0.125:release=1000``, where 0.125 is an
    absolute linear key level a -18 LUFS narration bus never crosses (so the
    compressor barely opened) and a 1000 ms release pinned the bed flat through
    every VO gap — an ambient bed never breathed. These defaults give it the
    same fast, HPF'd-key, short-release breathing shape ``MusicBed`` ships. No
    ``duck_db``/ratio knob: the ambient duck runs at a fixed ratio (mirroring
    ``MusicBed``'s effective ratio for its default ``duck_db=-12``), keeping the
    surface tight. Presence-is-opt-in: a caller sending none of these still gets
    the shipped breathing default.
    """

    url: str
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float | None = Field(
        default=None, description="End offset, seconds; None = run to end of video"
    )
    gain_db: float = Field(default=-24.0, description="Bed level under narration, dB")
    ambient_offset_lu: float | None = Field(
        default=None,
        ge=0,
        le=40,
        description=(
            "How many LU BELOW the measured narration this ambient cue should sit. When set, "
            "the compiler MEASURES the VO bus and this asset and computes the gain; "
            "``gain_db`` is then the fallback used only if measurement fails."
        ),
    )
    duck_attack_ms: float = Field(
        default=30.0,
        ge=1,
        le=1000,
        description="Sidechain attack, ms — how fast the bed dips when narration starts",
    )
    duck_release_ms: float = Field(
        default=700.0,
        ge=10,
        le=5000,
        description=(
            "Sidechain release, ms — how fast the bed swells back once narration stops. "
            "700 ms is long enough that inter-WORD gaps do NOT recover (which is what "
            "made a shorter release pump once per syllable) and short enough that a real "
            "inter-SENTENCE pause still breathes"
        ),
    )
    duck_threshold: float = Field(
        default=0.03,
        ge=0.001,
        le=1.0,
        description=(
            "Sidechain open threshold as an ABSOLUTE linear key level (0.03 ~= -30 dBFS), "
            "fed straight to ffmpeg sidechaincompress. Ordinary narration must cross it or "
            "the compressor never opens and the bed never ducks"
        ),
    )
    sidechain_hpf_hz: float = Field(
        default=150.0,
        ge=20,
        le=2000,
        description=(
            "High-pass applied to the sidechain KEY (not the audible narration) so the duck "
            "triggers on speech presence rather than rumble / plosives / room tone"
        ),
    )
    duck_open_lu_below_vo: float | None = Field(
        default=None,
        ge=0,
        le=40,
        description=(
            "When set, the sidechain open threshold is derived from the MEASURED VO level "
            "instead of the absolute ``duck_threshold``."
        ),
    )


class Transition(BaseModel):
    """A visual crossfade INTO a shot from the PREVIOUS one.

    A shot's ``transition`` describes how the cut ENTERING that shot behaves, so
    the first shot's ``transition`` is meaningless (no preceding shot to dissolve
    from) and is ignored. ``type`` selects an ffmpeg ``xfade`` transition;
    ``dissolve`` is an alias for the plain ``fade`` cross-dissolve. ``duration``
    is the overlap length in seconds; at render time it is clamped to the shorter
    of the two adjacent segments so a crossfade can never exceed either clip.

    When ANY shot in a compile carries a ``transition`` the compiler leaves the
    fast stream-copy concat path and joins segments through a re-encoding
    ``xfade``/``acrossfade`` chain instead; a compile with no transitions is
    byte-identical to before this field existed.
    """

    type: Literal["fade", "dissolve", "wipeleft", "slideright"] = Field(
        default="fade", description="xfade transition style ('dissolve' aliases 'fade')"
    )
    duration: float = Field(
        default=0.5,
        gt=0,
        le=5.0,
        description="Crossfade overlap, seconds; clamped to the shorter adjacent clip",
    )


class ColorAdjust(BaseModel):
    """Per-shot colour grade applied as an ffmpeg ``eq`` pass.

    Every numeric knob defaults to its identity value (a grade that changes
    nothing), so an all-default ``ColorAdjust`` with no ``preset`` emits no
    filter at all and the render stays byte-identical to before this field
    existed. ``preset`` expands to a fixed grade (warm/cool/bw/vivid/cinematic);
    any explicitly-set numeric field overrides the matching preset component.
    """

    brightness: float = Field(
        default=0.0, ge=-1, le=1, description="Additive brightness (-1..1; 0 = unchanged)"
    )
    contrast: float = Field(
        default=1.0, ge=0, le=2, description="Contrast multiplier (0..2; 1 = unchanged)"
    )
    saturation: float = Field(
        default=1.0,
        ge=0,
        le=3,
        description="Saturation multiplier (0..3; 1 = unchanged, 0 = greyscale)",
    )
    gamma: float = Field(
        default=1.0, ge=0.1, le=10, description="Gamma correction (0.1..10; 1 = unchanged)"
    )
    preset: Literal["warm", "cool", "bw", "vivid", "cinematic"] | None = Field(
        default=None,
        description=(
            "Named grade expanded to a fixed eq; explicitly-set numeric fields "
            "override the matching preset component. None = no preset."
        ),
    )


class TextOverlay(BaseModel):
    """A burned-in text title drawn on a shot via ffmpeg ``drawtext``.

    ``position`` anchors the text at one of nine safe-area points; an explicit
    ``x``/``y`` (pixels from the top-left) overrides the anchor. ``start``/``end``
    (shot-local seconds) gate the overlay's visibility via an ``enable=between``
    expression; both unset shows it for the whole shot.
    """

    text: str = Field(min_length=1, max_length=200, description="Overlay text (rendered one line)")
    position: Literal[
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    ] = Field(default="bottom-center", description="Nine-grid anchor; overridden by explicit x/y")
    x: int | None = Field(default=None, description="Explicit x (px from left); overrides position")
    y: int | None = Field(default=None, description="Explicit y (px from top); overrides position")
    font_size: int = Field(default=48, ge=8, le=200, description="Font size in px")
    color: str = Field(default="white", description="ffmpeg fontcolor (name or 0xRRGGBB)")
    start: float | None = Field(
        default=None, ge=0, description="Show from this shot-local second (None = from the start)"
    )
    end: float | None = Field(
        default=None, gt=0, description="Hide after this shot-local second (None = to the end)"
    )


class ShotAudio(BaseModel):
    gain_db: float = Field(
        default=0.0, ge=-60, le=12, description="Primary-audio volume adjust in dB"
    )
    mute: bool = Field(default=False, description="Drop the shot's primary audio (silence)")
    fade_in_s: float = Field(
        default=0.0, ge=0, le=30, description="Fade-in seconds from shot start"
    )
    fade_out_s: float = Field(default=0.0, ge=0, le=30, description="Fade-out seconds to shot end")


class CursorKey(BaseModel):
    """One keyframe of a synthetic cursor path (ORCH-3.2 directed footage).

    ``t`` is the shot-local second; ``x``/``y`` are the cursor tip position in
    the RENDER frame's pixel space (the same frame-pixel space the VP-2 capture
    producer emits — CSS px x capture_scale). ``click`` marks a click at this
    keyframe, which triggers a brief expanding click-pulse ring. Between
    keyframes the renderer smoothstep-eases the position. An absent
    ``cursor_track`` (the default) synthesises no cursor and is byte-identical.
    """

    t: float = Field(ge=0, description="Shot-local time of this keyframe, seconds")
    x: float = Field(description="Cursor-tip x in render-frame pixels")
    y: float = Field(description="Cursor-tip y in render-frame pixels")
    click: bool = Field(default=False, description="A click fires here → click-pulse ring")


class FocusKey(BaseModel):
    """One keyframe of a focus/auto-zoom track (ORCH-3.2 directed footage).

    ``bbox`` is ``(x, y, w, h)`` in RENDER-frame pixel space — the region the
    punch-in should frame (a clicked control, a highlighted panel). The renderer
    eases a ``zoompan`` punch-in centered on the bbox and back out. An absent
    ``focus_track`` (the default) applies no zoom and is byte-identical.
    """

    t: float = Field(ge=0, description="Shot-local time of this keyframe, seconds")
    bbox: tuple[float, float, float, float] = Field(
        description="Focus region (x, y, w, h) in render-frame pixels"
    )


class Callout(BaseModel):
    """One per-shot callout bound to a clicked element (ORCH-3.2 directed footage).

    A ``badge`` draws a tinted numbered disc glued to the element's top-left
    corner (its ``label`` is the short number/text centered on the disc); a
    ``label`` draws a short caption anchored to the element bbox; a ``spotlight``
    dims the bbox-complement to isolate the element; an ``arrow`` overlays a small
    triangle sprite pointing at the element. ``bbox`` is ``(x, y, w, h)`` in
    RECORDED source pixels — the SAME convention as ``FocusKey.bbox`` — so the
    renderer remaps it through the same fit transform before compositing. Callouts
    ride the PRE-zoom canvas so a focus punch-in magnifies them with the element.
    """

    at: float = Field(ge=0, description="Shot-local onset seconds")
    type: Literal["badge", "label", "spotlight", "arrow"]
    bbox: tuple[float, float, float, float] = Field(
        description="Clicked-element region (x, y, w, h) in RECORDED source pixels "
        "(same convention as FocusKey.bbox)"
    )
    label: str = Field(
        default="", max_length=120, description="L3 callout copy; badge → short number/text"
    )


class CaptionWord(BaseModel):
    """One word's shot-local ``[start, end)`` timing, for karaoke ``{\\k}`` highlight.

    Shot-local seconds, mapped onto the film bus with the SAME offset/÷speed/clamp
    math as the parent cue. Only consumed when a compile requests
    ``captions_karaoke`` in ``burned_in`` mode; otherwise ignored.
    """

    text: str
    start: float = Field(ge=0, description="Word onset, shot-local seconds")
    end: float = Field(gt=0, description="Word offset, shot-local seconds")

    @model_validator(mode="after")
    def _end_after_start(self) -> CaptionWord:
        if self.end <= self.start:
            raise ValueError("caption word end must be greater than start")
        return self


class CaptionCue(BaseModel):
    """One word- or phrase-timed caption, in the SHOT's local timeline (seconds).

    ``start``/``end`` are shot-local; the renderer maps them onto the film bus
    (applying the shot's ``speed`` retime and clamping to the shot's on-screen
    window) when it builds the burned-in ASS subtitle track. ``end`` must be
    strictly greater than ``start``.

    ``words`` — the cue's per-word timings — is optional; it powers karaoke
    ``{\\k}`` word-highlight when a compile sets ``captions_karaoke`` and renders
    ``burned_in``. ``None``/absent ⇒ the whole cue highlights as one phrase.
    """

    text: str
    start: float = Field(ge=0, description="Cue onset, shot-local seconds")
    end: float = Field(gt=0, description="Cue offset, shot-local seconds")
    words: list[CaptionWord] | None = Field(
        default=None,
        description=(
            "Optional per-word timings (shot-local) for karaoke highlight. Used "
            "only when the compile sets `captions_karaoke` in `burned_in` mode; "
            "None (default) ⇒ the cue highlights as a single phrase."
        ),
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> CaptionCue:
        if self.end <= self.start:
            raise ValueError("caption cue end must be greater than start")
        return self


class ShotCaptions(BaseModel):
    """The caption cues for one shot, addressed by its index in the shot list.

    ``shot_index`` is the 0-based position of the shot in ``CompileRequest.shots``.
    An empty ``cues`` list contributes nothing to the burned-in track.
    """

    shot_index: int = Field(ge=0, description="0-based index into CompileRequest.shots")
    cues: list[CaptionCue] = Field(default_factory=list)


class SplitScreen(BaseModel):
    layout: Literal["h", "v"] = Field(
        ..., description="hstack (h) side-by-side | vstack (v) stacked."
    )
    second_source_url: str = Field(
        ...,
        description="Resolved URL of the SECOND video composited beside the primary. "
        "The primary source drives the shot duration; both halves are letterboxed into their half-canvas.",
    )


class OverlayClip(BaseModel):
    """An animated graphic with a REAL ALPHA CHANNEL composited ON TOP of the shot.

    This is the true overlay layer — a lower-third that flies in over drone
    footage, a persistent logo bug — as opposed to a full-frame animated card,
    which needs nothing new (it is an ordinary ``Shot`` with a ``video_url``).

    ``url`` is the same reference shape as every other asset (presigned MinIO
    http(s), ``file://``, or a bare local path) and is fetched through the same
    per-job dedupe, so one logo bug reused on twenty shots downloads once. An
    unfetchable overlay DROPS that overlay and logs; it never fails the compile.

    THE PRODUCER SHIPS VP9 WEBM WITH ALPHA. Alpha does not ride in the pixel
    format — a correct file reports ``pix_fmt=yuv420p`` and carries the WebM
    container tag ``alpha_mode=1``, with the alpha plane in ``BlockAdditional``
    side data. Anything that "validates alpha" by looking for ``yuva420p`` will
    reject every correct build. See ``renderer._inject_overlay_clips`` for the
    decoder trap this costs on the ffmpeg side.

    Times are SHOT-LOCAL seconds on the shot's own (pre-``speed``) timeline —
    see ``retime_with_speed``.
    """

    url: str
    start_seconds: float = Field(
        default=0.0,
        ge=0,
        le=_MAX_SHOT_SECONDS,
        description="Shot-local onset, seconds; 0 (default) = visible from the shot's first frame",
    )
    duration_seconds: float | None = Field(
        default=None,
        gt=0,
        le=_MAX_SHOT_SECONDS,
        description=(
            "How long the overlay stays on screen, seconds. None (default) = the "
            "overlay's own natural length, after which it disappears (it does NOT "
            "freeze on its last frame)."
        ),
    )
    x: str = Field(
        default="0",
        max_length=64,
        pattern=_SAFE_OVERLAY_EXPR_PATTERN,
        description=(
            "ffmpeg overlay x expression in CANVAS pixels — a bare number, or an "
            "expression over W/H (canvas) and w/h (overlay), e.g. '(W-w)/2' to "
            "centre or 'W-w-64' to pin right. Charset-constrained (no ':' ',' or "
            "quotes) because it is interpolated into the filtergraph."
        ),
    )
    y: str = Field(
        default="0",
        max_length=64,
        pattern=_SAFE_OVERLAY_EXPR_PATTERN,
        description="ffmpeg overlay y expression in canvas pixels; same charset rules as ``x``",
    )
    scale: Literal["fit", "native"] = Field(
        default="fit",
        description=(
            "'fit' (default) scales the graphic to fit INSIDE the render canvas "
            "preserving its aspect (no crop, no pad — the alpha carries the "
            "framing), so one 1920x1080 authored asset lands correctly at every "
            "output tier. 'native' composites it at its authored pixel size."
        ),
    )
    opacity: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description=(
            "Uniform multiplier on the graphic's OWN alpha (1.0 = default, the "
            "authored alpha untouched and byte-identical; 0 = fully transparent)."
        ),
    )
    retime_with_speed: bool = Field(
        default=True,
        description=(
            "Whether ``start_seconds``/``duration_seconds`` are read on the shot's "
            "SOURCE timeline (True, default) or on the RENDERED one (False). The "
            "overlay is a separate ffmpeg input and is NOT touched by the shot's "
            "``speed`` retime, so True divides both times by ``speed`` — a cue "
            "authored against the source stays glued to the frame it was authored "
            "for. False pins the overlay to wall-clock output time instead (a "
            "logo bug that must appear 2s into the delivered segment regardless "
            "of how the picture was retimed). At ``speed=1.0`` the two are "
            "identical."
        ),
    )


class Shot(BaseModel):
    # id is interpolated into scratch FILE PATHS (``{id}-clip.mp4`` …) and INLINE into
    # ffmpeg filtergraphs (``drawtext=…:textfile={path}``). Constrain to a safe charset so
    # a caller-supplied id can neither traverse the filesystem (``/``) nor break out of the
    # filter graph (``:`` ``,`` quotes) nor be parsed as an ffmpeg flag. Storyboard/line
    # ids (``l1``, ``scene_3``, uuids) already satisfy this.
    id: str = Field(pattern=_SAFE_ID_PATTERN, max_length=128)
    line_id: str | None = None
    seconds: float = Field(
        gt=0,
        le=_MAX_SHOT_SECONDS,
        description=(
            "Requested wall-clock duration. For a silent visual-only shot this IS the "
            "length. When ``audio_url`` is set AND ``max_seconds`` is set (a shot the "
            "orchestrator's reconcile_timeline bounded) it is a FLOOR: the segment lasts "
            "``max(probed narration, seconds)``, so the narration is never cut and a "
            "plan that pins a footage beat to its measured clip (the orchestrator "
            "clamps ``seconds`` into reconcile_timeline's ``[min, max]`` window) is "
            "honoured instead of being shortened to the narration; the audio after "
            "the narration is silence. Without ``max_seconds`` a narrated shot's "
            "``seconds`` may be a storyboard guess and is ignored (narration length). "
            "A floor within one output frame of the "
            "narration is ignored. A clip that carries its OWN audio (no "
            "``audio_url``) still plays at its natural length. Bounded so one shot "
            "can't request a multi-year render."
        ),
    )
    max_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Upper bound (clip length + freeze-pad ceiling) the orchestrator's "
            "reconcile_timeline stamped for a footage beat carrying over-length "
            "narration. Advisory unless the ``freeze_pad_cap`` render tag is on: "
            "when set, the freeze-pad DRIFT is bounded to end at ``max_seconds`` "
            "instead of ramping over the whole narration, so an over-narrated "
            "footage clip stops drifting at the ceiling and settles for the tail "
            "(the video still fills the full audio — no black, no truncated audio). "
            "None (default) or the tag off is a no-op — byte-identical to before "
            "this field existed."
        ),
    )
    # Visual source, in precedence order: video_url > image_url > text card.
    video_url: str | None = Field(
        default=None,
        description="Pre-recorded clip (keeps its own audio unless audio_url overrides)",
    )
    image_url: str | None = Field(
        default=None,
        description="Still frame held for the shot duration (slide PNG / screenshot / AI image)",
    )
    audio_url: str | None = Field(
        default=None, description="Narration audio muxed over this shot; sets the shot duration"
    )
    description: str | None = Field(
        default=None, description="Text drawn on a solid card when no image_url/video_url is given"
    )
    caption: str | None = Field(default=None, description="Optional burned-in lower-third caption")
    fit: Literal["contain", "cover", "scroll"] = Field(
        default="contain",
        description=(
            "How the visual fills the frame. 'contain' (default) fits-inside + "
            "pads (letter/pillarbox) so nothing is cropped — for footage/slides/"
            "screenshots. 'cover' scales-to-fill + center-crops so the frame is "
            "full — for atmospheric AI-image backgrounds, killing the navy bars. "
            "'scroll' fits a tall full-page screenshot to the frame width and "
            "pans it top→bottom over the shot — for website-review captures that "
            "would otherwise letterbox to an unreadable vertical sliver."
        ),
    )
    transition_sfx: SoundCue | None = Field(
        default=None, description="One-shot SFX mixed in at this shot's cut-in (t=0)"
    )
    sfx: list[AccentCue] = Field(
        default_factory=list, description="Accent SFX mixed in at offsets within the shot"
    )
    clip_anchor: Literal["start", "end"] = Field(
        default="start",
        description=(
            "Where within a footage clip the shown window sits. 'start' (default) "
            "plays from the beginning (optionally after ``clip_lead_seconds``). "
            "'end' shows the clip's SETTLED TAIL — the last ``duration`` seconds — "
            "so a slow app load / zoom-in at the head is skipped and only the final "
            "hero state (a fitted floor plan, a loaded page) appears. Zero leading "
            "load-in / white frames. Ignored for image/text shots."
        ),
    )
    clip_lead_seconds: float = Field(
        default=0.0,
        ge=0,
        description=(
            "For ``clip_anchor='start'``: seconds to skip at the head of the "
            "footage clip (trim the app's white load-in) before the shown window. "
            "Bounded so the remaining clip still covers the shot duration."
        ),
    )
    clip_out_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Source-timeline OUT-point, in seconds from the head of the footage "
            "clip. Together with ``clip_lead_seconds`` (the in-point) it names an "
            "explicit window: ``clip_lead_seconds=4.0, clip_out_seconds=7.5`` "
            "shows exactly 4.0s to 7.5s of the source. That is the only way to "
            "cut SEVERAL shots out of ONE long recording; without it the "
            "out-point is merely implied by ``seconds`` and every shot replays "
            "from the same in-point. Measured on the SOURCE timeline, so it is "
            "independent of ``speed`` (a 3.5s window at speed=2.0 renders as a "
            "1.75s segment). Clamped to the real clip length; a window shorter "
            "than the shot's narration freeze-pads as usual. Must be greater "
            "than ``clip_lead_seconds``. Ignored for image/text shots. None "
            "(default) is a no-op, byte-identical to before this field existed."
        ),
    )
    target_fps: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Declared source frame rate of this shot's visual (VC-3). Feeds the "
            "master-cadence selector so the whole compile snaps onto one "
            "retime-cost-minimal output rate. Advisory — an omitted value simply "
            "doesn't bias the selection; the render still snaps to the master rate."
        ),
    )
    retime: str | None = Field(
        default=None,
        description=(
            "Optional per-shot retiming hint onto the master cadence: 'clean' "
            "(frame drop/dup), 'motion' (motion-compensated interpolation), or "
            "'budget' (cheap blend). Only bites when this shot's ``target_fps`` "
            "is a NON-integer ratio of the master cadence — an exact ratio is "
            "always the clean drop/dup. Advisory; unset/unknown falls back to "
            "the service-wide ``render_motion_blur`` (off by default = clean)."
        ),
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description=(
            "Per-clip playback speed factor. Retimes BOTH this shot's video "
            "(setpts) and its audio (atempo) so the rendered segment lasts "
            "1/speed of its source: 2.0 = twice as fast / half the length, 0.5 = "
            "half speed. 1.0 (default) is a no-op — byte-identical to before this "
            "field existed."
        ),
    )
    transition: Transition | None = Field(
        default=None,
        description=(
            "Crossfade INTO this shot from the previous one (xfade/acrossfade). "
            "Ignored on the first shot. Setting it on ANY shot switches the whole "
            "compile from the fast stream-copy concat to a re-encoding xfade join."
        ),
    )
    adjust: ColorAdjust | None = Field(
        default=None,
        description=(
            "Optional per-shot colour grade (ffmpeg eq), applied after scale/pad. "
            "None — or an all-default grade with no preset — is a no-op, "
            "byte-identical to before this field existed."
        ),
    )
    overlays: list[TextOverlay] = Field(
        default_factory=list,
        description=(
            "Burned-in text titles (drawtext) drawn on this shot after the grade. "
            "Empty (default) is a no-op."
        ),
    )
    focus_track: list[FocusKey] | None = Field(
        default=None,
        description=(
            "Auto-zoom track (ORCH-3.2): a punch-in eased onto the focus bbox and "
            "back out (ffmpeg zoompan), applied after fit/scale and before the look. "
            "None (default) applies no zoom — byte-identical to before this field."
        ),
    )
    cursor_track: list[CursorKey] | None = Field(
        default=None,
        description=(
            "Synthetic cursor path (ORCH-3.2): a brand-tinted disc overlaid on the "
            "shot, smoothstep-eased between keyframes, with a click-pulse ring on "
            "click keyframes. None (default) synthesises no cursor — byte-identical."
        ),
    )
    footage_viewport: tuple[int, int] | None = Field(
        default=None,
        description=(
            "(width, height) in pixels of the recorded source video the "
            "focus_track/cursor_track coordinates were captured in. When set and "
            "!= the render canvas, the renderer remaps those coords through the "
            "same decrease-scale+centered-pad transform used to fit the source, so "
            "cursor/zoom land on-target at any output tier. None => coords assumed "
            "already in render-canvas pixels (legacy/identity)."
        ),
    )
    callouts: list[Callout] | None = Field(
        default=None,
        max_length=6,
        description=(
            "Optional per-shot callouts (badge/label/spotlight/arrow) bound to the "
            "clicked element bbox in recorded px; None/[] ⇒ no-op, byte-identical. "
            "The renderer composites at most the first few (see _CALLOUT_MAX_PER_SHOT)."
        ),
    )
    audio: ShotAudio | None = Field(
        default=None,
        description="Per-shot primary-audio gain/mute/fade. None is a no-op — byte-identical.",
    )
    split: SplitScreen | None = Field(
        default=None,
        description="Composite two sources into the shot canvas via hstack/vstack (ORCH-3.2 T8). "
        "None (default) is a no-op — byte-identical to before this field existed. On a split shot the "
        "renderer disables cursor/callout/zoom/speed (their coords are in the primary's full-canvas px).",
    )
    overlay_clips: list[OverlayClip] = Field(
        default_factory=list,
        max_length=_MAX_OVERLAY_CLIPS,
        description=(
            "Animated graphics with a real alpha channel composited ON TOP of this "
            "shot's picture (lower-thirds, logo bugs), after the grade/titles. At "
            "most 2 per shot. Empty (default) is a no-op — no extra ffmpeg input, "
            "no extra filter, byte-identical to before this field existed. A "
            "FULL-FRAME animated card is not this: it is an ordinary shot with a "
            "``video_url``."
        ),
    )

    @model_validator(mode="after")
    def _clip_window_is_ordered(self) -> Shot:
        """An explicit out-point must sit AFTER the in-point.

        ``clip_out_seconds <= clip_lead_seconds`` describes an empty or inverted
        source window. Left unvalidated it would reach ``_clip_seek_args`` as a
        zero/negative span, which the renderer would clamp to "no bound" — i.e. a
        caller that meant "4.0s-3.5s" (a transposed pair) would silently get the
        WHOLE clip instead of a 422, and the defect would only surface as a wrong
        shot in a delivered mp4. Reject at the boundary, where it is cheap.
        ``clip_out_seconds is None`` (the default) skips the check entirely.
        """
        if self.clip_out_seconds is not None and self.clip_out_seconds <= self.clip_lead_seconds:
            raise ValueError("clip_out_seconds must be greater than clip_lead_seconds")
        return self


class MasteringProfile(BaseModel):
    """Per-job overrides for the encode/delivery master (VC-2, contract C11).

    Every field is optional; an omitted/out-of-range value degrades to the
    service default (never a 5xx) via the same clamp discipline as the compile
    ``tags``. These fold into the tag-resolution path in the compile route, so
    the picture master can be retargeted per run without a redeploy.
    """

    colorspace: str | None = Field(
        default=None, description="Delivery colorspace tag: bt709/bt601/bt2020/srgb/none"
    )
    deband: str | None = Field(default=None, description="Conditional deband policy: auto/on/off")
    bitrate_mode: str | None = Field(
        default=None, description="Rate control: auto (CRF only) / vbv / cbr / capped"
    )
    maxrate: str | None = Field(
        default=None, description="VBV peak bitrate (e.g. '12M'); only used under a VBV mode"
    )
    crf: int | None = Field(default=None, ge=0, le=51, description="Per-job libx264 CRF override")
    loudness_target: float | None = Field(
        default=None, description="Loudness-master integrated target (LUFS); clamped to [-30, -5]"
    )
    true_peak: float | None = Field(
        default=None, description="Loudness-master true-peak ceiling (dBTP); clamped to [-9, 0]"
    )
    platform: str | None = Field(
        default=None, description="Target delivery platform (youtube/instagram/tiktok/x/web)"
    )


class GradeSpec(BaseModel):
    """Whole-film colour grade applied as ONE post-concat re-encode pass (VC-5).

    Unlike the per-shot ``ColorAdjust`` (which folds an ``eq`` onto each segment
    before the stream-copy concat), this grades the ENTIRE finished timeline in a
    single pass over the delivery master — a film-wide "look" with split-toning,
    vignette, grain, sharpen, glow and an optional broadcast-legal clamp.

    Presence-is-opt-in and identity-is-no-op: ``enabled=False`` (the default), or
    an ``enabled`` grade whose every knob sits at its identity default, emits an
    EMPTY filter chain (``renderer._grade_vf`` → ``""``), so the compile keeps the
    ``-c copy`` fast path and stays byte-identical to a pre-VC-5 render. Only a
    grade that actually moves a knob triggers the whole-film re-encode.

    Field names + clamps mirror the orchestrator's ``finish.GradeSpec`` byte-for-
    byte so a request round-trips unchanged. Tints are accepted permissively (3-
    or 6-digit hex, leading ``#`` optional) — a stricter pattern here would 422 a
    valid orchestrator payload; an unparseable tint degrades to a neutral (no-op)
    split-tone rather than rejecting the compile.
    """

    enabled: bool = Field(
        default=False, description="Master opt-in; False (default) emits no grade (byte-identical)"
    )
    look: str = Field(
        default="none",
        description=(
            "Named base look expanded to a fixed eq (warm/cool/bw/vivid/cinematic "
            "reuse the per-shot presets); a free string — an unknown look is a "
            "no-op base, not a 422. 'none' (default) adds no base grade."
        ),
    )
    contrast: float = Field(
        default=1.0, ge=0.8, le=1.4, description="Whole-film contrast multiplier (1 = unchanged)"
    )
    saturation: float = Field(
        default=1.0, ge=0.6, le=1.4, description="Whole-film saturation multiplier (1 = unchanged)"
    )
    gamma: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Whole-film gamma (1 = unchanged)"
    )
    temperature: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Warm/cool bias (-1 cool .. +1 warm; 0 = neutral), applied as a per-channel gamma",
    )
    shadow_tint: str = Field(
        default="#000000",
        description="Split-tone colour pushed into the shadows; #000000 (default) = no shadow tint",
    )
    highlight_tint: str = Field(
        default="#ffffff",
        description="Split-tone colour pushed into the highlights; #ffffff (default) = no tint",
    )
    vignette: float = Field(
        default=0.0, ge=0, le=1, description="Corner darkening amount (0 = off)"
    )
    grain: float = Field(default=0.0, ge=0, le=1, description="Film-grain noise amount (0 = off)")
    glow: float = Field(default=0.0, ge=0, le=1, description="Soft bloom/glow amount (0 = off)")
    sharpen: float = Field(
        default=0.0, ge=0, le=1, description="Output sharpening amount (0 = off)"
    )
    grade_intensity: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Global dry/wet on the whole grade (1 = full, 0 = identity/no-op)",
    )
    match_sources: bool = Field(
        default=False,
        description=(
            "Bias the grade's luma/saturation midpoints toward the delivered "
            "film's own measured signalstats (best-effort; a measurement miss "
            "silently falls back to the un-biased grade)."
        ),
    )
    broadcast_safe: bool = Field(
        default=False,
        description=(
            "Append a best-effort legal-range (16..235) luma clamp as the LAST "
            "filter — the colour analog of the true-peak limiter, not a hard "
            "guarantee (x264 ringing can re-introduce a few illegal codes)."
        ),
    )
    lut_url: str | None = Field(
        default=None,
        description=(
            "Optional 3D-LUT reference. Carried on the contract + report for "
            "forward-compat; LUT application is not performed by this service."
        ),
    )


class FinishProfile(BaseModel):
    """Per-job finishing/delivery overrides (VC-5): the brand letterbox colour.

    Presence-is-opt-in, exactly like ``MasteringProfile``: an absent ``finish``
    (or an all-default one) changes no filter string, so the render stays
    byte-identical to before this field existed. The colour folds onto the
    per-shot filter chain — the final concat is a stream copy, so there is no
    whole-film finishing pass to hang it on.

    A ``broadcast_safe`` legal-luma flag was designed here and CUT: a pre-encode
    clamp cannot guarantee a decoded 16..235 master (libx264 ringing around text
    re-introduces illegal values, worsening with CRF) and the flag would have
    been a lie. See the note in ``renderer`` above ``_pad_color``.
    """

    letterbox_color: str | None = Field(
        default=None,
        # Anchored with ^/$ — pydantic v2 compiles patterns with the Rust `regex`
        # crate, which rejects \A and \Z as unrecognized escapes.
        pattern=r"^#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?$",
        description=(
            "Brand letterbox/pillarbox fill as #RRGGBB (or #RRGGBBAA), e.g. "
            "'#101820'. None (default) keeps the service's render_bg_color. The "
            "hex pattern is also the injection guard — this value is interpolated "
            "into the ffmpeg filtergraph's pad `color=` option."
        ),
    )
    grade: GradeSpec | None = Field(
        default=None,
        description=(
            "Whole-film colour grade (VC-5): a single post-concat re-encode over "
            "the delivered timeline. None — or a disabled/all-identity grade — is "
            "a no-op, byte-identical to before this field existed."
        ),
    )


class ShotBanding(BaseModel):
    """One shot's PER-SHOT banding measurement (S20), keyed by the orchestrator's
    line_id. Bucketed from a discarded timed signalstats pass so the encode-critic
    can decide deband per shot instead of only whole-file. ``banding`` is the
    shot's worst-frame 0..1 banding score; ``banding_residual_over_trigger``
    reuses the SAME threshold as the whole-file banding gate.

    ⚠️ The flag is published under TWO names carrying the identical value.
    ``banding_residual_over_trigger`` is the accurate one — it is measured on the
    DELIVERED file, after the conditional deband pass has already run, so it
    reports a RESIDUAL, not an owed action. ``needs_deband`` is the original name,
    kept as an alias because ``w2v_orchestrator``'s demo_video encode-critic pins
    it in a structured-output JSON schema. See ``mastering.bucket_banding``.

    Neither name gates. A true flag on any shot is surfaced at the report's top
    level as a ``banding_residual`` entry in ``MasteringReport.advisories`` —
    never in ``violations``, and never flips ``gate_pass``."""

    line_id: str | None = None
    banding: float = 0.0
    banding_residual_over_trigger: bool = False
    needs_deband: bool = False


class MasteringViolation(BaseModel):
    """ONE cleared mastering gate, as published on the wire.

    THE CONTRACT with the orchestrator-side consumer, and now a MODEL rather than
    a ``dict[str, Any]`` plus a hand-kept key tuple plus a test: the five field
    names below ARE the contract, and ``tier``'s two-value vocabulary is enforced
    by the type system instead of by convention.

    ``mastering.violations_payload`` is the single producer; it dumps this model,
    so the wire shape cannot drift from this declaration.

    ⛔ ``detail`` IS DELIBERATELY ABSENT. It exists on the internal
    ``mastering.MasteringGateViolation`` as operator prose for the job's ``error``
    string. A consumer parsing prose is re-deriving intent from a message, which
    is the exact defect the structured verdict exists to end — so it must not
    appear here, and the field-set assertion in
    ``tests/unit/test_av_quality_wave13.py`` pins that.

    ⚠️ ``value``/``limit`` MUST default to ``None`` and serialise as ``null``.
    They are ``None`` for the non-numeric checks (an absent colorspace tag, an
    unmeasurable loudness, an absent report), and a ``0.0`` default would turn
    "never measured" into "measured zero" — the same absence-as-evidence
    confusion ``master_measure_failed`` exists to prevent, in a field a consumer
    compares numerically.
    """

    stage: str
    metric: str
    value: float | None = None
    limit: float | None = None
    tier: Literal["structural", "tunable"]


#: The published contract's key set, DERIVED from the model above so the two can
#: never disagree. ``mastering._VIOLATION_CONTRACT_KEYS`` re-exports it.
VIOLATION_CONTRACT_KEYS: tuple[str, ...] = tuple(MasteringViolation.model_fields)


class NarrationPresenceClip(BaseModel):
    """One shot's content-aware narration-presence measurement: cross-correlation
    of the SOURCE narration clip the compiler fetched against the delivered
    master's own audio at that shot's [start, end) timeline window (padded to
    absorb AAC encoder delay / probe rounding). See
    ``mastering.narration_presence_report``.

    ``correlation_db``/``present`` are BOTH ``None`` when the clip could not be
    fetched, decoded, or windowed — an unmeasured clip, never a fabricated pass
    or fail (no-gate-on-a-gap, the same policy every other check in
    ``MasteringReport`` follows).

    ``correlation_db`` (#234) is now LAG-TOLERANT: the best reading found while
    searching up to ``mastering.NARRATION_PRESENCE_LAG_SEARCH_SECONDS`` of offset
    around the shot's own probed window, not a single fixed-offset reading.
    ``neighbor_correlation_db`` is the SAME clip's best (exact-window, no lag
    search) reading against its immediate neighbour shot(s)' windows — diagnostic
    AND load-bearing: ``present`` requires ``correlation_db`` to beat it by
    ``mastering.NARRATION_PRESENCE_IDENTITY_MARGIN_DB``, so a clip that merely
    has SOME audio in its own slot no longer passes if that audio matches the
    neighbouring slot just as well (or better)."""

    line_id: str | None = None
    correlation_db: float | None = None
    present: bool | None = None
    neighbor_correlation_db: float | None = None


class MasteringReport(BaseModel):
    """Read-only soft-gate over the delivered mux (VC-2). Reports the picture
    master's on-target state and which stages a strict pipeline would re-run.
    Under the DEFAULT ``mastering_gate_mode='advisory'`` this never blocks a
    render — ``gate_pass=False`` is informational and the artifact still uploads.
    Under the opt-in ``mastering_gate_mode='enforce'`` a cleared ``gate_pass``
    fails the compile instead, so a report with ``gate_pass=False`` can only ever
    be observed in advisory mode.

    ONE EXCEPTION, and it is deliberate: the ``av_coverage`` check (see
    ``av_gap_seconds`` below) fails the compile in BOTH modes. Every other check
    here is a quality judgement with a tunable ceiling; a master whose picture
    ends before its narration is a structurally broken artifact with no advisory
    reading, and it shipped silently for months precisely because this gate could
    do nothing but annotate it.

    LOUDNESS HONESTY: ``post_mux_verify`` does NOT re-measure loudness. Both
    ``integrated_lufs`` and ``true_peak_dbtp`` are REUSED from the mux-stage
    ``final_mux_loudness`` result, not freshly measured here. Only
    ``integrated_lufs`` is gated (a >3 LU deviation check); ``true_peak_dbtp``
    is reported advisory-only and is never inspected or gated."""

    colorspace_ok: bool | None = None
    integrated_lufs: float | None = Field(
        default=None,
        description=(
            "Reused mux-stage integrated loudness (LUFS) — not re-measured in "
            "post_mux_verify. This is the ONLY loudness figure subject to the "
            "gate (>3 LU deviation from target flips gate_pass)."
        ),
    )
    true_peak_dbtp: float | None = Field(
        default=None,
        description=(
            "Reused mux-stage true-peak (dBTP) — not re-measured here. REPORTED "
            "advisory-only: no pass/fail check is applied to it and it never "
            "affects gate_pass."
        ),
    )
    noise_floor_db: float | None = Field(
        default=None,
        description=(
            "Reused mux-stage astats noise floor (dB) — not re-measured here. "
            "The second measurement the speech-presence check needs: no "
            "threshold on integrated_lufs or silence_ratio alone can separate "
            "steady hiss from speech, because hiss never trips silencedetect's "
            "-50 dB gate and normalises to the same loudness as narration."
        ),
    )
    program_to_noise_floor_db: float | None = Field(
        default=None,
        description=(
            "integrated_lufs minus noise_floor_db: how far the delivered program "
            "sits above the master's OWN noise floor. LEVEL-INVARIANT (gain moves "
            "both terms equally), so it measures dynamics, not loudness. Below "
            "speech_presence_min_program_over_floor_db the master reads as a "
            "steady bed of noise with no speech in it. None when either input was "
            "unmeasurable — no verdict, never a fabricated 0.0."
        ),
    )
    speech_presence_min_program_over_floor_db: float | None = Field(
        default=None,
        description=(
            "The minimum program-over-noise-floor margin (dB) this run compared "
            "against, published so a consumer reads the threshold off the wire "
            "instead of hardcoding a second copy of it. None on a degraded "
            "report, where no comparison was made."
        ),
    )
    narration_presence: list[NarrationPresenceClip] = Field(
        default_factory=list,
        description=(
            "Per-shot content-aware presence: for every shot that carried an "
            "audio_url and was actually measured, whether its SOURCE narration "
            "clip's audio was found (cross-correlated) inside the delivered "
            "master at that shot's own timeline window. Supersedes "
            "speech_presence/program_to_noise_floor_db for the actual missing-"
            "clip defect: a whole-file loudness statistic cannot tell 'every "
            "clip present' from 'one clip missing, three present', and a "
            "digitally silent master makes that statistic read as clean. Empty "
            "when shot_timeline/shots were not supplied to post_mux_verify."
        ),
    )
    narration_presence_min_correlation_db: float | None = Field(
        default=None,
        description=(
            "The correlation-RMS-dB floor this run compared each shot's clip "
            "against, published so a consumer reads the threshold off the wire "
            "instead of hardcoding a second copy. None on a degraded report."
        ),
    )
    # Worst-frame banding score for the delivered master. ADVISORY-ONLY since
    # 2026-09-11: it is REPORTED and it still decides which shots get the deband
    # filter, but it no longer affects ``gate_pass``. The score has no usable
    # spatial term (see ``mastering.banding_score``) and fired on 89.2% of 120
    # measured masters, so it could not be calibrated into a verdict at any
    # ceiling. What still gates on the banding pass is its ABSENCE
    # ("banding_measurement" in ``rerun_stages``): an empty signalstats pass
    # leaves every picture-side signal unmeasured and must not read as clean.
    banding_scores: list[float] = Field(default_factory=list)
    # Per-shot banding (S20): one entry per shot when the compile computed shot
    # boundaries (deband active or emit_shot_timeline). Empty when no boundaries
    # were available. Advisory, like the whole-file scores above.
    per_shot: list[ShotBanding] = Field(default_factory=list)
    cfr_ok: bool | None = None
    # Measured cadence of the delivered master (VC-3). ``measured_fps`` is the
    # probed avg_frame_rate. ``duplicate_frame_ratio`` is a best-effort mpdecimate
    # estimate of how much of the film is duplicate (padded/held) frames. BOTH ARE
    # ADVISORY — neither affects ``gate_pass``. The ratio GATED between W1-3 and
    # 2026-09-11 and the gate was removed outright: duplicate frames are intentional
    # dwell (6500ms hold x a 25fps capture) and the ratio measured 0.643-0.998 on
    # 120 of 120 completed masters against a 0.10 ceiling, i.e. it convicted every
    # delivery. The setting behind it is gone too; see ``mastering.py``. Both fields
    # are None when the probe failed.
    measured_fps: float | None = None
    duplicate_frame_ratio: float | None = None
    # DEAD/BLANK-master gate (VC-2): ``luma_spread`` is the delivered master's
    # intra-frame luma spread (YMAX-YMIN) — ~0 for a uniform/blank (dead) frame,
    # ~200 for a legit static UI screen; None when unmeasured. ``gate_pass`` flips
    # false + "dead_frame" appears in rerun_stages below the floor.
    luma_spread: float | None = None
    # The FRACTION of sampled frames whose luma spread fell below the floor — the figure
    # the dead-frame gate actually decides on (`mastering.py:538-541`: gate_pass flips and
    # "dead_frame" joins rerun_stages when this crosses render_dead_frame_min_ratio).
    #
    # It was computed and returned by post_mux_verify but NEVER DECLARED here, so pydantic
    # silently dropped it from every response. A caller could see `rerun_stages:
    # ["dead_frame"]` and had no way to learn whether that was 11% of frames or 100% —
    # i.e. whether to re-render or to look at the source. `luma_spread` alone does not
    # answer it: that is the single worst frame, which is exactly the signal #152 stopped
    # gating on because one intentional fade-to-black failed a whole film.
    dead_frame_ratio: float | None = None
    # THE CUTOFF `dead_frame_ratio` ABOVE WAS COMPARED AGAINST
    # (`render_dead_frame_min_ratio`), published on EVERY report — including the
    # clean path, where a consumer most needs it and where no violation exists to
    # carry it as a `limit`.
    #
    # It exists because the orchestrator duplicated the number as its own
    # `shared/mastering_gate.py` constant: the metric was on the wire and the
    # threshold was not, so retuning it here silently left the orchestrator's
    # blank-picture verdict comparing against a stale 0.10. One threshold, two
    # repos, no link. None only on `degraded_report()`, which holds no Settings
    # and so cannot state a cutoff it never applied.
    dead_frame_min_ratio: float | None = None
    # A/V COVERAGE (the truncated-master gate). The delivered VIDEO stream's own
    # length and the delivered AUDIO stream's own length, probed separately, plus
    # the gap between them (``audio - video``; positive means narration plays over a
    # picture that has already ended). None when unmeasured — and an unmeasured gap
    # never gates, like every other check here.
    #
    # WHY THREE FIELDS AND NOT JUST THE VERDICT: the defect these exist for shipped
    # for months because the report carried ONE reduced duration scalar. That scalar
    # is `min(video_stream, container)` and it AGREES with the plan on a truncated
    # master, so no consumer of it could ever see the gap. Two independently probed
    # numbers is the smallest shape that can.
    #
    # ⚠️ Unlike every other check on this model, ``av_coverage`` is NOT advisory
    # under ``mastering_gate_mode='advisory'``. A gap above
    # ``mastering.AV_COVERAGE_TOLERANCE_SECONDS`` (0.5s — above real AAC tail
    # padding, far below the smallest observed truncation) raises
    # ``MasteringGateError`` in BOTH modes, so the compile fails and the master is
    # never uploaded. A ``gate_pass=False`` carrying ``"av_coverage"`` in
    # ``rerun_stages`` is therefore only ever observable on a stored/replayed report,
    # never on a delivered artifact.
    video_stream_seconds: float | None = None
    audio_stream_seconds: float | None = None
    av_gap_seconds: float | None = None
    # ADVISORY, read-only (``mastering.audio_pts_gaps``): holes in the delivered
    # master's audio packet timeline — how many, and the time they cover. A player
    # renders one as a dropout or skips it. Nothing gates on these and no switch
    # reads them. ``None`` = unmeasured, never "no holes".
    audio_pts_gap_count: int | None = None
    audio_pts_gap_ms_total: float | None = None
    # The pts-interval half of ``cfr_ok`` (``mastering.video_pts_gaps``): how many
    # of the delivered master's video frame intervals are off the nominal one by
    # more than half a frame (a missing frame slot at a stream-copy join is one),
    # and the first few as ``{"at_seconds", "interval_ms"}``. Any count above zero
    # forces ``cfr_ok`` False. ``None`` = unmeasured, never "no gaps" — and then
    # ``cfr_ok`` cannot be True either.
    video_pts_gap_count: int | None = None
    video_pts_gaps: list[dict[str, float]] | None = None
    # ``duplicate_frame_advisory: bool | None`` was DELETED here (W1-3) as a
    # redundant restatement of ``duplicate_frame_ratio > ceiling``. That comparison
    # no longer exists anywhere (2026-09-11): ``duplicate_frame_ratio`` alone is
    # what this report says about duplicate frames, and it says it advisorily.
    gate_pass: bool = True
    rerun_stages: list[str] = Field(default_factory=list)
    # THE STRUCTURED GATE VERDICT — one entry per cleared check, each carrying
    # the metric, the measured value, the ceiling it breached, and its TIER
    # (``"structural"`` = absence/breakage, armed; ``"tunable"`` = a degree
    # against a ceiling, reported while its switch is off).
    #
    # CONTRACT: the ``MasteringViolation`` model above — exactly five keys per
    # entry, no more, with ``tier`` enforced as a two-value Literal by the type
    # system rather than by a key tuple and a test. It was
    # ``list[dict[str, Any]]`` until the ``Any`` ratchet correctly refused it;
    # typing it is the better shape anyway and the wire JSON is identical.
    # ``mastering.violations_payload`` is the single producer (it dumps that
    # model). ``detail`` is deliberately EXCLUDED — see the model's docstring.
    #
    # ⚠️ DECLARED HERE ON PURPOSE. pydantic's default ``extra="ignore"`` silently
    # DROPS any key ``post_mux_verify`` returns that this model does not name —
    # that is exactly how ``dead_frame_ratio`` above was computed, returned, and
    # never seen by a single caller, and how ``violations`` itself was built
    # every single run and reached nobody. A returned key with no field is not a
    # field.
    #
    # WHY IT MATTERS: without it the only machine-readable output of the gate was
    # ``rerun_stages`` — bare stage NAMES, with no numbers, and mixing real gate
    # failures with advisory "this stage might help" appends (``"cfr"``). The
    # orchestrator re-derived the gate's intent by string-matching that list.
    #
    # ⛔ ``enforced_stages: list[str]`` WAS REMOVED HERE and must not come back.
    # It published ``sorted(mastering._ALWAYS_ENFORCED_STAGES)`` — a module
    # constant, byte-identical on every response, which the consumer already
    # mirrored as its own hardcoded fallback; a key whose value can never differ
    # from the reader's default is a cross-repo negotiation that cannot fail, not
    # a contract. Per-violation ``tier`` answers the question it was asked to
    # answer, for the violations a given run actually produced. Settings keeps
    # ``extra="ignore"``-equivalent leniency here via pydantic's default, so a
    # stored older report carrying the key still validates.
    violations: list[MasteringViolation] = Field(default_factory=list)
    # Measured-over-threshold signals that are deliberately NOT gated, same
    # five-key shape as ``violations`` but they never clear ``gate_pass`` and are
    # never armed (see ``mastering.banding_advisories``). Today: one
    # ``banding_residual`` entry whenever any shot's ``needs_deband`` is true.
    advisories: list[MasteringViolation] = Field(default_factory=list)
    # Which normalization mode `loudnorm` ACTUALLY ran in on the final master pass
    # ("linear" / "dynamic"), read back from ffmpeg's own end-of-run summary on
    # stderr. None when no master pass ran, or when the ffmpeg build printed no
    # summary (observation failed — NOT a downgrade).
    #
    # ⚠️ DECLARED HERE FOR THE SAME REASON AS `enforced_stages` ABOVE: pydantic's
    # default `extra="ignore"` would silently drop it and no caller would ever see
    # it. It exists because `linear=true` is a REQUEST — ffmpeg drops it for its
    # dynamic single-pass mode when the measured offset is out of range, without
    # warning and without changing its exit code. The two-pass linear master is the
    # whole point of this chain (the dynamic mode's continuous gain re-derivation
    # over a mostly-speech bus is the audible "digital" gain riding the owner
    # reported), so a chain that asked for linear and got dynamic is an INERT fix
    # that every filter-string assertion in the suite reports as green. A consumer
    # reading `"dynamic"` here on a job whose measurement succeeded knows the master
    # did not get the treatment it was configured for.
    master_normalization_type: str | None = None
    # WHETHER PASS 1 OF THE MASTER LOUDNESS MEASURE FAILED (C10). True only when a
    # measure pass actually ran and bailed; None means NOT MEASURED, never "fine".
    #
    # ⚠️ THE DEGRADATION THIS NAMES IS NOT COSMETIC: when the measure pass fails the
    # master ships UN-NORMALISED while the ducking back-solves from an assumed
    # -18 LUFS, so the mix is wrong in two directions at once. Without this field
    # that outcome is indistinguishable on the wire from a clean master.
    #
    # ⚠️ DECLARED HERE BECAUSE A SIGNAL THE SCHEMA DROPS IS A SIGNAL THAT DOES NOT
    # EXIST. All five bail paths log it and `renderer.get_master_loudnorm_evidence()`
    # already returns `measure_failed`, which the compile route folds in as
    # `master_measure_failed` -- but pydantic's default `extra="ignore"` silently
    # ate the key, so the whole fix was visible only in pod logs. That is the same
    # shape as the structurally-unreachable linear-loudnorm path and the published
    # wheel that lacked its own fix: implemented, merged, green, and INERT.
    #
    # ABSENCE IS THE THIRD STATE, on purpose. `get_master_loudnorm_evidence` emits
    # the key ONLY when a failure was recorded, so a clean render and a
    # mastering-off render both leave this None. Do not give it a `False` default:
    # that would assert a measurement succeeded on every job that never measured.
    master_measure_failed: bool | None = None

    # Whether the master pass ASKED loudnorm for `linear=true`, i.e. whether pass 1
    # produced a usable measurement set. None when no master pass ran.
    #
    # ⚠️ WITHOUT THIS, `master_normalization_type` ABOVE CANNOT BE READ. `"dynamic"`
    # has two completely different meanings: with `master_requested_linear=True` it
    # means "loudnorm REFUSED our linear request and rode the gain anyway" — the
    # inert-fix case that field exists to expose. With False it means "pass 1 failed
    # open and we never asked" — a measurement problem one layer up. The flag was
    # already computed at both call sites and handed to
    # `renderer._record_master_normalization_type`, but only ever as a LOG FIELD, so
    # no run row has ever carried it and the two cases were indistinguishable on the
    # record. The first production master on which the type was observable at all
    # (run `01a0a582-2531-7145-ba7d-a1295e92b232`) reported `"dynamic"` with nothing
    # to say which of the two it was.
    #
    # ⚠️ DECLARED HERE FOR THE SAME REASON AS EVERY FIELD ABOVE: pydantic's default
    # `extra="ignore"` silently DROPS a returned key with no field.
    master_requested_linear: bool | None = None

    # Pass 1's OWN loudnorm readings (its `print_format=json` block), carried so a
    # `master_normalization_type="dynamic"` row can be ATTRIBUTED rather than just
    # observed. af_loudnorm refuses `linear` under two independent conditions that
    # produce byte-identical reports unless these are present:
    #
    #   1. true-peak headroom:  master_input_tp + (master_lufs - master_input_i) > master_tp
    #   2. `master_input_lra == 0.0` (refused even with tens of dB of headroom)
    #
    # Measured against real ffmpeg at `I=-14:TP=-1:linear=true`: `input_i=-22` with
    # `input_tp=-3` -> Dynamic; the same `input_i` with `input_tp=-9` -> Linear. The
    # boundary is exact. See `renderer._loudnorm_filter` for the full derivation and
    # for why the target pair is NOT ours to change.
    #
    # None whenever pass 1 did not complete (every bail there fails open to the
    # single-pass dynamic chain, where `master_requested_linear` is False and there
    # is no refusal to explain), or for an individual reading that was absent or
    # non-finite — loudnorm prints `-inf` for a digitally silent bus and that cannot
    # cross this boundary.
    master_input_i: float | None = None
    master_input_tp: float | None = None
    master_input_lra: float | None = None
    master_target_offset: float | None = None

    # The post-master MISS correction (`renderer._correct_master_loudness_miss`):
    # the constant gain applied when the master landed further than the loudness
    # gate's tolerance from `master_lufs` (dynamic loudnorm on a film that opens on
    # silence ships ~7 LU under), plus the delivered loudness before and after.
    # None ⇒ no correction ran (the master landed, or mastering was off) — never
    # "corrected by 0". Declared for the same `extra="ignore"` reason as above.
    master_loudness_correction_db: float | None = None
    master_loudness_pre_correction_lufs: float | None = None
    master_loudness_post_correction_lufs: float | None = None

    # Which gate INPUTS could not be measured, sorted. Same defect shape as
    # ``dead_frame_ratio`` above: ``post_mux_verify`` has computed and returned this
    # since the gate gained its no-gate-on-a-gap policy, but it was never DECLARED
    # here, so pydantic's default ``extra="ignore"`` dropped it at the response
    # boundary and no caller ever saw it.
    #
    # It exists because ``gate_pass=True`` conflates two different states: "every
    # check ran and none was violated" (verified) and "the checks that RAN found
    # nothing" (partially verified). Concretely: if the mpdecimate pass dies,
    # ``duplicate_frame_ratio`` is None, the duplicate gate deliberately does not
    # fire on a gap, and the job reports ``gate_pass=True, rerun_stages=[]`` — a
    # frozen master could pass with nothing naming the check that never ran. A
    # caller could infer the gap from the ``None`` metric values, so the signal was
    # degraded rather than absent; this makes it explicit and names only the Nones
    # that were actually EXPECTED. Empty is the only state in which
    # ``gate_pass=True`` means "verified".
    #
    # ``colorspace_ok`` and ``cfr_ok`` are in this channel too, and the colorspace
    # one is the ironic case: on an unprobeable file the colorspace gate — the very
    # gate whose two-vocabulary comparison was just fixed — degrades to ``None``,
    # does not fire, and used to leave NOTHING naming it. That is the same fail-open
    # this field exists to close, so omitting the two booleans left the hole open
    # under the patch for it.
    #
    # ``colorspace_ok`` is listed ONLY when a ``render_colorspace`` was configured.
    # "The check was off" and "the check could not run" are different facts about
    # the master: the first says nothing was expected, the second says something was
    # expected and is missing. A disabled gate contributes nothing here, so a
    # ``mastering: off`` job does not report a spuriously unverified master.
    # ``cfr_ok`` has no disabled path (``framerate_coherence`` and the video pts
    # scan run on every verification), so ``None`` there always means one of its
    # two probes failed while the other found nothing wrong.
    unmeasured: list[str] = Field(default_factory=list)


class CompileRequest(BaseModel):
    shots: list[Shot] = Field(min_length=1, max_length=_MAX_SHOTS)
    tags: dict[str, str] | None = None
    cadence: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Per-job master-cadence override (VC-3). When set, the whole compile "
            "renders at this rate (snapped to the nearest legal cadence "
            "{24,25,30,50,60}), bypassing the auto retime-cost selector. Omitted "
            "→ auto selection from the shots' declared source rates / render_fps."
        ),
    )
    music: MusicBed | None = Field(
        default=None,
        description="Continuous music bed under the whole video, ducked under narration",
    )
    ambient: list[AmbientCue] = Field(
        default_factory=list, description="Continuous ambient bed segments on the global timeline"
    )
    master_lufs: float | None = Field(
        default=None,
        description=(
            "Per-job override for the loudness-master integrated-loudness target "
            "(LUFS). Folded into ``tags['master_lufs']`` and clamped to [-30, -5] "
            "via the same path as the tag; out-of-range values are ignored."
        ),
    )
    mastering: MasteringProfile | None = Field(
        default=None,
        description=(
            "Per-job encode/delivery master overrides (colorspace, deband, "
            "bitrate mode, CRF, loudness target, platform). Folded into the "
            "tag-resolution clamp path; out-of-range values degrade to defaults."
        ),
    )
    finish: FinishProfile | None = Field(
        default=None,
        description=(
            "Per-job finishing overrides (VC-5): the brand letterbox/pillarbox "
            "colour, applied on the per-shot filter chain. None (default) is a "
            "no-op — byte-identical to before this field existed."
        ),
    )
    captions: list[ShotCaptions] | None = Field(
        default=None,
        description=(
            "Word/phrase-timed burned-in captions (VC-6), one entry per shot "
            "carrying cues. Cues are mapped onto the film bus and rendered as a "
            "single libass ASS subtitle pass folded into the SAME whole-film "
            "grade re-encode. None (default) — or no emitted cues — is a no-op, "
            "byte-identical to before this field existed."
        ),
    )
    captions_render: Literal["burned_in", "overlay"] = Field(
        default="overlay",
        description=(
            "How the word-timed captions in `captions` are rendered. 'overlay' "
            "(default) muxes them as a selectable soft mov_text subtitle track "
            "(players toggle on/off) — no burn, no video re-encode. 'burned_in' "
            "bakes them into the picture in the whole-film re-encode and must be "
            "requested explicitly: subtitles are never burned in by default. "
            "Ignored when `captions` is None."
        ),
    )
    caption_mood: str | None = Field(
        default=None,
        description=(
            "Run-level mood token (e.g. 'energetic', 'calm', 'confident') that "
            "selects a per-video caption look (colour/outline/size/weight), so the "
            "caption styling tracks the same mood that drives the colour grade. "
            "None (default) or an unrecognised value ⇒ the default `Cap` style, "
            "byte-identical to before this field existed."
        ),
    )
    captions_karaoke: bool = Field(
        default=False,
        description=(
            "When True AND `captions_render` is 'burned_in', render each cue's "
            "per-word `words` as ASS {\\k} karaoke highlight; a cue with no words "
            "highlights as one phrase. No effect in 'overlay' mode (soft subtitle "
            "tracks cannot carry {\\k}) or when `captions` is None. False (default) "
            "⇒ plain phrase captions, byte-identical."
        ),
    )
    caption_font_family: Literal["DejaVu Sans", "Inter"] | None = Field(
        default=None,
        description=(
            "Font family for burned-in captions, from the fonts this service "
            "embeds: 'Inter' (Inter 4.1, OFL — SemiBold 600, Bold 700 when the "
            "caption mood is bold; matches the film cards) or 'DejaVu Sans' (the "
            "default). None (default) ⇒ the default font, byte-identical. "
            "Ignored in 'overlay' mode."
        ),
    )
    caption_text_colour: str | None = Field(
        default=None,
        pattern=_HEX_RGB_PATTERN,
        description=(
            "Burned-in caption text colour as '#RRGGBB' (e.g. '#F8FAFC'). "
            "Overrides the mood's text colour. None (default) ⇒ unchanged. Ignored "
            "in 'overlay' mode."
        ),
    )
    caption_box_colour: str | None = Field(
        default=None,
        pattern=_HEX_RGB_PATTERN,
        description=(
            "Burned-in caption box colour as '#RRGGBB' (e.g. '#0B1F3A'). When this "
            "or `caption_box_opacity` is set, each cue is drawn on ONE continuous "
            "box (libass BorderStyle=4) instead of the default per-line plate; the "
            "unset one falls back to black / the default plate opacity. None "
            "(default) ⇒ unchanged. Ignored in 'overlay' mode."
        ),
    )
    caption_box_opacity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Opacity of the caption box, 0 (clear) .. 1 (solid). See "
            "`caption_box_colour`. None (default) ⇒ unchanged."
        ),
    )
    emit_shot_timeline: bool = Field(
        default=False,
        description=(
            "When true, the response carries a per-shot `shot_timeline` "
            "(authoritative `[start,end)` per beat on the final mux). Off by "
            "default so the plain concat fast path stays probe-free."
        ),
    )


class ShotTimelineEntry(BaseModel):
    """One shot's absolute [start, end) span on the final rendered mux, keyed by the
    orchestrator's line_id. With crossfade transitions consecutive spans OVERLAP by the
    crossfade duration (each entry is that shot's full span on the bus)."""

    line_id: str | None = None
    start: float
    end: float


class CompileResponse(BaseModel):
    artifact_url: str
    duration_seconds: float
    object_key: str
    bucket: str
    # Read-only loudness/silence measurement of the delivered mux (integrated
    # LUFS, true peak dBTP, LRA, silence_ratio). Best-effort observability —
    # None when the measure pass failed or was skipped; never blocks a render.
    loudness: dict | None = None
    # Encode/delivery master soft-gate report (VC-2, contract C11). None for
    # audio concat or when verification could not run.
    mastering: MasteringReport | None = None
    # Auditable report of the VO-relative music-bed mix (T1). None when the
    # compile carried no music/ambient bed, or when no offset field was set
    # (the literal-gain path never measures). See ``overlay_audio_bed``.
    music_bed: dict | None = None
    # Resolved whole-film grade knobs + whether a grade re-encode engaged (VC-5).
    # None when the request carried no ``finish.grade``. Reports the REQUESTED
    # knobs, never a fabricated "measured grade".
    grade: dict | None = None
    # Authoritative per-shot ``[start, end)`` on the final rendered mux, keyed by
    # line_id (opt-in via request ``emit_shot_timeline``). None when the flag was
    # off — the probe-free fast path never builds it.
    shot_timeline: list[ShotTimelineEntry] | None = None


class AudioClip(BaseModel):
    """One narration clip to be stitched into the running audio file.

    ``audio_url`` is whatever the orchestrator's ``store_tts_clip`` returned
    (presigned http(s) MinIO URL or ``file://`` for the local backend) — the
    same reference shape the video ``Shot.audio_url`` accepts. ``seconds`` is
    advisory only; the concatenated output's duration is always re-probed.

    ``seconds`` IS A BILLING INPUT, NOT A RENDERING ONE, and that is the whole
    reason it is normalised rather than validated. ``renderer.concat_audio`` reads
    only ``audio_url`` — there is not one read of ``.seconds`` anywhere in
    ``renderer.py`` or ``mastering.py``. Its single reader is
    ``compile._declared_clip_seconds``, whose all-or-nothing sum becomes
    ``declared_duration_seconds`` → ``_billed_duration`` → the cost event, i.e. the
    nominal charge used when the delivered master could not be probed.
    """

    id: str = Field(pattern=_SAFE_ID_PATTERN, max_length=128)  # scratch-path + filtergraph safe
    audio_url: str
    seconds: float | None = Field(
        default=None,
        description=(
            "Advisory per-clip duration; output duration is probed. Non-positive "
            "values are normalised to null (unmeasured), never rejected."
        ),
    )

    @field_validator("seconds")
    @classmethod
    def _non_positive_is_unmeasured(cls, value: float | None) -> float | None:
        """Normalise a non-positive advisory duration to ``None`` at PARSE time.

        This field used to carry ``gt=0``. An ADVISORY field that HARD-REJECTS is a
        contradiction, and an expensive one: a ``0.0`` raised a 422 that failed the
        terminal stitch AFTER every clip's audio had already been generated and
        paid for. (Found by a peer session.)

        WHY NOT ``ge=0``, and this is not reconsiderable: it ACCEPTS ``0.0`` — the
        exact fabricated value this chain exists to stop producing — and then feeds
        it straight into the billing reader.

        WHY NOT A READER-SIDE SKIP: the 422 being removed was ACCIDENTALLY HOLDING A
        MONEY INVARIANT. Skipping non-positive clips in ``_declared_clip_seconds``
        would turn a loud failure into a silent UNDER-BILL — four 100s clips with
        one zeroed would sum to 300.0 and bill a nominal 300 for 400 seconds of
        delivered audio, with nothing anywhere saying a number had been dropped. An
        over-bill is permanent and an under-bill is invisible; neither is
        acceptable, and a reader-side skip also leaves the model still accepting a
        value that means nothing.

        Normalising to ``None`` here keeps ``_declared_clip_seconds``'
        all-or-nothing design intact: one unmeasured clip makes the WHOLE request
        unmeasured, so it is not billed at all rather than billed short. That is
        the same "report the absence AS an absence, never as 0.0" rule
        ``_billed_duration``'s ``_DURATION_UNMEASURED`` branch already follows.

        ``value <= 0`` and NOT ``not value``: ``-1.0`` is TRUTHY and survives a
        naive falsy check, and it is the value a caller's own subtraction bug
        produces. Non-finite is normalised on the SAME line rather than left to the
        comparison: ``nan`` compares False against everything (so ``<= 0`` misses
        it and it would poison the sum into a ``nan`` charge) and ``inf`` is
        strictly positive (so ``<= 0`` misses it and it would bill an unbounded
        duration — and is not JSON-safe on the way out either, the same reason
        ``parse_loudness`` scrubs ``-Infinity``). None of the three is a
        measurement, so all three become the absence.
        """
        if value is None:
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value


class AcxMaster(BaseModel):
    """Opt-in ACX program-loudness master for the combined-book concat (A12 seam).

    Presence (a non-None ``acx_master`` on the request) turns on the compile-tail
    two-pass loudnorm master; absence leaves the stitch byte-identical to today.
    Each override defaults to the service ``render_audiobook_*`` setting when None.
    """

    enabled: bool = True
    lufs_target: float | None = Field(
        default=None, ge=-30.0, le=-5.0, description="loudnorm I target; default -20 (ACX mid)"
    )
    tp: float | None = Field(
        default=None, ge=-9.0, le=-1.0, description="loudnorm true-peak ceiling dBTP; default -3"
    )
    lra: float | None = Field(
        default=None, ge=1.0, le=20.0, description="loudnorm LRA; default 11 for spoken word"
    )


class Chapter(BaseModel):
    """One m4b chapter marker at a cumulative ``start_ms`` offset.

    END is derived (the next chapter's start, or the file duration for the last),
    so the caller only supplies ordered ``start_ms`` + a title."""

    title: str = Field(min_length=1, max_length=512)
    start_ms: int = Field(ge=0)


class AcxReport(BaseModel):
    """Read-only ACX-compliance verdict re-measured on the delivered master.

    ``passed`` (serialised as ``pass``) is True iff integrated loudness is within
    the ACX window, true peak ≤ ceiling, and noise floor ≤ ceiling. A metric that
    could not be measured is None and fails ``passed`` — an unmeasurable master is
    not a compliant master."""

    model_config = ConfigDict(populate_by_name=True)

    integrated_lufs: float | None = None
    true_peak_dbtp: float | None = None
    noise_floor_db: float | None = None
    passed: bool = Field(serialization_alias="pass", validation_alias="pass")
    # Same field, same meaning and same None-is-not-fine rule as on
    # ``MasteringReport`` above (C10) -- see the long note there.
    #
    # IT MATTERS MORE HERE, not less. Every number on this model is a LOUDNESS
    # measurement, and ``passed`` is decided against the ACX window. A master that
    # shipped un-normalised because pass 1 failed is exactly the master whose ACX
    # verdict cannot be taken at face value, so the compliance report is the last
    # place the degradation should be invisible.
    master_measure_failed: bool | None = None


class AudioConcatRequest(BaseModel):
    """Stitch an ordered list of audio clips into a single WAV/MP3/m4b.

    Used by the orchestrator's ``audiobook_stitch`` stage twice: once per
    chapter (chunk clips, ``gap_ms=0``) and once for the combined book
    (per-chapter files, optionally a small inter-chapter ``gap_ms``).

    ``acx_master`` (opt-in) applies the compile-tail ACX loudness master to the
    combined book; ``output_format="m4b"`` + ``chapters`` packages a chaptered
    audiobook container. Both absent ⇒ byte-identical to the legacy WAV/MP3 concat.
    """

    clips: list[AudioClip] = Field(min_length=1, max_length=_MAX_CLIPS)
    output_format: Literal["wav", "mp3", "m4b"] = "wav"
    gap_ms: int = Field(
        default=0, ge=0, le=5000, description="Silence inserted between clips (filter path)"
    )
    acx_master: AcxMaster | None = None
    chapters: list[Chapter] | None = Field(
        default=None,
        max_length=_MAX_CLIPS,
        description="m4b chapter markers (ordered by start_ms); only honoured for output_format=m4b",
    )
    tags: dict[str, str] | None = None


class AudioConcatResponse(BaseModel):
    artifact_url: str
    duration_seconds: float
    object_key: str
    bucket: str
    # ACX-compliance report; None unless the request carried ``acx_master``.
    acx_report: AcxReport | None = None


class JobAccepted(BaseModel):
    """202 response from a submit endpoint — the render runs in the background.

    ``poll_url`` is the path the caller GETs (``/api/jobs/{job_id}``) until the
    job reaches a terminal ``status``. Mirrors the fleet's submit_job/poll_job
    convention (``w2v_common`` BaseServiceClient).
    """

    job_id: str
    status: Literal["processing"] = "processing"
    poll_url: str


class JobCancelRequest(BaseModel):
    """Cancel every LIVE job of the calling tenant whose Idempotency-Key starts with
    ``idempotency_key_prefix``.

    Prefix, not exact key: the orchestrator submits one compile per run PER STAGE under
    ``{run_id}:{stage}`` (plus an optional re-render discriminator), so a run-scoped
    cancel — the only kind a user can press — cannot enumerate the keys it must reap.
    Passing ``"{run_id}:"`` reaps that run's renders and nothing else. The prefix is
    matched literally (``%``/``_`` are escaped), so it can never widen into a wildcard.
    """

    idempotency_key_prefix: str = Field(min_length=1, max_length=255)


class JobCancelResponse(BaseModel):
    """What the cancel actually did — deliberately a COUNT of real effects, not an ack.

    ``cancelled`` is the job_ids this call flipped terminal (already-terminal and
    other-tenant jobs are not in it, so an empty list means nothing was live). ``killed``
    is how many in-flight renders this replica interrupted immediately; a job owned by a
    SIBLING replica is cancelled durably here and its ffmpeg dies on that pod's next lease
    beat, so ``killed < len(cancelled)`` is normal, not a failure.
    """

    cancelled: list[str]
    killed: int


class JobStatus(BaseModel):
    """Poll response. ``status`` drives the caller's poll loop:

    * ``processing`` — still rendering; keep polling.
    * ``completed`` — terminal success; the result fields are populated.
    * ``failed`` — terminal failure; ``error`` carries the reason.
    * ``cancelled`` — terminal; a caller cancelled the run that owns this render
      (``POST /api/jobs/cancel``). Distinct from ``failed`` on purpose: nothing broke, so
      it must not read as a render fault in logs, dashboards or the caller's error path.

    The result fields mirror Compile/AudioConcatResponse (both render kinds
    return the same shape), so one poll endpoint serves compile + concat-audio.
    """

    job_id: str
    status: Literal["processing", "completed", "failed", "cancelled"]
    artifact_url: str | None = None
    duration_seconds: float | None = None
    # The ACTUAL ffprobe'd duration of the encoded output (video compiles only;
    # None for audio concat). ``duration_seconds`` is the DECLARED/nominal sum of
    # shot durations; a material gap between the two signals a truncated or
    # short-rendered file, which the orchestrator surfaces as a degraded run.
    actual_duration_seconds: float | None = None
    # SHA-256 hex digest of the delivered mux (video compiles only; None for audio
    # concat). A stable content fingerprint the orchestrator records in the render
    # manifest's audit ledger.
    output_sha256: str | None = None
    object_key: str | None = None
    bucket: str | None = None
    # Actual encoded output dimensions (video compiles only; None for audio
    # concat). Lets the orchestrator surface the real 2K/4K resolution to the
    # e2e post-promote dimension check.
    width: int | None = None
    height: int | None = None
    # Read-only loudness/silence measurement of the delivered mux (video compiles
    # only; None for audio concat or when the best-effort measure pass failed):
    # {integrated_lufs, true_peak_dbtp, lra, silence_ratio}.
    loudness: dict | None = None
    # ACX-compliance report re-measured on the delivered audiobook master (concat
    # -audio path only; None unless the request carried ``acx_master``).
    acx_report: AcxReport | None = None
    # Encode/delivery master soft-gate report (VC-2, contract C11; video compiles
    # only). None for audio concat or when verification could not run.
    mastering: MasteringReport | None = None
    # Auditable report of the VO-relative music-bed mix (T1; video compiles
    # only). None when the compile carried no music/ambient bed, or when no
    # offset field was set (the literal-gain path never measures).
    music_bed: dict | None = None
    # Resolved whole-film grade knobs + whether a grade re-encode engaged (VC-5;
    # video compiles only). None when the request carried no ``finish.grade``.
    grade: dict | None = None
    # Authoritative per-shot ``[start, end)`` on the final rendered mux, keyed by
    # line_id (video compiles only; opt-in via request ``emit_shot_timeline``).
    # None when the flag was off — the probe-free fast path never builds it.
    shot_timeline: list[ShotTimelineEntry] | None = None
    error: str | None = None


class SafeRect(BaseModel):
    """A centered safe-area rectangle in FRAME PIXELS (top-left origin).

    Returned as overlay geometry so a caller can lay title/logo/CTA elements
    inside the broadcast-style safe margins without re-deriving them. The guides
    are metadata ONLY — they are never burned into the extracted PNG."""

    x: int = Field(ge=0, description="Left edge of the rect, pixels from frame left")
    y: int = Field(ge=0, description="Top edge of the rect, pixels from frame top")
    width: int = Field(ge=0, description="Rect width in pixels")
    height: int = Field(ge=0, description="Rect height in pixels")


class ThumbnailCandidate(BaseModel):
    """One extracted poster/thumbnail frame + its safe-area geometry.

    ``url`` is a presigned GET for the uploaded PNG. When the request carried a
    ``grade`` the frame was run through the IDENTICAL ``renderer._grade_vf`` chain
    the whole-film grade uses, so a poster cut from an ungraded hero source
    matches the delivered film's look (spec 07-color-finish §E)."""

    url: str
    timestamp_seconds: float = Field(description="Source timestamp the frame was sampled at")
    width: int
    height: int
    action_safe: SafeRect = Field(description="90% centered action-safe rect")
    title_safe: SafeRect = Field(description="80% centered title-safe rect")


class ThumbnailRequest(BaseModel):
    """Extract N grade-parity poster/thumbnail candidates from a video.

    Synchronous (unlike compile): frame extraction is sub-second, so the route
    renders inline and returns the candidates directly. ``grade`` is supplied when
    the ``video_url`` is an UNGRADED hero source that must be graded to match the
    film; omit it (None) when the input is already the graded delivery master and
    frames should be taken as-is."""

    video_url: str = Field(description="Source video (presigned MinIO http(s), file://, or path)")
    count: int = Field(
        default=3, ge=1, le=12, description="Number of evenly-spaced candidates to extract"
    )
    format: str | None = Field(
        default=None,
        description=(
            "Deliverable aspect (16x9/9x16/1x1/4x5) — retargets the poster canvas "
            "to the same dims the film ships on. None keeps the source dimensions."
        ),
    )
    grade: GradeSpec | None = Field(
        default=None,
        description=(
            "Whole-film grade to apply to each frame so an ungraded hero source "
            "matches the film look. None (or a disabled/identity grade) extracts "
            "the frame unchanged — the already-graded-input case."
        ),
    )
    tags: dict[str, str] | None = Field(
        default=None, description="Free-form tags; `run_id` namespaces the uploaded PNGs"
    )


class ThumbnailResponse(BaseModel):
    candidates: list[ThumbnailCandidate]
