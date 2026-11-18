"""A/V quality wave 2: the narration bus, the zoompan scaler, and the gate tier.

Wave 1 (#215) fixed only WHERE the limiter sat. What the owner still heard as
"digital" is upstream of that:

* Narration reaches this bus ALREADY normalised to -18 LUFS per clip by
  ``tts_backend``. The bus then compressed it 3:1, multiplied it by a LINEAR
  makeup of 2 (= +6 dB of blind boost), and handed the result to a single-pass
  ``loudnorm`` with no ``measured_*`` — i.e. ffmpeg's DYNAMIC mode, which
  re-derives gain continuously over a mostly-speech signal full of pauses. That
  is gain riding, and it is audible as pumping/breathing.
* ``zoompan`` resamples internally on every output frame and exposes no
  ``flags`` option, so swscale's BILINEAR default applied to every ken-burns
  drift and every focus punch — the ``_SWS_FLAGS`` lanczos constant only ever
  reached explicit ``scale=`` nodes and could not see inside zoompan.
* ``_ALWAYS_ENFORCED_STAGES`` was a tier the report never published, so the
  orchestrator mirrored it by name and pydantic would have dropped the key even
  if it had been returned (the ``dead_frame_ratio`` defect, exactly). ⚠️ WAVE-13
  replaced it with a per-violation ``tier`` on a published ``violations`` list;
  the tests below were rewritten accordingly, and the "stays advisory" pin was
  INVERTED by owner decision.
"""

from __future__ import annotations

from src.api.schemas import FocusKey, MasteringReport, Shot
from src.config import Settings, get_settings
from src.services import mastering, renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# Two-pass / linear loudnorm
# ---------------------------------------------------------------------------

_MEASURED = {
    "input_i": "-21.4",
    "input_tp": "-3.2",
    "input_lra": "9.1",
    "input_thresh": "-31.6",
}


def test_measured_values_put_loudnorm_in_linear_mode() -> None:
    """The whole fix: with pass-1 numbers, loudnorm applies ONE constant gain.

    Without ``linear=true`` ffmpeg ignores nothing but still runs its dynamic
    normaliser over the programme; the ``measured_*`` values alone are not
    enough, and neither is ``linear=true`` alone.
    """
    chain = renderer._master_filter(_settings(), _MEASURED)
    loudnorm = chain[chain.index("loudnorm=") : chain.index("aformat=")]

    assert "linear=true" in loudnorm, f"loudnorm must run linear, got: {loudnorm}"
    assert "measured_I=-21.4" in loudnorm, loudnorm
    assert "measured_TP=-3.2" in loudnorm, loudnorm
    assert "measured_LRA=9.1" in loudnorm, loudnorm
    assert "measured_thresh=-31.6" in loudnorm, loudnorm


def test_without_measurements_the_chain_falls_open_to_the_dynamic_pass() -> None:
    """Fail-open is deliberate: a measurement hiccup must not fail a paid compile.

    So ``None`` must reproduce today's single-pass chain — NOT emit a half-built
    ``linear=true`` with no ``measured_*``, which ffmpeg would reject outright.
    """
    chain = renderer._master_filter(_settings(), None)

    assert "loudnorm=" in chain
    assert "linear=true" not in chain, f"no measurement => no linear mode, got: {chain}"
    assert "measured_" not in chain, chain


def test_the_measure_pass_includes_the_compressor_it_will_run_behind() -> None:
    """Pass 1 must measure what pass 2's loudnorm actually receives.

    The compressor sits upstream of loudnorm in the delivery chain. Measuring
    without it would hand pass 2 the loudness of a signal that never reaches it,
    and the constant gain would be wrong by the compressor's own gain change.
    """
    measure = renderer._master_measure_filter(_settings())

    assert measure.index("acompressor=") < measure.index("loudnorm="), measure
    assert "print_format=json" in measure, measure
    # It must NOT normalise for real — pass 1 is discarded to `-f null -`, and a
    # `measured_*`/`linear` here would make it apply a gain it has not measured yet.
    assert "linear=true" not in measure, measure


# ---------------------------------------------------------------------------
# Softened bus dynamics
# ---------------------------------------------------------------------------


def test_compressor_defaults_are_softened_and_configurable() -> None:
    """ratio 3 -> 2 and makeup 2 -> 1 (unity).

    ``makeup`` is a LINEAR multiplier, so the old 2 was +6 dB of blind boost on a
    bus that already arrives levelled — gain loudnorm then had to ride back off.
    """
    settings = _settings()
    assert settings.master_compressor_ratio == 2.0
    assert settings.master_compressor_makeup == 1.0

    chain = renderer._master_filter(settings)
    comp = chain[chain.index("acompressor=") : chain.index("loudnorm=")]
    assert ":ratio=2:" in comp, f"ratio must be the softened 2, got: {comp}"
    assert ":makeup=1" in comp, f"makeup must be unity, got: {comp}"
    assert "threshold=-24dB" in comp, comp


def test_compressor_and_limiter_track_their_settings_fields() -> None:
    """Promoted to Settings means the SETTING decides, not a literal in a string."""
    chain = renderer._master_filter(
        _settings(
            master_compressor_threshold_db=-18.0,
            master_compressor_ratio=4.0,
            master_compressor_makeup=1.5,
            master_limiter_attack_ms=7.0,
            master_limiter_release_ms=120.0,
        )
    )

    assert "threshold=-18dB" in chain, chain
    assert ":ratio=4:" in chain, chain
    assert ":makeup=1.5" in chain, chain
    limiter = chain[chain.index("alimiter=") :]
    assert ":attack=7:" in limiter, limiter
    assert ":release=120:" in limiter, limiter


def test_master_loudness_target_is_the_streaming_reference() -> None:
    """-14 LUFS: the value YouTube/Spotify normalise TO, so anything hotter is
    turned back down on playback and only the extra compression survives."""
    settings = _settings()
    assert settings.master_lufs == -14.0
    assert settings.master_lra > 7.0, "the old 7 LU squeezed below speech's natural range"
    assert f"I={settings.master_lufs}" in renderer._master_filter(settings)


def test_render_crf_is_the_near_transparent_cell_for_2k() -> None:
    assert _settings().render_crf == 18


# ---------------------------------------------------------------------------
# zoompan is an unflagged scaler
# ---------------------------------------------------------------------------


def _focus_shot() -> Shot:
    return Shot(
        id="s1",
        seconds=4.0,
        focus_track=[FocusKey(t=0.0, bbox=(100.0, 50.0, 200.0, 120.0))],
    )


def test_focus_zoompan_is_fed_a_lanczos_prescaled_frame() -> None:
    """zoompan has no `flags` option, so it upscales BILINEAR unless the frame it
    receives is already bigger than its output."""
    settings = _settings()
    chain = renderer._focus_zoom_filter("[0:v]scale=1:1", _focus_shot(), settings, 4.0)

    assert "zoompan=" in chain, chain
    prescale = chain[: chain.index("zoompan=")]
    expected = (
        f"scale={settings.render_width * renderer._ZOOMPAN_PRESCALE}"
        f":{settings.render_height * renderer._ZOOMPAN_PRESCALE}:{renderer._SWS_FLAGS}"
    )
    assert expected in prescale, f"missing lanczos prescale before zoompan, got: {prescale}"
    assert "flags=lanczos" in prescale, prescale


def test_focus_punch_centre_is_scaled_into_the_prescaled_space() -> None:
    """The coordinate-space half of the fix, and the one that breaks silently.

    The bbox centre is in RENDER-CANVAS pixels. Prescaling doubles ``iw``/``ih``,
    so an unscaled centre punches at half the intended position — a wrong-framed
    shot that still renders cleanly and so never trips anything else.
    """
    chain = renderer._focus_zoom_filter("[0:v]scale=1:1", _focus_shot(), _settings(), 4.0)

    # bbox (100, 50, 200, 120) -> centre (200, 110) -> x2 prescale -> (400, 220).
    assert "400-iw/zoom/2" in chain, f"punch x must be prescale-scaled, got: {chain}"
    assert "220-ih/zoom/2" in chain, f"punch y must be prescale-scaled, got: {chain}"


def test_no_zoompan_in_this_module_rides_the_bilinear_default() -> None:
    """The GENERAL guard — a zoompan added later without a prescale is the same bug.

    Mirrors ``test_no_scale_in_this_module_rides_the_bicubic_default``: compare
    every emitted ``zoompan=`` against the prescale helper in the same window,
    so the next one is caught automatically instead of shipping soft.
    """
    import inspect

    lines = inspect.getsource(renderer).splitlines()
    bare = []
    for i, line in enumerate(lines):
        if 'f"zoompan=' not in line:
            continue
        if "_zoompan_prescale" not in "".join(lines[max(0, i - 3) : i + 1]):
            bare.append(line.strip())
    assert not bare, f"zoompan= without a lanczos prescale: {bare}"


# ---------------------------------------------------------------------------
# violations (WAVE-13) — SUPERSEDES enforced_stages
# ---------------------------------------------------------------------------


def test_violations_survives_the_response_model() -> None:
    """A returned key with no declared field is not a field — pydantic's default
    ``extra="ignore"`` drops it, which is how ``dead_frame_ratio`` was lost, and
    how ``violations`` itself was built every run and reached nobody."""
    entry = {
        "stage": "av_coverage",
        "metric": "av_gap_seconds",
        "value": 9.4,
        "limit": 0.5,
        "tier": "structural",
    }
    report = MasteringReport.model_validate(
        {"gate_pass": False, "rerun_stages": ["av_coverage"], "violations": [entry]}
    )

    # Compared as a ROUND TRIP, not by identity: `violations` is typed as
    # `MasteringViolation` (it was `dict[str, Any]` until the `Any` ratchet
    # refused it), so `model_validate` parses this dict INTO a model instance and
    # an instance is never `==` a dict. Dumping back is the stronger assertion
    # anyway — it proves the wire shape survives the model in BOTH directions,
    # which is the property this test exists to pin.
    assert [v.model_dump(mode="json") for v in report.violations] == [entry]
    assert report.model_dump()["violations"] == [entry]


def test_enforced_stages_is_gone_and_does_not_come_back() -> None:
    """``enforced_stages`` published a module constant identical on every
    response, which the consumer already mirrored as its own hardcoded fallback —
    a cross-repo negotiation whose two sides can never differ. Per-violation
    ``tier`` replaced it. Re-adding the field would re-introduce the drift."""
    assert "enforced_stages" not in MasteringReport.model_fields
    assert "enforced_stages" not in mastering.degraded_report()
    assert not hasattr(mastering, "_ALWAYS_ENFORCED_STAGES")


# ---------------------------------------------------------------------------
# master_normalization_type
# ---------------------------------------------------------------------------


def test_master_normalization_type_survives_the_response_model() -> None:
    """Same `extra="ignore"` trap as `enforced_stages` above: the renderer can
    record that ffmpeg silently ran `dynamic` instead of the `linear=true` it was
    asked for, and if the field is not DECLARED here pydantic drops it at the
    response boundary and no consumer ever learns the master went untreated."""
    report = MasteringReport.model_validate(
        {"gate_pass": True, "rerun_stages": [], "master_normalization_type": "dynamic"}
    )

    assert report.master_normalization_type == "dynamic"
    assert report.model_dump()["master_normalization_type"] == "dynamic"


def test_master_normalization_type_defaults_to_none() -> None:
    """A job with no master pass (mastering off, no audio) keeps today's report
    shape — the key is absent-as-None, not a fabricated "linear"."""
    assert MasteringReport.model_validate({"gate_pass": True}).master_normalization_type is None
    assert (
        MasteringReport.model_validate(mastering.degraded_report()).master_normalization_type
        is None
    )


def test_the_compiler_is_no_longer_advisory_but_tunables_still_are() -> None:
    """⚠️ THIS TEST WAS INVERTED IN WAVE-13, BY OWNER DECISION. It used to pin
    "the compiler stays advisory"; the owner has decided to ARM the gates, and
    runs may now fail. What is still pinned is the SHAPE of that decision: the
    whole-gate ``enforce`` mode stays OFF (arming is per gate, so a bad threshold
    is backed out one switch at a time), every STRUCTURAL gate is armed, and
    every TUNABLE gate is not — because a tunable ceiling with no measured fleet
    distribution is how this project built a signal that would have fired on
    96.7% of masters."""
    settings = _settings()
    assert not mastering.gate_enforced(settings)

    armed = {
        stage for stage in mastering._GATE_KILL_SWITCHES if mastering.gate_armed(settings, stage)
    }
    tunable = {s for s, t in mastering._GATE_TIERS.items() if t == "tunable"}

    # VC-W14 armed two more STRUCTURAL gates, each on a ceiling that is a
    # property rather than a percentile: ``true_peak`` at DIGITAL FULL SCALE
    # (V3 — the comparison ``true_peak_dbtp`` never had), and ``measure``
    # (V4 — the two-pass loudness master could not be built, self-reported by
    # the render so no flaky probe can manufacture it). ``bed_calibration``
    # (V6) is structural and deliberately left DISARMED: its trigger convicts
    # the loudness PROBE at an unmeasured rate.
    assert armed == {"av_coverage", "dead_frame", "verify", "true_peak"}
    assert not (armed & tunable), "a tunable gate must never be armed by default"
    # colorspace is structural but deliberately NOT armed: measured FALSE on
    # 23.3% of n=120 masters, an upstream tagging bug this service cannot fix.
    assert mastering.gate_tier("colorspace") == "structural"
    assert not mastering.gate_armed(settings, "colorspace")
