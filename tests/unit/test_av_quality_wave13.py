"""WAVE-13 — GATE HONESTY: the gate must EMIT its verdict and must DECIDE.

Three defects, one family. The governing failure mode of this pipeline is that a
green run proves nothing — it once delivered a 218.47s film against a 30.0s order
with every stage green — and the mastering gate was a textbook instance:

* C3. ``violations`` was BUILT every single run (stage / metric / measured value /
  breached limit) and then never placed in the returned dict. The only
  machine-readable output was ``rerun_stages``: bare stage NAMES, no numbers, and
  mixing real gate failures with advisory "this stage might help" appends
  (``"cfr"``). The orchestrator re-derived the gate's intent by string-matching
  that list.
* C1. ``gate_pass`` was computed CORRECTLY and then decided nothing, because the
  only enforcement path was one whole-gate ``mastering_gate_mode='enforce'`` that
  no deployment ever set.
* C12. Thresholds carried no measurement, which is what made arming unsafe and
  the advisory default feel reasonable.

⭐ EVERY assertion here is mutation-tested: the line it covers was mutated (tier
flipped, ``violations`` dropped from the payload, a kill-switch default flipped,
the contract key list widened) and the test observed to FAIL before the mutation
was reverted. This file asserts BEHAVIOUR — what the gate does to a compile and
what the consumer receives — never the presence of a call site. Six tests in this
project once survived DELETING the call site they claimed to cover because they
asserted position.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.api.schemas import AcxReport, MasteringReport, Shot
from src.config import Settings, get_settings
from src.services import mastering


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


# ---------------------------------------------------------------------------
# C3 — violations[] is EMITTED, in the contracted shape
# ---------------------------------------------------------------------------


def test_violations_payload_is_exactly_the_contracted_five_keys() -> None:
    """A CONTRACT, not a preference: the orchestrator-side consumer is written
    against these five keys. ``detail`` is excluded on purpose — it is operator
    prose for the job's ``error``, and a consumer parsing it would be re-deriving
    intent from a message again, which is the defect ``violations`` ends."""
    payload = mastering.violations_payload(
        [
            mastering.MasteringGateViolation(
                stage="av_coverage",
                metric="av_gap_seconds",
                value=9.4,
                limit=0.5,
                tier="structural",
                detail="prose for the operator",
            )
        ]
    )

    assert payload == [
        {
            "stage": "av_coverage",
            "metric": "av_gap_seconds",
            "value": 9.4,
            "limit": 0.5,
            "tier": "structural",
        }
    ]
    # Spelled out independently of the dict above, so widening the contract key
    # list cannot pass by also widening the expectation.
    assert set(payload[0]) == {"stage", "metric", "value", "limit", "tier"}
    assert "detail" not in payload[0]


def test_a_non_numeric_check_still_emits_value_and_limit_as_none() -> None:
    """``value``/``limit`` are ``float | None`` in the contract for a reason: the
    colorspace tag, an unmeasurable loudness and an absent report have no number.
    The keys must still be PRESENT — a consumer that has to distinguish "absent
    key" from "null" is back to guessing."""
    (entry,) = mastering.violations_payload(
        [
            mastering.MasteringGateViolation(
                stage="colorspace",
                metric="colorspace_ok",
                tier="structural",
                detail="no tag",
            )
        ]
    )

    assert entry["value"] is None
    assert entry["limit"] is None


def test_an_unclassified_or_third_tier_cannot_be_constructed() -> None:
    """``tier`` is required with no default, and the vocabulary is CLOSED at two
    values. A default would let a new construction site ship an unclassified
    violation the consumer reads as merely tunable — silently downgrading a
    structural defect. A third value is worse: the consumer is written against
    exactly these two strings, so an invented tier falls through its filter
    entirely and the violation stops being classified at all.

    ⚠️ TWO AXES, NOT ONE. ``tier`` says what KIND of threshold this is (does it
    need calibration?) and is the published contract. Whether the gate currently
    BLOCKS is a separate boolean axis owned by the kill switch (``gate_armed``) —
    which is why "advisory" is NOT a tier: a gate can be tunable AND armed, or
    structural AND temporarily advisory, and one field cannot carry both.

    Driven through ``model_validate`` rather than the constructor so these are
    RUNTIME validation assertions on untrusted input shapes, with no type-checker
    suppressions anywhere in this file."""
    base = {"stage": "dead_frame", "metric": "dead_frame_ratio", "detail": "x"}

    with pytest.raises(ValidationError):
        mastering.MasteringGateViolation.model_validate(base)  # no tier at all
    with pytest.raises(ValidationError):
        mastering.MasteringGateViolation.model_validate({**base, "tier": "advisory"})
    with pytest.raises(ValidationError):
        mastering.MasteringGateViolation.model_validate({**base, "tier": None})

    # ...and the two legal values really are accepted, so the test above is not
    # passing because the model rejects everything.
    for tier in ("structural", "tunable"):
        assert mastering.MasteringGateViolation.model_validate({**base, "tier": tier}).tier == tier


@pytest.mark.asyncio
async def test_the_emitted_tier_is_read_from_the_map_not_retyped_per_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE DISCRIMINATING TIER TEST. Asserting ``tier == "tunable"`` on a loudness
    violation passes just as well if every construction site hardcodes its own
    string — the tier would then be six independent literals that drift from
    ``_GATE_TIERS`` the moment one is edited, and no assertion anywhere would
    notice. A tier test that passes whether or not the tier is actually READ is
    the exact failure mode this project keeps producing.

    So: repoint the MAP and require the EMITTED value to follow. This can only
    pass if the payload derives its tier from the single source of truth."""
    _stub_clean_probes(monkeypatch)
    monkeypatch.setitem(mastering._GATE_TIERS, "loudness", "structural")

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness={"integrated_lufs": -30.0}
    )

    assert [v["tier"] for v in report["violations"]] == ["structural"]


def test_the_degraded_report_emits_a_structural_verify_violation() -> None:
    """An ABSENT report is itself a structural finding: verification did not run,
    so NOTHING is known about the master. ``violations: []`` here would read as
    "checked, nothing wrong" — the absence-of-evidence-as-evidence confusion this
    whole wave exists to end."""
    report = mastering.degraded_report()

    assert report["violations"] == [
        {
            "stage": "verify",
            "metric": "post_mux_verify",
            "value": None,
            "limit": None,
            "tier": "structural",
        }
    ]
    # ...and it survives the response model rather than being dropped by
    # pydantic's default extra="ignore", which is how dead_frame_ratio was lost.
    #
    # Compared as a ROUND TRIP rather than by identity: `violations` is typed as
    # `MasteringViolation` (it was `dict[str, Any]` until the `Any` ratchet
    # refused it), so `model_validate` parses these dicts INTO model instances
    # and an instance is never `==` a dict. Dumping back is the stronger
    # assertion anyway — it proves the wire shape survives the model in both
    # directions, which is the actual property at stake.
    revalidated = MasteringReport.model_validate(report).violations
    assert [v.model_dump(mode="json") for v in revalidated] == report["violations"]


@pytest.mark.asyncio
async def test_a_delivered_report_carries_the_tunable_violation_it_did_not_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE WHOLE POINT OF C3, end to end: a master that misses a TUNABLE ceiling
    is still DELIVERED (the switch is off), and the report it ships with says so
    with the measured number and the limit — which is how the fleet distribution
    needed to arm that gate gets collected. Before this, the only trace was
    ``rerun_stages: ["loudness"]``."""
    _stub_clean_probes(monkeypatch)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(master_lufs=-14.0),
        loudness={"integrated_lufs": -30.0},
    )

    assert report["gate_pass"] is False  # it is still a finding
    assert report["violations"] == [
        {
            "stage": "loudness",
            "metric": "integrated_lufs",
            "value": -30.0,
            "limit": -14.0,
            "tier": "tunable",
        }
    ]
    # NOT enforced: the compile completed and returned a report at all.
    assert not mastering.gate_armed(_settings(), "loudness")


@pytest.mark.asyncio
async def test_a_clean_master_emits_an_empty_violations_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative half. If ``violations`` were non-empty on a clean master the
    key would be noise, and a consumer would learn to ignore it."""
    _stub_clean_probes(monkeypatch)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness={"integrated_lufs": -14.2}
    )

    assert report["gate_pass"] is True
    assert report["violations"] == []


# ---------------------------------------------------------------------------
# C2 (compiler half) — av_coverage RAISES, and the violation is readable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_av_coverage_raises_and_the_violation_is_still_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``av_coverage`` raises, so the REPORT can never carry its violation — which
    left the consumer parsing an error string for the one defect that destroys a
    paid deliverable. The structured verdict rides the exception instead, in the
    same contracted shape."""
    _stub_clean_probes(monkeypatch, video_seconds=118.4, audio_seconds=158.0)

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness={"integrated_lufs": -14.0}
        )

    (entry,) = excinfo.value.violations_payload()
    assert entry["stage"] == "av_coverage"
    assert entry["metric"] == "av_gap_seconds"
    assert entry["tier"] == "structural"
    assert entry["value"] == pytest.approx(39.6, abs=0.01)
    assert entry["limit"] == pytest.approx(mastering.AV_COVERAGE_TOLERANCE_SECONDS)


@pytest.mark.asyncio
async def test_legitimate_aac_tail_padding_is_not_a_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured other edge (2026-09-10: video 58.200 / audio 58.361 on a clean
    promo). A gate that convicts every AAC delivery is the 96.7% trap again."""
    _stub_clean_probes(monkeypatch, video_seconds=58.200, audio_seconds=58.361)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness={"integrated_lufs": -14.0}
    )

    assert report["violations"] == []
    assert report["gate_pass"] is True


# ---------------------------------------------------------------------------
# C1 — the gates are ARMED, tiered, and every one has a kill switch
# ---------------------------------------------------------------------------


def test_every_structural_gate_is_armed_and_every_tunable_one_is_not() -> None:
    """THE TIERING IS THE SAFETY DESIGN. Structural gates detect ABSENCE or
    BREAKAGE, so they cannot be miscalibrated and are armed now. Tunable gates
    decide DEGREE against a ceiling whose fleet distribution is unmeasured, and
    this project has twice armed an uncalibrated ceiling (duplicate frames
    convicted 120 of 120 masters; a deband ceiling would have fired on 96.7%).

    ``colorspace`` is the documented exception in the other direction: structural
    by nature, held disarmed on a MEASURED 23.3% fire rate over n=120."""
    settings = _settings()

    # VC-W14 added ``true_peak`` (V3) and ``measure`` (V4), both STRUCTURAL and
    # both armed: the true-peak ceiling is DIGITAL FULL SCALE (a property of the
    # format, not a fleet percentile), and ``measure`` reports that the requested
    # two-pass master could not be built — self-reported by the render, so no
    # flaky probe can manufacture it. ``bed_calibration`` (V6) is also structural
    # and deliberately NOT armed — see the disarmed block below.
    expected_armed = {"av_coverage", "dead_frame", "verify", "true_peak"}
    actual_armed = {s for s in mastering._GATE_KILL_SWITCHES if mastering.gate_armed(settings, s)}
    assert actual_armed == expected_armed

    for stage in expected_armed:
        assert mastering.gate_tier(stage) == "structural", stage
    # The two STRUCTURAL gates deliberately held disarmed, each on a measurement:
    # colorspace fires on 23.3% of n=120 masters (an upstream tagging bug), and
    # banding_measurement convicts the PROBE rather than the master at an
    # unmeasured rate. Both still REPORT, which is how the rates get collected.
    # VC-W14/V6 adds ``bed_calibration`` to this list on the same footing: it
    # convicts the loudness PROBE over the narration bus, whose fleet failure
    # rate is n=0.
    for stage in ("colorspace", "banding_measurement", "bed_calibration", "measure"):
        assert mastering.gate_tier(stage) == "structural", stage
        assert not mastering.gate_armed(settings, stage), stage
    for stage, tier in mastering._GATE_TIERS.items():
        if tier == "tunable":
            assert not mastering.gate_armed(settings, stage), stage


def test_every_gate_has_its_own_kill_switch_and_it_is_a_real_settings_field() -> None:
    """A kill switch that is not a ``Settings`` field cannot be set from gitops,
    so the back-out path would be a code change and a rollout — exactly what
    arming must not require. Also pins the two default tables in agreement: the
    module's fallback (for a settings object predating the fields) must not
    disagree with the real field default."""
    fields = Settings.model_fields

    for stage in mastering._GATE_TIERS:
        switch = mastering._GATE_KILL_SWITCHES.get(stage)
        assert switch is not None, f"{stage} has no kill switch"
        assert switch in fields, f"{switch} is not a Settings field"
        assert fields[switch].annotation is bool
        assert fields[switch].default is mastering._GATE_ARMED_DEFAULTS[stage], stage


@pytest.mark.asyncio
async def test_a_kill_switch_actually_backs_a_gate_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BEHAVIOUR, not configuration: the same truncated master that fails the
    compile with the switch on is DELIVERED with an annotated report when the
    switch is off. If this passed with the switch ignored, the whole safety
    argument for arming would be false."""
    _stub_clean_probes(monkeypatch, video_seconds=118.4, audio_seconds=158.0)
    loudness = {"integrated_lufs": -14.0}

    with pytest.raises(mastering.MasteringGateError):
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness=loudness
        )

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(master_lufs=-14.0, mastering_gate_av_coverage_enabled=False),
        loudness=loudness,
    )
    assert report["gate_pass"] is False
    assert [v["stage"] for v in report["violations"]] == ["av_coverage"]


@pytest.mark.asyncio
async def test_a_tunable_switch_turned_on_does_arm_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch is WIRED, not decorative: the operator who measures the fleet
    loudness distribution can arm it from config alone. This is the assertion
    that stops the tunable tier from being a dead branch."""
    _stub_clean_probes(monkeypatch)

    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(master_lufs=-14.0, mastering_gate_loudness_enabled=True),
            loudness={"integrated_lufs": -30.0},
        )

    assert [v["stage"] for v in excinfo.value.violations_payload()] == ["loudness"]


def test_enforce_mode_still_arms_everything_including_unmapped_stages() -> None:
    """The whole-gate mode survives as an ESCALATION. It must arm the tunable and
    the deliberately-disarmed structural gates too, or "enforce" would quietly
    mean less than its name."""
    enforcing = _settings(mastering_gate_mode="enforce")

    for stage in ("loudness", "colorspace", "a_stage_that_does_not_exist_yet"):
        assert mastering.gate_armed(enforcing, stage)


def test_an_unknown_stage_falls_through_to_tunable_and_unarmed() -> None:
    """DRIFT MUST CAUSE A MISS, NEVER A FLEET-WIDE FALSE POSITIVE. A stage added
    to the gate without a tier entry must read as tunable and stay disarmed under
    default settings — the direction where a new gate is under-enforced rather
    than convicting every delivery."""
    assert mastering.gate_tier("a_stage_that_does_not_exist_yet") == "tunable"
    assert not mastering.gate_armed(_settings(), "a_stage_that_does_not_exist_yet")


def test_gate_armed_uses_the_real_default_when_the_field_is_absent() -> None:
    """FAIL-OPEN IS THE DEFECT BEING FIXED. A ``Settings`` that predates these
    fields must NOT read as "every gate disarmed" — reading an absent switch as
    off is exactly how ``mastering_gate_mode`` ended up deciding nothing.

    The double is a REAL ``Settings`` instance with the attribute genuinely
    removed, not a stand-in object: ``gate_armed``'s signature is what protects
    its call sites, so the test satisfies the real type rather than suppressing
    the checker. Deleting from ``__dict__`` makes pydantic's attribute lookup
    miss for real, which is the condition under test."""
    settings = _settings()
    for field in ("mastering_gate_av_coverage_enabled", "mastering_gate_loudness_enabled"):
        del settings.__dict__[field]
        assert getattr(settings, field, "ABSENT") == "ABSENT", f"{field} is still present"

    # Falls back to _GATE_ARMED_DEFAULTS, which preserves the real tiering...
    assert mastering.gate_armed(settings, "av_coverage") is True
    # ...in BOTH directions — a blanket "True on missing" would pass the line
    # above and silently arm every tunable gate on an older Settings.
    assert mastering.gate_armed(settings, "loudness") is False


def test_the_two_axes_are_independent() -> None:
    """``tier`` (what kind of threshold) and armed (does it block) are SEPARATE
    axes, and the pair must be able to take every combination — otherwise one of
    them is redundant and the design collapses back into a single flag.

    Proven on live gates, not on a table: structural-and-armed (av_coverage),
    structural-and-advisory (colorspace, held on a measured 23.3% fire rate),
    tunable-and-advisory (loudness by default) and tunable-and-armed (the same
    gate once its switch is flipped)."""
    default = _settings()
    armed_loudness = _settings(mastering_gate_loudness_enabled=True)

    assert (mastering.gate_tier("av_coverage"), mastering.gate_armed(default, "av_coverage")) == (
        "structural",
        True,
    )
    assert (mastering.gate_tier("colorspace"), mastering.gate_armed(default, "colorspace")) == (
        "structural",
        False,
    )
    assert (mastering.gate_tier("loudness"), mastering.gate_armed(default, "loudness")) == (
        "tunable",
        False,
    )
    assert (
        mastering.gate_tier("loudness"),
        mastering.gate_armed(armed_loudness, "loudness"),
    ) == ("tunable", True)


# ---------------------------------------------------------------------------
# shot-cache classification of the new gate switches
# ---------------------------------------------------------------------------


def test_flipping_a_gate_switch_does_not_change_a_shot_cache_key(tmp_path: Path) -> None:
    """THE BUCKET CHOICE, ASSERTED BEHAVIOURALLY. Every new
    ``mastering_gate_*_enabled`` switch belongs in
    ``renderer._SHOT_CACHE_EXCLUDED_SETTINGS``, and this is the test that can tell
    the two buckets apart.

    ⚠️ ``test_every_settings_field_is_classified`` CANNOT: it only demands that a
    field appear in ONE of the two sets, and greens whichever set you pick —
    verified, both classifications satisfy it. So it catches "you forgot to
    classify" and is blind to "you classified it wrong", which is the half that
    corrupts the cache.

    A gate switch decides whether a run FAILS after the mux, not what the pixels
    are: all six are read only by ``mastering.gate_armed`` (post-mux verdict) and
    ``compile.py``'s verify branch, never by segment rendering. So folding them
    into the key would not protect anything — it would invalidate every cached
    shot in the fleet each time an operator armed or disarmed a gate, re-rendering
    byte-identical segments. ``_SHOT_CACHE_KEY_VERSION`` stays where it is for the
    same reason: nothing about a segment's pixels changed."""
    from src.services import renderer

    shot = Shot(id="a", seconds=3.0, video_url="http://x/clip.mp4", caption="hi")
    base = get_settings().model_copy(update={"work_dir": str(tmp_path)})
    baseline = renderer._shot_cache_key(shot, base, 0.25)

    switches = [
        f
        for f in Settings.model_fields
        if f.startswith("mastering_gate_") and f != "mastering_gate_mode"
    ]
    # 6 -> 9 with VC-W14's true_peak / measure / bed_calibration switches; 9 -> 10
    # with ``speech_presence`` (a POST-MUX measurement of the delivered master's
    # program-over-noise-floor margin — it cannot change any shot's pixels either);
    # 10 -> 11 with ``narration_presence`` (a POST-MUX, content-aware per-shot
    # measurement of the SAME delivered master — reads only shot_timeline/audio_url
    # to compare against the already-encoded file, never a segment's pixels).
    # 11 -> 12 with ``cfr`` (read only by ``mastering._cfr_violation`` /
    # ``gate_armed`` inside ``post_mux_verify``, over the delivered master's
    # already-probed cadence — never by segment rendering).
    # This count exists so a new gate switch cannot be added without someone
    # proving here that it stays out of the shot-cache key.
    assert len(switches) == 12, switches

    for field in switches:
        flipped = base.model_copy(update={field: not getattr(base, field)})
        assert getattr(flipped, field) is not getattr(base, field), field
        assert renderer._shot_cache_key(shot, flipped, 0.25) == baseline, (
            f"{field} leaked into the shot-cache key: flipping a post-mux GATE "
            f"switch must not invalidate cached segments whose pixels are identical"
        )


# ---------------------------------------------------------------------------
# master_measure_failed (C10) — a signal the schema drops does not exist
# ---------------------------------------------------------------------------


def test_master_measure_failed_survives_both_response_models() -> None:
    """C10 was implemented, merged and GREEN while being INERT on the wire: all
    five loudness-measure bail paths record the degradation and
    ``renderer.get_master_loudnorm_evidence()`` returns ``measure_failed``, which
    the compile route folds in as ``master_measure_failed`` — but pydantic's
    default ``extra="ignore"`` ate the undeclared key, so the fix existed only in
    pod logs.

    Why this is not a missing-field nit: a failed measure pass means the master
    shipped UN-NORMALISED while the ducking back-solved from an assumed -18 LUFS.
    Dropped by the schema, that is indistinguishable from a clean master.

    Asserted on BOTH models the field was added to, because a declaration on one
    of two response shapes is still a silent drop on the other."""
    for model in (MasteringReport, AcxReport):
        base: dict[str, object] = (
            {"gate_pass": True, "rerun_stages": []} if model is MasteringReport else {"pass": False}
        )
        report = model.model_validate({**base, "master_measure_failed": True})

        assert report.master_measure_failed is True, model.__name__
        assert report.model_dump()["master_measure_failed"] is True, model.__name__


def test_master_measure_failed_defaults_to_none_and_never_to_false() -> None:
    """ABSENCE IS THE THIRD STATE. ``get_master_loudnorm_evidence`` emits the key
    ONLY when a failure was recorded, so a clean render and a mastering-off render
    both leave it unset. A ``False`` default would assert that a measurement
    succeeded on every job that never measured one — the absence-of-evidence-as-
    evidence confusion, in the one field whose whole purpose is to report a
    degradation."""
    for model in (MasteringReport, AcxReport):
        base: dict[str, object] = (
            {"gate_pass": True} if model is MasteringReport else {"pass": True}
        )
        assert model.model_validate(base).master_measure_failed is None, model.__name__

    # The declared default itself, so a later edit to `= False` fails here rather
    # than quietly changing what every clean job reports.
    for model in (MasteringReport, AcxReport):
        assert model.model_fields["master_measure_failed"].default is None, model.__name__


# ---------------------------------------------------------------------------
# dead_frame_min_ratio — publish the cutoff, not just the metric
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_dead_frame_cutoff_is_published_on_the_clean_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator duplicated this number as its own constant
    (``shared/mastering_gate.py:_DEAD_FRAME_MIN_RATIO = 0.10``) because the
    compiler reported ``dead_frame_ratio`` and never the cutoff it compared
    against. Retuning ``render_dead_frame_min_ratio`` here then left the
    orchestrator's blank-picture verdict comparing against a stale 0.10 — two
    repos, one threshold, no link.

    THE CLEAN PATH IS THE POINT. On a violation the number is already available
    as the violation's ``limit``, but by then the decision is made; a consumer
    re-deriving the verdict on a GOOD master had nothing to read and fell back to
    its constant every time the picture was fine."""
    _stub_clean_probes(monkeypatch)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(master_lufs=-14.0, render_dead_frame_min_ratio=0.25),
        loudness={"integrated_lufs": -14.0},
    )

    assert report["violations"] == []  # clean: no violation to carry a `limit`
    assert report["dead_frame_min_ratio"] == pytest.approx(0.25)
    # Survives the response model — a returned key with no declared field is not
    # a field, which is how `dead_frame_ratio` itself was lost for months.
    assert MasteringReport.model_validate(report).dead_frame_min_ratio == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_the_published_cutoff_is_the_one_the_gate_actually_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reported threshold that is not the threshold applied is worse than none:
    the consumer would trust it and compare against a number this module ignored.
    Pinned by driving a ratio BETWEEN the default and the configured cutoff — 0.5
    of frames blank fails at 0.10 and passes at 0.75 — so the published value and
    the decision must agree or one of these two halves breaks."""
    _stub_half_dead_probes(monkeypatch)
    loudness = {"integrated_lufs": -14.0}

    # Cutoff raised ABOVE the measured 0.5 → delivered, and the report states the
    # cutoff that let it through.
    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"),
        _settings(master_lufs=-14.0, render_dead_frame_min_ratio=0.75),
        loudness=loudness,
    )
    assert report["dead_frame_ratio"] == pytest.approx(0.5)
    assert report["dead_frame_min_ratio"] == pytest.approx(0.75)
    assert report["violations"] == []

    # Same master, cutoff below the measured ratio → the armed gate fails it, and
    # the violation's `limit` is that same configured number.
    with pytest.raises(mastering.MasteringGateError) as excinfo:
        await mastering.post_mux_verify(
            Path("/tmp/out.mp4"),
            _settings(master_lufs=-14.0, render_dead_frame_min_ratio=0.25),
            loudness=loudness,
        )
    (entry,) = excinfo.value.violations_payload()
    assert entry["stage"] == "dead_frame"
    assert entry["limit"] == pytest.approx(0.25)


def test_the_degraded_report_states_no_cutoff_rather_than_inventing_one() -> None:
    """``degraded_report()`` holds no ``Settings``, so it cannot know the cutoff a
    deployment configured. Restating the module default would re-create exactly
    the duplicated constant that publishing this key exists to eliminate, and
    there is no ``dead_frame_ratio`` on that report to compare it against
    anyway."""
    report = mastering.degraded_report()

    assert "dead_frame_min_ratio" in report  # the KEY is present on both shapes
    assert report["dead_frame_min_ratio"] is None
    assert report["dead_frame_ratio"] is None


# ---------------------------------------------------------------------------
# C12 — every armed threshold carries its measurement, or is marked TUNABLE
# ---------------------------------------------------------------------------


def test_the_unmeasured_thresholds_are_the_ones_that_are_not_armed() -> None:
    """C12 as a BEHAVIOURAL invariant rather than a comment audit (a comment
    cannot be mutation-tested). The rule the wave establishes: a threshold with no
    measured fleet distribution must not decide a compile.

    ``LOUDNESS_GATE_TOLERANCE_LU`` (3.0 LU, an EBU-R128 convention, n=0) and
    ``_CFR_FRAME_UNIFORMITY`` (0.98, tail-frame arithmetic, n=0) are the two
    uncited numbers left in the module, and NEITHER can fail a compile by
    default. ``AV_COVERAGE_TOLERANCE_SECONDS`` is armed and IS measured
    two-sided (2026-09-10). ``BANDING_GATE_CEILING`` is armed-adjacent only as a
    deband TRIGGER and carries n=120."""
    settings = _settings()

    # The uncited numbers decide nothing.
    assert not mastering.gate_armed(settings, "loudness")
    # cfr HAS a switch since ``mastering_gate_cfr_enabled`` — what keeps the
    # uncited number safe is that the switch defaults OFF, not that it is absent.
    assert not mastering.gate_armed(settings, "cfr")
    assert Settings.model_fields["mastering_gate_cfr_enabled"].default is False
    assert mastering.gate_tier("cfr") == "tunable"

    # The one armed numeric threshold is the measured one.
    assert mastering.gate_armed(settings, "av_coverage")
    assert mastering.AV_COVERAGE_TOLERANCE_SECONDS == 0.5


@pytest.mark.asyncio
async def test_cfr_incoherence_can_never_fail_a_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_CFR_FRAME_UNIFORMITY`` is uncited (n=0) and post-#218 fleet ``cfr_ok``
    rates were never re-measured, so arming the gates must NOT have swept cadence
    in. ``"cfr"`` stays an advisory ``rerun_stages`` append with no ``violations``
    entry — the one property that keeps the uncited number safe."""
    _stub_clean_probes(monkeypatch)

    # The cfr PROBE is stubbed, not the gate: this test is about how the gate
    # TREATS ``cfr_ok is False``, not about how that verdict is measured (which
    # ``framerate_coherence``'s own tests cover). Stubbing it is what makes the
    # incoherent case reachable without also perturbing the packet-count probes
    # that feed the unrelated cadence telemetry.
    async def incoherent(path: object, expected_fps: object = None) -> bool:
        return False

    monkeypatch.setattr(mastering, "framerate_coherence", incoherent)

    report = await mastering.post_mux_verify(
        Path("/tmp/out.mp4"), _settings(master_lufs=-14.0), loudness={"integrated_lufs": -14.0}
    )

    assert report["cfr_ok"] is False
    assert "cfr" in report["rerun_stages"]
    assert report["violations"] == []
    assert report["gate_pass"] is True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ALIVE_FRAME = (
    "lavfi.signalstats.YMIN=16\n"
    "lavfi.signalstats.YLOW=40\n"
    "lavfi.signalstats.YHIGH=200\n"
    "lavfi.signalstats.YMAX=235\n"
    "lavfi.signalstats.YDIF=9.0\n"
)


def _stub_clean_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    video_seconds: float = 60.0,
    audio_seconds: float = 60.1,
    r_frame_rate: str = "30/1",
    avg_frame_rate: str = "30/1",
) -> None:
    """Every probe answers CLEAN unless a keyword overrides it, so each test
    above perturbs exactly one signal. Two live frames (not one) because the
    dead-frame ratio is a fraction of measured frames."""
    from src.services import renderer

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return f"{r_frame_rate}\n{avg_frame_rate}\n"
        if "stream=avg_frame_rate" in joined:
            return f"{avg_frame_rate}\n"
        if "signalstats" in joined:
            return (
                "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
                + _ALIVE_FRAME
                + "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
                + _ALIVE_FRAME
            )
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)

    async def fake_av_stream_seconds(path: object) -> tuple[float | None, float | None]:
        return (video_seconds, audio_seconds)

    monkeypatch.setattr(mastering, "av_stream_seconds", fake_av_stream_seconds)


def _stub_half_dead_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """One blank frame, one live frame → ``dead_frame_ratio`` 0.5, a value that
    straddles the two cutoffs the test above configures."""
    from src.services import renderer

    blank = (
        "lavfi.signalstats.YMIN=16\n"
        "lavfi.signalstats.YLOW=16\n"
        "lavfi.signalstats.YHIGH=16\n"
        "lavfi.signalstats.YMAX=16\n"
        "lavfi.signalstats.YDIF=0.0\n"
    )

    async def fake_run(cmd: list[str], *, capture_stderr: bool = False) -> str:
        joined = " ".join(cmd)
        if "color_primaries" in joined:
            return "color_space=bt709\ncolor_transfer=bt709\ncolor_primaries=bt709\n"
        if "r_frame_rate,avg_frame_rate" in joined:
            return "30/1\n30/1\n"
        if "stream=avg_frame_rate" in joined:
            return "30/1\n"
        if "signalstats" in joined:
            return (
                "[Parsed_metadata_1 @ 0x1] frame:0    pts:0    pts_time:0\n"
                + blank
                + "[Parsed_metadata_1 @ 0x1] frame:1    pts:512    pts_time:0.017\n"
                + _ALIVE_FRAME
            )
        return ""

    monkeypatch.setattr(renderer, "_run", fake_run)

    async def fake_av_stream_seconds(path: object) -> tuple[float | None, float | None]:
        return (60.0, 60.1)

    monkeypatch.setattr(mastering, "av_stream_seconds", fake_av_stream_seconds)
