"""Settings — env-driven, no defaults that talk to real infra in tests."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import DatabaseSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = Field(default=8200, description="Bind port")
    log_level: str = Field(default="INFO")

    # Durable compile-job store + Idempotency-Key dedup index. Reads DATABASE_URL
    # (env_prefix DATABASE_). Persisting the store is what makes a re-submitted
    # render dedupe across a pod restart instead of firing a second PAID compile.
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    minio_endpoint: str = Field(
        default="",
        description=(
            "host:port for the MinIO server (no scheme). No hardcoded "
            "default; supplied via gitops-production env per "
            "docs/config-externalization-policy.md."
        ),
    )
    minio_access_key: str = Field(default="")
    minio_secret_key: str = Field(default="")
    minio_secure: bool = Field(default=False, description="Use TLS for MinIO")
    minio_bucket: str = Field(
        default="w2v-compiled-videos",
        description="Bucket where compiled MP4s are uploaded",
    )
    minio_audio_bucket: str = Field(
        default="w2v-audiobooks",
        description="Bucket where stitched audiobook WAV/MP3 artifacts are uploaded",
    )
    minio_url_expiry_seconds: int = Field(
        default=3600,
        ge=60,
        le=604800,  # SigV4 hard max is 7 days; a larger expiry yields URLs MinIO 403s
        description="Presigned-URL expiry on the returned artifact_url",
    )
    # WHY THIS IS NOT ``minio_url_expiry_seconds``. ``prune()`` deletes terminal
    # compile_jobs rows, and that row IS the ``(tenant_id, idempotency_key)`` dedup
    # index (uq_compile_jobs_tenant_idempotency_key). Prune was previously driven by
    # the PRESIGN expiry (3600s), so a resend of the same key after the row vanished
    # found no conflict, inserted fresh, returned action="spawn" and paid for a SECOND
    # full render plus a second _emit_cost.
    #
    # 7200 is DERIVED, not guessed: the orchestrator resumes a run whose updated_at is
    # up to PIPELINE_RECONCILER_MAX_AGE_SECONDS (default 7200) old, and its compile
    # idempotency key is resume-stable (`{run_id}:{stage}`), so 2h is the real outer
    # bound on a same-key resend. Normal re-dispatch is ~90s (reconciler_interval) and
    # was never at risk; the exposure was a 3600-7200s dispatch gap — a wedged rollout,
    # a lost node, drained replicas. Keep this >= the orchestrator's value.
    compile_job_retention_seconds: int = Field(
        default=7200,
        ge=3600,
        le=604800,
        description=(
            "How long a TERMINAL compile_jobs row survives before prune() evicts it. "
            "This is the idempotency dedup window, NOT an artifact-URL lifetime."
        ),
    )

    # Bounded so a misconfigured env can't produce a degenerate canvas: fps=0 would
    # divide-by-zero in the cadence math, and non-positive / absurd dims blow up ffmpeg.
    render_width: int = Field(default=1920, ge=16, le=7680)
    render_height: int = Field(default=1080, ge=16, le=4320)
    render_fps: int = Field(default=30, ge=1, le=240)
    render_cadence_mode: str = Field(
        default="auto",
        description=(
            "Master-cadence selection policy (VC-3). 'auto'/'on' pick a SINGLE "
            "output frame rate that minimizes retiming cost across the shots' "
            "declared source rates (Shot.target_fps), falling back to render_fps "
            "when no rates are declared — so a default compile stays render_fps "
            "(byte-identical). 'fixed'/'off' pins the render to render_fps and "
            "skips selection (also how an explicit request 'cadence' is honoured)."
        ),
    )
    render_cadence_candidates: str = Field(
        default="24,25,30,50,60",
        description=(
            "Comma-separated legal delivery cadences the auto master-cadence "
            "selector may choose from. Values are snapped to the nearest legal "
            "rate {24,25,30,50,60}; a bad entry is dropped."
        ),
    )
    render_vfr_epsilon: float = Field(
        default=0.02,
        ge=0.0,
        description=(
            "Tolerance (fps) for treating a measured average frame rate as equal "
            "to the declared cadence when reasoning about VFR/coherence. Advisory "
            "for cadence reporting; the master soft-gate owns the actual check."
        ),
    )
    mastering_gate_mode: Literal["advisory", "enforce"] = Field(
        default="advisory",
        description=(
            "What a CLEARED mastering soft-gate (post_mux_verify's gate_pass) does "
            "to the render. 'advisory' (DEFAULT, and the only mode any deployment "
            "runs today) keeps the historical behaviour byte-for-byte: the report "
            "is attached to the response, gate_pass=False is informational, and the "
            "artifact still uploads. 'enforce' makes the gate REAL — a cleared "
            "gate_pass fails the compile with an error naming the offending metric, "
            "its measured value and its ceiling, and a verification that CRASHED "
            "fails CLOSED (an unverified master is a failed render, never a "
            "degraded_report() indistinguishable from a pass). Enforce is opt-in "
            "per deployment, and STILL NOT ARMED ANYWHERE. The old reason was that "
            "gate_pass measured false on essentially every master; the two checks "
            "responsible (duplicate_frames 120/120, banding 89.2% over n=120 "
            "masters) were removed from the verdict on 2026-09-11, leaving "
            "av_coverage + dead_frame + loudness + colorspace. On the same "
            "measurements that is ~26% false, and ~23 points of it is colorspace "
            "tagging — an upstream bug to fix at the source, not a ceiling to "
            "widen. Arming enforce is a separate, later decision that belongs to "
            "the owner, and it should follow that fix, not precede it."
        ),
    )
    # ---------------------------------------------------------------------
    # PER-GATE KILL SWITCHES (mastering soft-gate arming)
    # ---------------------------------------------------------------------
    #
    # ``mastering_gate_mode`` above is the WHOLE-gate escalation and stays
    # 'advisory'. These six switches are the real arming surface: each one arms
    # or disarms ONE gate, so a threshold that turns out to be wrong is backed
    # out by flipping a single env var in gitops — no code change, no image
    # build, no rollout. The tiering (which default to True) lives in
    # ``mastering._GATE_TIERS`` / ``_GATE_ARMED_DEFAULTS`` and its rationale is
    # documented there; the defaults MUST agree with this file and
    # ``tests/unit/test_av_quality_wave13.py`` pins that agreement.
    #
    # STRUCTURAL (default ON): these detect ABSENCE or BREAKAGE, not degree —
    # blank picture, audio outrunning picture, a picture never measured at all,
    # a verification that did not run. There is no ratio, gap or ceiling at
    # which shipping such a master is correct, so they need no fleet
    # calibration to be trustworthy.
    mastering_gate_av_coverage_enabled: bool = Field(
        default=True,
        description=(
            "ARM the A/V coverage gate: fail the compile when the delivered "
            "audio outruns the delivered picture by more than "
            "mastering.AV_COVERAGE_TOLERANCE_SECONDS (0.5s). STRUCTURAL and the "
            "only armed gate with a measured two-sided separation (2026-09-10: "
            "0.161s of legitimate AAC tail padding on a clean promo vs a "
            "smallest real truncation of 8.8s). This gate already failed the "
            "compile in both modes before the switch existed; setting it false "
            "restores pre-2026-08 behaviour, in which a master whose picture "
            "ends mid-sentence uploads and bills like a clean one."
        ),
    )
    mastering_gate_dead_frame_enabled: bool = Field(
        default=True,
        description=(
            "ARM the dead/blank-master gate: fail the compile when more than "
            "render_dead_frame_min_ratio of MEASURED frames sit below "
            "render_dead_frame_luma_spread_floor. STRUCTURAL — a uniform-luma "
            "delivery is a dead render, not a low-quality one. ⚠️ Its two "
            "thresholds are DERIVED (fade-duration arithmetic, see their own "
            "descriptions) rather than measured against a fleet distribution, "
            "which is exactly what this switch is for: if the fleet turns out "
            "to carry legitimate long blank passages, disarm here and re-derive "
            "rather than widening the ratio in code."
        ),
    )
    mastering_gate_banding_measurement_enabled: bool = Field(
        default=False,
        description=(
            "ARM the measurement-GAP gate: fail the compile when the per-frame "
            "signalstats pass produced NO frames. Note this is NOT the banding "
            "SCORE gate (retired 2026-09-11, 89.2% of n=120 masters) — it "
            "carries no ceiling at all, and an empty pass leaves luma_spread, "
            "dead_frame_ratio and every banding score None so every picture-side "
            "gate 'passes' on zero evidence. DEFAULT OFF anyway, and on "
            "measurement rather than caution: a fire here convicts the PROBE, not "
            "the master, its fleet rate has never been measured (n=0), and "
            "arming it broke four pre-existing tests that have nothing to do with "
            "picture quality simply because their fixture file is not probe-able. "
            "The finding is still REPORTED (gate_pass=False, a 'structural' "
            "violations entry), which is how the rate gets collected. "
            "mastering_gate_verify_enabled covers the unambiguous half: no report "
            "at all."
        ),
    )
    mastering_gate_verify_enabled: bool = Field(
        default=True,
        description=(
            "ARM the absent-report gate: fail the compile when post_mux_verify "
            "itself could not run (it crashed, the probe binary failed). "
            "STRUCTURAL — an unverified master is not a clean master, and "
            "mastering.degraded_report() is byte-for-byte indistinguishable "
            "from a report whose checks all ran and passed except for this "
            "violation. Setting it false restores the historical fail-open: the "
            "artifact uploads and bills with nothing known about it."
        ),
    )
    mastering_gate_true_peak_enabled: bool = Field(
        default=True,
        description=(
            "ARM the true-peak gate (VC-W14/V3): fail the compile when the "
            "delivered master's true peak exceeds mastering."
            "TRUE_PEAK_CEILING_DBTP (0.0 dBTP). STRUCTURAL: the ceiling is "
            "DIGITAL FULL SCALE, a property of the format, not a fleet "
            "percentile — which is why it needs no calibration and is NOT set "
            "at master_tp (-1.0), where the limiter merely aims. true_peak_dbtp "
            "was measured and published on every report since the gate was "
            "written and compared to nothing at all. A job with mastering off "
            "runs no limiter and is excluded from the gate entirely, not "
            "convicted. Setting this false restores that "
            "measured-but-never-read behaviour."
        ),
    )
    mastering_gate_measure_enabled: bool = Field(
        default=False,
        description=(
            "ARM the pass-1-measurement gate (VC-W14/V4): FAIL the compile when "
            "the two-pass loudness MEASUREMENT crashed and the master silently "
            "degraded to single-pass DYNAMIC loudnorm. STRUCTURAL by shape (it "
            "reports that the requested master could not be BUILT) but DEFAULT "
            "OFF, and the reasoning is the point: a dynamically-normalised "
            "master is a SHIPPABLE film — worse than requested, not "
            "undeliverable — and the pass-1 failure rate over this fleet is "
            "UNMEASURED (n=0). Arming it would refuse paid deliverables at an "
            "unknown rate over a defect whose output a viewer may not notice, "
            "which is the 96.7% trap in a new costume. THE FIX SHIPS ANYWAY: "
            "with this false the violation still clears gate_pass and appears "
            "in report.violations with its tier, so the degradation is finally "
            "VISIBLE (it used to publish gate_pass=True and no note at all) and "
            "the n gets collected. Arm once that n exists."
        ),
    )
    # NOT ARMED — the condition convicts the PROBE, not the master (n=0).
    mastering_gate_bed_calibration_enabled: bool = Field(
        default=False,
        description=(
            "ARM the bed-calibration gate (VC-W14/V6): fail the compile when a "
            "job REQUESTED VO-relative music-bed levelling or ducking "
            "(music_offset_lu / duck_open_lu_below_vo) and the narration bus "
            "could not be measured, so the bed gain AND the sidechain open "
            "threshold both fell back to literals. STRUCTURAL by shape but "
            "DEFAULT OFF: the trigger is a final_mux_loudness failure whose "
            "fleet rate is UNMEASURED (n=0), and arming it would destroy paid "
            "deliverables on a transient probe fault. It ALREADY clears "
            "gate_pass and appears in report.violations with its tier while "
            "this is false — which was the actual defect (it used to be a log "
            "line and an unread boolean) and is how the n gets collected."
        ),
    )
    # NOT ARMED — a MEASURED refusal, not caution.
    mastering_gate_colorspace_enabled: bool = Field(
        default=False,
        description=(
            "ARM the BT.709 colorspace-tag gate. Structural by nature (the tag "
            "is present or it is not) but DEFAULT OFF on measurement: over the "
            "same n=120 completed-master fleet sample (2026-08-27 -> 09-11) "
            "colorspace_ok measured FALSE on 23.3%. That is an upstream TAGGING "
            "regression to fix at its source, and arming here would fail ~1 "
            "delivery in 4 on a defect this service neither causes nor can "
            "remediate. Flip on AFTER the tagging fix lands and the rate is "
            "re-measured to ~0."
        ),
    )
    # NOT ARMED — and, unlike every sibling switch, OFF means "no violation at
    # all", not "reported but not enforced": see the description.
    mastering_gate_cfr_enabled: bool = Field(
        default=False,
        description=(
            "ARM the constant-frame-rate gate: fail the compile when the "
            "delivered master measures cfr_ok=False (r_frame_rate and "
            "avg_frame_rate disagree, the cadence sits off render_fps, or the "
            "video pts timeline has an irregular interval such as a missing "
            "frame slot — see mastering.framerate_coherence, "
            "mastering.video_pts_gaps and mastering._cfr_verdict). TUNABLE (the frame-uniformity "
            "threshold is uncited) and DEFAULT OFF. While this is false cfr "
            "stays ADVISORY exactly as before the switch existed: cfr_ok is "
            "published, 'cfr' is appended to rerun_stages, and NO violation is "
            "recorded and gate_pass is NOT cleared — so the report is "
            "byte-identical to the pre-switch one. (That differs from "
            "colorspace, which reports its violation while disarmed; changing "
            "what a disarmed cfr publishes was not the point of adding the "
            "switch.) A cfr_ok of None (probe gap) never fires in either "
            "state. Flip on by config once the fleet cfr_ok=False rate is "
            "measured at ~0."
        ),
    )
    # STRUCTURAL and default OFF. The threshold HAS a measured two-sided
    # separation (synthesized fixtures, 2026-09-16 — see
    # mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB), but the FLEET
    # distribution of program_to_noise_floor_db over real masters is n=0, and a
    # densely scored or wall-to-wall-music master could read a narrow margin
    # while being perfectly narrated. A gate must never destroy a paid
    # deliverable; it reports until the n exists.
    mastering_gate_speech_presence_enabled: bool = Field(
        default=False,
        description=(
            "ARM the speech-presence gate: fail the compile when the delivered "
            "program sits less than "
            "mastering.SPEECH_PRESENCE_MIN_PROGRAM_OVER_FLOOR_DB (1.5 dB) above "
            "the master's own astats noise floor — the shape of a master "
            "carrying only hiss and no speech, which shipped to production once "
            "and which no single-number threshold on integrated_lufs or "
            "silence_ratio can detect (hiss never trips silencedetect's -50 dB "
            "gate and normalises to the same loudness as narration). STRUCTURAL, "
            "threshold measured over fixtures, FLEET n=0. The violation is "
            "REPORTED (report.violations, tier 'structural') while this is "
            "false, which is how the n gets collected."
        ),
    )
    # NOT ARMED — STRUCTURAL, threshold measured on real production audio. #234
    # (2026-09-18) made the measure lag-tolerant + line-identity-aware (see
    # mastering.NARRATION_PRESENCE_LAG_SEARCH_SECONDS /
    # NARRATION_PRESENCE_IDENTITY_MARGIN_DB) and raised calibration to fleet n=3
    # masters / 11 measurable lines, all correctly classified — still not the
    # n=120 this project requires before arming any threshold.
    mastering_gate_narration_presence_enabled: bool = Field(
        default=False,
        description=(
            "ARM the narration-presence gate: fail the compile when a shot's "
            "SOURCE narration clip cannot be found (cross-correlated) inside "
            "the delivered master at that shot's own timeline window — the "
            "content-aware, per-shot successor to mastering_gate_speech_"
            "presence_enabled, which reads a whole-file loudness statistic that "
            "cannot tell 'every clip present' from 'one clip missing, three "
            "present' and reads a digitally silent master as clean speech. "
            "STRUCTURAL, threshold measured on real production audio (see "
            "mastering.NARRATION_PRESENCE_MIN_CORRELATION_RMS_DB), FLEET n=1. "
            "The violation is REPORTED (report.violations, tier 'structural') "
            "while this is false, which is how the n gets collected."
        ),
    )
    # TUNABLE (default OFF): a quality judgement about DEGREE, decided against a
    # ceiling whose fleet distribution is UNMEASURED. This project has twice
    # built a 'signal' by arming an uncalibrated ceiling — duplicate frames
    # convicted 120 of 120 masters, a mastering deband ceiling would have fired
    # on 96.7% — so the switch is wired and left off until the n exists.
    mastering_gate_loudness_enabled: bool = Field(
        default=False,
        description=(
            "ARM the loudness-tolerance gate: fail the compile when the "
            "delivered integrated loudness sits further than "
            "mastering.LOUDNESS_GATE_TOLERANCE_LU (3.0 LU) from master_lufs, or "
            "is unmeasurable. TUNABLE and PROVENANCE: NONE (n=0) — 3.0 LU is an "
            "EBU-R128-flavoured convention, never measured against this "
            "pipeline's masters. Measure the integrated_lufs distribution over a "
            "fleet sample and record the n at the constant BEFORE arming. The "
            "violation is still REPORTED (report.violations, tier 'tunable') "
            "while this is false, which is how the n gets collected."
        ),
    )
    # ⛔ ``render_duplicate_frame_ratio_ceiling`` WAS REMOVED 2026-09-11 and must
    # not come back. It configured a gate on the delivered master's duplicate-frame
    # ratio; duplicate frames are intentional dwell (6500ms hold x a 25fps capture),
    # and across 120 of 120 completed masters the ratio ran 0.643-0.998 against the
    # 0.10 default — i.e. the gate convicted every delivery ever made. The setting
    # went with the predicate so there is no knob left to "re-arm": re-introducing
    # the gate would need a NEW measurement that can tell dwell from a frozen
    # render, not a new ceiling. ``duplicate_frame_ratio`` is still probed and still
    # published in the mastering report as telemetry. Settings ignores unknown env
    # (``extra="ignore"``), so a deployment still setting
    # RENDER_DUPLICATE_FRAME_RATIO_CEILING is harmless. See
    # ``src/services/mastering.py`` where ``_duplicate_frames`` used to live.
    render_dead_frame_min_ratio: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of MEASURED frames that must fall below "
            "render_dead_frame_luma_spread_floor before the master is treated as "
            "dead. The gate used to fire on a SINGLE frame (min across the file), "
            "which fails an entire film on one legitimate fade-to-black or dark "
            "title card. signalstats measures EVERY frame, so at 30fps a 1s fade is "
            "~1.4% of a 72s master and a 2s fade ~2.8%, while a master blank for "
            "part of its length — the case the per-frame check exists to catch — "
            "sits far above 10% (half-blank = 50%). 0.0 restores the old "
            "any-single-frame behaviour."
        ),
    )
    render_dead_frame_luma_spread_floor: float = Field(
        default=8.0,
        ge=0.0,
        description=(
            "Below this intra-frame luma spread (YMAX-YMIN) the delivered master "
            "is treated as blank/dead (uniform frame) and the mastering soft-gate "
            "fails. A legitimate static UI screen has spread ~200; only a truly "
            "uniform/blank frame falls below ~8."
        ),
    )
    render_motion_blur: bool = Field(
        default=False,
        description=(
            "When a shot must be retimed onto the master cadence across a "
            "non-integer frame-rate ratio, use motion-compensated interpolation "
            "(minterpolate) instead of a plain frame drop/dup. Off by default so "
            "the clean fps= resample (byte-identical) is used."
        ),
    )
    render_crf: int = Field(
        default=18,
        ge=0,
        le=51,
        description=(
            "libx264 constant-rate-factor for shot encodes (lower = sharper, "
            "bigger). 18 is the near-transparent cell for the 2K/1440p canvas "
            "this fleet actually renders at — the previous 20 was chosen when "
            "the docstring itself already said 'drop toward 18 for 2K/4K', and "
            "the render canvas has been 2K since. Per-job overridable via the "
            "compile 'crf' tag."
        ),
    )
    render_intermediate_crf_delta: int = Field(
        default=6,
        ge=0,
        le=51,
        description=(
            "How far BELOW 'render_crf' an INTERMEDIATE re-encode runs. A "
            "segment that is re-encoded for picture polish (deband / "
            "source-match) is already a render_crf x264 encode, so running "
            "generation 2 at the same render_crf is pure generational loss: "
            "x264 re-quantises material that is already quantised. Encoding "
            "the intermediate ~6 CRF points lower makes that pass "
            "near-transparent to generation 1 (each 6 CRF points is roughly a "
            "2x bitrate change, i.e. the intermediate carries the detail the "
            "delivery cell would have thrown away a second time). The cost is "
            "a larger TEMP file, which never leaves /tmp on any path that "
            "re-encodes downstream. Set to 0 to restore the pre-#wave11 "
            "behaviour exactly (intermediates at the delivery cell). The "
            "DELIVERED master encode never takes this delta - render_crf "
            "alone owns the delivery cell."
        ),
    )
    render_preset: str = Field(
        default="medium",
        description=(
            "libx264 -preset for shot encodes. 'veryfast' (the old hardcoded "
            "value) is visibly soft on text/gradients; 'medium'/'slow' encode "
            "sharper at higher CPU cost. Only applies when render_encoder is "
            "libx264 (hardware encoders ignore -preset/-crf)."
        ),
    )
    render_encoder: str = Field(
        default="libx264",
        description=(
            "Video encoder for shot encodes. 'libx264' (default) is the "
            "CPU/software path and honours render_preset + render_crf. "
            "'h264_videotoolbox' (Apple media engine) and 'h264_nvenc' (NVIDIA) "
            "are hardware paths that ignore preset/CRF and use bitrate instead "
            "(render_hw_bitrate) — set one of those ONLY on a deployment whose "
            "host actually has that encoder (e.g. an Apple-silicon or NVENC "
            "host). Lets the compile offload to a hardware encoder via a single "
            "env flip without a code change."
        ),
    )
    render_hw_bitrate: str = Field(
        default="6M",
        description=(
            "Target video bitrate for the hardware encoder paths "
            "(h264_videotoolbox / h264_nvenc), which are bitrate- not CRF-driven. "
            "Unused for libx264. Raise toward 10M+ for 2K/4K."
        ),
    )
    render_colorspace: str = Field(
        default="bt709",
        description=(
            "Delivery colorspace tag written onto every per-shot encode "
            "(additive colr atom: -color_primaries/-color_trc/-colorspace). "
            "'bt709' is the HD-delivery default; 'bt601'/'bt2020'/'srgb' retarget "
            "for other pipelines; '' / 'none' disables tagging (pre-master "
            "behaviour). Per-job overridable via the mastering profile. The colr "
            "atom survives the downstream concat + audio-remux stream-copies, so "
            "no re-assert on the master mux is needed."
        ),
    )
    render_deband: str = Field(
        default="auto",
        description=(
            "Conditional deband policy for the picture master. 'auto'/'on' "
            "measure each rendered segment's signalstats and apply an ffmpeg "
            "deband pass ONLY to a shot whose smooth-gradient banding risk trips "
            "(clean shots stay a single-pass, byte-identical encode). "
            "'off'/'false' skips measurement entirely — unconditional single pass."
        ),
    )
    render_source_match: str = Field(
        default="auto",
        description=(
            "Per-shot source-matching policy ('auto'/'on' active, 'off' disabled): "
            "a post-render pass equalizes each shot's exposure toward the film's "
            "median luma so mixed sources feel like one camera. Gated additionally "
            "on the per-request finish.grade.match_sources."
        ),
    )
    source_match_tolerance: float = Field(
        default=6.0,
        description=(
            "Max YAVG (0-255) a shot may deviate from the film median before "
            "source-matching corrects it — the ±6/255 'one camera' AC."
        ),
    )
    render_bitrate_mode: str = Field(
        default="auto",
        description=(
            "Rate-control mode for the CRF/bitrate ladder. 'auto' (default) is "
            "pure CRF (render_crf) with NO VBV caps — byte-identical to the "
            "pre-master encode. 'vbv'/'cbr'/'capped' emit -maxrate/-bufsize from "
            "render_vbv_maxrate/render_vbv_bufsize for a hard-capped delivery."
        ),
    )
    render_vbv_maxrate: str = Field(
        default="",
        description=(
            "VBV peak bitrate (-maxrate, e.g. '12M') for a capped delivery. Only "
            "emitted when render_bitrate_mode selects a VBV mode; empty = no cap "
            "(the default), so nothing is added to the encode args."
        ),
    )
    render_vbv_bufsize: str = Field(
        default="",
        description=(
            "VBV buffer size (-bufsize, e.g. '24M'). Only emitted alongside "
            "render_vbv_maxrate under a VBV mode; empty derives ~2x maxrate."
        ),
    )
    render_output_platform: str = Field(
        default="youtube",
        description=(
            "Target delivery platform for the mastering profile (reporting / "
            "future platform-specific ladder tuning). Reported on the master "
            "tags; does not by itself change the default encode."
        ),
    )
    master_enabled: bool = Field(
        default=True,
        description=(
            "Apply a compressor+limiter+loudnorm master to the FINAL audio of "
            "every compile (bed path folds it into overlay_audio_bed; bed-less "
            "path gets a dedicated audio-only pass). Lifts films to a dense, "
            "platform-competitive loudness instead of the ~-22 LUFS raw mix. "
            "Default-ON fleet-wide; disable per-job via the 'mastering' tag "
            "(=off/false/0) or globally via env."
        ),
    )
    master_lufs: float = Field(
        default=-14.0,
        description=(
            "loudnorm integrated-loudness target (I), LUFS, for the final "
            "master. -14 is the streaming-platform reference (YouTube/Spotify "
            "both normalise TO it, so anything hotter is turned back DOWN on "
            "playback and only the extra compression survives). The previous "
            "-12.5 asked a bus that already arrives at -18 LUFS per narration "
            "clip for another ~5.5 dB of gain riding, which is what made the "
            "narration breathe. Per-job overridable via the 'master_lufs' tag."
        ),
    )
    master_tp: float = Field(
        default=-1.0, description="loudnorm true-peak ceiling (TP), dBTP, for the final master"
    )
    master_lra: float = Field(
        default=11.0,
        description=(
            "loudnorm target loudness range (LRA) for the final master. 11 LU "
            "is EBU R128's normal-range figure; the previous 7 asked loudnorm "
            "to squeeze the film's dynamics into a range narrower than speech "
            "naturally occupies, on top of the bus compressor. In the two-pass "
            "linear mode this chain now runs, LRA is the range the measurement "
            "is judged against rather than a per-sample squeeze."
        ),
    )
    # --- Final-master bus dynamics (the compressor + limiter around loudnorm) --
    # Promoted from hardcoded filter-string literals so the bus can be tuned
    # without a code change. Narration reaches this bus ALREADY normalised to
    # -18 LUFS per clip by tts_backend, so the compressor is a gentle glue stage,
    # not the leveller it was built as.
    master_compressor_threshold_db: float = Field(
        default=-24.0,
        description="acompressor threshold (dB) on the final master bus.",
    )
    master_compressor_ratio: float = Field(
        default=2.0,
        ge=1.0,
        le=20.0,
        description=(
            "acompressor ratio on the final master bus. Softened from 3 to 2: "
            "the bus is fed pre-normalised narration, so 3:1 was squashing "
            "already-levelled speech."
        ),
    )
    master_compressor_makeup: float = Field(
        default=1.0,
        ge=1.0,
        le=64.0,
        description=(
            "acompressor makeup gain on the final master bus. LINEAR, not dB — "
            "the previous 2 was +6 dB of blind boost ahead of loudnorm, which "
            "loudnorm then had to ride back off. 1 is unity."
        ),
    )
    master_limiter_attack_ms: float = Field(
        default=5.0,
        gt=0.0,
        description="alimiter attack (ms) on the final master bus.",
    )
    master_limiter_release_ms: float = Field(
        default=50.0,
        gt=0.0,
        description="alimiter release (ms) on the final master bus.",
    )

    # --- Audiobook ACX program master (A12 seam) -------------------------------
    # These drive the compile-tail ACX loudness master on the concat-audio path
    # ONLY when a request carries an ``acx_master`` block. They never touch the
    # video ``_master_filter`` seam above (which stays -12.5 LUFS for film), and
    # a concat-audio request WITHOUT ``acx_master`` is byte-identical to today.
    render_audiobook_lufs_target: float = Field(
        default=-20.0,
        description=(
            "loudnorm integrated-loudness target (I), LUFS, for the audiobook "
            "program master. -20 sits in the middle of the ACX -18..-23 window."
        ),
    )
    render_audiobook_lufs_min: float = Field(
        default=-23.0, description="ACX integrated-loudness floor (LUFS); acx_report.pass gate"
    )
    render_audiobook_lufs_max: float = Field(
        default=-18.0, description="ACX integrated-loudness ceiling (LUFS); acx_report.pass gate"
    )
    render_audiobook_tp: float = Field(
        default=-3.0, description="ACX true-peak ceiling (TP), dBTP; acx_report.pass gate"
    )
    render_audiobook_lra: float = Field(
        default=11.0,
        description="loudnorm LRA for the spoken-word audiobook master (wider than film's 7)",
    )
    render_audiobook_noise_floor_db: float = Field(
        default=-60.0, description="ACX max measured noise floor (dB RMS); acx_report.pass gate"
    )

    render_bg_color: str = Field(
        default="0x111827", description="Background fill, AARRGGBB or 0xRRGGBB"
    )
    render_letterbox_color: str = Field(
        default="",
        description=(
            "Brand letterbox/pillarbox fill (VC-5). Overrides render_bg_color for "
            "the pad/canvas colour ONLY; empty (the default) keeps render_bg_color, "
            "so an unset value is byte-identical. Per-job overridable via the "
            "request-level `finish` profile, which validates the hex shape."
        ),
    )
    render_text_color: str = Field(default="white")
    render_font: str = Field(
        default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        description="Path to a TrueType font visible inside the container",
    )

    caption_font_family: str = Field(
        default="DejaVu Sans",
        description=(
            "Font FAMILY name (not a path) libass resolves for burned-in captions "
            "(VC-6). Must be a family the container's fontconfig can find "
            "(fonts-dejavu-core ships 'DejaVu Sans')."
        ),
    )
    caption_font_size_frac: float = Field(
        default=0.045,
        gt=0,
        description="Caption font size as a fraction of render_height (VC-6).",
    )
    caption_margin_v_frac: float = Field(
        default=0.06,
        ge=0,
        description="Caption bottom margin as a fraction of render_height (VC-6).",
    )
    caption_drawtext_font_size_frac: float = Field(
        default=44 / 1080,
        gt=0,
        description=(
            "Per-shot drawtext caption font size as a fraction of render_height. "
            "Same pattern as caption_font_size_frac (the film-level ASS caption "
            "system) but a separate key: the drawtext overlay is a visually "
            "distinct system with its own historical pixel size, and this default "
            "is chosen so render_height=1080 reproduces that exact legacy value "
            "(44px) rather than the ASS system's own 1080p size (49px)."
        ),
    )
    caption_drawtext_margin_v_frac: float = Field(
        default=60 / 1080,
        ge=0,
        description=(
            "Per-shot drawtext caption bottom margin as a fraction of "
            "render_height. Same pattern as caption_margin_v_frac; kept as a "
            "separate key for the same reason as caption_drawtext_font_size_frac "
            "(reproduces the legacy 60px value at render_height=1080)."
        ),
    )
    caption_outline_px: int = Field(
        default=2,
        ge=0,
        description="Caption text outline thickness in pixels (VC-6).",
    )
    caption_primary: str = Field(
        default="&H00FFFFFF",
        description="Caption primary (fill) colour in ASS &HAABBGGRR form (VC-6).",
    )
    caption_outline_colour: str = Field(
        default="&H00000000",
        description="Caption outline colour in ASS &HAABBGGRR form (VC-6).",
    )
    caption_secondary: str = Field(
        default="&H00808080",
        description=(
            "Caption karaoke un-swept (SecondaryColour) colour in ASS &HAABBGGRR "
            "form — dim mid-grey; inert on non-karaoke renders (VC-6)."
        ),
    )
    caption_box_enabled: bool = Field(
        default=True,
        description=(
            "Draw a semi-opaque backplate behind every burned-in ASS caption line "
            "(libass BorderStyle=3, an opaque box per line). A thin glyph outline "
            "alone is illegible over dense UI footage (showcase spike run "
            "01a0fd4d-9948, 13s frame: white text sat directly on app text). "
            "False restores the bare-outline look."
        ),
    )
    caption_box_colour: str = Field(
        default="&H50000000",
        description=(
            "Caption backplate colour in ASS &HAABBGGRR form. Alpha 0x50 = ~69% "
            "opaque black: dark enough to hold white text over any UI, light "
            "enough that the footage still reads through."
        ),
    )
    caption_box_padding_frac: float = Field(
        default=0.25,
        ge=0,
        description=(
            "Backplate padding around the text as a fraction of the caption font "
            "size (BorderStyle=3 draws the box Outline px beyond the glyphs). "
            "Scales with render_height via the font size, so the plate keeps the "
            "same proportions at 1080p and 4k."
        ),
    )
    caption_max_line_chars: int = Field(
        default=42,
        ge=1,
        description="Greedy wrap width (chars) before a caption line breaks (VC-6).",
    )

    work_dir: str = Field(
        default="/tmp/video-compiler",  # noqa: S108  # nosec B108 — ephemeral container scratch dir
        description="Scratch dir for per-request ffmpeg work; cleared after each compile",
    )

    shot_cache_max_bytes: int = Field(
        default=1024**3,  # 1 GiB
        ge=0,
        description=(
            "Soft size cap on the content-addressed per-shot render cache "
            "(work_dir/shot-render-cache), which — unlike the per-job scratch — "
            "survives across compiles so the orchestrator's re-compile loop only "
            "re-renders changed shots. Before publishing a segment the cache evicts "
            "oldest entries (by mtime) until under this size, so it can't grow "
            "unbounded and fill work_dir on the long-lived single pod. 0 disables "
            "the cache (get + put become no-ops). Tune via gitops env."
        ),
    )

    max_concurrent_ffmpeg_subprocesses: int = Field(
        default=6,
        ge=1,
        description=(
            "Per-pod cap on concurrent ffmpeg/ffprobe subprocesses. This is the "
            "intra-job parallelism lever: shots within one compile fan out and "
            "encode concurrently up to this many at a time (across all in-flight "
            "jobs too), so a K-shot compile takes ~ceil(K/N) encode-waves instead "
            "of K serial encodes. Each concurrent full-resolution libx264 encode "
            "holds ~1GiB of decode/encode buffers, so this MUST stay in lockstep "
            "with the pod memory limit (≈1GiB headroom per slot) or ffmpeg "
            "OOM-kills (exit=-9). Default 6 pairs with the 12Gi/6-core pod; tune "
            "via gitops env."
        ),
    )

    max_concurrent_compile_jobs: int = Field(
        default=8,
        ge=1,
        description=(
            "Per-pod cap on simultaneously-accepted compile/concat jobs. Each "
            "accepted job spawns a background render that holds a scratch dir "
            "under work_dir; an unbounded burst of submits would fill /tmp and "
            "OOM the pod. Once this many jobs are 'processing', further submits "
            "get a 429 (caller retries). The reaper's re-dispatch path enforces "
            "this same cap (see reaper_service.reap_once), so crash-recovery can't "
            "over-admit past it. Single-replica capacity lever — the render itself "
            "is an in-process asyncio task pinned to this pod, even though job "
            "STATE is durable in Postgres."
        ),
    )
    job_work_timeout_seconds: int = Field(
        default=3600,
        ge=60,
        description=(
            "Hard wall-clock ceiling for a single compile/concat render before "
            "it is force-failed. Below the orchestrator's compile wait ceiling "
            "(7500s) and its presign TTL (7800s) so a healthy render never trips "
            "it; exists so a wedged ffmpeg can't leave a job 'processing' forever."
        ),
    )
    ffmpeg_subprocess_timeout_seconds: int = Field(
        default=1800,
        ge=30,
        description=(
            "Per-ffmpeg-subprocess wall-clock deadline; below "
            "job_work_timeout_seconds so one wedged subprocess dies before the "
            "whole job."
        ),
    )

    # --- Crash-resumption reaper (durable dispatch) ---
    reaper_enabled: bool = Field(
        default=True,
        description=(
            "Run the per-pod reaper that reclaims crash-orphaned compile jobs (a "
            "row whose owning pod died mid-render) and re-runs them from the "
            "persisted job_options. Also drives periodic prune. Disable only to "
            "pin the pre-reaper behaviour (a killed render stays stuck until an "
            "orchestrator resend)."
        ),
    )
    reaper_interval_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Seconds between reaper reclaim/prune sweeps.",
    )
    job_lease_seconds: int = Field(
        default=120,
        ge=15,
        description=(
            "Claim-lease TTL for an in-flight render. A live render renews it every "
            "lease/3; a row still 'processing' past its lease (owning pod died) is "
            "reclaimable. Comfortably above one reaper interval + a render's slowest "
            "heartbeat gap so a healthy render is never reclaimed under itself."
        ),
    )
    reclaim_max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Poison-job cap: a job reclaimed this many times without completing is "
            "hard-failed instead of re-run (a crash-looping render can't be reclaimed "
            "and re-billed forever). A reclaim of a row a graceful drain interrupted "
            "(interrupt_reason='drain') is charged ZERO attempts by "
            "reclaim_next_compile_job, so N rollouts do not spend N of these lives; the "
            "wall-clock reclaim_max_lifetime_seconds is what bounds that path."
        ),
    )
    reclaim_max_lifetime_seconds: int = Field(
        default=7200,
        ge=1,
        description=(
            "Wall-clock ceiling on a job's total RUNNING life across reclaims "
            "(anchored on first start). A job running longer is failed rather than "
            "reclaimed. CANNOT BE DISABLED (ge=1, was ge=0/'0 = off'): since a reclaim "
            "of a drain-interrupted row is charged ZERO attempts, the attempt cap does "
            "NOT bound a drain->rerun loop and this wall clock is the ONLY backstop "
            "left. At 0 a rollout-flapping paid render was reclaimed and re-billed "
            "forever with attempts stuck at 0. The reclaim SQL still treats 0 as "
            "'disabled' (tests pass it directly); this field simply cannot supply it."
        ),
    )
    prune_interval_seconds: float = Field(
        default=1800.0,
        gt=0,
        description=(
            "Seconds between periodic prune sweeps (flip zombie 'processing' + evict "
            "aged terminal rows) on a long-lived pod. Startup also prunes once."
        ),
    )
    max_fetch_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=0,
        description=(
            "Max DECODED bytes for a single streamed shot-asset fetch; 0 = unlimited. "
            "Raised 512 MiB -> 4 GiB for tenant drone footage. Safe to raise only "
            "because the fetch streams to disk a chunk at a time (safe_get_to_path) "
            "instead of buffering the body in RAM -- the old buffered path peaked at "
            "~2x the file, so a 4 GiB source would have cost ~8 GiB of pod RSS per "
            "concurrent fetch against a 12Gi limit. The cap is counted mid-stream, so "
            "an oversized (or gzip-bombed) asset aborts before it is on disk."
        ),
    )
    allow_local_file_inputs: bool = Field(
        default=False,
        description=(
            "Accept file:// and bare-path shot/clip inputs at submit. Defaults FALSE "
            "(secure-by-default): a hostile image_url/audio_url like 'file:///etc/passwd' "
            "is rejected with 400 instead of being read into the render (LFI guard). "
            "Tests and local dev that submit file:// fixtures opt in via "
            "ALLOW_LOCAL_FILE_INPUTS=true; prod leaves it unset so the guard is armed "
            "without depending on a gitops env being remembered."
        ),
    )

    metering_redis_url: str = Field(
        default="",
        description=(
            "Redis URL for the cost-bus cost-event stream (a SEPARATE "
            "Redis from any service-local cache). Empty = metering disabled "
            "(emits silently no-op). Per-minute billing of compiled output; "
            "supplied via gitops-production env."
        ),
        validation_alias="METERING_REDIS_URL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
