"""The final-audio master chain: filter ORDER, the true-peak ceiling, and codec bits.

Every assertion here is anchored to a measurement taken from a real delivered
master (run ``01a08f66-7e66-712d-a80e-74ad92f62991``), not to a preference:

* the shipped chain put ``alimiter`` BEFORE ``loudnorm``, where it cannot bound
  the delivered peak — loudnorm adds ~5.5 dB after it and its own true-peak
  limiter is advisory in single-pass dynamic mode. Reconstructed on that run's
  own narration, the shipped order landed **-0.15 dBTP against a -1.0 ceiling**;
  with the limiter last, **-0.90 dBTP** at unchanged loudness (-12.55 LUFS) and
  unchanged range (2.70 LU).
* ``alimiter``'s ``level`` option DEFAULTS TO TRUE and auto-normalizes back to
  full scale, so ``limit=`` bounds nothing while it is on: a +12 dB signal
  through ``alimiter=limit=0.95`` measured peak **1.0000**, and **0.9500** only
  with ``level=disabled``.
* ``loudnorm`` emits 192 kHz whatever its input, so the limiter must be preceded
  by a rate/layout pin or it does its true-peak work at the wrong rate.
* the AAC encode carried no ``-b:a``: one generation measured 39.39 dB SNR at
  ~126 kbps, against 50.14 dB at ``192k`` — and narration is encoded 2-3 times.
"""

from __future__ import annotations

import re

from src.config import Settings, get_settings
from src.services import renderer


def _settings(**overrides: object) -> Settings:
    return get_settings().model_copy(update=overrides)


def test_limiter_runs_after_loudnorm() -> None:
    """The ordering is the whole defect: a limiter before loudnorm bounds nothing."""
    chain = renderer._master_filter(_settings())
    assert chain, "master chain should be enabled by default"
    assert chain.index("loudnorm=") < chain.index("alimiter="), (
        f"alimiter must come AFTER loudnorm, got: {chain}"
    )


def test_limiter_disables_auto_level() -> None:
    """`level` defaults TRUE, which re-normalizes to full scale and voids `limit`."""
    chain = renderer._master_filter(_settings())
    limiter = chain[chain.index("alimiter=") :]
    assert "level=disabled" in limiter, f"alimiter must disable auto-level, got: {limiter}"


def test_limiter_ceiling_is_derived_from_master_tp() -> None:
    """The configured dBTP must be the value actually enforced, not a hardcoded 0.95.

    ⚠️ The ceiling now sits `_LIMITER_TP_HEADROOM_DB` BELOW `master_tp`, and that
    is the point of the constant rather than drift: `alimiter`'s `limit` is a
    SAMPLE-peak bound while the mastering gate measures TRUE peak, which is always
    >= it. Aiming the sample ceiling AT `master_tp` delivered -0.93/-0.87 dBTP
    against a -1.0 gate, i.e. every master failing by 0.07 dB. What must hold is
    that the ceiling TRACKS `master_tp` — that a per-job dBTP is still the value
    being enforced — not that it equals it exactly."""
    chain = renderer._master_filter(_settings(master_tp=-3.0))
    m = re.search(r"alimiter=limit=([0-9.]+)", chain)
    assert m, f"no alimiter limit in: {chain}"
    expected = 10.0 ** ((-3.0 - renderer._LIMITER_TP_HEADROOM_DB) / 20.0)
    assert abs(float(m.group(1)) - expected) < 1e-4, (
        f"limit should track master_tp, got {m.group(1)}"
    )
    # And it must be BELOW the configured true-peak ceiling, not at or above it.
    assert float(m.group(1)) < 10.0 ** (-3.0 / 20.0)


def test_rate_and_layout_pinned_before_the_limiter() -> None:
    """loudnorm emits 192 kHz; the limiter must not do true-peak work there."""
    chain = renderer._master_filter(_settings())
    pin = chain.index("aformat=sample_rates=48000")
    assert chain.index("loudnorm=") < pin < chain.index("alimiter="), (
        f"aformat must sit between loudnorm and alimiter, got: {chain}"
    )


def test_master_filter_empty_when_disabled() -> None:
    assert renderer._master_filter(_settings(master_enabled=False)) == ""


def test_audio_encode_sets_an_explicit_bitrate() -> None:
    """Native AAC at its own default measured 39.39 dB SNR; 192k measured 50.14 dB."""
    args = renderer._audio_codec_args()
    assert "-b:a" in args, f"audio encode must pin a bitrate, got: {args}"
    assert args[args.index("-b:a") + 1] == "192k"


# ---------------------------------------------------------------------------
# Scaler quality. Shots routinely arrive UNDER the canvas — a real delivered
# demo_video recorded 1920x1080 and mastered at 2560x1440 — and swscale's unset
# default is bicubic. Measured on that master's own source frame, variance-of-
# Laplacian for the 1920 -> 2560 upscale: bilinear 28.22, bicubic 43.33,
# spline 45.83, lanczos 47.57.
# ---------------------------------------------------------------------------


def test_scale_pad_filter_uses_lanczos_in_both_fit_modes() -> None:
    for fit in ("contain", "cover"):
        chain = renderer._scale_pad_filter(_settings(), None, fit=fit)
        assert "scale=" in chain, f"{fit}: no scale in {chain}"
        for frag in chain.split(","):
            if frag.startswith("scale="):
                assert "flags=lanczos" in frag, f"{fit}: scale must pin a scaler, got: {frag}"


def test_no_scale_in_this_module_rides_the_bicubic_default() -> None:
    """A new `scale=` added without flags silently reverts to bicubic — catch it here."""
    import inspect

    lines = inspect.getsource(renderer).splitlines()
    bare = []
    for i, line in enumerate(lines):
        if 'f"scale=' not in line:
            continue
        # A filter string may be split across adjacent literal lines, with the
        # flags on a continuation (see the motion-graphics "fit" branch), so the
        # window — not the single line — is what must carry them.
        window = "".join(lines[i : i + 3])
        # ``_sws_flags(settings)`` is the colour-matrix-aware wrapper around the
        # ``_SWS_FLAGS`` constant (VC colour-matrix fix) — either spelling pins a
        # real scaler; only a genuinely bare ``scale=`` should trip this.
        if "_SWS_FLAGS" not in window and "_sws_flags(" not in window:
            bare.append(line.strip())
    assert not bare, f"scale= filters without _SWS_FLAGS/_sws_flags: {bare}"
