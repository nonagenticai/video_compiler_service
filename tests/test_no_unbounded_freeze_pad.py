"""CI-enforced guard: an UNBOUNDED ``tpad`` clone must never reappear in ``src/``.

THE DEFECT (#232): ``tpad=stop_mode=clone:stop=-1`` asks ffmpeg to clone the last
frame FOREVER, relying entirely on an external ``-t`` to ever terminate the
stream. On ffmpeg 7.1.5 this livelocked a real production render — a cover-fit
video input a few milliseconds short of its narration-driven segment target,
paired with a second finite audio input in the same graph, froze at 71M dropped
frames and did not respond to SIGTERM.

The fix (``renderer._cover_pad_stop_duration`` / ``_bounded_cover_pad``, and the
split-screen half in ``_split_filter``, which used to carry its own independent
``stop=-1``) replaces every clone with a FINITE ``stop_duration`` ceiling'd at
the segment's own known duration. This is a plain source-text ratchet, not an
AST walk: the string ``stop=-1`` has no legitimate spelling anywhere in
``src/`` — not in an f-string, not in a comment describing the current
behaviour (history/docstring references spell it with a space, ``stop = -1``,
for exactly this reason) — so a bare substring scan is sufficient and cannot be
quietly defeated by, say, building the string via concatenation without also
showing up as this exact substring in the constructed ffmpeg arg.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

_FORBIDDEN = "stop=-1"


def test_no_unbounded_tpad_clone_anywhere_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _FORBIDDEN in line:
                offenders.append(f"{path.relative_to(_SRC.parent)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "an unbounded `tpad=stop_mode=clone:stop=-1` (or a reference to the literal "
        "sentinel) reappeared in src/ — this is the exact filter shape that livelocked "
        "ffmpeg 7.1.5 in production (#232). Every last-frame clone must be FINITE, sized "
        "off the segment's own known duration via `renderer._cover_pad_stop_duration` / "
        "`_bounded_cover_pad`:\n" + "\n".join(offenders)
    )


def test_guard_pattern_is_not_vacuous() -> None:
    """Self-check: prove the scanner actually matches the forbidden string."""
    assert _FORBIDDEN in "tpad=stop_mode=clone:stop=-1"
