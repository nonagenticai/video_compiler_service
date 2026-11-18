"""Guard: no ffmpeg argv in ``src/`` may carry the removed ``-vsync`` option.

``-vsync`` was deprecated in ffmpeg 5.1 and current ffmpeg REJECTS it (Homebrew
9.x failed every render). CI's apt ffmpeg (6.1) and the image's (7.1) still
accept it, so nothing in CI would notice a reintroduction until a base-image
bump silently broke production renders. ``-fps_mode[:v]`` (since 5.1) is the
replacement with identical semantics.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_no_vsync_string_literal_in_src() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # an argv element IS the option token (``"-vsync"`` / ``"-vsync:v"``);
            # prose in docstrings/comments is not flagged.
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (node.value == "-vsync" or node.value.startswith("-vsync:"))
            ):
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
    assert offenders == [], f"removed ffmpeg option -vsync used at: {offenders}"
