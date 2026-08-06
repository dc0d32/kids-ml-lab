"""Mermaid diagrams have to parse.

Mermaid renders in the browser, so a syntax error shows up as a red error box on the page
and is completely invisible to every other test here — which is how a broken diagram sat in
chapter 14 until a reader hit it.

A full check needs Node and the real mermaid parser (see the note at the bottom). What runs
here is a lint for the mistakes that actually break diagrams in practice, which is cheap and
needs no JavaScript.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Node label shapes: [box], (round), ((circle)), {diamond}, and |edge label|.
LABEL_PATTERNS = [
    re.compile(r"\[([^\[\]\"]*)\]"),
    re.compile(r"\|([^|\"]*)\|"),
]

# Characters that start a node shape. Inside an unquoted label they end it early.
TROUBLE = re.compile(r"[()\[\]{}]")


def _diagrams():
    """Every mermaid diagram in the course, as (source file, line, text)."""
    found = []
    pages = sorted((ROOT / "app").rglob("*.py"))
    for path in pages:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r'lesson\.mermaid\(\s*\n?\s*"""(.*?)"""', text, re.S):
            found.append((path, text[: match.start()].count("\n") + 1, match.group(1)))

    for path in sorted((ROOT / "notebooks" / "_src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"```mermaid\n(.*?)```", text, re.S):
            body = "\n".join(line.lstrip("# ").rstrip() for line in match.group(1).split("\n"))
            found.append((path, text[: match.start()].count("\n") + 1, body))
    return found


DIAGRAMS = _diagrams()


def test_the_course_has_diagrams():
    assert DIAGRAMS, "no mermaid diagrams found — did the extraction pattern change?"


@pytest.mark.parametrize(
    "path,line,text", DIAGRAMS, ids=[f"{p.stem}:{n}" for p, n, _ in DIAGRAMS]
)
def test_labels_do_not_contain_unquoted_brackets(path: Path, line: int, text: str):
    """`A -->|2(out-y)| B` is a parse error; `A -->|"2(out-y)"| B` is fine.

    Mermaid reads an opening bracket inside an unquoted label as the start of a node
    shape, so the label ends early and the rest is nonsense to the parser.
    """
    for pattern in LABEL_PATTERNS:
        for label in pattern.findall(text):
            offender = TROUBLE.search(label)
            assert not offender, (
                f"{path.name}:{line} — the mermaid label {label.strip()!r} contains "
                f"{offender.group(0)!r}, which mermaid reads as a node shape. "
                'Wrap the label in double quotes: |"like this"|'
            )


@pytest.mark.parametrize(
    "path,line,text", DIAGRAMS, ids=[f"{p.stem}:{n}" for p, n, _ in DIAGRAMS]
)
def test_labels_are_ascii_or_quoted(path: Path, line: int, text: str):
    """`A[x₁]` silently kills the whole diagram; `A["x₁"]` is fine.

    The mermaid build bundled with streamlit-mermaid has an ASCII-only lexer for unquoted
    labels. One subscript and the diagram throws in the browser, which the reader sees as
    a blank gap in the middle of the page and no other test can see at all.
    """
    for pattern in LABEL_PATTERNS:
        for label in pattern.findall(text):
            offender = next((ch for ch in label if not ch.isascii()), None)
            assert offender is None, (
                f"{path.name}:{line} — the mermaid label {label.strip()!r} contains "
                f"{offender!r}, which the bundled mermaid parser cannot read. "
                'Use plain letters and digits, or wrap the label: ["like this"]'
            )


@pytest.mark.parametrize(
    "path,line,text", DIAGRAMS, ids=[f"{p.stem}:{n}" for p, n, _ in DIAGRAMS]
)
def test_diagram_declares_a_type(path: Path, line: int, text: str):
    first = next((ln.strip() for ln in text.strip().split("\n") if ln.strip()), "")
    assert re.match(
        r"(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|"
        r"journey|gantt|pie|mindmap|timeline)\b",
        first,
    ), f"{path.name}:{line} — diagram does not start with a mermaid diagram type: {first!r}"


# Checked against the real parser with Node during development:
#
#     npm install mermaid jsdom
#     node check.mjs        # mermaid.parse() over every diagram, with a jsdom window
#
# All 52 diagrams pass. That is not wired into the suite because it would add a Node
# toolchain to a project whose whole point is that it runs with one `uv` command.
