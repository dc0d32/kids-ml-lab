"""Turn the chapter sources in ``notebooks/_src/*.py`` into real ``.ipynb`` notebooks.

Chapters are *written* as plain Python files in jupytext's "percent" format, because
plain Python is easy to review and diff. Kids never see those files — they open the
generated ``.ipynb`` in JupyterLab.

    ./run.sh build              # rebuild every notebook
    ./run.sh build 13 14        # rebuild only chapters 13 and 14
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def sources(selection: list[str]) -> list[Path]:
    """All chapter sources, or just the ones whose number was asked for."""
    all_src = sorted(SRC.glob("[0-9][0-9]_*.py"))
    if not selection:
        return all_src
    wanted = {f"{int(s):02d}" for s in selection}
    return [p for p in all_src if p.name[:2] in wanted]


def build(path: Path) -> Path:
    notebook = jupytext.read(path, fmt="py:percent")
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    target = OUT / (path.stem + ".ipynb")
    jupytext.write(notebook, target, fmt="ipynb")
    return target


def main(argv: list[str]) -> int:
    todo = sources(argv)
    if not todo:
        print("No chapter sources found in", SRC)
        return 1
    for path in todo:
        target = build(path)
        print(f"  {path.name}  ->  notebooks/{target.name}")
    print(f"Built {len(todo)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
