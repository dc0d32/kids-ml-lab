"""Every chapter notebook must execute cleanly, and quickly.

The "quickly" part is a house rule, not a nicety: if a chapter stops fitting in the
budget, the *dataset* shrinks — never the explanation.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

# Wall-clock seconds. Chapters that train something get more room, but not much.
DEFAULT_BUDGET = 60
BUDGETS = {
    16: 120,   # deeper nets, several trainings
    17: 120,   # PyTorch import alone is slow
    18: 120,   # digits MLP
    19: 240,   # the small CNN
    21: 120,   # PCA over all the digits
    23: 180,   # MLP language model
    24: 300,   # the tiny Transformer
}

notebooks = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))


def budget_for(path: Path) -> int:
    return BUDGETS.get(int(path.name[:2]), DEFAULT_BUDGET)


def test_at_least_one_notebook_exists():
    assert notebooks, "no notebooks found — run ./run.sh build"


@pytest.mark.parametrize("path", notebooks, ids=lambda p: p.stem)
def test_notebook_runs_within_budget(path: Path):
    nb = nbformat.read(path, as_version=4)
    budget = budget_for(path)

    # Tell the notebooks nobody is watching, so ipywidgets does not sit waiting
    # for a browser that will never connect.
    os.environ["KIDSML_HEADLESS"] = "1"

    started = time.perf_counter()
    NotebookClient(nb, timeout=budget * 2, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    elapsed = time.perf_counter() - started

    assert elapsed < budget, (
        f"{path.name} took {elapsed:.0f}s, over its {budget}s budget. "
        "Shrink the dataset or the number of training steps — not the explanation."
    )


@pytest.mark.parametrize("path", notebooks, ids=lambda p: p.stem)
def test_notebook_is_in_sync_with_its_source(path: Path):
    """The .ipynb files are generated. A stale one means someone forgot ./run.sh build."""
    import jupytext

    source = NOTEBOOKS / "_src" / (path.stem + ".py")
    assert source.exists(), f"missing source for {path.name}"

    fresh = jupytext.read(source, fmt="py:percent")
    current = nbformat.read(path, as_version=4)

    fresh_src = [c.source for c in fresh.cells]
    current_src = [c.source for c in current.cells]
    assert fresh_src == current_src, (
        f"{path.name} is out of date with notebooks/_src/{source.name} — run ./run.sh build"
    )


@pytest.mark.parametrize("path", notebooks, ids=lambda p: p.stem)
def test_notebook_offers_its_workbook(path: Path):
    """A notebook reader should get the same questions the app reader gets."""
    source = NOTEBOOKS / "_src" / (path.stem + ".py")
    chapter = int(path.name[:2])
    assert f"workbook.render({chapter})" in source.read_text(encoding="utf-8"), (
        f"{source.name} never calls workbook.render({chapter})"
    )
