"""Every Streamlit chapter page must run without blowing up.

This uses Streamlit's own test harness, which really executes the page script and
captures any exception — a plain HTTP 200 would not tell us anything useful.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from kidsml.ui import CHAPTERS, page_filename

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "app" / "pages"

# Some chapters train a small model on first render, so give them room.
TIMEOUT_SECONDS = 180

existing = sorted(p for p in PAGES.glob("[0-9][0-9]_*.py"))


def test_at_least_one_page_exists():
    assert existing, "no chapter pages found in app/pages/"


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_page_runs_without_error(page: Path):
    at = AppTest.from_file(str(page), default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception, f"{page.name} raised: {[e.message for e in at.exception]}"


def test_home_runs_without_error():
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda c: f"ch{c[0]:02d}")
def test_page_is_named_as_the_course_map_says(chapter):
    """The sidebar order comes from the filenames, so they must match kidsml.ui."""
    number = chapter[0]
    expected = PAGES / page_filename(number)
    if not any(p.name.startswith(f"{number:02d}_") for p in existing):
        pytest.skip(f"chapter {number:02d} not built yet")
    assert expected.exists(), f"expected page file {expected.name}"
