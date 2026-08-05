"""Every Streamlit chapter page must run without blowing up — on every step.

Chapters are stepped, one idea per screen, so rendering the first screen proves almost
nothing. These tests click all the way through each chapter.

This uses Streamlit's own test harness, which really executes the page script and
captures any exception. A plain HTTP 200 would not tell us anything useful.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from kidsml.ui import CHAPTERS, page_filename

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "app" / "pages"

# Some chapters train a small model on first render, so give them room.
TIMEOUT_SECONDS = 240

# A chapter with more screens than this has almost certainly lost the reader.
MAX_STEPS = 30

existing = sorted(p for p in PAGES.glob("[0-9][0-9]_*.py"))


def test_at_least_one_page_exists():
    assert existing, "no chapter pages found in app/pages/"


def _next_button(at):
    for button in at.button:
        if "Next" in button.label:
            return button
    return None


def _walk(page: Path):
    """Click Next until it runs out. Returns (app, number of steps seen)."""
    at = AppTest.from_file(str(page), default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception, f"{page.name} step 1 raised: {[e.message for e in at.exception]}"

    steps = 1
    while steps <= MAX_STEPS:
        button = _next_button(at)
        if button is None or button.disabled:
            return at, steps
        button.click().run()
        steps += 1
        assert not at.exception, (
            f"{page.name} step {steps} raised: {[e.message for e in at.exception]}"
        )

    pytest.fail(f"{page.name} has more than {MAX_STEPS} steps — split it or trim it")


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_every_step_of_the_page_runs(page: Path):
    _walk(page)


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_page_is_broken_into_steps(page: Path):
    """One idea per screen. A page with two steps is still a wall."""
    _, steps = _walk(page)
    assert steps >= 5, f"{page.name} has only {steps} step(s) — that is still a wall of page"


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


# ---------------------------------------------------------------------------
# Shape of a chapter, checked by reading the source rather than running it
# ---------------------------------------------------------------------------

import re  # noqa: E402

BEAT_ORDER = ["hook", "byhand", "seeit", "play", "forreal", "challenge"]


def _beats_of(page: Path) -> list[str]:
    return re.findall(r'@lesson\.step\([^)]*beat="(\w+)"', page.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_steps_run_through_the_beats_in_order(page: Path):
    """Hook, then by hand, then see it, then play, then for real, then challenge.

    Several steps may share a beat, but a chapter must never go backwards — a reader
    who has reached 'In real code' should not be dropped back into 'Your turn'.
    """
    beats = _beats_of(page)
    assert beats, f"{page.name} declares no beats"

    positions = [BEAT_ORDER.index(b) for b in beats]
    assert positions == sorted(positions), (
        f"{page.name} jumps back through the beats: {beats}"
    )


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_chapter_asks_before_it_tells(page: Path):
    """Predict-then-reveal is the pattern that stops a chapter being a slideshow.

    Two per chapter is the floor, placed in front of the genuinely surprising moments.
    """
    count = page.read_text(encoding="utf-8").count("lesson.predict(")
    assert count >= 2, f"{page.name} has {count} prediction(s) — ask before you tell"


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_no_deprecated_streamlit_width_argument(page: Path):
    assert "use_container_width" not in page.read_text(encoding="utf-8"), (
        f"{page.name} uses use_container_width, which Streamlit has deprecated. "
        'Use width="stretch" or width="content".'
    )


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_chapter_has_a_little_kid_corner(page: Path):
    """The 4th grader is owed one per chapter, in the styled box rather than buried."""
    assert "lesson.kid_corner(" in page.read_text(encoding="utf-8"), (
        f"{page.name} has no lesson.kid_corner() — the younger sibling gets left out"
    )
