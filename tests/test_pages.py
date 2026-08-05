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


DEMONSTRATIVE = re.compile(
    r"\b(these|those)\b|\bthe (table|chart|picture|plot|graph) (above|below)\b"
    r"|\bshown (here|above|below)\b",
    re.I,
)
DRAWS = ("lesson.show(", "st.dataframe(", "st.table(", "st.image(", "lesson.mermaid(")


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_predictions_do_not_ask_about_invisible_things(page: Path):
    """If a question says "these", the reader has to be able to see them.

    `lesson.predict` withholds everything after the gate. That is correct for the
    *reveal* and wrong for the *setup* — chapter 03 once asked "can any line split these
    four points?" with the four points drawn only after the answer was locked in.
    """
    source = page.read_text(encoding="utf-8")

    for block in re.split(r"(?=@lesson\.step\()", source):
        if "lesson.predict(" not in block:
            continue
        gate = block.find("if guess is None")
        if gate == -1:
            continue

        question = re.search(r"lesson\.predict\(\s*\n?\s*[\"']([^\"']+)", block)
        if not question or not DEMONSTRATIVE.search(question.group(1)):
            continue

        before_gate = block[:gate]
        title = re.search(r'@lesson\.step\("([^"]+)"', block)
        assert any(call in before_gate for call in DRAWS), (
            f"{page.name}, step {title.group(1) if title else '?'}: the question says "
            f"{question.group(1)[:60]!r} but nothing is drawn before the prediction gate"
        )


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_markdown_inside_styled_boxes_actually_renders(page: Path):
    """`**bold**` must come out bold, not with the asterisks showing.

    Streamlit parses markdown only when it owns the whole block. Wrapping text in our own
    <div> for styling makes it raw HTML, so `lesson.say` renders the markdown itself
    first. Every chapter is full of bold, so this is worth a real check.
    """
    at = AppTest.from_file(str(page), default_timeout=TIMEOUT_SECONDS).run()

    steps = 1
    while steps <= MAX_STEPS:
        for element in at.markdown:
            value = element.value
            if "kml-say" in value or "kml-look" in value or "kml-jargon" in value:
                assert "**" not in value, (
                    f"{page.name} step {steps}: literal ** in a styled box — markdown is "
                    f"not being rendered.\n    {value[:160]}"
                )
        button = _next_button(at)
        if button is None or button.disabled:
            return
        button.click().run()
        steps += 1


VISUAL_OR_INTERACTIVE = (
    "lesson.show(", "lesson.mermaid(", "lesson.workbook(",
    "st.dataframe(", "st.table(", "st.plotly_chart(", "st.image(",
    "st.line_chart(", "st.bar_chart(", "st.area_chart(", "st.metric(", "st.code(",
    "st.slider(", "st.radio(", "st.selectbox(", "st.multiselect(", "st.checkbox(",
    "st.button(", "st.text_input(", "st.number_input(", "st.file_uploader(",
    "st_canvas(", "st.select_slider(", "st.toggle(",
)


@pytest.mark.parametrize("page", existing, ids=lambda p: p.stem)
def test_every_screen_has_something_to_look_at(page: Path):
    """A screen of pure prose in the middle of a chapter is a wasted click.

    It is usually one of two bugs: an idea introduced a screen before its picture, or one
    thought sliced in half to pad the step count. Both make a reader hold something
    abstract across a page turn, and that is where the kids reported losing the thread.

    Exempt: the opening hook (it may be setting a scene before any data exists) and the
    challenge screens (a list of dares is the right shape for those).
    """
    source = page.read_text(encoding="utf-8")

    # Pages define module-level helpers that draw. A call to one of those counts as a
    # visual, so resolve them first — checking call sites alone reported a false failure
    # on a screen that draws its chart through a helper.
    drawing_helpers = set()
    for match in re.finditer(r"^def (\w+)\(.*?(?=^def |\Z)", source, re.S | re.M):
        if any(item in match.group(0) for item in VISUAL_OR_INTERACTIVE):
            drawing_helpers.add(match.group(1) + "(")

    blocks = re.split(r"(?=@lesson\.step\()", source)
    steps = [b for b in blocks if b.startswith("@lesson.step(")]

    for index, block in enumerate(steps):
        title = re.search(r'@lesson\.step\("([^"]+)"', block).group(1)
        beat = re.search(r'beat="(\w+)"', block)
        beat = beat.group(1) if beat else "play"

        if index == 0 or beat == "challenge":
            continue

        body = block[block.find("def _"):]
        shows_something = any(item in body for item in VISUAL_OR_INTERACTIVE) or any(
            helper in body for helper in drawing_helpers
        )
        assert shows_something, (
            f"{page.name}: screen {title!r} has nothing to look at and nothing to move. "
            "Either bring its picture onto this screen, or merge it with the next one."
        )


def test_the_stylesheet_is_responsive():
    """Zoom and window resizing should behave like any other modern site.

    That means no fixed pixel width on the reading column, type that scales, columns that
    stack when the window is narrow, and nothing that can force a horizontal scrollbar.
    Checked here because it is invisible to every other test we have — AppTest reports
    element values, not layout.
    """
    from kidsml.lesson import _STYLE

    required = {
        "a shrinkable reading column": "max-width: min(",
        "type that scales with the window": "clamp(",
        "a narrow-window breakpoint": "@media (max-width:",
        "columns that stack when narrow": "flex-direction: column",
        "columns allowed to shrink": "min-width: 0",
        "images that never overflow": "max-width: 100% !important",
        "respect for reduced-motion": "prefers-reduced-motion",
    }
    for description, token in required.items():
        assert token in _STYLE, f"stylesheet is missing {description} ({token!r})"

    assert "max-width: 980px" not in _STYLE, (
        "the reading column has a hard pixel width again; use a rem-based min() so it "
        "shrinks with the window"
    )


def test_the_stylesheet_centres_and_animates():
    """Buttons and radio rows centre their text, and things move when you touch them.

    Layout is invisible to AppTest, so the properties get asserted directly. Button
    labels in particular sat high, because the label is a block inside a taller button.
    """
    from kidsml.lesson import _STYLE

    required = {
        "buttons centre their label": "align-items: center !important",
        "button labels are not block-level": ".stButton > button p",
        "list markers are styled": "li::marker",
        "buttons respond to hover": ".stButton > button:hover",
        "sliders respond to hover": "[data-testid=\"stSlider\"]:hover",
        "reveal-on-scroll where supported": "animation-timeline: view()",
        "reveal is scoped to the reading column": ".block-container [data-testid=\"stMarkdown\"]",
        "keyboard focus stays visible": ":focus-visible",
    }
    for description, token in required.items():
        assert token in _STYLE, f"stylesheet is missing {description} ({token!r})"
