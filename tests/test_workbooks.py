"""Every chapter's interactive workbook must load, and must agree with itself.

The self-agreement check matters more than it sounds: a question whose own stated answer
fails its own checker would tell a kid they were wrong when they were right, which is
about the worst thing this project could do.
"""

from __future__ import annotations

import pytest

from kidsml import workbook as wb
from kidsml.ui import CHAPTERS

chapters_with_workbooks = wb.available()


def test_at_least_one_workbook_exists():
    assert chapters_with_workbooks, "no workbooks found in worksheets/"


@pytest.mark.parametrize("chapter", chapters_with_workbooks, ids=lambda c: f"ch{c:02d}")
def test_workbook_loads(chapter: int):
    book = wb.load(chapter)
    assert book is not None, f"worksheets/{chapter:02d}_*.py has no WORKBOOK"
    assert book.chapter == chapter, "workbook says it belongs to a different chapter"
    assert book.questions, "a workbook with no questions is not a workbook"


@pytest.mark.parametrize("chapter", chapters_with_workbooks, ids=lambda c: f"ch{c:02d}")
def test_every_question_accepts_its_own_answer(chapter: int):
    book = wb.load(chapter)
    for i, q in enumerate(book.questions):
        assert q.kind in {"number", "choice", "text", "open"}, f"q{i}: bad kind {q.kind!r}"
        assert q.why, f"q{i}: every question needs a `why` — that is where the teaching happens"

        if q.kind == "open" or q.answer is None:
            continue
        candidate = q.answer[0] if isinstance(q.answer, (list, tuple)) else q.answer
        assert q.check(candidate) is True, (
            f"chapter {chapter:02d} question {i + 1} rejects its own stated answer "
            f"({candidate!r})"
        )
        if q.kind == "choice":
            assert q.choices, f"q{i}: a choice question needs choices"
            assert str(q.answer) in [str(c) for c in q.choices], (
                f"q{i}: the correct answer is not one of the offered choices"
            )


@pytest.mark.parametrize("chapter", chapters_with_workbooks, ids=lambda c: f"ch{c:02d}")
def test_workbook_has_a_little_kid_corner(chapter: int):
    """Chapter 24 is a wrap-up, but every teaching chapter owes the 4th grader something."""
    book = wb.load(chapter)
    if chapter == 24:
        return
    assert book.kid_corner, f"chapter {chapter:02d} has no Little Kid Corner"


def test_no_printable_worksheets_remain():
    """There is no printer in this house. Workbooks are Python, rendered on screen."""
    leftovers = sorted(p.name for p in wb.WORKSHEET_DIR.glob("*.md"))
    assert not leftovers, (
        f"printable worksheets left behind: {leftovers}. "
        "Convert them to interactive workbooks (see worksheets/00_guessing_machine.py)."
    )


@pytest.mark.parametrize("chapter", CHAPTERS, ids=lambda c: f"ch{c[0]:02d}")
def test_built_chapters_have_a_workbook(chapter):
    """If a chapter's page exists, its workbook should too."""
    from tests.test_pages import PAGES

    number = chapter[0]
    if not any(p.name.startswith(f"{number:02d}_") for p in PAGES.glob("[0-9][0-9]_*.py")):
        pytest.skip(f"chapter {number:02d} not built yet")
    assert number in chapters_with_workbooks, f"chapter {number:02d} has a page but no workbook"
