"""Interactive workbooks — the "do it by hand" part, on screen.

There is no printer in this house, so nothing here is a handout. Each chapter's workbook
is a list of questions that render as real inputs: type an answer, press check, and get
told not just *whether* you were right but *why the question was asked*.

A chapter's questions live in ``worksheets/NN_slug.py`` as a module-level ``WORKBOOK``.
Keeping them in their own file means each chapter owns its own questions and nothing has
to be edited in a shared file to add one.

Use it from a Streamlit page::

    from kidsml import workbook
    workbook.render(5)

...or from a notebook::

    from kidsml import workbook
    workbook.render(5)          # works in both; it detects where it is running
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path

WORKSHEET_DIR = Path(__file__).resolve().parent.parent / "worksheets"


@dataclass
class Question:
    """One question.

    ``kind`` decides the input widget and how the answer is checked:

    ``"number"``
        Type a number. Correct if within ``tolerance`` of ``answer``.
    ``"choice"``
        Pick from ``choices``. ``answer`` is the correct entry.
    ``"text"``
        Type a word or short phrase. Compared case- and space-insensitively; ``answer``
        may be a list, in which case any of them counts.
    ``"open"``
        No right answer. There is a box to think in, and pressing check reveals how a
        grown-up would answer. Use this for "why do you think..." questions.
    """

    prompt: str
    kind: str = "open"
    answer: object = None
    choices: list[str] | None = None
    tolerance: float = 1e-6
    hint: str = ""
    why: str = ""
    table: object = None  # optional pandas DataFrame shown above the input

    def check(self, given) -> bool | None:
        """True / False, or None when the question has no single right answer."""
        if self.kind == "open" or self.answer is None:
            return None
        if self.kind == "number":
            try:
                value = float(given)
            except (TypeError, ValueError):
                return False
            return math.isclose(value, float(self.answer), abs_tol=self.tolerance)
        if self.kind == "choice":
            return str(given) == str(self.answer)
        accepted = self.answer if isinstance(self.answer, (list, tuple)) else [self.answer]
        cleaned = str(given).strip().lower()
        return any(cleaned == str(a).strip().lower() for a in accepted)


@dataclass
class Workbook:
    """Everything the reader works through by hand for one chapter."""

    chapter: int
    title: str
    intro: str = ""
    questions: list[Question] = field(default_factory=list)
    kid_corner: str = ""
    closing: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_cache: dict[int, Workbook | None] = {}


def load(chapter: int) -> Workbook | None:
    """Find and import ``worksheets/NN_*.py`` for a chapter. None if there isn't one."""
    if chapter in _cache:
        return _cache[chapter]

    matches = sorted(WORKSHEET_DIR.glob(f"{chapter:02d}_*.py"))
    if not matches:
        _cache[chapter] = None
        return None

    spec = importlib.util.spec_from_file_location(f"workbook_{chapter:02d}", matches[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    book = getattr(module, "WORKBOOK", None)
    _cache[chapter] = book
    return book


def available() -> list[int]:
    """Chapter numbers that have a workbook."""
    return sorted(int(p.name[:2]) for p in WORKSHEET_DIR.glob("[0-9][0-9]_*.py"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


def render(chapter: int) -> None:
    """Show the workbook for a chapter, in whichever environment we're running in."""
    book = load(chapter)
    if book is None:
        return
    if _in_streamlit():
        _render_streamlit(book)
    else:
        _render_notebook(book)


def _render_streamlit(book: Workbook) -> None:
    import streamlit as st

    st.markdown(f"### 📝 {book.title}")
    if book.intro:
        st.markdown(book.intro)
    st.caption("Grab some scrap paper. Work it out, then type your answer in.")

    for i, q in enumerate(book.questions):
        key = f"wb_{book.chapter}_{i}"
        st.markdown(f"**{i + 1}.** {q.prompt}")
        if q.table is not None:
            st.dataframe(q.table, hide_index=True, width="content")

        if q.kind == "number":
            given = st.text_input("Your answer", key=key, placeholder="a number")
        elif q.kind == "choice":
            given = st.radio("Your answer", q.choices, key=key, index=None, horizontal=True)
        else:
            given = st.text_input("Your answer", key=key, placeholder="in your own words")

        if q.hint:
            with st.expander("Stuck? Nudge me"):
                st.markdown(q.hint)

        if st.button("Check", key=key + "_go"):
            verdict = q.check(given) if given not in (None, "") else None
            if verdict is True:
                st.success("✅ That's it.")
            elif verdict is False:
                st.error(f"❌ Not quite. The answer is **{q.answer}**.")
            elif q.kind == "open":
                st.info("💭 Here's one way to think about it:")
            else:
                st.warning("Type something in first.")
            if q.why:
                st.markdown(f"> {q.why}")
        st.divider()

    if book.kid_corner:
        st.info(f"🧸 **Little Kid Corner**\n\n{book.kid_corner}")
    if book.closing:
        st.markdown(book.closing)


def _render_notebook(book: Workbook) -> None:
    """ipywidgets version: same questions, same instant feedback, inside JupyterLab."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError:  # plain python, e.g. during tests
        _render_plain(book)
        return

    display(Markdown(f"### 📝 {book.title}"))
    if book.intro:
        display(Markdown(book.intro))
    display(Markdown("_Grab some scrap paper. Work it out, then type your answer in._"))

    for i, q in enumerate(book.questions):
        display(Markdown(f"**{i + 1}.** {q.prompt}"))
        if q.table is not None:
            display(q.table)

        if q.kind == "choice":
            box = widgets.RadioButtons(options=q.choices, value=None, description="")
        else:
            box = widgets.Text(placeholder="your answer")

        button = widgets.Button(description="Check", button_style="primary")
        out = widgets.Output()

        def on_click(_, q=q, box=box, out=out):
            out.clear_output()
            with out:
                verdict = q.check(box.value) if box.value not in (None, "") else None
                if verdict is True:
                    display(Markdown("✅ **That's it.**"))
                elif verdict is False:
                    display(Markdown(f"❌ Not quite. The answer is **{q.answer}**."))
                elif q.kind == "open":
                    display(Markdown("💭 Here's one way to think about it:"))
                else:
                    display(Markdown("Type something in first."))
                if q.why:
                    display(Markdown(f"> {q.why}"))

        button.on_click(on_click)
        display(widgets.VBox([box, button, out]))

    if book.kid_corner:
        display(Markdown(f"> 🧸 **Little Kid Corner** — {book.kid_corner}"))
    if book.closing:
        display(Markdown(book.closing))


def _render_plain(book: Workbook) -> None:
    """Last-resort text rendering, used when nothing interactive is available."""
    print(f"=== {book.title} ===")
    for i, q in enumerate(book.questions):
        print(f"\n{i + 1}. {q.prompt}")
        if q.answer is not None:
            print(f"   answer: {q.answer}")
        if q.why:
            print(f"   why: {q.why}")
