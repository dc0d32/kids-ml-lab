"""One idea per screen.

A chapter used to be one long page you scrolled. Scrolling lets you skim, and skimming
is how a reader arrives at the end of a chapter having understood none of it.

This module turns a chapter into a sequence of **steps**. One screen shows one idea, a
picture of it, and something to do. You move on when you're ready, and the page never
shows you a wall of anything.

The three habits that make it work:

* **Small.** A step is 1-3 sentences and one thing to look at. If a step needs scrolling,
  it should have been two steps.
* **Predict, then reveal.** Ask what they think will happen *before* showing them. A
  wrong prediction followed by a surprise teaches more than a correct one that was never
  in doubt. That's :func:`predict`.
* **Nothing is passive.** Every step has a slider to move, a question to answer, or a
  picture that only makes sense once they've looked for something specific.

Written like this::

    from kidsml import lesson

    lesson.begin(3)

    @lesson.step("Four points, four answers", beat="hook")
    def _():
        lesson.say("Here is the smallest hard problem in machine learning.")
        ...

    lesson.finish()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import streamlit as st

from kidsml.plots import use_house_style
from kidsml.ui import CHAPTER_BY_NUMBER

# The six beats, in order, with the label shown in the progress trail.
BEATS = {
    "hook": ("🎣", "The Hook"),
    "byhand": ("✏️", "By Hand"),
    "seeit": ("👀", "See It"),
    "play": ("🎛️", "Play"),
    "forreal": ("💻", "For Real"),
    "challenge": ("🏆", "Challenge"),
}
BEAT_ORDER = list(BEATS)


@dataclass
class Step:
    title: str
    beat: str
    run: Callable[[], None]


@dataclass
class _Lesson:
    chapter: int = 0
    steps: list[Step] = field(default_factory=list)


_current = _Lesson()


# ---------------------------------------------------------------------------
# Setting up
# ---------------------------------------------------------------------------


def begin(chapter: int) -> None:
    """Start a chapter page. Call once, at the top."""
    global _current
    _current = _Lesson(chapter=chapter)

    _, _, title, idea, part = CHAPTER_BY_NUMBER[chapter]
    st.set_page_config(page_title=f"{chapter:02d} · {title}", page_icon="🧪", layout="wide")
    use_house_style()
    _inject_style()

    st.markdown(
        f"<div class='kml-chapter-head'>"
        f"<div class='kml-part'>{part}</div>"
        f"<h1>{title}</h1>"
        f"<p class='kml-idea'>{idea}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def step(title: str, beat: str = "play"):
    """Register one screen. The decorated function draws it."""
    if beat not in BEATS:
        raise ValueError(f"unknown beat {beat!r}; pick one of {BEAT_ORDER}")

    def decorate(func: Callable[[], None]) -> Callable[[], None]:
        _current.steps.append(Step(title=title, beat=beat, run=func))
        return func

    return decorate


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _state_key() -> str:
    return f"kml_step_{_current.chapter}"


def _index() -> int:
    return int(st.session_state.get(_state_key(), 0))


def _go(delta: int) -> None:
    total = len(_current.steps)
    st.session_state[_state_key()] = max(0, min(total - 1, _index() + delta))


def _jump(target: int) -> None:
    st.session_state[_state_key()] = target


def finish() -> None:
    """Draw the current step, the progress trail, and the navigation. Call once, at the end."""
    steps = _current.steps
    if not steps:
        st.warning("This chapter has no steps yet.")
        return

    index = min(_index(), len(steps) - 1)
    current = steps[index]

    _draw_trail(steps, index)

    st.markdown(
        f"<div class='kml-step-title'>"
        f"<span class='kml-step-count'>Step {index + 1} of {len(steps)}</span>"
        f"<h2>{current.title}</h2>"
        f"</div>",
        unsafe_allow_html=True,
    )

    current.run()

    _draw_nav(steps, index)


def _draw_trail(steps: list[Step], index: int) -> None:
    """A slim bar showing which of the six beats this step belongs to."""
    beats_used = []
    for s in steps:
        if s.beat not in beats_used:
            beats_used.append(s.beat)
    here = steps[index].beat

    pieces = []
    for beat in beats_used:
        icon, label = BEATS[beat]
        css = "kml-beat kml-beat-on" if beat == here else "kml-beat"
        pieces.append(f"<span class='{css}'>{icon} {label}</span>")

    st.markdown(f"<div class='kml-trail'>{''.join(pieces)}</div>", unsafe_allow_html=True)
    st.progress((index + 1) / len(steps))


def _draw_nav(steps: list[Step], index: int) -> None:
    st.markdown("<div class='kml-nav-space'></div>", unsafe_allow_html=True)
    left, middle, right = st.columns([1, 2, 1])

    with left:
        st.button(
            "← Back",
            key=f"back_{_current.chapter}_{index}",
            disabled=index == 0,
            on_click=_go,
            args=(-1,),
            use_container_width=True,
        )

    with middle:
        if index == len(steps) - 1:
            st.button(
                "↺ Start this chapter again",
                key=f"restart_{_current.chapter}",
                on_click=_jump,
                args=(0,),
                use_container_width=True,
            )
        else:
            st.caption(f"Next up: {steps[index + 1].title}")

    with right:
        st.button(
            "Next →",
            key=f"next_{_current.chapter}_{index}",
            disabled=index == len(steps) - 1,
            on_click=_go,
            args=(1,),
            type="primary",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Things to put inside a step
# ---------------------------------------------------------------------------


def say(markdown: str) -> None:
    """Prose, held to a readable column width. Keep it to a few sentences."""
    st.markdown(f"<div class='kml-say'>{markdown}</div>", unsafe_allow_html=True)


def predict(question: str, choices: list[str], correct: int | None = None,
            why: str = "", key: str = "") -> str | None:
    """Ask what they think will happen, *before* showing them.

    Returns ``None`` until they commit to an answer, so the step can hold the reveal
    back::

        guess = lesson.predict("Will any straight line work?", ["Yes", "No"], correct=1,
                               why="No line can split opposite corners.", key="xor")
        if guess is not None:
            ...draw the reveal...

    Being wrong here is the point, so the feedback never scolds.
    """
    slot = f"kml_predict_{_current.chapter}_{key or question[:20]}"

    st.markdown(f"<div class='kml-predict'><b>{question}</b></div>", unsafe_allow_html=True)

    if slot not in st.session_state:
        picked = st.radio("", choices, index=None, key=slot + "_radio", label_visibility="collapsed")
        if st.button("Lock it in", key=slot + "_go", type="primary", disabled=picked is None):
            st.session_state[slot] = picked
            st.rerun()
        return None

    chosen = st.session_state[slot]
    st.markdown(f"You said: **{chosen}**")

    if correct is not None:
        if chosen == choices[correct]:
            st.success("Nice — that's what happens.")
        else:
            st.info(f"Turns out it's **{choices[correct]}**. Worth being surprised by.")
    if why:
        st.markdown(f"> {why}")
    return chosen


def look_for(what: str) -> None:
    """Point at what matters in the picture above. A figure with no pointer is decoration."""
    st.markdown(f"<div class='kml-look'>👀 <b>Look for:</b> {what}</div>", unsafe_allow_html=True)


def aha(body: str) -> None:
    st.success(f"💡 **Aha!**\n\n{body}")


def careful(body: str) -> None:
    st.warning(f"⚠️ **Careful**\n\n{body}")


def kid_corner(body: str) -> None:
    st.info(f"🧸 **Little Kid Corner**\n\n{body}")


def jargon(term: str, plain: str) -> None:
    """Name the thing only *after* the idea has landed."""
    st.markdown(
        f"<div class='kml-jargon'>📖 Grown-ups call this <b>{term}</b>. {plain}</div>",
        unsafe_allow_html=True,
    )


def figure(width: float = 6.4, height: float = 4.8):
    return plt.subplots(figsize=(width, height))


def show(fig, clear: bool = True) -> None:
    st.pyplot(fig, use_container_width=False)
    if clear:
        plt.close(fig)


def mermaid(diagram: str, height: int = 300) -> None:
    from streamlit_mermaid import st_mermaid

    st_mermaid(diagram.strip(), height=f"{height}px")


def workbook(chapter: int | None = None) -> None:
    """Drop this chapter's interactive questions into the current step."""
    from kidsml import workbook as wb

    wb.render(_current.chapter if chapter is None else chapter)


def controls():
    """A left column for knobs and a right column for the picture they change."""
    return st.columns([1, 2], gap="large")


# ---------------------------------------------------------------------------
# Looks
# ---------------------------------------------------------------------------

_STYLE = """
<style>
  /* Reading width. Long lines are the fastest way to lose a 13-year-old. */
  .kml-say, .kml-jargon, .kml-look, .kml-predict { max-width: 68ch; }
  .kml-say { font-size: 1.06rem; line-height: 1.65; margin-bottom: 0.6rem; }
  .kml-say p { margin-bottom: 0.7rem; }

  .kml-chapter-head { margin-bottom: 0.4rem; }
  .kml-chapter-head h1 { margin: 0.1rem 0 0.2rem 0; font-size: 2.0rem; letter-spacing: -0.01em; }
  .kml-part { text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem;
              color: #64748B; font-weight: 700; }
  .kml-idea { color: #475569; font-size: 1.12rem; margin: 0 0 0.7rem 0; max-width: 70ch; }

  .kml-trail { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.5rem 0 0.4rem 0; }
  .kml-beat { font-size: 0.74rem; padding: 0.16rem 0.6rem; border-radius: 999px;
              background: #F1F5F9; color: #94A3B8; font-weight: 600; }
  .kml-beat-on { background: #10B981; color: white; }

  .kml-step-title { margin: 0.9rem 0 0.5rem 0; }
  .kml-step-title h2 { margin: 0.1rem 0 0 0; font-size: 1.45rem; letter-spacing: -0.01em; }
  .kml-step-count { font-size: 0.72rem; color: #94A3B8; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.08em; }

  .kml-predict { background: #EFF6FF; border-left: 4px solid #3B82F6;
                 padding: 0.75rem 1rem; border-radius: 6px; margin: 0.6rem 0; }
  .kml-look { background: #FFFBEB; border-left: 4px solid #F59E0B;
              padding: 0.6rem 1rem; border-radius: 6px; margin: 0.5rem 0; font-size: 0.96rem; }
  .kml-jargon { background: #F8FAFC; border-left: 4px solid #CBD5E1;
                padding: 0.6rem 1rem; border-radius: 6px; margin: 0.6rem 0;
                font-size: 0.94rem; color: #475569; }

  .kml-nav-space { margin-top: 1.6rem; border-top: 1px solid #E2E8F0; padding-top: 0.4rem; }

  /* The progress bar reads better thin. */
  .stProgress > div > div > div { height: 5px; }

  #MainMenu, footer { visibility: hidden; }
</style>
"""


def _inject_style() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)
