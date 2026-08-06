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

import textwrap
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import streamlit as st
from markdown_it import MarkdownIt

from kidsml.plots import use_house_style
from kidsml.ui import CHAPTER_BY_NUMBER

# The six beats, in order, with the label shown in the progress trail.
BEATS = {
    "hook": ("🎣", "Start here"),
    "byhand": ("✏️", "Work it out"),
    "seeit": ("👀", "Take a look"),
    "play": ("🎛️", "Your turn"),
    "forreal": ("💻", "In real code"),
    "challenge": ("🏆", "Go further"),
}
BEAT_ORDER = list(BEATS)



# Streamlit parses markdown — but only when it is handling the whole block. The moment we
# wrap text in our own <div> for styling, everything inside is treated as raw HTML and
# `**like this**` renders with the asterisks showing. So we do the markdown ourselves
# first. (markdown-it-py ships with Streamlit, so this costs no new dependency.)
_MARKDOWN = MarkdownIt("commonmark", {"breaks": True, "html": True})


def _dedent(text: str) -> str:
    """Strip the indentation a triple-quoted string picks up from sitting in a function.

    Four leading spaces mean "code block" in markdown, so an indented block of prose comes
    out in a monospace box with its asterisks showing instead of as a paragraph with bold.
    """
    lines = text.strip("\n").split("\n")
    body = [line for line in lines[1:] if line.strip()]

    # A triple-quoted string often starts right after the quotes, so its first line has no
    # indent while the rest do. Line it up before measuring the common prefix.
    if body and lines[0].strip() and not lines[0].startswith(" "):
        pad = min(len(line) - len(line.lstrip()) for line in body)
        lines[0] = " " * pad + lines[0]

    return textwrap.dedent("\n".join(lines))


def _as_html(text: str) -> str:
    """Markdown to HTML, for text that has to sit inside one of our styled boxes."""
    return _MARKDOWN.render(_dedent(text))


def _inline_html(text: str) -> str:
    """Same, for a short phrase that should not become its own paragraph."""
    return _MARKDOWN.renderInline(_dedent(text).replace("\n", " ").strip())


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
    st.set_page_config(page_title=f"{chapter:02d} · {title}", page_icon="🧪", layout="centered")
    use_house_style()
    _inject_style()

    st.markdown(
        f"<div class='kml-chapter-head'>"
        f"<div class='kml-part'>{part}</div>"
        f"<h1>{title}</h1>"
        f"<p class='kml-idea'>{_inline_html(idea)}</p>"
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
            width="stretch",
        )

    with middle:
        if index == len(steps) - 1:
            st.button(
                "↺ Start this chapter again",
                key=f"restart_{_current.chapter}",
                on_click=_jump,
                args=(0,),
                width="stretch",
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
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Things to put inside a step
# ---------------------------------------------------------------------------


def say(markdown: str) -> None:
    """Prose. Keep it to a few sentences.

    Handed straight to Streamlit rather than wrapped in our own ``<div>``. Wrapping made
    the element size to its content, and a block with a max-width inside a fit-content
    parent collapses to the width of its longest word — which is exactly how the text
    ended up one word per line. The column width now comes from the page container.
    """
    st.markdown(_dedent(markdown))


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

    st.markdown(
        f"<div class='kml-box kml-predict'><b>{_inline_html(question)}</b></div>",
        unsafe_allow_html=True,
    )

    if slot not in st.session_state:
        picked = st.radio(
            "Your prediction", choices, index=None,
            key=slot + "_radio", label_visibility="collapsed",
        )
        if st.button("Lock it in", key=slot + "_go", type="primary", disabled=picked is None):
            st.session_state[slot] = picked
            st.rerun()
        return None

    chosen = st.session_state[slot]
    st.markdown(f"You said: **{chosen}**")

    if correct is not None:
        if chosen == choices[correct]:
            st.markdown(
                "<div class='kml-box kml-right'><b>✅ Nice — that's what happens.</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='kml-box kml-surprise'><b>🙃 Turns out it's "
                f"{_inline_html(choices[correct])}.</b> Worth being surprised by.</div>",
                unsafe_allow_html=True,
            )
    if why:
        st.markdown(f"> {why}")
    return chosen


def look_for(what: str) -> None:
    """Point at what matters in the picture above. A figure with no pointer is decoration."""
    st.markdown(
        f"<div class='kml-box kml-look'>👀 <b>Look for:</b> {_inline_html(what)}</div>",
        unsafe_allow_html=True,
    )


def aha(body: str) -> None:
    """The moment worth stopping on. Green, like the good news it is."""
    st.markdown(
        f"<div class='kml-box kml-aha'><b>💡 Aha!</b>{_as_html(body)}</div>",
        unsafe_allow_html=True,
    )


def careful(body: str) -> None:
    """A trap worth pointing out before they fall into it."""
    st.markdown(
        f"<div class='kml-box kml-careful'><b>⚠️ Careful</b>{_as_html(body)}</div>",
        unsafe_allow_html=True,
    )


def kid_corner(body: str) -> None:
    """The 🧸 box: the same idea with no algebra in it, for the younger sibling."""
    st.markdown(
        f"<div class='kml-box kml-kid'><b>🧸 Little Kid Corner</b>{_as_html(body)}</div>",
        unsafe_allow_html=True,
    )


def jargon(term: str, plain: str) -> None:
    """Name the thing only *after* the idea has landed."""
    st.markdown(
        f"<div class='kml-box kml-jargon'>📖 Grown-ups call this <b>{term}</b>. "
        f"{_inline_html(plain)}</div>",
        unsafe_allow_html=True,
    )


def figure(width: float = 6.4, height: float = 4.8):
    return plt.subplots(figsize=(width, height))


def show(fig, clear: bool = True) -> None:
    """Render a matplotlib figure and free it.

    Stretched to the container rather than drawn at its natural pixel size, so a wide
    figure inside a narrow column scales down instead of overflowing. The page is a
    centred column now, so natural size is often wider than the space available.
    """
    fig.patch.set_alpha(0)
    st.pyplot(fig, width="stretch")
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
  /* ---------------------------------------------------------------- layout */
  /* One centred column that shrinks with the window. Everything is in rem or a
     viewport unit so browser zoom and a resized window behave like any other
     site — no fixed pixel widths anywhere in the reading column. */
  .block-container {
      max-width: min(62rem, 100%) !important;
      padding-top: clamp(1rem, 3vw, 2.2rem) !important;
      padding-bottom: 4rem !important;
      padding-left: clamp(0.9rem, 4vw, 3rem) !important;
      padding-right: clamp(0.9rem, 4vw, 3rem) !important;
      margin: 0 auto !important;
  }

  /* Anything we emit as raw HTML must stretch, not shrink-wrap. A block with a
     max-width inside a fit-content parent collapses to its longest word. */
  .block-container [data-testid="stMarkdown"] { width: 100% !important; }
  .block-container [data-testid="stMarkdownContainer"] { width: 100% !important; }
  .kml-box { width: 100%; box-sizing: border-box; }

  /* --------------------------------------------------------------- reading */
  [data-testid="stMarkdownContainer"] p {
      font-size: clamp(0.98rem, 0.55vw + 0.86rem, 1.09rem);
      line-height: 1.72;
      color: #D5DEE9;
      margin-bottom: 0.75rem;
  }
  [data-testid="stMarkdownContainer"] li { font-size: 1.05rem; line-height: 1.66; color: #D5DEE9; }
  [data-testid="stMarkdownContainer"] strong { color: #F0F6FC; font-weight: 650; }
  [data-testid="stMarkdownContainer"] code {
      background: #1B212E; color: #9FD8B4; padding: 0.1em 0.36em; border-radius: 4px;
  }

  /* ------------------------------------------------------------------ head */
  .kml-chapter-head { text-align: center; margin-bottom: 0.2rem; }
  .kml-chapter-head h1 {
      margin: 0.2rem 0 0.3rem 0;
      font-size: clamp(1.5rem, 3.2vw + 0.7rem, 2.15rem); letter-spacing: -0.02em;
      color: #F0F6FC; line-height: 1.15;
  }
  .kml-part {
      text-transform: uppercase; letter-spacing: 0.13em; font-size: 0.7rem;
      color: #7D8899; font-weight: 700;
  }
  .kml-idea {
      color: #9FB0C4; font-size: clamp(0.95rem, 1vw + 0.72rem, 1.12rem);
      margin: 0 auto 0.5rem auto; max-width: 60ch;
  }

  /* ----------------------------------------------------------------- trail */
  .kml-trail {
      display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.7rem 0 0.5rem 0;
      justify-content: center;
  }
  .kml-beat {
      font-size: 0.72rem; padding: 0.18rem 0.66rem; border-radius: 999px;
      background: #1B212E; color: #6E7A8C; font-weight: 600;
      transition: all 0.35s ease;
  }
  .kml-beat-on {
      background: #34D399; color: #08130E;
      box-shadow: 0 0 14px rgba(52, 211, 153, 0.35);
  }

  /* ------------------------------------------------------------ step title */
  .kml-step-title { text-align: center; margin: 1.1rem 0 1.1rem 0; }
  .kml-step-title h2 {
      margin: 0.25rem 0 0 0;
      font-size: clamp(1.18rem, 1.9vw + 0.66rem, 1.58rem);
      letter-spacing: -0.015em; color: #F0F6FC;
  }
  .kml-step-count {
      font-size: 0.7rem; color: #6E7A8C; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.11em;
  }

  /* ------------------------------------------------------------------ boxes */
  .kml-box { padding: 0.85rem 1.1rem; border-radius: 10px; margin: 0.85rem 0; }
  .kml-predict {
      background: linear-gradient(135deg, #14243B 0%, #16283F 100%);
      border-left: 3px solid #60A5FA; color: #DCE7F5; font-size: 1.05rem;
  }
  .kml-look {
      background: #241F14; border-left: 3px solid #D9A441;
      color: #E8DCC2; font-size: 1.0rem;
  }
  .kml-jargon {
      background: #171B26; border-left: 3px solid #3A4152;
      color: #9FB0C4; font-size: 0.97rem;
  }

  /* Green means "good news, stop and enjoy this": the aha moment, the Little Kid
     Corner, and a correct prediction. */
  .kml-aha, .kml-kid, .kml-right {
      background: linear-gradient(135deg, #10241C 0%, #132E22 100%);
      border-left: 3px solid #34D399;
      color: #C9EBDA;
      box-shadow: 0 0 22px rgba(52, 211, 153, 0.10);
  }
  .kml-aha b, .kml-kid b, .kml-right b { color: #6EE7B7; }
  .kml-careful {
      background: #241F14; border-left: 3px solid #F59E0B; color: #E8DCC2;
  }
  .kml-careful b { color: #FBBF24; }
  .kml-surprise {
      background: #221A2E; border-left: 3px solid #A78BFA; color: #DDD3F0;
  }
  .kml-wrong {
      background: #2A1719; border-left: 3px solid #F87171; color: #F3D2D2;
  }
  .kml-wrong b { color: #FCA5A5; }
  .kml-surprise b { color: #C4B5FD; }

  /* The heading line inside a box, then its prose. */
  .kml-box > b { display: block; margin-bottom: 0.4rem; font-size: 1.02rem; }
  .kml-box p { margin: 0 0 0.5rem 0; }
  .kml-box p:last-child { margin-bottom: 0; }
  .kml-aha, .kml-kid { animation: kmlPop 0.42s cubic-bezier(0.22, 1, 0.36, 1) both,
                                  kmlGlow 1.8s ease-in-out 1; }

  /* --------------------------------------------------------------- figures */
  [data-testid="stImage"] { display: flex; justify-content: center; }
  [data-testid="stImage"] img { background: transparent; border-radius: 8px; }
  [data-testid="stDataFrame"], [data-testid="stTable"] { margin: 0 auto; }
  [data-testid="stMetric"] {
      background: #171B26; border: 1px solid #262C3A; border-radius: 10px;
      padding: 0.65rem 0.9rem; text-align: center;
  }

  /* ------------------------------------------------------------------- nav */
  .kml-nav-space {
      margin-top: 2rem; border-top: 1px solid #262C3A; padding-top: 0.6rem;
  }
  /* Button labels were sitting high because the label is a block inside a taller
     button box. Make the button a centring flex container and the label a plain
     inline run. */
  .stButton > button {
      display: inline-flex !important;
      align-items: center !important;
      justify-content: center !important;
      min-height: 2.6rem;
      line-height: 1.15 !important;
      padding: 0.5rem 1.1rem !important;
      border-radius: 10px;
      font-weight: 600;
      transition: transform 0.16s cubic-bezier(0.22, 1, 0.36, 1),
                  box-shadow 0.16s ease, border-color 0.16s ease, background 0.2s ease;
  }
  .stButton > button p,
  .stButton > button div,
  .stButton > button span {
      margin: 0 !important; line-height: 1.15 !important;
      display: inline !important;
  }
  .stButton > button:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 8px 22px rgba(52, 211, 153, 0.24);
      border-color: #34D399;
  }
  .stButton > button:active:not(:disabled) {
      transform: translateY(0) scale(0.985);
      box-shadow: 0 2px 8px rgba(52, 211, 153, 0.2);
  }
  .stButton > button:focus-visible {
      outline: 2px solid #34D399; outline-offset: 2px;
  }

  /* ------------------------------------------------- vertical centring, misc */
  /* Radio and checkbox rows: dot and text on the same middle line. */
  [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
      display: flex !important; align-items: center !important;
      gap: 0.45rem; min-height: 1.9rem; line-height: 1.3 !important;
      transition: color 0.16s ease, transform 0.16s ease;
  }
  [data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {
      margin: 0 !important; line-height: 1.3 !important;
  }
  [data-testid="stRadio"] label:hover { color: #F0F6FC; transform: translateX(2px); }

  /* List items: the marker and the first line should share a baseline. */
  [data-testid="stMarkdownContainer"] ul,
  [data-testid="stMarkdownContainer"] ol { padding-left: 1.35rem; margin-bottom: 0.8rem; }
  [data-testid="stMarkdownContainer"] li { margin: 0.3rem 0; padding-left: 0.15rem; }
  [data-testid="stMarkdownContainer"] li > p { margin: 0 !important; }
  [data-testid="stMarkdownContainer"] li::marker { color: #34D399; }

  /* --------------------------------------------------------- control polish */
  [data-testid="stSlider"] [role="slider"] {
      transition: transform 0.16s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.16s ease;
  }
  [data-testid="stSlider"]:hover [role="slider"] {
      transform: scale(1.18);
      box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.16);
  }
  [data-baseweb="select"] > div {
      border-radius: 9px !important;
      transition: border-color 0.16s ease, box-shadow 0.16s ease;
  }
  [data-baseweb="select"] > div:hover {
      border-color: #34D399 !important;
      box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.12);
  }
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
      border-radius: 9px; transition: border-color 0.16s ease, box-shadow 0.16s ease;
  }
  [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus {
      box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.16);
  }
  [data-testid="stMetric"] {
      transition: transform 0.18s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.18s ease;
  }
  [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #34D399; }
  .kml-beat { cursor: default; }
  .kml-beat:hover { color: #B8C4D4; }
  .kml-box { transition: transform 0.18s ease, box-shadow 0.18s ease; }
  .kml-box:hover { transform: translateX(2px); }
  [data-testid="stExpander"] summary { transition: color 0.16s ease; }
  [data-testid="stExpander"] summary:hover { color: #34D399; }

  .stProgress > div > div > div {
      height: 6px; border-radius: 999px;
      transition: width 0.55s cubic-bezier(0.22, 1, 0.36, 1);
  }

  /* ------------------------------------------------------------ animations */
  /* Every screen arrives with a small rise-and-fade. Streamlit re-runs the
     script on each click, so this replays on every Next — which is the point. */
  @keyframes kmlRise {
      from { opacity: 0; transform: translateY(14px); }
      to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes kmlPop {
      0%   { opacity: 0; transform: scale(0.94); }
      60%  { opacity: 1; transform: scale(1.02); }
      100% { opacity: 1; transform: scale(1); }
  }
  @keyframes kmlGlow {
      0%, 100% { box-shadow: 0 0 0 rgba(52, 211, 153, 0); }
      50%      { box-shadow: 0 0 20px rgba(52, 211, 153, 0.28); }
  }

  /* Scoped to the reading column. Applied globally these also animated the sidebar
     nav on every rerun, which made the chapter list twitch and re-space itself
     every time you clicked. */
  .block-container .kml-step-title,
  .block-container [data-testid="stMarkdown"],
  .block-container [data-testid="stImage"],
  .block-container [data-testid="stDataFrame"],
  .block-container [data-testid="stMetric"],
  .block-container .stPlotlyChart {
      animation: kmlRise 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
  }
  /* Stagger, so the screen assembles itself instead of snapping into place. */
  .block-container [data-testid="stVerticalBlock"] > div:nth-child(1) { animation-delay: 0.00s; }
  .block-container [data-testid="stVerticalBlock"] > div:nth-child(2) { animation-delay: 0.04s; }
  .block-container [data-testid="stVerticalBlock"] > div:nth-child(3) { animation-delay: 0.08s; }
  .block-container [data-testid="stVerticalBlock"] > div:nth-child(4) { animation-delay: 0.12s; }
  .block-container [data-testid="stVerticalBlock"] > div:nth-child(5) { animation-delay: 0.16s; }

  .block-container .kml-box { animation: kmlPop 0.36s cubic-bezier(0.22, 1, 0.36, 1) both; }
  .block-container [data-testid="stAlert"] { animation: kmlPop 0.4s cubic-bezier(0.22, 1, 0.36, 1) both; }

  /* Reveal each block as it scrolls into view, on browsers that support
     scroll-driven animations. Elements already on screen at load land at the end
     of the range, so they simply appear — no flash of hidden content, and no
     JavaScript. Everywhere else the entrance animation above still runs. */
  @supports (animation-timeline: view()) {
      /* Scoped to the reading column. Anything in a scrolling or clipped ancestor
         can fail to resolve a view timeline and would then sit at zero opacity. */
      .block-container [data-testid="stMarkdown"],
      .block-container [data-testid="stImage"],
      .block-container [data-testid="stDataFrame"],
      .block-container [data-testid="stMetric"],
      .block-container .stPlotlyChart,
      .block-container [data-testid="stAlert"],
      .block-container [data-testid="stTable"],
      .block-container .stButton {
          animation: kmlReveal linear both;
          animation-timeline: view();
          animation-range: entry 0% entry 55%;
      }
  }
  @keyframes kmlReveal {
      from { opacity: 0; transform: translateY(22px) scale(0.985); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
  }

  /* Kids who get motion sick, and school machines with reduced motion on. */
  @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
          animation-duration: 0.001s !important;
          transition-duration: 0.001s !important;
          animation-timeline: auto !important;
          transform: none !important;
      }
  }

  /* --------------------------------------------------------- responsiveness */
  /* Never let anything force the page wider than the window. */
  [data-testid="stMarkdown"] img, [data-testid="stImage"] img,
  .stPlotlyChart, [data-testid="stDataFrame"] {
      max-width: 100% !important;
      height: auto;
  }
  [data-testid="stDataFrame"] { overflow-x: auto; }
  pre, code { white-space: pre-wrap; word-break: break-word; }

  /* Knobs-beside-picture becomes knobs-above-picture on a narrow window, rather
     than two columns squeezed until neither works. */
  @media (max-width: 46rem) {
      [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
      [data-testid="stHorizontalBlock"] > div {
          width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important;
      }
      .kml-box { padding: 0.7rem 0.85rem; }
      .kml-trail { gap: 0.25rem; }
      .kml-beat { font-size: 0.66rem; padding: 0.14rem 0.5rem; }
  }

  /* Columns must be allowed to shrink; the default min-width keeps them wide and
     pushes a horizontal scrollbar onto the whole page. */
  [data-testid="stHorizontalBlock"] > div { min-width: 0; }

  /* ------------------------------------------------------- the chapter list */
  /* Nothing in the sidebar animates. It is furniture, not content, and a nav that
     re-lays-itself-out on every click is maddening. */
  [data-testid="stSidebar"] * { animation: none !important; }

  [data-testid="stSidebarNav"] { padding-top: 0.4rem; }
  [data-testid="stSidebarNav"] ul { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
  [data-testid="stSidebarNav"] li {
      margin: 0 !important; padding: 0 !important; list-style: none !important;
  }
  [data-testid="stSidebarNav"] li::marker { content: none !important; }
  [data-testid="stSidebarNav"] a {
      display: flex !important; align-items: center !important;
      min-height: 1.95rem !important;
      margin: 1px 0 !important;
      padding: 0.24rem 0.6rem !important;
      border-radius: 7px;
      line-height: 1.2 !important;
      transition: background 0.14s ease, color 0.14s ease, padding-left 0.14s ease;
  }
  [data-testid="stSidebarNav"] a span,
  [data-testid="stSidebarNav"] a p {
      margin: 0 !important; line-height: 1.2 !important; font-size: 0.9rem;
  }
  [data-testid="stSidebarNav"] a:hover {
      background: #1B212E; color: #F0F6FC; padding-left: 0.85rem !important;
  }
  [data-testid="stSidebarNav"] a[aria-current="page"] {
      background: #16302A; color: #6EE7B7;
      box-shadow: inset 2px 0 0 #34D399;
  }

  #MainMenu, footer { visibility: hidden; }
</style>
"""


def apply_style() -> None:
    """Inject the course stylesheet. Chapters get this from :func:`begin`; the landing
    page calls it directly so the whole app looks like one thing."""
    st.markdown(_STYLE, unsafe_allow_html=True)


def _inject_style() -> None:
    apply_style()
