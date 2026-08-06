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

import re
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

    Drawn at its natural size. Stretching it to the container scaled a 704px figure up to
    884px, which enlarged every label and title with it and left the charts shouting next
    to the prose. The CSS caps images at 100% of the column, so a narrow window still
    scales them *down* — which is the direction that does no harm.
    """
    fig.patch.set_alpha(0)
    st.pyplot(fig, width="content")
    if clear:
        plt.close(fig)


# streamlit-mermaid exposes no theme setting, so the theme is declared in the diagram
# source itself. Without this every diagram renders on mermaid's default white card,
# which on a dark page is a bright rectangle in the middle of the reading column.
_MERMAID_THEME = (
    '%%{init: {"theme":"base",'
    '"flowchart":{"useMaxWidth":true,"htmlLabels":false,"padding":10,"nodeSpacing":38,'
    '"rankSpacing":44,"curve":"basis"},'
    '"themeVariables":{'
    '"background":"#000000",'
    '"fontFamily":"ui-sans-serif, system-ui, Helvetica, Arial, sans-serif","fontSize":"17px",'
    '"primaryColor":"#131315","primaryTextColor":"#EDEDF0","primaryBorderColor":"#34D399",'
    '"secondaryColor":"#0B0B0D","secondaryTextColor":"#EDEDF0","secondaryBorderColor":"#3C3C44",'
    '"tertiaryColor":"#0B0B0D","tertiaryTextColor":"#EDEDF0","tertiaryBorderColor":"#3C3C44",'
    '"lineColor":"#A8A8B2","textColor":"#EDEDF0",'
    '"mainBkg":"#131315","nodeBorder":"#34D399","clusterBkg":"#0B0B0D",'
    '"edgeLabelBackground":"#000000"'
    "}}}%%\n"
)


def _estimated_height(diagram: str) -> int:
    """Guess how tall a diagram needs to be.

    The component reserves a fixed 424px whatever it draws, so the height is forced from
    CSS — which means we have to work it out here, and a bad guess either leaves a hole
    under the picture or slices the bottom off it.

    Mermaid lays a flowchart out in ranks: everything with no arrow coming in sits in the
    first rank, everything they point at sits in the second, and so on. Top-down, the
    number of ranks is the height. Left-to-right, the *widest* rank is the height.

    Biased to over-estimate: slack under a diagram is untidy, a clipped diagram is broken.
    """
    lines = [ln.strip() for ln in diagram.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("%%")]
    if not lines:
        return 200

    match = re.match(r"(?:graph|flowchart)\s+([A-Za-z]+)", lines[0])
    if match is None:
        return 420
    direction = match.group(1).upper()

    # Edges look like `A --> B`, `A -. label .-> B`, `A ---|yes| B`. Grab the plain
    # identifier at each end, ignoring whatever label shape is wrapped around it.
    body = "\n".join(lines[1:])
    edges = re.findall(
        r"([A-Za-z][A-Za-z0-9_]*)\s*(?:\[[^\]]*\]|\(\([^)]*\)\)|\([^)]*\)|\{[^}]*\})?\s*"
        r"[-=.]+[->ox]*\s*(?:\|[^|]*\|)?\s*"
        r"([A-Za-z][A-Za-z0-9_]*)",
        body,
    )
    nodes = set(re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*[\[\(\{]", body))
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    if not nodes:
        return 200

    # Longest path from the start, with loops broken. A "predict, measure, adjust, repeat"
    # diagram has an arrow going back to the top, and following it forever would report a
    # diagram twenty ranks deep. Nodes still waiting when nothing has an empty in-tray are
    # part of a loop, so one of them is released and the walk carries on.
    remaining = {(a, b) for a, b in edges if a != b}
    rank = {name: 0 for name in nodes}
    waiting = set(nodes)
    while waiting:
        ready = [n for n in waiting if not any(b == n for _, b in remaining)]
        if not ready:
            ready = [sorted(waiting)[0]]
        for name in ready:
            for a, b in list(remaining):
                if a == name:
                    rank[b] = max(rank[b], rank[name] + 1)
                    remaining.discard((a, b))
            waiting.discard(name)

    ranks = max(rank.values()) + 1
    widest = max(list(rank.values()).count(level) for level in set(rank.values()))

    if direction in ("LR", "RL"):
        return min(70 + 64 * widest, 760)
    return min(70 + 78 * ranks, 900)


# The mermaid build bundled with streamlit-mermaid has an ASCII-only lexer for unquoted
# node labels. One `x₁` and the whole diagram fails to parse — and because the failure is
# a browser-side exception, the page simply shows a blank gap where the picture should be.
# Every label with a character outside ASCII therefore gets quoted on the way through.
_LABEL_SHAPES = [
    (re.compile(r"\[([^\[\]\"]+)\]"), "[", "]"),
    (re.compile(r"\(\(([^()\"]+)\)\)"), "((", "))"),
    (re.compile(r"\{([^{}\"]+)\}"), "{", "}"),
    (re.compile(r"\|([^|\"]+)\|"), "|", "|"),
]


def quote_awkward_labels(diagram: str) -> str:
    """Wrap any node or edge label containing a non-ASCII character in double quotes."""

    def fix(match: re.Match, open_mark: str, close_mark: str) -> str:
        label = match.group(1)
        if label.isascii():
            return match.group(0)
        return f'{open_mark}"{label}"{close_mark}'

    for pattern, open_mark, close_mark in _LABEL_SHAPES:
        diagram = pattern.sub(lambda m, o=open_mark, c=close_mark: fix(m, o, c), diagram)
    return diagram


def mermaid(diagram: str, height: int | None = None) -> None:
    """Draw a Mermaid diagram — for flows, pipelines and "what connects to what".

    Use this when the thing being explained is *structure* rather than data: how a
    prediction flows through a model, what order the steps happen in, how one chapter
    leads to the next. Use matplotlib when the thing being explained is *numbers*.

    The notebook equivalent is a fenced ```mermaid block inside a markdown cell, which
    JupyterLab renders natively. Keep the two in step.

    Keep diagrams small — six or seven boxes. A diagram that needs studying is a diagram
    that has stopped helping.
    """
    import hashlib

    from streamlit_mermaid import st_mermaid

    diagram = quote_awkward_labels(diagram.strip())

    # The component ignores the height it is given and reserves 424px whatever the
    # diagram, which left a large hole under every small flowchart. Streamlit puts a
    # `st-key-<key>` class on a keyed container, so the iframe can be sized from CSS.
    if height is None:
        height = _estimated_height(diagram)

    key = "kmlmmd" + hashlib.md5(diagram.encode()).hexdigest()[:10]
    with st.container(key=key):
        st.markdown(
            f"<style>.st-key-{key} iframe {{ height: {height}px !important; "
            f"min-height: 0 !important; }}</style>",
            unsafe_allow_html=True,
        )
        st_mermaid(_MERMAID_THEME + diagram, height=f"{height}px")


def workbook(chapter: int | None = None) -> None:
    """Drop this chapter's interactive questions into the current step."""
    from kidsml import workbook as wb

    wb.render(_current.chapter if chapter is None else chapter)


def controls():
    """A left column for knobs and a right column for the picture they change."""
    return st.columns([1, 2], gap="large")


def regenerate(label: str = "🎲 Try another one", key: str = "", help: str | None = None) -> int:
    """A button that hands back a different number every time it is pressed.

    Anything that rolls dice — generating text, sampling a picture, redrawing a canvas —
    needs a way to ask for another go. Streamlit only re-runs the script when *something*
    changes, so a step that samples once looks frozen: you change the prompt, and the same
    words come back because the sampling was cached against the old roll.

    Feed the returned number in as a seed (or as part of a cache key) and the reader gets a
    button that visibly does something::

        roll = lesson.regenerate(key="babble")
        words = make_words(prompt, seed=roll)
    """
    slot = f"kml_roll_{_current.chapter}_{key or label}"
    if slot not in st.session_state:
        st.session_state[slot] = 0

    def bump() -> None:
        st.session_state[slot] += 1

    st.button(label, key=slot + "_btn", on_click=bump, help=help, type="primary")
    return int(st.session_state[slot])


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
      color: #D6D6DB;
      margin-bottom: 0.75rem;
  }
  [data-testid="stMarkdownContainer"] li { font-size: 1.05rem; line-height: 1.66; color: #D6D6DB; }
  [data-testid="stMarkdownContainer"] strong { color: #FAFAFA; font-weight: 650; }
  [data-testid="stMarkdownContainer"] code {
      background: #131315; color: #9AD6B0; padding: 0.1em 0.36em; border-radius: 4px;
  }

  /* ------------------------------------------------------------------ head */
  .kml-chapter-head { text-align: center; margin-bottom: 0.2rem; }
  .kml-chapter-head h1 {
      margin: 0.2rem 0 0.3rem 0;
      font-size: clamp(1.5rem, 3.2vw + 0.7rem, 2.15rem); letter-spacing: -0.02em;
      color: #FAFAFA; line-height: 1.15;
  }
  .kml-part {
      text-transform: uppercase; letter-spacing: 0.13em; font-size: 0.7rem;
      color: #8A8A93; font-weight: 700;
  }
  .kml-idea {
      color: #A0A0A9; font-size: clamp(0.95rem, 1vw + 0.72rem, 1.12rem);
      margin: 0 auto 0.5rem auto; max-width: 60ch;
  }

  /* ----------------------------------------------------------------- trail */
  .kml-trail {
      display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.7rem 0 0.5rem 0;
      justify-content: center;
  }
  .kml-beat {
      font-size: 0.72rem; padding: 0.18rem 0.66rem; border-radius: 999px;
      background: #131315; color: #7A7A83; font-weight: 600;
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
      letter-spacing: -0.015em; color: #FAFAFA;
  }
  .kml-step-count {
      font-size: 0.7rem; color: #7A7A83; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.11em;
  }

  /* ------------------------------------------------------------------ boxes */
  .kml-box { padding: 0.85rem 1.1rem; border-radius: 10px; margin: 0.85rem 0; }
  .kml-predict {
      background: linear-gradient(135deg, #06101B 0%, #08131F 100%);
      border-left: 3px solid #60A5FA; color: #DCE4EE; font-size: 1.05rem;
  }
  .kml-look {
      background: #15110A; border-left: 3px solid #D9A441;
      color: #E4D9C0; font-size: 1.0rem;
  }
  .kml-jargon {
      background: #0B0B0D; border-left: 3px solid #2B2B30;
      color: #A0A0A9; font-size: 0.97rem;
  }

  /* Green means "good news, stop and enjoy this": the aha moment, the Little Kid
     Corner, and a correct prediction. */
  .kml-aha, .kml-kid, .kml-right {
      background: linear-gradient(135deg, #04140D 0%, #071A11 100%);
      border-left: 3px solid #34D399;
      color: #C6E8D6;
      box-shadow: 0 0 22px rgba(52, 211, 153, 0.10);
  }
  .kml-aha b, .kml-kid b, .kml-right b { color: #6EE7B7; }
  .kml-careful {
      background: #15110A; border-left: 3px solid #F59E0B; color: #E4D9C0;
  }
  .kml-careful b { color: #FBBF24; }
  .kml-surprise {
      background: #110B18; border-left: 3px solid #A78BFA; color: #DBD2EC;
  }
  .kml-wrong {
      background: #170809; border-left: 3px solid #F87171; color: #F0D0D0;
  }
  .kml-wrong b { color: #FCA5A5; }
  .kml-surprise b { color: #C4B5FD; }

  /* Only the boxes whose bold text is a *heading* get their own line. In the
     look-for and jargon boxes the bold run sits mid-sentence, and making it a
     block stranded the emoji on the line above it. */
  .kml-aha > b, .kml-kid > b, .kml-careful > b {
      display: block; margin-bottom: 0.35rem; font-size: 1.02rem;
  }
  .kml-look > b, .kml-jargon > b, .kml-surprise > b, .kml-right > b, .kml-wrong > b {
      display: inline;
  }
  /* The emoji and the words after it stay on one line. */
  .kml-look, .kml-jargon { display: block; }
  .kml-look > b { margin-right: 0.15rem; }
  .kml-box p { margin: 0 0 0.5rem 0; }
  .kml-box p:last-child { margin-bottom: 0; }
  .kml-aha, .kml-kid { animation: kmlPop 0.42s cubic-bezier(0.22, 1, 0.36, 1) both,
                                  kmlGlow 1.8s ease-in-out 1; }

  /* --------------------------------------------------------------- figures */
  [data-testid="stImage"] { display: flex; justify-content: center; }
  [data-testid="stImage"] img { background: transparent; border-radius: 8px; }
  [data-testid="stDataFrame"], [data-testid="stTable"] { margin: 0 auto; }
  [data-testid="stMetric"] {
      background: #0B0B0D; border: 1px solid #1E1E21; border-radius: 10px;
      padding: 0.65rem 0.9rem; text-align: center;
  }

  /* ------------------------------------------------------------------- nav */
  .kml-nav-space {
      margin-top: 2rem; border-top: 1px solid #1E1E21; padding-top: 0.6rem;
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

  /* The primary button is a bright green slab. Streamlit puts white text on it, which
     is about 1.8:1 against #34D399 — unreadable. Very dark green ink on the same slab
     is over 10:1 and still looks like the same button. */
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="stBaseButton-primary"],
  .stButton > button[kind="primaryFormSubmit"] {
      background: #34D399 !important;
      border-color: #34D399 !important;
      color: #04160E !important;
      font-weight: 750;
  }
  .stButton > button[kind="primary"] *,
  .stButton > button[data-testid="stBaseButton-primary"] *,
  .stButton > button[kind="primaryFormSubmit"] * { color: #04160E !important; }
  .stButton > button[kind="primary"]:hover:not(:disabled),
  .stButton > button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
      background: #6EE7B7 !important; border-color: #6EE7B7 !important;
  }
  /* A disabled Next (the last screen) must not look like a live one. */
  .stButton > button[kind="primary"]:disabled,
  .stButton > button[data-testid="stBaseButton-primary"]:disabled {
      background: #14201A !important; border-color: #1E2B25 !important;
      color: #4A5A52 !important;
  }
  .stButton > button[kind="primary"]:disabled * { color: #4A5A52 !important; }

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
  [data-testid="stRadio"] label:hover { color: #FAFAFA; transform: translateX(2px); }

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
  .kml-beat:hover { color: #C0C0C7; }
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

  /* AMOLED means AMOLED everywhere. Streamlit lifts the sidebar to its secondary
     colour by default, which leaves a visible grey slab beside a black page. */
  [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
      background: #000000 !important;
      border-right: 1px solid #161619;
  }
  [data-testid="stSidebarHeader"], [data-testid="stSidebarUserContent"] {
      background: transparent !important;
  }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background: #000000 !important;
  }

  /* Diagrams sit in the middle of the reading column like everything else. */
  [data-testid="stCustomComponentV1"] { margin: 0 auto; display: block; }

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
      background: #131315; color: #FAFAFA; padding-left: 0.85rem !important;
  }
  [data-testid="stSidebarNav"] a[aria-current="page"] {
      background: #06150F; color: #6EE7B7;
      box-shadow: inset 2px 0 0 #34D399;
  }

  /* Streamlit's own chrome. The Deploy button is meaningless here and sat in the
     top-right of every chapter. */
  #MainMenu, footer, header [data-testid="stToolbar"],
  [data-testid="stDecoration"], [data-testid="stStatusWidget"],
  .stAppDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
  header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
</style>
"""


def apply_style() -> None:
    """Inject the course stylesheet. Chapters get this from :func:`begin`; the landing
    page calls it directly so the whole app looks like one thing."""
    st.markdown(_STYLE, unsafe_allow_html=True)


def _inject_style() -> None:
    apply_style()
