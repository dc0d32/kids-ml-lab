"""Shared Streamlit furniture.

Every chapter page looks the same: a title, the six beats in the same order, the same
little coloured boxes. This module owns that look so the pages themselves stay short
and are mostly about the actual idea.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from kidsml.plots import use_house_style

# ---------------------------------------------------------------------------
# The course map. Single source of truth for the sidebar, Home page and tests.
# ---------------------------------------------------------------------------

CHAPTERS = [
    # (number, slug, title, one-line big idea, part)
    (0, "guessing_machine", "The Guessing Machine",
     "A computer can learn a rule from examples — and you can race it.", "Part 0 · What even is this?"),

    (1, "lines_that_predict", "Lines That Predict",
     "y = w·x + b, and the idea of 'how wrong am I?'", "Part 1 · Classical models"),
    (2, "lines_that_decide", "Lines That Decide",
     "One line can split the whole world in two.", "Part 1 · Classical models"),
    (3, "ruler_not_enough", "When a Ruler Isn't Enough",
     "Some things a straight line can never do, no matter how you turn it.", "Part 1 · Classical models"),
    (4, "maybe_probably", "Maybe, Probably, Definitely",
     "Squishing any number into a probability.", "Part 1 · Classical models"),
    (5, "twenty_questions", "Twenty Questions",
     "Decision trees ask their way to an answer.", "Part 1 · Classical models"),
    (6, "crowd_of_trees", "A Crowd of Trees",
     "Many weak guessers beat one strong one.", "Part 1 · Classical models"),
    (7, "widest_road", "The Widest Road",
     "Don't just separate — separate with the biggest gap.", "Part 1 · Classical models"),
    (8, "model_zoo", "The Model Zoo",
     "Which model when, and how not to fool yourself.", "Part 1 · Classical models"),

    (9, "real_data", "Real Data, Real Mess",
     "Penguins, mushrooms, monsters and bikes.", "Part 2 · Escaping Flatland"),
    (10, "models_go_wrong", "Where Models Go Wrong",
     "Bias, leakage, and being confidently wrong.", "Part 2 · Escaping Flatland"),

    (11, "arrows_and_grids", "Arrows and Grids",
     "A matrix isn't a box of numbers. It's an instruction for moving space.",
     "Part 3 · Neural networks"),
    (12, "one_neuron", "One Neuron",
     "It's Chapter 2 plus a squish. That's all.", "Part 3 · Neural networks"),
    (13, "how_it_learns", "How a Neuron Learns",
     "Backprop by hand, then in 30 lines of NumPy.", "Part 3 · Neural networks"),
    (14, "two_layers", "Two Layers, Three Neurons",
     "Hidden neurons each draw a line — together they bend.", "Part 3 · Neural networks"),
    (15, "deeper_wider", "Deeper and Wider",
     "More layers, different squishes, and over-studying.", "Part 3 · Neural networks"),
    (16, "in_pytorch", "Same Thing, in PyTorch",
     "Nothing magic — we check its gradients against ours.", "Part 3 · Neural networks"),

    (17, "pictures_are_numbers", "Pictures Are Just Numbers",
     "Read a digit off a grid of numbers, then teach a net to.", "Part 4 · Seeing"),
    (18, "sliding_window", "The Sliding Window",
     "Convolutions by pencil, then a tiny CNN.", "Part 4 · Seeing"),

    (19, "your_neighbors", "You Are Like Your Neighbors",
     "The model that does no training at all.", "Part 5 · Without answers"),
    (20, "sorting_without_labels", "Sorting Without Labels",
     "k-means — and your photo squeezed to 5 colours.", "Part 5 · Without answers"),
    (21, "squishing_dimensions", "Squishing Dimensions",
     "PCA is picking the best shadow to cast.", "Part 5 · Without answers"),

    (22, "bigram_babbler", "The Bigram Babbler",
     "Count letter pairs, roll a die, invent words.", "Part 6 · Making things up"),
    (23, "giving_it_memory", "Giving It a Memory",
     "It discovers vowels on its own. Nobody told it.", "Part 6 · Making things up"),
    (24, "paying_attention", "Paying Attention",
     "A tiny Transformer, with its attention shown live.", "Part 6 · Making things up"),
    (25, "so_what_now", "So What Now?",
     "The whole map, the honest limits, and what to build next.", "Part 6 · Making things up"),
]

CHAPTER_BY_NUMBER = {c[0]: c for c in CHAPTERS}


def chapter_slug(number: int) -> str:
    return CHAPTER_BY_NUMBER[number][1]


def page_filename(number: int) -> str:
    """The Streamlit page filename for a chapter, e.g. ``03_ruler_not_enough.py``."""
    return f"{number:02d}_{chapter_slug(number)}.py"


def notebook_filename(number: int) -> str:
    """The notebook filename for a chapter, e.g. ``03_ruler_not_enough.ipynb``."""
    return f"{number:02d}_{chapter_slug(number)}.ipynb"


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


def page_setup(number: int) -> None:
    """Call at the very top of a chapter page. Sets the title, icon and plot style."""
    _, _, title, idea, part = CHAPTER_BY_NUMBER[number]
    st.set_page_config(page_title=f"Ch {number:02d} · {title}", page_icon="🧪", layout="wide")
    use_house_style()
    st.caption(part)
    st.title(f"Chapter {number:02d} · {title}")
    st.markdown(f"### {idea}")
    st.divider()


# The six beats. Each one prints a consistent little header.
_BEATS = {
    "hook": ("🎣", "Start here"),
    "byhand": ("✏️", "Work it out"),
    "seeit": ("👀", "Take a look"),
    "play": ("🎛️", "Your turn"),
    "forreal": ("💻", "In real code"),
    "challenge": ("🏆", "Go further"),
}


def beat(name: str, subtitle: str = "") -> None:
    """Start one of the six beats of a chapter."""
    icon, label = _BEATS[name]
    st.divider()
    st.header(f"{icon} {label}")
    if subtitle:
        st.markdown(f"*{subtitle}*")


def little_kid_corner(body: str) -> None:
    """The 🧸 box: the same idea, explained without any algebra."""
    st.markdown(
        f"<div class='kml-box kml-kid'><b>🧸 Little Kid Corner</b><br>{body}</div>",
        unsafe_allow_html=True,
    )


def aha(body: str) -> None:
    """The moment worth stopping on."""
    st.markdown(
        f"<div class='kml-box kml-aha'><b>💡 Aha!</b><br>{body}</div>",
        unsafe_allow_html=True,
    )


def careful(body: str) -> None:
    """A trap worth pointing out before they fall into it."""
    st.markdown(
        f"<div class='kml-box kml-careful'><b>⚠️ Careful</b><br>{body}</div>",
        unsafe_allow_html=True,
    )


def jargon(term: str, plain: str) -> None:
    """Introduce a piece of vocabulary *after* the idea, never before."""
    st.markdown(
        f"> 📖 **Grown-ups call this: _{term}_.** {plain}"
    )


def workbook(number: int) -> None:
    """Render this chapter's interactive workbook, right here on the page.

    There is no printer in this house, so the "do it by hand" questions live on screen:
    scrap paper for the working out, then type the answer in and get told why.
    """
    from kidsml import workbook as wb

    book = wb.load(number)
    if book is None:
        return
    st.divider()
    wb.render(number)


def worksheet_link(number: int) -> None:
    """Backwards-compatible alias for :func:`workbook`."""
    workbook(number)


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def show(fig, clear: bool = True) -> None:
    """Render a matplotlib figure and free it, so long pages don't eat all the memory."""
    st.pyplot(fig, width="content")
    if clear:
        plt.close(fig)


def mermaid(diagram: str, height: int = 320) -> None:
    """Draw a Mermaid diagram — for flows, pipelines and "what connects to what".

    Use this when the thing being explained is *structure* rather than data: how a
    prediction flows through a model, what order the steps happen in, how one chapter
    leads to the next. Use matplotlib when the thing being explained is *numbers*.

    The notebook equivalent is a fenced ```mermaid block inside a markdown cell, which
    JupyterLab renders natively. Keep the two in step.

    Keep diagrams small — six or seven boxes. A diagram that needs studying is a diagram
    that has stopped helping.
    """
    from streamlit_mermaid import st_mermaid

    st_mermaid(diagram.strip(), height=f"{height}px")


def figure(width: float = 6.0, height: float = 5.0):
    """A new figure in the house size. Returns ``(fig, ax)``."""
    return plt.subplots(figsize=(width, height))


def two_figures(width: float = 5.2, height: float = 4.6):
    """Side-by-side axes, for 'before and after' comparisons."""
    return plt.subplots(1, 2, figsize=(width * 2, height))


# ---------------------------------------------------------------------------
# Common controls
# ---------------------------------------------------------------------------


def shape_picker(default: str = "moons", key: str = "shape", include: tuple | None = None) -> str:
    """The 'pick a dataset shape' dropdown used all over Part 1."""
    from kidsml.datasets import SHAPE_BLURB, TOY_SHAPES

    options = list(include or TOY_SHAPES)
    choice = st.selectbox("Dataset shape", options, index=options.index(default), key=key)
    st.caption(SHAPE_BLURB[choice])
    return choice


def noise_slider(default: float = 0.2, key: str = "noise") -> float:
    return st.slider("How messy is the data?", 0.0, 0.6, default, 0.05, key=key)


def sample_slider(default: int = 200, key: str = "n") -> int:
    return st.slider("How many points?", 40, 600, default, 20, key=key)


def seed_slider(default: int = 0, key: str = "seed") -> int:
    return st.slider("Shuffle the data (random seed)", 0, 20, default, 1, key=key)
