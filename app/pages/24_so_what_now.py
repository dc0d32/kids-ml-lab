"""Chapter 24 · So What Now?"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.plots import ACCENT, COOL, MUTED, WARM

ui.page_setup(24)


def course_map_figure():
    fig, ax = ui.figure(10, 6)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.8, 4.8)
    ax.axis("off")

    parts = [
        ("Start", 0, 4.0, [0]),
        ("Classical", 1, 3.2, [1, 2, 3, 4, 5, 6, 7, 8]),
        ("Messy data", 2, 2.4, [9, 10]),
        ("Neural nets", 3, 3.2, [11, 12, 13, 14, 15]),
        ("Seeing", 4, 2.4, [16, 17]),
        ("No labels", 5, 1.6, [18, 19, 20]),
        ("Making things up", 6, 3.2, [21, 22, 23, 24]),
    ]

    for name, x, y, chapters in parts:
        ax.text(x, 4.55, name, ha="center", va="center", fontsize=10, weight="bold")
        for j, number in enumerate(chapters):
            yy = y - 0.35 * j
            colour = ACCENT if number in {1, 2, 11, 13, 17, 23} else COOL
            if number in {18, 19, 20}:
                colour = WARM
            if number in {0, 24}:
                colour = MUTED
            ax.scatter([x], [yy], s=420, c=colour, edgecolors="white", linewidths=1.4, zorder=3)
            ax.text(x, yy, f"{number:02d}", ha="center", va="center", color="white", weight="bold", fontsize=9)

    main_path = [(1, 3.2), (1, 2.85), (3, 3.2), (3, 2.5), (4, 2.05), (6, 2.5)]
    for (x1, y1), (x2, y2) in zip(main_path, main_path[1:]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2.2})
    ax.text(3.2, 0.15, "green path: one neuron → layers → vision → Transformer", color=ACCENT, ha="center", fontsize=10)
    ax.text(5.1, 0.55, "red branch: learning without labels", color=WARM, ha="center", fontsize=10)
    return fig


# ---------------------------------------------------------------------------
ui.beat("hook", "Look what you built.")

st.markdown(
    """
You started with a secret-rule game. You ended by training a tiny Transformer.

That is not a toy achievement. It is the map of modern machine learning, built piece by
piece with small numbers you could touch.
"""
)

ui.little_kid_corner("You did not get one magic wand. You filled a backpack with tools. Now you know which tool to pull out.")

# ---------------------------------------------------------------------------
ui.beat("byhand", "Your inventory.")

built = [
    "a linear model",
    "a perceptron trained by hand",
    "decision trees",
    "ensembles",
    "an SVM",
    "backprop from scratch, checked by numerical gradients",
    "a CNN",
    "k-means",
    "PCA",
    "a bigram generator",
    "an embedding MLP",
    "a Transformer",
]
st.markdown("**You have personally built:**")
for item in built:
    st.markdown(f"- {item}")

# ---------------------------------------------------------------------------
ui.beat("seeit", "One picture of the course.")

fig = course_map_figure()
ui.show(fig)
st.caption("The green line is the neural-network road. The red branch is learning without answer labels.")

# ---------------------------------------------------------------------------
ui.beat("play", "Where is this already in your life?")

uses = {
    "Recommendations": "Chapters 18-20: find things near things you already like, or group people/items by pattern.",
    "Photo search": "Chapters 16-17 and 20: pictures are numbers, CNNs spot patterns, PCA can shrink them.",
    "Autocomplete": "Chapters 21-23: guess the next letter or text piece over and over.",
    "Spam filters": "Chapters 04, 08, 10: probability, model choice, and checking failure modes.",
    "Voice assistants": "Chapters 13, 17, 23: layers for sound patterns plus language models for text.",
    "Game AI": "Chapters 00, 05, 06, 11: rules from examples, trees, crowds, and small neural nets.",
    "Translation": "Chapter 23: attention helps connect words far apart across languages.",
}
choice = st.selectbox("Pick a real-world thing", list(uses))
st.info(uses[choice])

# ---------------------------------------------------------------------------
ui.beat("forreal", "The honest limits.")

st.markdown(
    """
**Hallucination.** Your Chapter 23 model was never trained to be right. It was trained to
produce likely-looking text. Looking right and being right are different targets.

**Bias.** A model copies its data. Chapter 10 already warned you about lopsided examples.

**Confidently wrong.** Out-of-distribution inputs can still get a confident answer. Chapter
00 and Chapter 10 both showed that.

**No world inside.** A language model knows patterns in text. It does not know the world the
way you do.
"""
)

st.markdown(
    """
Being smart about AI:

- Check things that matter.
- Treat it as a tool, not an oracle.
- Do not put private stuff into it.
- Using it to learn is great. Using it to avoid learning is a bad trade.
"""
)

# ---------------------------------------------------------------------------
ui.beat("challenge", "Ten weekend projects.")

projects = pd.DataFrame(
    [
        ["Train the babbler on your own writing", "21-23", "It will sound weirdly like you."],
        ["Photo sorter for your room", "16-20", "Use your own pictures and cluster them."],
        ["Predict your bus arrival", "01, 08, 10", "Collect data for two weeks and test honestly."],
        ["Tiny game bot", "00, 05, 11", "Teach it from examples of your moves."],
        ["Music mood clusters", "18-20", "Group songs by features you choose."],
        ["Mushroom safety explainer", "05, 08, 10", "Accuracy is not enough when mistakes matter."],
        ["Handwritten symbol reader", "16-17", "Make a mini alphabet of your own symbols."],
        ["Name generator for a fantasy team", "21-22", "Tune temperature and pick the best accidents."],
        ["Bias detective", "10", "Build a lopsided dataset and catch the failure."],
        ["Attention poem machine", "23", "Train on poems you are allowed to use and inspect heads."],
    ],
    columns=["project", "chapters", "why it is interesting"],
)
st.dataframe(projects, hide_index=True, use_container_width=True)

st.markdown(
    """
What to learn next: algebra makes model formulas easier to move around, vectors make data
feel natural, and derivatives unlock the grown-up version of backprop. A GPU changes the
scale: bigger batches, bigger models, more experiments before dinner.
"""
)

st.markdown("🧸 **Little Kid Corner:** Pick one tool from the backpack and use it on something in your own room.")

ui.worksheet_link(24)
