# %% [markdown]
# # Chapter 25 · So What Now?
#
# ### The whole map, the honest limits, and what to build next.
#
# *Part 6 · Making things up*
#
# ---
#
# No new algorithm. This is the victory lap.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml.plots import ACCENT, COOL, MUTED, WARM, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# You started with a secret-rule game. You ended by training a tiny Transformer.
#
# That is not a toy achievement. It is the map of modern machine learning, built piece by
# piece with small numbers you could touch.
#
# > 🧸 **Little Kid Corner** — You did not get one magic wand. You filled a backpack with
# > tools. Now you know which tool to pull out.

# %% [markdown]
# ## ✏️ Work it out
#
# **You have personally built:**
#
# - a linear model
# - a perceptron trained by hand
# - decision trees
# - ensembles
# - an SVM
# - backprop from scratch, checked by numerical gradients
# - a CNN
# - k-means
# - PCA
# - a bigram generator
# - an embedding MLP
# - a Transformer

# %% [markdown]
# ## 👀 Take a look
#
# One picture of the whole course.

# %%
def course_map_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
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

fig = course_map_figure()
plt.show()

# %% [markdown]
# ## 🎛️ Your turn
#
# Where these ideas already are in your life.

# %%
uses = pd.DataFrame(
    [
        ["Recommendations", "Chapters 19-20", "Find things near things you already like, or group people/items by pattern."],
        ["Photo search", "Chapters 17-17 and 20", "Pictures are numbers, CNNs spot patterns, PCA can shrink them."],
        ["Autocomplete", "Chapters 22-23", "Guess the next letter or text piece over and over."],
        ["Spam filters", "Chapters 04, 08, 10", "Probability, model choice, and checking failure modes."],
        ["Voice assistants", "Chapters 14, 17, 23", "Layers for sound patterns plus language models for text."],
        ["Game AI", "Chapters 00, 05, 06, 11", "Rules from examples, trees, crowds, and small neural nets."],
        ["Translation", "Chapter 24", "Attention helps connect words far apart across languages."],
    ],
    columns=["thing", "chapters", "connection"],
)
uses

# %% [markdown]
# ## 💻 In real code
#
# The honest limits:
#
# **Hallucination.** Your Chapter 24 model was never trained to be right. It was trained to
# produce likely-looking text. Looking right and being right are different targets.
#
# **Bias.** A model copies its data. Chapter 10 already warned you about lopsided examples.
#
# **Confidently wrong.** Out-of-distribution inputs can still get a confident answer.
# Chapter 00 and Chapter 10 both showed that.
#
# **No world inside.** A language model knows patterns in text. It does not know the world
# the way you do.
#
# Being smart about AI:
#
# - Check things that matter.
# - Treat it as a tool, not an oracle.
# - Do not put private stuff into it.
# - Using it to learn is great. Using it to avoid learning is a bad trade.

# %% [markdown]
# ## 🏆 Go further
#
# Ten weekend projects.

# %%
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
projects

# %% [markdown]
# What to learn next: algebra makes model formulas easier to move around, vectors make data
# feel natural, and derivatives unlock the grown-up version of backprop. A GPU changes the
# scale: bigger batches, bigger models, more experiments before dinner.
#
# 🧸 **Little Kid Corner:** Pick one tool from the backpack and use it on something in your
# own room.

# %%
from kidsml import workbook
workbook.render(25)

# %% [markdown]
# ---
# **Next up:** keep building. Tiny, personal projects beat giant plans.
