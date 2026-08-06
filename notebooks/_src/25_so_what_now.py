# %% [markdown]
# # Chapter 25 · So What Now?
#
# ### The whole map, the honest limits, and what to build next.
#
# *Part 6 · Making things up*
#
# ---
#
# No new algorithm. This is the victory lap, the map spread on the table, and the safety check.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml.plots import ACCENT, COOL, INK, MUTED, PANEL, WARM, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# You opened with a secret-rule game. You ended by training a tiny Transformer.
#
# That is not a toy achievement. It is the map of modern machine learning, built brick by
# brick with small numbers you could touch. Nice work!
#
# > 🧸 **Little Kid Corner** — You did not get one magic wand. You filled a backpack with
# > tools. Now you can reach in and pick the tool for the job.

# %% [markdown]
# ## ✏️ Work it out
#
# **You have personally built this backpack of tools. Shake it and it rattles:**
#
# - a linear model: a line that turns numbers into a score
# - a perceptron: a hand-trained yes/no neuron
# - decision trees: question ladders that split the data
# - ensembles: crowds of models voting or correcting each other
# - an SVM: the widest safe street between classes
# - backprop: blame passing, checked by numerical gradients
# - a CNN: shared sliding-window filters for pictures
# - k-means: moving centres that make unlabeled piles
# - PCA: best-shadow squishing that keeps spread
# - a bigram generator: one-letter-back counting
# - an embedding MLP: three-letter memory with learned letter addresses
# - a Transformer: attention choosing which earlier clues to use

# %% [markdown]
# ## 👀 Take a look
#
# One picture of the whole course. Follow the roads like a treasure map.

# %%
def course_map_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.8, 4.8)
    ax.axis("off")

    parts = [
        ("Start", 0, 4.0, [0]),
        ("Classical", 1, 3.2, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("Messy data", 2, 2.4, [10, 11]),
        ("Neural nets", 3, 3.55, [12, 13, 14, 15, 16, 17]),
        ("Seeing", 4, 2.4, [18, 19]),
        ("No labels", 5, 2.4, [20, 21]),
        ("Making things up", 6, 3.2, [22, 23, 24, 25]),
    ]

    for name, x, y, chapters in parts:
        ax.text(x, 4.55, name, ha="center", va="center", fontsize=10, weight="bold")
        for j, number in enumerate(chapters):
            yy = y - 0.35 * j
            colour = ACCENT if number in {1, 2, 13, 15, 19, 24} else COOL
            if number in {20, 21}:
                colour = WARM
            if number in {0, 25}:
                colour = MUTED
            ax.scatter([x], [yy], s=420, c=colour, edgecolors=PANEL, linewidths=1.4, zorder=3)
            ax.text(x, yy, f"{number:02d}", ha="center", va="center", color=INK, weight="bold", fontsize=9)

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
# Where these ideas already are in your life. The map keeps showing up in the wild. Big
# chat systems usually guess **tokens** — chunks of text — while ours used letters so the
# gears stayed visible.

# %%
uses = pd.DataFrame(
    [
        ["Recommendations", "Chapters 08 and 20", "Find things near things you already like, or group people and items by pattern."],
        ["Photo search", "Chapters 18, 19, 21", "Pictures are numbers, CNNs spot patterns, PCA can shrink them."],
        ["Autocomplete", "Chapters 22-24", "Guess the next letter or text piece over and over."],
        ["Spam filters", "Chapters 04, 09, 11", "Probability, model choice, and checking failure modes."],
        ["Voice assistants", "Chapters 15, 19, 24", "Layers, sliding windows for sound-like patterns, and language models for text."],
        ["Game AI", "Chapters 00, 05, 06, 15", "Rules from examples, trees, crowds, and small neural nets."],
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
# produce likely-looking text. Looking right and being right are different targets, so
# important answers need checking.
#
# **Bias.** A model copies patterns from its data. If the examples are lopsided, missing
# people, or unfair, that shape can come along for the ride. Chapter 10 gave you the
# warning lights.
#
# **Confidently wrong.** Out-of-distribution inputs can still get confident answers.
# Chapter 00 showed confidence from too few clues; Chapter 11 showed the edge of the map.
#
# **No world inside.** A language model knows patterns in text. It does not have a lived-in
# world the way you do.
#
# In the app, start the Chapter 24 model with a factual-looking prompt like `the moon is
# made of`. Watch for the important split: it can keep the *shape* of an answer without
# checking whether the answer is true.
#
# Being smart about AI:
#
# - Check things that matter.
# - Treat it as a tool, not an oracle.
# - Keep private stuff out of the machine.
# - Using it to learn is great. Using it to dodge learning is a bad trade.

# %% [markdown]
# ## 🏆 Go further
#
# A pile of weekend projects. A ⭐ marks a good first one. Every project says which chapters
# it uses and where the data comes from — all of it is stuff you can collect yourself or a
# set already bundled in this course.
#
# **🍿 Start here — an afternoon each (chapters 00-09)**
#
# - ⭐ **Secret-rule game.** You invent a yes/no rule; a friend guesses it from your examples. *Ch 00. Data: make it up on paper.*
# - ⭐ **Sweet or salty?** Rate 15 snacks on a couple of numbers, predict the label. *Ch 01, 05. Data: rate them yourself.*
# - **Penguin species guesser.** Name the species from body measurements. *Ch 05, 08. Data: bundled `penguins`.*
# - **Will it fly?** Build a decision tree by hand for made-up creatures. *Ch 05. Data: bundled `creatures`.*
# - **Monster boss detector.** Is this card a boss, from its stats? *Ch 05, 06. Data: bundled `monsters`.*
#
# **🖼️ Play with pictures (chapters 18-19)**
#
# - ⭐ **Handwritten-digit reader.** Train on the digits set, then feed it your own scribbles. *Ch 18, 19. Data: bundled digits.*
# - ⭐ **Your own symbol alphabet.** Invent 5 symbols, draw each 10 times, train a reader. *Ch 18, 19. Data: draw them in a paint app, save small.*
# - **Fashion sorter.** Tell a shirt from a sneaker. *Ch 19. Data: Fashion-MNIST, already cached in Ch 19.*
# - **Filter explorer.** Slide a 3x3 filter over your photo and watch edges pop. *Ch 19. Data: any photo you took.*
# - **Confusion detective.** Which two digits does your model mix up most? *Ch 18. Data: bundled digits.*
#
# **💬 Play with words (chapters 22-24)**
#
# - ⭐ **Babbler trained on you.** Feed it your own writing; it will sound weirdly like you. *Ch 22-24. Data: paste your text into a file.*
# - ⭐ **Rhyme machine.** Train the tiny Transformer on rhymes and read the wobbles. *Ch 24. Data: bundled `rhymes`.*
# - **Fantasy team-name generator.** Tune temperature, keep the best accidents. *Ch 22-23. Data: bundled `names` or a list you type.*
# - **Pet-name inventor.** Feed it 100 real pet names, generate new ones. *Ch 22-23. Data: type the list.*
# - **Attention peek.** Generate a line, then find which earlier letters it leaned on. *Ch 24. Data: bundled `rhymes`/`fables`.*
#
# **🔍 Find patterns with no answer key (chapters 08, 20, 21)**
#
# - ⭐ **Music mood clusters.** Group your songs by features you pick, like tempo and loudness. *Ch 20. Data: rate 30 of your songs.*
# - ⭐ **Penguin islands.** Cluster penguins with the species label hidden, then peek. *Ch 20. Data: bundled `penguins`.*
# - **Photo sorter for your room.** Cluster your own pictures into piles. *Ch 20, 21. Data: your phone photos, shrunk small.*
# - **Squish-and-see.** PCA your symbol images down to 2D and hunt for clumps. *Ch 21. Data: bundled digits or your own symbols.*
# - **Odd one out.** Run k-means, then find the point farthest from every centre. *Ch 08, 20. Data: any small table you collect.*
#
# **🔬 Be a scientist about it (chapters 09, 11)**
#
# - ⭐ **Predict your bus.** Log arrivals for two weeks, then test honestly on days you held out. *Ch 01, 09, 11. Data: log it yourself.*
# - ⭐ **Bias detective.** Build a lopsided dataset on purpose and catch the failure. *Ch 10, 11. Data: make a slanted table.*
# - **Mushroom safety, honestly.** High accuracy is not enough when a mistake is deadly. *Ch 05, 10, 11. Data: bundled `mushrooms`.*
# - **Data-leakage trap.** Sneak the answer into a feature and watch the score look too good. *Ch 11. Data: any bundled table.*
# - **Break your own model.** Find the input that fools it most, then explain why. *Ch 11. Data: your own project's model.*
#
# **🛠️ Build something someone else can use**
#
# - ⭐ **Rock-paper-scissors bot** that learns your habits and starts beating you. *Ch 00, 05. Data: your own moves as you play.*
# - ⭐ **"Is this spam?" filter** for a club or family chat. *Ch 04, 09. Data: label 50 of your own messages.*
# - **Bike-day advisor.** Guess the rental count from the forecast. *Ch 01, 10. Data: bundled `bikes`.*
# - **Flashcard picker** that guesses which card you are about to miss. *Ch 08, 10. Data: log your own right/wrong.*
# - **Name-my-pet app.** Type a vibe, get invented names back. *Ch 22-23. Data: bundled `names`.*

# %% [markdown]
# What to learn next: algebra makes model formulas easier to move around, vectors make data
# feel natural, and derivatives unlock the grown-up version of backprop. A GPU changes the
# scale: bigger batches, bigger models, more experiments before dinner.
#
# 🧸 **Little Kid Corner:** Pick one tool from the backpack and use it on something in your
# own room. Tiny project, real sparks!

# %%
from kidsml import workbook
workbook.render(25)

# %% [markdown]
# ---
# **Next up:** keep building. Tiny, personal projects beat giant plans!
