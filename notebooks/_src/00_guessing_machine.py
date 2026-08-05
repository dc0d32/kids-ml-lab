# %% [markdown]
# # Chapter 00 · The Guessing Machine
#
# ### A computer can learn a rule from examples — and you can race it.
#
# *Part 0 · What even is this?*
#
# ---
#
# This notebook is the same chapter as the app, but with the code showing.
# Run a cell with **Shift + Enter**. Change the numbers. Break things. That's the point.

# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from kidsml.plots import use_house_style
from kidsml.zeeps import RULES, all_zeeps, encode, label_with, learning_curve, pretty

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# I am thinking of a **secret rule**.
#
# The rule decides whether a creature is a **zeep** or **not a zeep**. Every creature
# has three things about it: a **shape**, a **colour**, and a **size**. That's all you get.
#
# I won't tell you the rule. Instead I'll show you some creatures I've already sorted.
# Your job: work out the rule from the examples.
#
# That sentence — *work out the rule from the examples* — **is machine learning**.
# There is nothing else. Everything in the other 24 chapters is a different way of
# doing exactly this.
#
# > 🧸 **Little Kid Corner** — Think of the game where someone says *hot* or *cold*
# > while you look for a hidden toy. They never tell you where it is. You work it out
# > from their hints. Here the hints are the examples.

# %% [markdown]
# ### The whole world, all 18 creatures of it
#
# Three shapes × three colours × two sizes. That's every creature that can exist here.
# You can look at the entire universe of this problem in one go — which almost never
# happens again in this course.

# %%
zeeps = all_zeeps()
pretty(zeeps)

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Below are six creatures I have already sorted. **Cover the rest of this notebook with
# your hand.** Look only at the table, and work out my rule.

# %%
SECRET = "big_square"          # ← don't peek at what this means yet!
N_EXAMPLES = 6
SHUFFLE = 0

labels = label_with(zeeps, SECRET)

rng = np.random.default_rng(SHUFFLE)
order = rng.permutation(len(zeeps))
shown, hidden = order[:N_EXAMPLES], order[N_EXAMPLES : N_EXAMPLES + 3]

examples = pretty(zeeps.iloc[shown]).copy()
examples["zeep?"] = np.where(labels[shown], "✅ zeep", "❌ not a zeep")
examples

# %% [markdown]
# Now, **on paper**, write down your answer for each of these three. Don't scroll yet.

# %%
quiz = pretty(zeeps.iloc[hidden])
quiz

# %% [markdown]
# Written them down? Good. Here are the real answers.

# %%
truth = labels[hidden]
answers = quiz.copy()
answers["the truth"] = np.where(truth, "zeep", "not a zeep")
print("The secret rule was:", RULES[SECRET])
answers

# %% [markdown]
# ## 👀 See It
#
# Now the computer plays the same game. It gets **exactly what you got** — those same
# six examples and nothing else.
#
# But it has never heard of shapes or colours or zeeps. All it sees is a table of
# numbers. `circle` became 0, `square` became 1, `triangle` became 2. It has no idea
# what those mean, and it does not need to.

# %%
X = encode(zeeps)

what_the_computer_sees = pd.DataFrame(X[shown], columns=["shape", "colour", "size"])
what_the_computer_sees["answer"] = labels[shown].astype(int)
what_the_computer_sees

# %% [markdown]
# > 📖 **Grown-ups call this:** the columns it uses to guess are the **features**, and
# > the column it's trying to guess is the **label**.

# %% [markdown]
# ## 💻 For Real
#
# Here is the entire machine learning program. Three lines.

# %%
model = DecisionTreeClassifier(random_state=0)   # 1. pick a kind of guesser
model.fit(X[shown], labels[shown])               # 2. show it the examples
machine = model.predict(X[hidden]).astype(bool)  # 3. ask it about new ones

race = quiz.copy()
race["the computer says"] = np.where(machine, "zeep", "not a zeep")
race["the truth"] = np.where(truth, "zeep", "not a zeep")
race

# %%
print(f"The computer got {int((machine == truth).sum())} out of 3.")
print("How did you do?")

# %% [markdown]
# That's it. That's a machine learning program.
#
# Every single chapter from here on changes **line 1** — a different kind of guesser —
# or changes what goes into **line 2**. That's the whole course.

# %% [markdown]
# ## 🎛️ Play With It
#
# Now the most important picture in the course. We give the computer 1 example, then 2,
# then 3... all the way to 17, and each time we ask how often it gets the rest right.
#
# We repeat each one 60 times with different examples and average, so that one lucky
# deal doesn't fool us.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.5))

for rule in RULES:
    curve = learning_curve(rule, n_repeats=60)
    ax.plot(curve["n"], curve["accuracy"], marker="o", markersize=3.5, label=RULES[rule])

ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1.5)
ax.text(1, 0.515, "pure guessing", color="#94A3B8", fontsize=9)
ax.set_xlabel("how many examples the computer got to see")
ax.set_ylabel("how often it is right")
ax.set_title("More examples → better guesses")
ax.set_ylim(0.35, 1.02)
ax.legend(fontsize=8, loc="lower right")
plt.show()

# %% [markdown]
# > 💡 **Aha!**
# >
# > Nobody made the computer smarter. Nobody changed its program. Every line of code
# > was identical for every point on that graph.
# >
# > **The only thing that changed was how many examples it got to look at.**
# >
# > That is why people who work on AI spend most of their time thinking about data,
# > not about code.
#
# > ⚠️ **Careful**
# >
# > Look at the left end. With one or two examples the computer is barely better than
# > flipping a coin — but it still answers with total confidence. A model never says
# > *I don't know*. It just guesses. Remember that when you get to Chapter 10.
#
# Notice also that the five rules are **not equally hard**. "It is red" is learned
# almost immediately. "Exactly one of red / big" needs far more examples for the same
# score. Some patterns are just harder to spot — for you *and* for the machine.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. **Beat the machine.** Change `N_EXAMPLES` to `3` and re-run. Can you still get all
#    three right? Can the computer? Why is it so much harder for it than for you?
#
# 2. **Change the secret rule.** Set `SECRET` to `"red_xor_big"` and run the whole thing
#    again. Look back at the graph — why does this rule need so many more examples?
#
# 3. **Break it on purpose.** Try `SHUFFLE` values until all six of your examples are
#    zeeps. Now what can the computer possibly learn? What would *you* learn?
#
# 4. **Invent your own rule.** Open `kidsml/zeeps.py`, add an entry to `RULES` and
#    `_RULE_FUNCS`, and see where your rule lands on the graph. Can you invent one the
#    computer can *never* learn from 17 examples?
#
# 5. 🧸 **Little Kid Corner:** Play this with a real person. Think of a secret rule about
#    the things in your room — *anything blue*, or *anything you can eat*. Point at five
#    things and say yes or no for each. See how many they need before they get it.

# %% [markdown]
# ---
# **Next up:** Chapter 01 · *Lines That Predict* — where the guesses stop being
# yes-or-no and start being numbers.
