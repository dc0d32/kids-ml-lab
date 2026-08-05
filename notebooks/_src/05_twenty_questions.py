# %% [markdown]
# # Chapter 05 · Twenty Questions
#
# ### Decision trees ask their way to an answer.
#
# *Part 1 · Classical models*
#
# ---
#
# In Chapter 3, a straight line failed on circles and XOR. We had two escapes:
# **invent new features**, or use a **bendy model**. This is the first bendy model.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml.datasets import load_table
from kidsml.plots import decision_boundary, use_house_style
from kidsml.trees import (
    creature_feature_names,
    creature_split_table,
    fit_creature_tree,
    fit_tree_shape,
    mushroom_tree,
    plot_decision_tree,
    shallow_mushroom_scores,
    tree_depth_scores,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# Think of **Twenty Questions** or **Guess Who**.
#
# Is it bigger than a cat? Does it have wings? You split the pile, then ask another
# question. A decision tree is that game. The clever part is choosing the first question.
#
# > 🧸 **Little Kid Corner** — Put toy animals in a pile. Ask one yes/no question, like
# > *does it have wings?* Move the yes toys left and the no toys right. Keep asking until
# > each pile has one answer.
#
# > 📖 **Grown-ups call this:** a **decision tree** is a model that asks yes/no questions
# > until it reaches an answer.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Here are ten made-up creatures. We want to guess `can_fly`.

# %%
creatures = load_table("creatures")
creatures

# %% [markdown]
# A mixed bucket is messy. A clean bucket is good.
#
# For two answers, the bucket mix score is:
#
# `1 - p_yes² - p_no²`
#
# A bucket with 3 flyers and 3 non-flyers has `1 - (3/6)² - (3/6)² = 0.5`.
# A bucket with 4 flyers and 0 non-flyers has `1 - 1² - 0² = 0`.
#
# Now try every possible first question.

# %%
splits = creature_split_table()
splits

# %%
print("Best first question:", splits.iloc[0]["first question"])

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(5)

# %% [markdown]
# ## 👀 See It
#
# Now sklearn builds a tree from the same ten rows. It picks the same first question.

# %%
model, X_creatures, y_creatures = fit_creature_tree(max_depth=3)
fig, ax = plt.subplots(figsize=(11, 5))
plot_decision_tree(model, creature_feature_names(), ["cannot fly", "can fly"], ax=ax)
plt.show()

# %%
print("Computer's first split:", creature_feature_names()[int(model.tree_.feature[0])])

# %% [markdown]
# ## 🎛️ Play With It
#
# Change `DEPTH`. A tree bends by making **stairs**: horizontal and vertical cuts.
# Depth 1 is a stump. Depth 20 can carve a tiny box around almost every point.

# %%
SHAPE = "moons"
NOISE = 0.20
N = 220
SEED = 1
DEPTH = 4

model, X_train, X_test, y_train, y_test = fit_tree_shape(SHAPE, DEPTH, n=N, noise=NOISE, seed=SEED)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
decision_boundary(model.predict, X_train, y_train, ax=ax, steps=180, shade_confidence=False, title=f"max_depth = {DEPTH}")
plt.show()
print("train accuracy:", round(model.score(X_train, y_train), 3))
print("test accuracy:", round(model.score(X_test, y_test), 3))

# %%
scores = tree_depth_scores(SHAPE, n=N, noise=NOISE, seed=SEED)
scores

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(scores["max_depth"], scores["train accuracy"], marker="o", label="train")
ax.plot(scores["max_depth"], scores["test accuracy"], marker="o", label="test")
ax.set_xlabel("tree depth")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()

# %% [markdown]
# > 💡 **Aha!** A tree is not smooth-bendy. It is **blocky-bendy**. Each question cuts only
# > left-right or up-down. Enough little cuts can carve almost any shape.
#
# > ⚠️ **Careful** A depth-20 tree can get 100% on its own dots by memorising tiny boxes.
# > That is like memorising last year's test answers. It feels great until the questions change.

# %% [markdown]
# ## 💻 For Real
#
# Mushrooms are a real table of words. A column that says `smell` cannot be a number, so
# we make one yes/no column per smell. That is one-hot encoding.

# %%
mush_model, X_train, X_test, y_train, y_test, mush_scores, text = mushroom_tree(max_depth=4)
mush_scores

# %%
shallow_mushroom_scores()

# %%
print(text[:2200])

# %% [markdown]
# Smell sits at the top. Real mushroom guides talk about smell too. The data and the
# foragers agree. In this bundled table, depth 4 is the first tree that gets almost all
# test mushrooms right.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Find the shallowest mushroom tree that stays above **95%** on the test set.
# 2. Find the tree depth where training and test accuracy are farthest apart.
# 3. Turn one mushroom rule into a sentence you could tell a person.
# 4. 🧸 **Little Kid Corner:** Play Guess Who with animals. What is the best first question?
#
# ---
# **Next up:** Chapter 06 · *A Crowd of Trees* — where many small guessers vote and fix
# each other's mistakes.
