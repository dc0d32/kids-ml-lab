# %% [markdown]
# # Chapter 06 · A Crowd of Trees
#
# ### Many weak guessers beat one strong one.
#
# *Part 1 · Classical models*
#
# ---
#
# Chapter 3 said a straight line fails on circles and XOR. Chapter 5 escaped with one
# blocky tree. This chapter uses a whole crowd of blocky trees.

# %%
import matplotlib.pyplot as plt

from kidsml.plots import decision_boundary, use_house_style
from kidsml.trees import (
    boosting_trace,
    fit_tree_and_forest,
    forest_vote_counts,
    monster_models,
    tiny_boosting_table,
    tiny_vote_table,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# At a party, ask everyone to guess how many jellybeans are in a jar. One person may be
# wildly wrong. The **average** of many guesses is often spooky-good.
#
# Trees can do that too. There are two crowd tricks: **vote**, or **take turns fixing
# mistakes**.
#
# > 🧸 **Little Kid Corner** — Ask five people where a hidden toy is. If four point under
# > the couch, check there first. The crowd vote is stronger than one noisy guess.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Crowd trick 1: each tiny tree votes red or blue. Tally the majority.

# %%
tiny_vote_table()

# %% [markdown]
# Crowd trick 2: fix what is left over. The leftover mistake is `actual - current guess`.

# %%
tiny_boosting_table()

# %% [markdown]
# > 📖 **Grown-ups call this:** an **ensemble** is a model made by combining many smaller
# > models.
#
# > 📖 **Grown-ups call this:** a **residual** is the leftover mistake.

# %% [markdown]
# ## 👀 See It
#
# One tree draws a jagged staircase. A forest is still made of trees, but the vote smooths
# the jagged edges.

# %%
X, y, tree, forest = fit_tree_and_forest(shape="moons", n_estimators=30, noise=0.25, seed=4)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
decision_boundary(tree.predict, X, y, ax=axes[0], steps=150, shade_confidence=False, title="one tree")
decision_boundary(forest.predict, X, y, ax=axes[1], steps=150, shade_confidence=False, title="30 trees voting")
plt.show()

# %%
forest_vote_counts(forest, [0.0, 0.0])

# %% [markdown]
# ## 🎛️ Play With It
#
# Gradient boosting is easier to see in 1D. Fit a stump. Plot the leftover mistakes. Fit
# the next stump to those leftovers. Add it on. Repeat.

# %%
trace = boosting_trace(n_steps=20, learning_rate=0.25, max_depth=1, seed=0)
stage = trace["stages"][11]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].scatter(trace["x"], trace["y"], color="#3B82F6", edgecolors="white")
axes[0].plot(trace["x_grid"], stage["running_grid"], color="#10B981")
axes[0].set_title("running prediction")
axes[1].scatter(trace["x"], stage["residual"], color="#EF4444", edgecolors="white")
axes[1].axhline(0, color="#94A3B8", linewidth=1.4)
axes[1].set_title("leftover mistakes")
axes[2].plot(trace["x_grid"], stage["newest_grid"], color="#10B981")
axes[2].set_title("newest little tree")
plt.show()

# %% [markdown]
# > 💡 **Aha!** A smooth-looking curve can be built out of many tiny step shapes.
#
# > ⚠️ **Careful** Boosting learns in order. Too many strong fixes can chase noise and
# > wobble around the points.
#
# Forests are independent and hard to mess up. Boosting is sequential, often a bit
# stronger, and easier to overfit. On table data, boosted trees win many competitions.
# That surprises people who expected the answer to always be neural networks.

# %% [markdown]
# ## 💻 For Real
#
# The monster table was generated from a secret rule, with 5% of labels flipped on purpose.

# %%
scores, importances, secret = monster_models()
scores

# %%
importances

# %%
print(secret)

# %% [markdown]
# Attack, magic, and speed rise to the top. Element, home, and height do not matter much.
# That matches the secret rule. A model scoring 100% would be suspicious because some
# labels are deliberately wrong.

# %% [markdown]
# ## �� Challenge
#
# 1. How few trees does the forest need before it beats the one tree?
# 2. Make boosting overfit the wiggle: use many steps and deeper tiny trees.
# 3. Find a monster feature the model ignores. Does the secret rule agree?
# 4. 🧸 **Little Kid Corner:** Guess jellybeans with a group. Compare one guess with the average.
#
# ---
# **Next up:** Chapter 07 · *The Widest Road* — where separating is not enough; we want
# the biggest safe gap.
