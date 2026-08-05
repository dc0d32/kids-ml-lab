# %% [markdown]
# # Chapter 08 · The Model Zoo
#
# ### Which model when, and how not to fool yourself.
#
# *Part 1 · Classical models*
#
# ---
#
# You now know several guessers: lines, probabilities, trees, crowds, and widest roads.
# Which one should you use? The honest answer is: **try them and see**.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml.plots import use_house_style
from kidsml.trees import (
    MODEL_PERSONALITIES,
    deep_tree_train_test,
    fold_scores,
    lopsided_baseline,
    penguin_leaderboard,
    plot_folds,
    plot_zoo,
    split_bounce_scores,
)

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# *See* is harder than it sounds. This chapter is mostly about not lying to yourself.
#
# > 🧸 **Little Kid Corner** — If you test a bike, a scooter, and skates, use the same
# > hill for all three. A fair race needs fair rules.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Cut 10 rows into 5 folds of 2 rows. Each round hides one fold as the test set.

# %%
fold_table = pd.DataFrame(
    {
        "round": [1, 2, 3, 4, 5],
        "test rows": ["1, 2", "3, 4", "5, 6", "7, 8", "9, 10"],
        "score": [0.80, 0.70, 0.90, 0.80, 0.60],
    }
)
fold_table

# %%
print("average =", (0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5)

# %% [markdown]
# > 📖 **Grown-ups call this:** **cross-validation** means taking turns hiding different
# > chunks, then reporting the average and spread.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(8)

# %% [markdown]
# ## 👀 See It
#
# Here is the model zoo. The same data goes to every model. No model wins on every shape.

# %%
fig = plot_zoo(shape="moons", n=180, noise=0.20, seed=0)
plt.show()

# %%
pd.DataFrame({"model": list(MODEL_PERSONALITIES), "personality": list(MODEL_PERSONALITIES.values())})

# %% [markdown]
# > 💡 **Aha!** No model wins on every shape. That is not a cop-out. It is the real state
# > of the field.

# %% [markdown]
# ## 🎛️ Play With It
#
# If I let you study the exact test questions, your score means nothing.

# %%
pd.DataFrame([deep_tree_train_test()])

# %% [markdown]
# > ⚠️ **Careful** Evaluating on the training data is a fake victory. A deep tree can
# > score 100% there and still miss new points.

# %%
bounce = split_bounce_scores(test_size=0.30, max_seed=10)
bounce

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(bounce["seed"], bounce["test accuracy"], marker="o")
ax.set_xlabel("split seed")
ax.set_ylabel("test accuracy")
plt.show()

# %%
scores = fold_scores()
fig, ax = plt.subplots(figsize=(7, 3.6))
plot_folds(ax=ax)
plt.show()
print("fold scores:", [round(float(s), 3) for s in scores])
print("mean:", round(float(scores.mean()), 3), "spread:", round(float(scores.std()), 3))

# %%
print("lopsided baseline accuracy:", lopsided_baseline())

# %% [markdown]
# A useless model can score high when one class is much more common. Chapter 10 digs into
# that trap. A score without a spread is half a fact.

# %% [markdown]
# ## 💻 For Real
#
# Here is an honest penguin leaderboard: mean ± spread from five folds, sorted.

# %%
penguin_leaderboard()

# %% [markdown]
# If two means are closer than their spreads, do not brag that one crushed the other.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Find a shape where logistic regression wins or ties.
# 2. Find a seed where a bad model looks good.
# 3. Make two models swap places by changing only the split seed.
# 4. 🧸 **Little Kid Corner:** Race three toys down the same ramp three times. Report the average and the wiggle.
#
# ---
# **Next up:** Chapter 09 · *Real Data, Real Mess* — where the tables stop behaving neatly.
