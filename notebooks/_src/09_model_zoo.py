# %% [markdown]
# # Chapter 09 · The Model Zoo
#
# ### Which model when, and how not to fool yourself.
#
# *Part 1 · Classical models*
#
# ---
#
# You now know logistic regression from Chapter 4, decision trees from Chapter 5, random
# forests and boosting from Chapter 6, support vector machines from Chapter 7, and nearest
# neighbours from Chapter 8. Which
# one should you use? The honest answer is: **try them and see**. But *see* is harder than
# it sounds, because a model can look brilliant on the rows it studied and stumble on new rows.

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
# ## 🎣 Start here
#
# Open the model zoo gate. This chapter is about fair races. Same hill. Same timer. We compare models on data
# they did not train on, repeat the race, and always ask what a boring baseline could
# score.
#
# ```mermaid
# graph TD
#     A[all labelled rows] --> B[training rows]
#     A --> C[hidden test rows]
#     B --> D[train model]
#     D --> E[predict hidden rows]
#     C --> E
#     E --> F[test score]
# ```
#
# Notice the wall between training rows and hidden test rows. If the model studies the
# test rows, the score stops being evidence about new data.
#
# > 🧸 **Little Kid Corner** — If you test a bike, a scooter, and skates, use the same
# > hill for all three. A fair race needs fair rules.

# %% [markdown]
# ## ✏️ Work it out
#
# One train/test split can be lucky. Maybe the easy rows landed in the test set. Maybe the
# hard rows did. Cross-validation turns that one race into several smaller races.
#
# Cut 10 rows into 5 folds of 2 rows. Each round hides one fold as the test set and trains
# on the other four folds. Every row gets a turn being hidden.
#
# ```mermaid
# graph TD
#     A[10 rows] --> B[5 folds]
#     B --> C[round 1: fold 1 tests]
#     B --> D[round 2: fold 2 tests]
#     B --> E[more rounds rotate]
#     C --> F[average and spread]
#     D --> F
#     E --> F
# ```
#
# The rotation is the point. A bouncy test score is not a bug; it is the scoreboard
# rattling and warning you that one split is a shaky fact.

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
# Average arithmetic: `(0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 3.80 / 5 = 0.76`.
#
# > 📖 **Grown-ups call this:** a **fold** is one chunk of rows held out for testing
# > during one round of cross-validation.
#
# > 📖 **Grown-ups call this:** **cross-validation** means taking turns hiding different
# > chunks, then reporting the average and spread.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(9)

# %% [markdown]
# ## 👀 Take a look
#
# Here is the model zoo. Every model gets the same training rows and the same hidden test
# rows. That keeps the race fair.
#
# Decode the labels before you read the mini-plots: `logistic` means logistic regression,
# `tree` means decision tree, `forest` means random forest, `boosting` is Chapter 6's line
# of little tree fixes, and `linear SVM` / `rbf SVM` are Chapter 7 roads.
#
# One guest is new: **k-nearest neighbors** (kNN). It stores the training rows and asks the
# nearby points to vote on a new row.
#
# Watch the shapes. Logistic likes straight-ish borders. Trees stack boxes. Boosting stacks
# little fixes. RBF SVMs pour smooth islands. kNN listens to nearby points. No personality
# wins every kind of problem.

# %%
fig = plot_zoo(shape="moons", n=180, noise=0.20, seed=0)
plt.show()

# %% [markdown]
# Look at both parts of each mini-plot: the boundary shape and the test score in the
# title. A model can have the wrong personality for one shape and the right personality
# for another.

# %%
pd.DataFrame({"model": list(MODEL_PERSONALITIES), "personality": list(MODEL_PERSONALITIES.values())})

# %% [markdown]
# > 💡 **Aha!** No model wins on every shape. That is not a cop-out. It is the real state
# > of the field.

# %% [markdown]
# ## 🎛️ Your turn
#
# Scoring a model on its own training data is like taking a practice test after memorising
# the answer key. It may tell you the model stored the rows. It does not tell you whether
# it learned a pattern that works on new rows.

# %%
pd.DataFrame([deep_tree_train_test()])

# %% [markdown]
# > ⚠️ **Careful** Evaluating on the training data is a fake victory. A deep tree can
# > score 100% there by memorising tiny boxes, then miss new points that do not land in
# > those boxes.
#
# Now change only the split seed. The model type and dataset stay the same; the rows
# assigned to the hidden test set change. If the test score jumps, that is useful
# information: this single split was noisy.

# %%
bounce = split_bounce_scores(test_size=0.30, max_seed=10)
bounce

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(bounce["seed"], bounce["test accuracy"], marker="o")
ax.set_xlabel("split seed")
ax.set_ylabel("test accuracy")
plt.show()

# %% [markdown]
# Cross-validation exists because of that bounce. Instead of trusting one split, it
# rotates through several hidden chunks and reports the average **and** the spread. A
# score without a spread tells you only half the story: how big the number was, but not
# how much it wobbles when you change the split.

# %%
scores = fold_scores()
fig, ax = plt.subplots(figsize=(7, 3.6))
plot_folds(ax=ax)
plt.show()
print("fold scores:", [round(float(s), 3) for s in scores])
print("mean:", round(float(scores.mean()), 3), "spread:", round(float(scores.std()), 3))

# %%
print("lopsided baseline accuracy:", lopsided_baseline())
pd.DataFrame(
    {
        "guesser": ["always common class", "fancy model"],
        "score": [f"{lopsided_baseline():.0%}", "92%"],
        "what it proves": ["the data is lopsided", "only 2 points above the floor"],
    }
)

# %% [markdown]
# A useless model can score high when one class is much more common. Check the baseline
# first, or you may throw confetti for a model that learned nothing. Chapter 11 digs into
# that trap.
#
# > 📖 **Grown-ups call this:** a **baseline** is a boring score from a model that does
# > not learn, such as always predicting the most common class.
#
# > 📖 **Grown-ups call this:** **class imbalance** means one answer pile is much bigger
# > than another, so accuracy can look high without useful learning.

# %% [markdown]
# ## 💻 In real code
#
# Here is the penguin race with fair rules. Each row is one Palmer penguin, the columns are
# measurements such as beak, flipper, island, and weight, and the target is species.
#
# Every model gets the same five folds. The table reports mean plus spread, and the
# baseline is included because a fancy model has to beat the boring answer before it earns
# applause.

# %%
penguin_leaderboard()

# %% [markdown]
# If two means are closer than their spreads, do not brag that one crushed the other. Read
# `0.96 ± 0.03` as a small cloud of possible scores, not one magic number.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find a shape where logistic regression wins or ties.
# 2. Find a seed where a bad model looks good.
# 3. Make two models swap places by changing only the split seed.
# 4. 🧸 **Little Kid Corner:** Race three toys down the same ramp three times. Report the average and the wiggle.
#
# ---
# **Next up:** Chapter 10 · *Real Data, Real Mess* — where the tables stop behaving neatly.
