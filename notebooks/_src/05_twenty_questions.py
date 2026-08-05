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
# **invent new features**, or use a **bendy model**. This chapter is the first bendy
# model: it bends by asking questions.

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
# ## 🎣 Start here
#
# Think of **Twenty Questions** or **Guess Who**. The model gets a clipboard. Do not start with one giant rule.
# Grab one useful yes/no question, split the pile, then dive into each smaller pile
# with the next question.
#
# That is why a tree is a flowchart with sneakers on. The clever part is not the
# drawing; it is choosing which question leaves the next step with less mess.
#
# ```mermaid
# graph TD
#     A[Start with one pile] --> B{Has wings?}
#     B -->|yes| C{Has feathers?}
#     B -->|no| D[mostly cannot fly]
#     C -->|yes| E{Lives in water?}
#     C -->|no| F[can fly]
# ```
#
# Notice the tree asks **one question at a time**. A row never answers every question in
# the diagram; it walks one path until it reaches an answer.
#
# > 🧸 **Little Kid Corner** — Put toy animals in a pile. Ask one yes/no question, like
# > *does it have wings?* Move the yes toys left and the no toys right. Keep asking until
# > each pile has one answer.
#
# > 📖 **Grown-ups call this:** a **decision tree** is a model that asks yes/no questions
# > until it reaches an answer.

# %% [markdown]
# ## ✏️ Work it out
#
# Here are ten made-up creatures. We want to guess `can_fly`, but the tree is only allowed
# to start with **one** column. It tries possible first questions such as `has_wings` and
# `lives_in_water`.

# %%
creatures = load_table("creatures")
creatures

# %% [markdown]
# A bucket is **mixed** when different answers are stuck together like cereal in one
# bowl. Six animals with 3 flyers and 3 non-flyers is soupy. Four flyers and 0
# non-flyers is clean. Grown-ups also call a one-answer bucket **pure**.
#
# The Gini impurity score says: pick two random animals from the bucket. How likely are you to
# be surprised by two different answers? For two answers, the score is:
#
# `1 - p_yes² - p_no²`
#
# Half-and-half gives `1 - (3/6)² - (3/6)² = 1 - 9/36 - 9/36 = 0.5`. A clean bucket gives
# `1 - 1² - 0² = 0`. Lower means less mess left for the next question.
#
# > 📖 **Grown-ups call this:** **Gini impurity** is a bucket-mess score. `0`
# > means pure: every row in the bucket has the same answer.

# %%
splits = creature_split_table()
splits

# %% [markdown]
# The weighted mix column counts both buckets. For `has_wings`, the yes bucket has 6
# animals and mix `1 - (4/6)² - (2/6)² = 0.444`, while the no bucket has 4 animals and mix
# `0`. So the split score is `(6×0.444 + 4×0) / 10 = 0.267`.

# %%
print("Best first question:", splits.iloc[0]["first question"])

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(5)

# %% [markdown]
# ## 👀 Take a look
#
# The computer is not guessing from feelings. It tries the same kind of split table, picks
# the least-mixed question, and repeats that inside the new buckets. The split table did,
# as the adults apparently say, let him cook.
#
# Each split uses one column because a tree question has one job: send the row left or
# right. Later questions can use different columns, but only after the row has reached
# that branch.

# %%
model, X_creatures, y_creatures = fit_creature_tree(max_depth=3)
fig, ax = plt.subplots(figsize=(11, 5))
plot_decision_tree(model, creature_feature_names(), ["cannot fly", "can fly"], ax=ax)
plt.show()

# %% [markdown]
# Read the picture from top to bottom. At every box, the tree asks one yes/no question and
# sends the row down exactly one branch.

# %%
print("Computer's first split:", creature_feature_names()[int(model.tree_.feature[0])])

# %% [markdown]
# ## 🎛️ Your turn
#
# Change `DEPTH`. Chapter 3 asked for a bendy boundary. A tree can bend, but not like a
# smooth rubber band. On a two-column picture, `x1 <= 0.4` makes a vertical cut and
# `x2 <= -0.2` makes a horizontal cut.
#
# That is why the boundary becomes a staircase. More depth means more questions, and
# more questions stack more little rectangles onto the picture.

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

# %% [markdown]
# Notice the boundary is made only of horizontal and vertical cuts. The tree cannot draw a
# diagonal line; it can only stack enough stair steps to fake one.

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
# > 💡 **Aha!** A tree is **blocky-bendy**. Each question makes one straight cut, but a
# > chain of small cuts can wrap around moons, circles, or XOR without inventing new
# > features.
#
# > ⚠️ **Careful** A deep tree can score better on its own training dots by fencing tiny
# > boxes around awkward points. That is memorising: it learns *this exact dot goes red*
# > instead of a rule that helps on the next dot. Chapter 8 turns this worry into a fair
# > test.

# %% [markdown]
# ## 💻 In real code
#
# Real tables often contain words. Here the table is real-ish mushrooms: one row per
# mushroom, field-guide clues such as smell and cap shape, and a target column `edible`
# that says safe or poisonous.
#
# A column that says `smell = almond` cannot go straight into a tree as a sentence, so we
# turn it into yes/no columns like `smell_almond`, `smell_fishy`, and `smell_none`.
#
# This is called one-hot encoding. It gives the tree the same kind of question it already
# knows how to ask: is this column 0 or 1?

# %%
mushrooms = load_table("mushrooms")
smell_rows = mushrooms[mushrooms["smell"].isin(["almond", "fishy", "none"])].drop_duplicates("smell")
smell_rows = smell_rows[["smell", "edible"]].reset_index(drop=True)
encoded_smells = pd.get_dummies(smell_rows["smell"], prefix="smell").astype(int)
pd.concat([smell_rows, encoded_smells], axis=1)

# %% [markdown]
# **Look for:** one word turning into several yes/no switches. Each row has one smell
# switch turned on.

# %%
mush_model, X_train, X_test, y_train, y_test, mush_scores, text = mushroom_tree(max_depth=4)
mush_scores

# %%
shallow_mushroom_scores()

# %%
print(text[:2200])

# %% [markdown]
# Look at the top question before you read the whole printed tree. Smell sits near the
# top, and real mushroom guides talk about smell too. That agreement is a good sign: the
# model found a clue a human forager would recognise.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find the shallowest mushroom tree that stays above **95%** on the test set.
# 2. Find the tree depth where training and test accuracy are farthest apart.
# 3. Turn one mushroom rule into a sentence you could tell a person.
# 4. 🧸 **Little Kid Corner:** Play Guess Who with animals. What is the best first question?
#
# ---
# **Next up:** Chapter 06 · *A Crowd of Trees* — where many small guessers vote and fix
# each other's mistakes.
