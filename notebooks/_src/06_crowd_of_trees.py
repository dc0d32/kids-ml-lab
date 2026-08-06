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
from IPython.display import Image

from kidsml import boostanim
from kidsml.datasets import load_table
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
# ## 🎣 Start here
#
# Shake the jellybean jar and ask a crowd. One person rockets
# too high. Another dives low. The **average** can land closer than most individual
# guesses because the high and low mistakes cancel.
#
# But that only works if the guesses are different. If everyone copied the same wrong
# number from the same person, averaging would repeat the same mistake.
#
# Forests create useful disagreement on purpose: each tree sees a random sample of rows
# and is allowed to consider random columns while it grows.
#
# ```mermaid
# graph TD
#     A[training table] --> B[random row samples]
#     A --> C[random column choices]
#     B --> D[many different trees]
#     C --> D
#     D --> E[majority vote]
# ```
#
# Notice why two trees can disagree even though they came from the same table. Tree 1 may
# miss row 17. Tree 2 may not get the `speed` column at a split. Their mistakes land in
# different puddles, and voting can steady them.
#
# > 📖 **Grown-ups call this:** a **random forest** is a crowd of decision
# > trees trained with random row samples and random column choices, then combined by vote.
#
# > 🧸 **Little Kid Corner** — Ask five people where a hidden toy is. If four point under
# > the couch, check there first. The crowd vote is stronger than one noisy guess when
# > people are not copying each other.

# %% [markdown]
# ## ✏️ Work it out
#
# Here five tiny trees vote red or blue. For point A, the tally is red, red, blue, red,
# red: `4 red` versus `1 blue`, so the crowd says red.
#
# Voting helps when errors point in different directions. If one tree overreacts to a
# noisy dot and another tree never saw that dot, the majority can swat away the odd vote.
# If all five trees learned the same bad rule, the vote will not save us.

# %%
tiny_vote_table()

# %% [markdown]
# Crowd trick 2 is different. Instead of independent trees voting at the end, boosting
# lines trees up in order. Each new tiny tree looks at what the current team still gets
# wrong.
#
# The leftover is called a residual: `actual answer - current guess`. If the real answer
# is 2 and the current guess is 5, the residual is `2 - 5 = -3`. The next tree learns to
# push that guess downward.

# %%
tiny_boosting_table()

# %% [markdown]
# In the table, point C starts at 5 but should be 8, so the leftover is `8 - 5 = 3`. If
# we add half of that fix, `5 + 1.5 = 6.5`. The new leftover is `8 - 6.5 = 1.5`, smaller
# than before.
#
# > 📖 **Grown-ups call this:** an **ensemble** is a model made by combining many smaller
# > models.
#
# > 📖 **Grown-ups call this:** **boosting** is an ensemble that trains models in order,
# > with each new model fixing the leftovers from the team so far.
#
# > 📖 **Grown-ups call this:** a **residual** is the leftover mistake: actual answer
# > minus current guess.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(6)

# %% [markdown]
# ## 👀 Take a look
#
# Chapter 5 gave us one blocky-bendy tree. Here the forest keeps that same building block,
# then lets many versions vote.
#
# The forest is not magic smoothing paint. It is many stair-step boundaries laid over the
# same problem, with random differences between them. The final vote can look calmer than
# any one tree.

# %%
X, y, tree, forest = fit_tree_and_forest(shape="moons", n_estimators=30, noise=0.25, seed=4)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
decision_boundary(tree.predict, X, y, ax=axes[0], steps=150, shade_confidence=False, title="one tree")
decision_boundary(forest.predict, X, y, ax=axes[1], steps=150, shade_confidence=False, title="30 trees voting")
plt.show()

# %% [markdown]
# Look at the edges. The single tree often has sharp little bites. The forest is still
# made of blocky cuts, but the vote can sand down lonely mistakes.

# %%
forest_vote_counts(forest, [0.0, 0.0])

# %% [markdown]
# ## 🎛️ Your turn
#
# Boosting is easier to see on one wiggly line. Start with a plain guess. Measure the
# leftovers. Fit a small tree to those leftovers. Add a small amount of that new tree to
# the running prediction. Then repeat.
#
# ```mermaid
# graph TD
#     A[predict] --> B[measure leftovers]
#     B --> C[fit tiny tree]
#     C --> D[add small fix]
#     D --> E[better prediction]
#     E --> B
# ```
#
# The loop works because the next tree is not trying to relearn the whole answer. It
# learns the part the team still misses. Lots of small corrections can build a curve that
# one tiny tree could never draw!
#
# A depth-1 tiny tree has one split. Grown-ups call that tiny tree a **stump**: short,
# blunt, and useful in a crowd.
#
# Watch the running total build itself. The green line starts as one flat step and gains a
# new stair every round, and the total creeps toward the blue dots. The fixes pile up — they
# do not replace each other.

# %%
Image(data=boostanim.staircase_gif_bytes(learning_rate=0.25, max_depth=1, seed=4))

# %% [markdown]
# Now freeze one moment and pull it apart into three panels.

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
# Read the three panels left to right. The middle panel is what remains wrong; the right
# panel is the newest fix; the left panel is the total after the fixes have been added.
#
# > 💡 **Aha!** A smooth-looking curve can be built out of many tiny step shapes.

# %%
trace = boosting_trace(n_steps=50, learning_rate=0.45, max_depth=3, seed=8)
early = trace["stages"][11]
late = trace["stages"][-1]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, stage, title in zip(axes, [early, late], ["12 fixes: pattern", "50 strong fixes: noise hunt"]):
    ax.scatter(trace["x"], trace["y"], color="#3B82F6", edgecolors="white")
    ax.plot(trace["x_grid"], stage["running_grid"], color="#10B981")
    ax.set_title(title)
plt.show()

# %% [markdown]
# Look at the extra bends in the right panel. Those wiggles are the model answering
# individual noisy dots.
#
# > ⚠️ **Careful** Boosting overfits more easily than a forest because it keeps staring at
# > the current mistakes. If some leftovers are noise from bad labels or wiggly data,
# > strong late trees may chase that noise instead of the real pattern.
#
# Forests are independent and sturdy. Boosting is sequential, often a bit stronger, and
# easier to overfit. On table data, boosted trees win many competitions. That surprises
# people who expected the answer to always be neural networks.

# %% [markdown]
# ## 💻 In real code
#
# Meet the monster table before the model touches it. Each row is a trading-card creature.
# The columns include words like `element` and `home`, plus battle stats such as `attack`,
# `magic`, `speed`, `height_cm`, and `weight_kg`. The target is `is_boss`: yes or no.
#
# The table was generated from a secret rule, with 5% of labels flipped on purpose. That
# means some training answers are lies. A perfect training score would be suspicious,
# because a model would have to learn the lies too.

# %%
load_table("monsters").groupby("is_boss", group_keys=False).head(2)[
    ["name", "element", "attack", "defense", "magic", "speed", "is_boss"]
]

# %%
scores, importances, secret = monster_models()
scores

# %%
importances

# %%
print(secret)

# %% [markdown]
# Look for the tallest bars before revealing the rule. Attack, magic, and speed rise to
# the top; element, home, and height matter much less. That matches the secret rule.
#
# > 📖 **Grown-ups call this:** **feature importance** is a score for how much a trained
# > model leaned on each column or group of columns.

# %% [markdown]
# ## 🏆 Go further
#
# 1. How few trees does the forest need before it beats the one tree?
# 2. Make boosting overfit the wiggle: use many steps and deeper tiny trees.
# 3. Find a monster feature the model ignores. Does the secret rule agree?
# 4. 🧸 **Little Kid Corner:** Guess jellybeans with a group. Compare one guess with the average.
#
# ---
# **Next up:** Chapter 07 · *The Widest Road* — where separating is not enough; we want
# the biggest safe gap.
