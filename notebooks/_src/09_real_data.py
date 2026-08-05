# %% [markdown]
# # Chapter 09 · Real Data, Real Mess
#
# ### Penguins, mushrooms, Pokémon and bikes.
#
# *Part 2 · Escaping Flatland*
#
# ---
#
# We are leaving Flatland. The first eight chapters used two-column toy worlds you could
# draw on one plot. Real tables do not fit in one picture, so we need a new routine:
# inspect, clean, split, train, and compare with a boring baseline.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml import datasets, realdata, workbook
from kidsml.plots import ACCENT, confusion_grid, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Every dataset so far had **exactly two columns**, so we could draw the whole thing.
#
# Real data has twenty columns. Some are numbers. Some are words. Some cells are blank.
# You cannot draw the whole table on one neat graph.
#
# So the question changes. Instead of *can I draw the whole world?*, you ask: what kind
# of column is this, what is missing, and what would a boring guess score before my model
# learns anything?
#
# ```mermaid
# graph LR
#     A[Raw table] --> B[Encode words]
#     B --> C[Fill or drop gaps]
#     C --> D[Split rows]
#     D --> E[Train model]
#     E --> F[Score against baseline]
# ```
#
# Notice the order. We do not train first and clean later. The model can only learn from
# the table we hand it, so every cleanup choice becomes part of the experiment.
#
# > 🧸 **Little Kid Corner** — Imagine sorting a huge box of trading cards. You cannot
# > hold every card in the air at once. You look at one clue at a time.

# %%
penguins = datasets.load_table("penguins")
penguins.head(8)

# %%
print("rows:", len(penguins))
print("columns:", len(penguins.columns))
print("columns:", list(penguins.columns))

# %% [markdown]
# ## ✏️ Work it out
#
# Real tables give you three chores before the model deserves attention.
#
# ### 1. Word columns have to become yes/no columns

# %%
before, after = realdata.weather_one_hot_demo()
before

# %%
after

# %% [markdown]
# A model does arithmetic. It can compare `4 > 1`, multiply by 3, and split a tree at
# `weather < 2.5`. It cannot multiply by the word `storm`.
#
# Do **not** number weather as clear=1, misty=2, rain=3, storm=4 unless that order is
# real. The model may treat storm as four times clear, or misty as halfway between clear
# and rain. A tree may group clear and misty together because `weather <= 2.5`.
#
# The yes/no columns avoid the fake ruler. `weather = storm` is either 0 or 1, and it is
# not larger, warmer, or halfway between anything.
#
# > 📖 **Grown-ups call this:** **one-hot encoding** means turning each possible word into
# > its own 0-or-1 column.
#
# ### 2. Blanks are not free

# %%
missing = realdata.penguin_missing_rows()
missing.head(12)

# %%
realdata.penguin_missing_scores()

# %% [markdown]
# Dropping blank rows says, "these penguins never existed." That can erase the exact kind
# of penguin your measuring tools had trouble with.
#
# Filling blanks with an average tells a different lie: "this missing beak was perfectly
# ordinary." That keeps the row, but it hides the weirdness. Pick the lie you understand.
#
# ### 3. A boring guess comes first
#
# Before you are impressed by 82%, ask what the boring guess gets for free.
#
# If 80 out of 100 mushrooms are safe, a model that says **safe every time** scores 80%.
# A fancy model at 82% only bought two extra correct answers. A fancy model at 96% bought
# sixteen. Same scorecard, very different story.

# %%
realdata.all_dataset_scores()

# %% [markdown]
# > 📖 **Grown-ups call this:** a **baseline** is a boring score from a model that does
# > not learn. For classes, it says the most common answer every time.

# %% [markdown]
# ## 👀 Take a look
#
# Pick any bundled table and inspect it before training. Shape first. Column kinds next.
# Missing cells next. Then ask how lopsided the target is.
#
# Look first for giant piles. A lopsided target is where accuracy starts lying, because
# the most common answer may already score well.

# %%
DATASET = "mushrooms"  # try: penguins, mushrooms, monsters, bikes
info = realdata.table_overview(DATASET)
print(datasets.blurb_of(DATASET))
print("target:", info["target"])
print("shape:", info["rows"], "rows ×", info["columns"], "columns")
info["head"]

# %%
info["dtypes"]

# %%
info["missing"]

# %%
info["target_counts"].head(10)

# %% [markdown]
# ## 🎛️ Your turn
#
# The Feature Draft: pick the columns your model may use. This is not a guessing contest
# where the computer is always right; it is a draft.
#
# You choose a team of clues, train, then compare your hunch with what the model leaned
# on. If one column dominates the bar chart, ask whether it is a real clue or a sneaky
# shortcut.

# %%
DRAFT_DATASET = "monsters"
FEATURE_TEAM = ["element", "home", "attack", "magic", "speed"]

result = realdata.train_table_model(DRAFT_DATASET, features=FEATURE_TEAM)
print("baseline:", round(result["baseline_score"], 3))
print("model:", round(result["model_score"], 3))
print("rows used:", result["rows_used"])
result["importances"]

# %%
fig, ax = plt.subplots(figsize=(7, 3.8))
imp = result["importances"].sort_values("importance")
ax.barh(imp["column"], imp["importance"])
ax.set_xlabel("importance")
ax.set_title("What the forest leaned on")
plt.show()

# %% [markdown]
# Look for one tall bar. That column carried most of the model's decision, so it deserves
# a human sanity check.

# %%
print(datasets.MONSTER_SECRET_RULE)

# %% [markdown]
# Try your own drafts:
#
# - remove `speed`
# - use only `attack` and `magic`
# - add a weird column and see whether it helps
#
# On mushrooms, try to find the single question that gets you furthest.

# %% [markdown]
# ## 💻 In real code
#
# Two plots are worth building into your reflexes: a confusion matrix for classes, and
# predicted-versus-actual for numbers.

# %%
penguin_info = realdata.penguin_confusion()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
confusion_grid(penguin_info["cm"], labels=penguin_info["labels"], ax=ax, title="Penguin species confusion")
plt.show()
print("test accuracy:", round(penguin_info["score"], 3))

# %%
xcol, ycol = penguin_info["top"]
fig, ax = plt.subplots(figsize=(6, 4.6))
for species, part in penguin_info["data"].groupby("species"):
    ax.scatter(part[xcol], part[ycol], label=species, s=45, edgecolors="white", linewidths=0.8)
ax.set_xlabel(xcol)
ax.set_ylabel(ycol)
ax.set_title("The two most important measurements")
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# Look for species whose dots overlap. The confusion matrix above and this scatter plot
# are telling the same story: mistakes usually live where the measurements overlap.

# %%
bike = realdata.bike_regression()
rows = bike["rows"]
fig, ax = plt.subplots(figsize=(5.8, 5.0))
ax.scatter(rows["rentals"], rows["predicted"], s=35, alpha=0.85, edgecolors="white", linewidths=0.7)
lo = min(rows["rentals"].min(), rows["predicted"].min())
hi = max(rows["rentals"].max(), rows["predicted"].max())
ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", label="perfect")
ax.set_xlabel("actual rentals")
ax.set_ylabel("predicted rentals")
ax.set_title("Predicted versus actual")
ax.legend()
plt.show()
print("bike model R²:", round(bike["result"]["model_score"], 3))

# %%
bike["worst"]

# %% [markdown]
# Look for points far from the dashed perfect line. Those are not random embarrassments;
# they are clues. The worst bike mistakes are not random dots. Look at the dates. Rentals
# grew over the years this table covers, and our weather-only model did not know that story.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find the **smallest set of columns** that still beats 95% of the full model's score.
# 2. Find a column that makes the model worse.
# 3. On mushrooms, find the single question that gets you furthest.
# 4. On bikes, find one wrong prediction and explain it using date or weather.
# 5. 🧸 **Little Kid Corner:** Sort real cards or toys using three clues. Then hide one clue.
#    Which clue did you miss most?

# %%
workbook.render(9)

# %% [markdown]
# ---
# **Next up:** Chapter 10 · *Where Models Go Wrong* — when a shiny score is not an honest score.
