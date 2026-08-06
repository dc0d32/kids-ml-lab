# %% [markdown]
# # Chapter 04 · Maybe, Probably, Definitely
#
# ### Squishing any number into a probability.
#
# *Part 1 · Classical models*
#
# ---
#
# This notebook is the same chapter as the app, but with the code showing.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from kidsml.datasets import load_table, toy_shape
from kidsml.linear import logistic_proba, sigmoid
from kidsml.nn_numpy import log_loss
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Chapter 3 showed why a hard line sometimes needs help: some boundaries must
# bend. But even when a straight boundary is good enough, Chapter 2's perceptron
# has another problem.
#
# It says red or blue and never wavers. A point sitting on the boundary should not
# sound certain. It should say: **honestly, I have no idea**.
#
# This chapter keeps the line score, then turns it into confidence.
#
# > 📖 **Grown-ups call this:** **logistic regression** — a straight-line score followed
# > by an S-curve that turns it into a probability.

# %%
X_shrug, y_shrug = toy_shape("blobs", n=120, noise=0.25, seed=10)
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.scatter(
    X_shrug[:, 0],
    X_shrug[:, 1],
    c=np.where(y_shrug == 1, WARM, COOL),
    s=38,
    alpha=0.75,
    edgecolors="white",
    linewidths=0.4,
)
draw_line(2, -2, 0, ax=ax)
ax.scatter([0], [0], s=190, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
ax.set_title("The boundary is the unsure place")
plt.show()

# %% [markdown]
# **Look for:** the circled spot on the line. A hard model must pick a side there,
# even though it is exactly on the fence.

# %% [markdown]
# ## ✏️ Work it out
#
# Start with the same raw line score **z**. Positive should lean red. Negative
# should lean blue. A score near 0 should mean a shrug.
#
# So we need a squish machine: any score in, a **probability** out. A probability
# is a promise from **0 to 1**: 0 means 0% red, 1 means 100% red, and 0.5 means
# the model is split right down the middle.
#
# The S-curve we use is **sigmoid(z) = 1 / (1 + e^-z)**. The **e** is a
# particular number, about **2.718**; it makes a smooth, balanced S-curve, and you
# do not need to memorize it.
#
# At **z = 0**, the arithmetic is **1 / (1 + e⁰) = 1 / (1 + 1) = 0.5**, so the
# boundary lands exactly on the 50/50 shrug.

# %%
z = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
p = sigmoid(z)
pd.DataFrame({"z": z, "sigmoid(z)": np.round(p, 3)})

# %% [markdown]
# We wanted three habits before we picked a formula: low scores near 0, high scores
# near 1, and score 0 exactly at 0.5. We also want no sudden cliff, because training
# needs smooth slopes to walk on.
#
# The sigmoid S-curve has those habits. For example, **sigmoid(2) ≈ 0.88** and
# **sigmoid(-2) ≈ 0.12**.
#
# It also has a training-friendly meaning: adding 1 to the score multiplies the
# red-vs-blue odds by the same amount each time. That steady rule is why this
# particular S-curve shows up everywhere.
#
# ```mermaid
# flowchart TD
#     A[Point features] --> B[Raw score z]
#     B --> C[Sigmoid S-curve]
#     C --> D[Probability]
#     D --> E{p >= 0.5?}
#     E -->|yes| F[red]
#     E -->|no| G[blue]
# ```
#
# Notice where the threshold sits: p = 0.5 happens exactly when the raw score z is 0.

# ## 👀 Take a look
#
# The slider in the app changes `w`. Here, change `w` in code. Large `w` makes the
# S-curve look like Chapter 2's hard step. Small `w` makes a model that shrugs for
# almost every score.

# %%
w = 1.0
xs = np.linspace(-6, 6, 300)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot(xs, sigmoid(w * xs), color=ACCENT)
ax.scatter(z, sigmoid(w * z), color=WARM, edgecolors="white", zorder=3)
ax.axhline(0.5, color="#94A3B8", linestyle="--")
ax.set_xlabel("line score z")
ax.set_ylabel("probability of red")
ax.set_title("The S-curve")
plt.show()

# %% [markdown]
# Look at the dashed 0.5 line. Scores near zero land near that line, which is the
# shrug zone. That dashed line is the model saying, “maybe.”

# %% [markdown]
# ## 🎛️ Your turn
#
# The probability is curved, but the decision boundary stays straight. Why? The
# model says red when **p ≥ 0.5**, and sigmoid reaches 0.5 exactly at **z = 0**.
#
# The set of points with **z = w1·x1 + w2·x2 + b = 0** is the same straight line as
# before. The new thing is the fade of confidence around it.

# %%
X, y = toy_shape("blobs", n=220, noise=0.28, seed=6)
w1, w2, b = 2.0, 2.0, 0.0
prob = logistic_proba(X, w1, w2, b)
pd.DataFrame({"average penalty": [round(log_loss(prob, y), 3)]})

fig, ax = plt.subplots(figsize=(6, 5))
decision_boundary(lambda G: logistic_proba(G, w1, w2, b), X, y, ax=ax, shade_confidence=True)
draw_line(w1, w2, b, ax=ax)
ax.set_title("The fade is uncertainty")
plt.show()

# %% [markdown]
# Notice that the black boundary is perfectly straight, even though the shading
# changes smoothly. For a concrete score, use **w1 = 2**, **w2 = -1**, **b = 0.5**,
# and point **(1, 3)**:
#
# **z = 2(1) + (-1)(3) + 0.5 = -0.5**, so **sigmoid(-0.5) ≈ 0.38**. That means
# "38% red," not a hard no.
#
# A probability is not a decoration; training treats it like a promise and charges a
# penalty when the promise breaks. The table below shows how expensive broken promises get. Crunch!

# %%
rows = []
for pred, truth in [(0.9, 1), (0.6, 1), (0.1, 1), (0.99, 0), (0.5, 0)]:
    rows.append({"predicted red": pred, "true answer": truth, "penalty": round(log_loss([pred], [truth]), 2)})
pd.DataFrame(rows)

# %% [markdown]
# If the truth is red, predicting **0.6** gives penalty about **-log(0.6) = 0.51**.
# Predicting **0.1** gives **-log(0.1) = 2.30**.
#
# If the truth is blue and the model says **0.99 red**, the true class only got
# probability **0.01**, so the penalty is **-log(0.01) = 4.61**. Being unsure and
# wrong is forgivable. Being certain and wrong is expensive.
#
# > 📖 **Grown-ups call this:** **log loss** — the penalty score for probability
# > promises. Confident wrong answers cost the most.

# %% [markdown]
# ## 💻 In real code
#
# Meet some real penguins. These are **real penguins** — 344 of them, waddling around
# three islands near Antarctica, each one caught, measured, and released by researchers.
#
# There are three kinds here: **Adelie**, **Chinstrap**, and **Gentoo**. Our yes-or-no
# question is: **is this one a Gentoo?** The model sees two measurements: flipper length
# and weight.

# %%
penguins = load_table("penguins").dropna(subset=["species", "flipper_length_mm", "weight_g"]).copy()
penguins.groupby("species", group_keys=False).head(2)[["species", "island", "flipper_length_mm", "weight_g"]]

# %% [markdown]
# Gentoos are the big ones — around 217 mm of flipper and 5 kg, against about 190 mm and
# 3.7 kg for the other two. Now fit a model that estimates how likely each penguin is to
# be Gentoo.
#
# The circled penguin is the closest to 50/50. That is not failure; it is useful honesty
# about a hard call. Think of it as a warning light for a bird that is genuinely hard to call.

# %%
penguins["is_gentoo"] = (penguins["species"] == "Gentoo").astype(int)
features = ["flipper_length_mm", "weight_g"]
model = LogisticRegression().fit(penguins[features], penguins["is_gentoo"])
probs = model.predict_proba(penguins[features])[:, 1]
penguins["gentoo_probability"] = probs
uncertain = penguins.iloc[np.argmin(np.abs(probs - 0.5))]

fig, ax = plt.subplots(figsize=(7, 5))
colors = np.where(penguins["is_gentoo"] == 1, WARM, COOL)
ax.scatter(penguins["flipper_length_mm"], penguins["weight_g"], c=colors, s=32, alpha=0.75, edgecolors="white")
ax.scatter([uncertain["flipper_length_mm"]], [uncertain["weight_g"]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
ax.set_xlabel("flipper length (mm)")
ax.set_ylabel("weight (g)")
ax.set_title("The circled penguin is the biggest shrug")
plt.show()

print("most uncertain species:", uncertain["species"])
print("probability Gentoo:", round(uncertain["gentoo_probability"], 3))
print("score:", round(accuracy_score(penguins["is_gentoo"], probs >= 0.5), 3))

# %%
sample = penguins.iloc[[0, len(penguins) // 2, int(np.argmin(np.abs(probs - 0.5))), -1]][["species", "gentoo_probability"]]
sample

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(4)

# %% [markdown]
# ## 🏆 Go further
#
# 1. **Find a confident mistake.** Look for a penguin with a high probability and the
#    wrong answer.
# 2. **Crank w.** Make the S-curve a cliff. Does confidence mean correctness?
# 3. **Find the shrug zone.** Move a point near the boundary and watch the probability
#    approach 50%.
# 4. 🧸 **Little Kid Corner:** Stand near one side of a room and say "probably red team."
#    Stand on the middle tape line and say "I do not know." That middle shrug is the idea.
#
# ---
# **Next up:** Chapter 05 · *Twenty Questions* — where a model asks its way to an answer.
