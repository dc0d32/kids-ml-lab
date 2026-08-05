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
# ## 🎣 The Hook
#
# The chapter-2 line says red or blue and never wavers.
#
# But a point sitting right on the line should say: **honestly, I have no idea**.
#
# How do we make a model admit that?
#
# > 📖 **Grown-ups call this:** **logistic regression** — a straight-line score followed
# > by an S-curve that turns it into a probability.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Compute or look up the S-curve for these seven values.
#
# **sigmoid(z) = 1 / (1 + e^-z)**
#
# Hint: at z = 0, e⁰ = 1, so sigmoid(0) = 1/2 exactly.

# %%
z = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
p = sigmoid(z)
pd.DataFrame({"z": z, "sigmoid(z)": np.round(p, 3)})

# %% [markdown]
# Look at those seven points and join them in your head. You have drawn the S-curve.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(4)

# %% [markdown]
# ## 👀 See It
#
# The slider in the app changes `w`. Here, change `w` in code. Large `w` makes a cliff.
# Small `w` makes a shrug.

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
# Chapter 2 is what happens when this curve becomes a near-vertical cliff.

# %% [markdown]
# ## 🎛️ Play With It
#
# The boundary is still a straight line. The confidence around it is new.

# %%
X, y = toy_shape("blobs", n=220, noise=0.28, seed=6)
w1, w2, b = 2.0, 2.0, 0.0
prob = logistic_proba(X, w1, w2, b)
print("log loss:", round(log_loss(prob, y), 3))

fig, ax = plt.subplots(figsize=(6, 5))
decision_boundary(lambda G: logistic_proba(G, w1, w2, b), X, y, ax=ax, shade_confidence=True)
draw_line(w1, w2, b, ax=ax)
ax.set_title("The fade is uncertainty")
plt.show()

# %% [markdown]
# Confidence is a promise. A wrong 99% promise gets punished hard.

# %%
rows = []
for pred, truth in [(0.9, 1), (0.6, 1), (0.1, 1), (0.99, 0), (0.5, 0)]:
    rows.append({"predicted red": pred, "true answer": truth, "penalty": round(log_loss([pred], [truth]), 2)})
pd.DataFrame(rows)

# %% [markdown]
# ## 💻 For Real
#
# Predict whether a penguin is a Gentoo from two measurements: flipper length and weight.

# %%
penguins = load_table("penguins").dropna(subset=["species", "flipper_length_mm", "weight_g"]).copy()
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
# ## 🏆 Challenge
#
# 1. **Find a confident mistake.** Look for a penguin with a high probability and the
#    wrong answer.
# 2. **Crank w.** Make the S-curve a cliff. Does confidence mean correctness?
# 3. **Find the shrug zone.** Move a point near the boundary and watch the probability
#    approach 50%.
# 4. 🧸 **Little Kid Corner:** Stand near one side of a room and say "probably red team."
#    Stand on the middle tape line and say "I don't know." That middle shrug is the idea.
#
# ---
# **Next up:** Chapter 05 · *Twenty Questions* — where a model asks its way to an answer.
