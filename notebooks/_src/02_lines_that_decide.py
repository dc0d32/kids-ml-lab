# %% [markdown]
# # Chapter 02 · Lines That Decide
#
# ### One line can split the whole world in two.
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
from sklearn.linear_model import Perceptron

from kidsml.datasets import toy_shape, two_blobs_tiny
from kidsml.linear import mistake_count, perceptron_history, predict_side, score_line
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, decision_boundary, draw_line, scatter_2d, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# Chapter 1 used a line to answer **how much?** The line gave a dollar amount.
#
# Same line, new question: **which side?** A point on one side becomes blue. A
# point on the other side becomes red. The model still computes a number first,
# but now the sign of that number makes the decision.
#
# > 📖 **Grown-ups call this:** a **perceptron** — an old-school model that decides which
# > side of a line a point is on.
#
# ```mermaid
# flowchart LR
#     A[Point x1 and x2] --> B[Weighted sum z]
#     B --> C{z > 0?}
#     C -->|yes| D[red]
#     C -->|no| E[blue]
# ```
#
# Read the diagram left to right: the perceptron is a score machine followed by a sign check.

# %%
X_tiny, y_tiny = two_blobs_tiny()
pd.DataFrame({"x1": X_tiny[:, 0], "x2": X_tiny[:, 1], "answer": np.where(y_tiny, "red", "blue")})

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Try this score rule on five points:
#
# **score = 1·x1 + 1·x2 - 8**
#
# Positive score means red. Negative score means blue. For point **(6, 5)** the
# arithmetic would be **1(6) + 1(5) - 8 = 3**, so the guess is red.

# %%
w_start = np.array([1.0, 1.0])
b_start = -8.0
scores = score_line(X_tiny[:5], w_start[0], w_start[1], b_start)
pd.DataFrame(
    {
        "x1": X_tiny[:5, 0],
        "x2": X_tiny[:5, 1],
        "score": scores,
        "guess": np.where(scores > 0, "red", "blue"),
        "truth": np.where(y_tiny[:5] == 1, "red", "blue"),
    }
)

# %% [markdown]
# Now make the line too strict: **w = (1, 1), b = -20**. The same red point gets
# **1(6) + 1(5) - 20 = -9**, so the model guesses blue.
#
# When a red point is missed, the perceptron adds the point's coordinates to the
# weights and adds 1 to **b**.

# %%
w_bad = np.array([1.0, 1.0])
b_bad = -20.0
w_after, b_after, was_wrong = perceptron_step(w_bad, b_bad, X_tiny[5], y_tiny[5])
print("wrong?", was_wrong)
print("new w:", w_after)
print("new b:", b_after)

# %% [markdown]
# Why does adding the point help? The score for that same point jumps from **-9**
# to **7(6) + 6(5) - 19 = 53**. The point is now strongly on the red side.
#
# Changing **b** is different from changing **w**. It adds the same amount to every
# point's score, so the boundary slides without turning.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(2)

# %% [markdown]
# ## 👀 See It
#
# The circled red point caused the update. Watch how one correction changes the boundary.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, w_now, b_now, title in [
    (axes[0], w_bad, b_bad, "Before"),
    (axes[1], w_after, b_after, "After"),
]:
    scatter_2d(X_tiny, y_tiny, ax=ax)
    ax.scatter([X_tiny[5, 0]], [X_tiny[5, 1]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
    draw_line(w_now[0], w_now[1], b_now, ax=ax)
    ax.set_title(title)
plt.show()

# %% [markdown]
# Notice that the line did not learn a whole rule in one step. It made one wrong
# point less wrong.

# %%
hist = perceptron_history(X_tiny, y_tiny, w=(0, 0), b=0, steps=12)
hist

# %%
step = min(4, len(hist) - 1)
row = hist.iloc[step]
fig, ax = plt.subplots(figsize=(6, 4.6))
decision_boundary(lambda G: predict_side(G, row.w1, row.w2, row.b), X_tiny, y_tiny, ax=ax, shade_confidence=False)
draw_line(row.w1, row.w2, row.b, ax=ax)
ax.set_title(f"Step {step}: {int(row.mistakes)} mistake(s)")
plt.show()

# %% [markdown]
# Follow the mistake count. Training stops only when a straight line can make every
# point happy.

# %% [markdown]
# ## 🎛️ Play With It
#
# The boundary is the place where **w1·x1 + w2·x2 + b = 0**. The **w** arrow sticks
# straight out from that line.
#
# Why perpendicular? If you walk along the boundary, the score must stay 0. Moving
# in the **w** direction changes the score fastest, so **w** cannot point along the
# line. It points across the line, toward red.

# %%
X, y = toy_shape("blobs", n=180, noise=0.25, seed=2)
w1, w2, b = 1.0, 1.0, 0.0
print("mistakes:", mistake_count(X, y, w1, w2, b))

fig, ax = plt.subplots(figsize=(6, 5))
decision_boundary(lambda G: predict_side(G, w1, w2, b), X, y, ax=ax, shade_confidence=False)
draw_line(w1, w2, b, ax=ax)
ax.arrow(0, 0, w1 * 0.25, w2 * 0.25, color=ACCENT, width=0.025, length_includes_head=True)
ax.text(w1 * 0.28, w2 * 0.28, "w arrow", color=ACCENT)
ax.set_title("w points toward red")
plt.show()

# %% [markdown]
# Move `b` and the line glides in parallel. Move `w1` or `w2` and the line rotates.

# %% [markdown]
# ## 💻 For Real
#
# scikit-learn has a Perceptron too. On clean blobs, a straight separator exists,
# so the model can settle.

# %%
model = Perceptron(max_iter=1000, random_state=0).fit(X, y)
print("blobs score:", model.score(X, y))

# %% [markdown]
# Now look at shapes where the ruler is in trouble. A perceptron keeps fixing the
# first mistake it sees. If the data overlaps, or if the correct boundary must
# bend, one fix can undo an earlier fix. Then the weights keep moving because zero
# mistakes is not available.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, shape in zip(axes, ["moons", "circles"]):
    Xr, yr = toy_shape(shape, n=220, noise=0.18, seed=4)
    m = Perceptron(max_iter=1000, random_state=0).fit(Xr, yr)
    decision_boundary(lambda G, mm=m: mm.predict(G), Xr, yr, ax=ax, shade_confidence=False, title=f"{shape}: {m.score(Xr, yr):.0%} right")
plt.show()

# %% [markdown]
# Notice the leftover mistakes. The algorithm is not lazy; one straight line has
# run out of road.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. **Beat the algorithm.** Find slider values with zero mistakes on blobs.
# 2. **Set b to 0.** What can the line no longer do?
# 3. **Make overlap.** Add noise and watch the perceptron fail to settle.
# 4. 🧸 **Little Kid Corner:** Lay a pencil between two piles of toys. One side is red
#    team, the other is blue team. Move the pencil until nobody is on the wrong side.
#
# ---
# **Next up:** Chapter 03 · *When a Ruler Isn't Enough* — where circles and XOR break
# the ruler.
