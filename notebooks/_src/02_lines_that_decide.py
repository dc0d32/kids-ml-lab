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
from IPython.display import Image
from sklearn.linear_model import Perceptron

from kidsml.datasets import toy_shape, two_blobs_tiny
from kidsml.linear import mistake_count, perceptron_history, predict_side, score_line
from kidsml.lineanim import correction_gif_bytes
from kidsml.nn_numpy import perceptron_step
from kidsml.plots import ACCENT, decision_boundary, draw_line, scatter_2d, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Flip the line from price tag to referee. Chapter 1 used a line to answer **how much?** The line gave a dollar amount.
#
# Same line, new question: **which side?** Use this when the answer has two
# buckets: puppy or grown dog, blue or red, yes or no.
#
# The model still computes a number first. That number is a **weighted sum**:
# multiply each measurement by its weight, add the bias, then check the sign.
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

# %% [markdown]
# ### Ten dogs at the park
#
# Here is the thing we want to decide. Ten dogs, and for each one we wrote down two
# numbers: **how tall** it is in hand-spans, and **how heavy** it is in bags of sugar.
#
# Some are puppies. Some are fully grown. Nobody wrote that down — that's the bit we
# want the line to work out.

# %%
X_tiny, y_tiny = two_blobs_tiny()
pd.DataFrame(
    {
        "how tall (x1)": X_tiny[:, 0],
        "how heavy (x2)": X_tiny[:, 1],
        "really a": np.where(y_tiny == 1, "grown dog", "puppy"),
    }
)

# %% [markdown]
# Notice what changed since Chapter 1. There, each thing had **one** number — the weeks
# you'd been saving — so the data sat on a number line. Now each dog has **two** numbers,
# so every dog is a dot on a **map**.
#
# That's the whole reason this chapter needs a line instead of a threshold. On a number
# line you split things with a single point. On a map you split them with a line.

# %%
fig, ax = plt.subplots(figsize=(5.4, 4.4))
scatter_2d(X_tiny, y_tiny, ax=ax, size=110)
ax.set_xlabel("how tall (hand-spans)")
ax.set_ylabel("how heavy (bags of sugar)")
ax.set_title("Ten dogs, two measurements each")
plt.show()

# %% [markdown]
# **Look for:** the empty gap running diagonally between the two clumps. Puppies are
# small on both measurements, grown dogs are big on both. Any line through that gap does
# the job.
#
# ### Guess a line
#
# So let's guess one. The simplest idea in the world: **add the two numbers together, and
# if the total is more than 8, call it a grown dog.** Written the way the model writes it:
#
# **score = 1·x1 + 1·x2 − 8**
#
# The two **1**s say how much each measurement counts — here, equally. The **−8** is where
# we set the bar. Score above zero means the total beat 8.
#
# > 📖 **Grown-ups call this:** the two multipliers are the **weights** and the number on
# > the end is the **bias**. Same pair you met in Chapter 1, doing the same jobs — the
# > weights set the tilt, the bias slides it.

# %%
w_start = np.array([1.0, 1.0])
b_start = -8.0

fig, ax = plt.subplots(figsize=(5.4, 4.4))
scatter_2d(X_tiny, y_tiny, ax=ax, size=110)
ax.set_xlabel("how tall (hand-spans)")
ax.set_ylabel("how heavy (bags of sugar)")
draw_line(w_start[0], w_start[1], b_start, ax=ax, label="x1 + x2 = 8")
ax.legend(loc="lower left", fontsize=9)
ax.set_title("Our guess, drawn on the map")
plt.show()

# %% [markdown]
# We picked those three numbers by eye. The rest of the chapter is about getting a
# machine to pick them instead.

# %% [markdown]
# ## ✏️ Work it out
#
# Time to check the guess. Run **score = 1·x1 + 1·x2 − 8** on five of the dogs and see
# whether the sign matches the truth.
#
# Take the dog at **(6, 5)**: **1(6) + 1(5) − 8 = 3**. Positive, so the line calls it a
# grown dog.

# %%
# A mix of both answers, including the dog the worked example uses. The first five rows
# would all have been puppies, leaving nothing to compare against.
hand_rows = [0, 3, 5, 7, 9]

scores = score_line(X_tiny[hand_rows], w_start[0], w_start[1], b_start)
pd.DataFrame(
    {
        "how tall (x1)": X_tiny[hand_rows, 0],
        "how heavy (x2)": X_tiny[hand_rows, 1],
        "score": scores,
        "line says": np.where(scores > 0, "grown dog", "puppy"),
        "really a": np.where(y_tiny[hand_rows] == 1, "grown dog", "puppy"),
    }
)

# %% [markdown]
# Now make the line too strict on purpose: **w = (1, 1), b = -20**. The same red point
# gets **1(6) + 1(5) - 20 = -9**, so the model guesses blue. Starting with one clear
# mistake lets us watch one correction happen.
#
# When a red point is missed, the perceptron adds the point's coordinates to the
# weights and adds 1 to **b**.

# %%
w_bad = np.array([1.0, 1.0])
b_bad = -20.0
w_after, b_after, was_wrong = perceptron_step(w_bad, b_bad, X_tiny[5], y_tiny[5])
pd.DataFrame(
    {
        "line": ["before", "after"],
        "w1": [w_bad[0], w_after[0]],
        "w2": [w_bad[1], w_after[1]],
        "b": [b_bad, b_after],
        "missed point?": [was_wrong, False],
        "score for (6, 5)": [
            score_line(X_tiny[[5]], w_bad[0], w_bad[1], b_bad)[0],
            score_line(X_tiny[[5]], w_after[0], w_after[1], b_after)[0],
        ],
    }
)

# %% [markdown]
# Why does adding the point help? The score for that same point jumps from **-9**
# to **7(6) + 6(5) - 19 = 53**. Wham — now the point is strongly on the red side!
#
# Changing **b** is different from changing **w**. It adds the same amount to every
# point's score, so the boundary slides without turning.

# ## 👀 Take a look
#
# The circled red point caused the update. Watch one correction happen: the missed dog
# flashes, then the boundary sweeps down from the strict line up in the corner until that
# dog sits on the red side. One update, one point fixed.

# %%
Image(data=correction_gif_bytes())

# %% [markdown]
# Prefer it held still? Here is the same before and after, side by side.

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
# ## 🎛️ Your turn
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
# ## 💻 In real code
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
# mistakes is locked behind a wall.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, shape in zip(axes, ["moons", "circles"]):
    Xr, yr = toy_shape(shape, n=220, noise=0.18, seed=4)
    m = Perceptron(max_iter=1000, random_state=0).fit(Xr, yr)
    decision_boundary(lambda G, mm=m: mm.predict(G), Xr, yr, ax=ax, shade_confidence=False, title=f"{shape}: {m.score(Xr, yr):.0%} right")
plt.show()

# %% [markdown]
# Notice the leftover mistakes. The algorithm is not lazy; one straight line has
# run out of road. The ruler hit the curb!

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(2)

# %% [markdown]
# ## 🏆 Go further
#
# 1. **Beat the algorithm, no cap.** Find slider values with zero mistakes on blobs.
# 2. **Set b to 0.** What can the line no longer do?
# 3. **Make overlap.** Add noise and watch the perceptron fail to settle.
# 4. 🧸 **Little Kid Corner:** Lay a pencil between two piles of toys. One side is red
#    team, the other is blue team. Move the pencil until nobody is on the wrong side.
#
# ---
# **Next up:** Chapter 03 · *When a Ruler Isn't Enough* — where circles and XOR break
# the ruler.
