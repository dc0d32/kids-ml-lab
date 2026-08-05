# %% [markdown]
# # Chapter 01 · Lines That Predict
#
# ### y = w·x + b, and the idea of "how wrong am I?"
#
# *Part 1 · Classical models*
#
# ---
#
# This notebook is the same chapter as the app, but with the code showing.
# Run a cell with **Shift + Enter**. Change the numbers. Break things. That's the point.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from kidsml.datasets import allowance, load_table
from kidsml.linear import gradient_descent_line, mse_for_line, squared_error_table
from kidsml.plots import ACCENT, loss_surface, regression_fit, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Chapter 0 showed the whole course in one sentence: learn a rule from examples.
# This time the answer is not **zeep** or **not zeep**. The answer is a number.
#
# If you saved for **20 weeks**, how much money would be in your piggy bank? You
# already have a plan: put a line near the dots, then read the line's height at
# week 20.
#
# That line is a model. It is a tiny machine: feed in a week number, get out a
# dollar prediction.
#
# > 📖 **Grown-ups call this:** **linear regression** — a line used to predict a number.

# %%
weeks, dollars = allowance()
pd.DataFrame({"weeks saved": weeks, "dollars": dollars})

# %% [markdown]
# ## ✏️ Work it out
#
# Try the line **dollars = 3 × weeks + 5** on four rounded rows. For week 3, the
# prediction is **3 × 3 + 5 = 14** dollars. The rounded real amount is 15 dollars,
# so the mistake is **15 - 14 = 1**.
#
# One mistake is not enough to judge the line. We need one score for all four rows,
# so we square each mistake and add the squares.

# %%
hand_y = np.round(dollars[:4])
hand = squared_error_table(weeks[:4], hand_y, w=3, b=5)
hand

# %%
print("Total squared mistake:", hand["mistake²"].sum())

# Why square instead of adding the raw mistakes? First, signs can hide errors:
# **+2 + (-2) = 0**, even though the line missed twice. Squaring says
# **2² + (-2)² = 4 + 4 = 8**, so both misses count.
#
# Second, big mistakes get louder. A miss of 8 becomes **8² = 64**. A miss of 2
# becomes **2² = 4**. The first miss is four times as far away, but it costs
# sixteen times as much. That pressure pulls the best line away from giant misses.

# %% [markdown]
# ## 👀 Take a look
#
# The squared mistake is not a mystery word. On the graph, it is a real square.
# A taller miss makes a taller square, and that square's area is the number
# in the table. The red block grows right in front of you!

# %%
fig, ax = plt.subplots(figsize=(7, 4.6))
regression_fit(weeks, dollars, 3, 5, ax=ax, show_errors=True, show_squares=True)
ax.set_xlabel("weeks saved")
ax.set_ylabel("dollars")
ax.set_title("Squared error means square area")
plt.show()

# %% [markdown]
# Look at the red boxes: one small vertical miss makes a tiny area, while one tall
# miss would make a huge area.

# %% [markdown]
# ## 🎛️ Your turn
#
# A line has two knobs. **w** is dollars per week, so it tilts the line. **b** is
# the starting height, so it slides the line up or down.
#
# Every pair of knob settings gets its own average squared mistake. If we draw all
# those scores as a map, the good settings form a low valley. Tiny knob moves
# make tiny prediction moves, so the score changes smoothly instead of rattling around.
# The valley has, as I understand it, excellent aura.

# %%
w = 3.0
b = 5.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
regression_fit(weeks, dollars, w, b, ax=axes[0], show_errors=True, show_squares=False)
axes[0].set_xlabel("weeks saved")
axes[0].set_ylabel("dollars")
axes[0].set_title(f"Average squared mistake = {mse_for_line(weeks, dollars, w, b):.2f}")

W, B, Z = loss_surface(weeks, dollars, w_range=(0, 6), b_range=(-5, 15))
axes[1].contourf(W, B, Z, levels=24, cmap="viridis")
axes[1].contour(W, B, Z, levels=12, colors="white", alpha=0.35, linewidths=0.7)
axes[1].scatter([w], [b], s=110, c=ACCENT, edgecolors="white", zorder=5)
axes[1].set_xlabel("w")
axes[1].set_ylabel("b")
axes[1].set_title("You are here on the hill")
plt.show()

# %% [markdown]
# Notice the valley shape on the right. Many terrible lines live up on the walls,
# and the best line sits near the low floor.
#
# A **gradient** is an arrow made from slopes. It says, "if you nudge **w** this
# way and **b** that way, the average squared mistake rises fastest." Grown-ups
# call that mistake score **loss**. To learn, the computer walks the opposite way:
# downhill.

# %%
path = gradient_descent_line(weeks, dollars, w=0, b=0, lr=0.01, steps=90)

fig, ax = plt.subplots(figsize=(6, 4.6))
ax.contourf(W, B, Z, levels=24, cmap="viridis")
ax.plot(path["w"], path["b"], marker="o", markersize=2.5, color="white")
ax.scatter([path["w"][-1]], [path["b"][-1]], s=110, c=ACCENT, edgecolors="white")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_title("Gradient descent walks down the valley")
plt.show()

# %% [markdown]
# Follow the white dots: each step measures the slope, then nudges the two numbers
# toward lower error.
#
# ```mermaid
# flowchart LR
#     A[Choose w and b] --> B[Predict dollars]
#     B --> C[Measure squared mistakes]
#     C --> D[Find downhill direction]
#     D --> E[Nudge w and b]
#     E --> B
# ```
#
# The loop is the whole training story: fit, measure, adjust, then try again. Around it goes!

# %% [markdown]
# Here is the learning loop in code. Read it slowly. It is a prediction, two
# slopes, and two nudges inside a loop.

# %%
x = weeks
y = dollars
w = 0.0
b = 0.0
lr = 0.01

for step in range(8):
    pred = w * x + b
    loss = np.mean((pred - y) ** 2)
    w_slope = np.mean(2 * (pred - y) * x)
    b_slope = np.mean(2 * (pred - y))
    w = w - lr * w_slope
    b = b - lr * b_slope
    print(step, round(loss, 2), round(w, 2), round(b, 2))

# %% [markdown]
# > 💡 **Aha!**
# >
# > The model **is two numbers**: `w` and `b`. Training is the loop that chooses them.

# %% [markdown]
# ## 💻 In real code
#
# scikit-learn does the same job without showing every downhill step. It still
# returns the same two things: a slope **w** and a starting height **b**.

# %%
model = LinearRegression().fit(weeks.reshape(-1, 1), dollars)
print("w =", round(model.coef_[0], 2))
print("b =", round(model.intercept_, 2))

# %% [markdown]
# Now use one real feature: temperature. A warmer day often means more bike
# rentals, so a line can help. But real life also has rain, holidays, seasons, and
# luck, so the dots will not sit neatly on the line.

# %%
bikes = load_table("bikes").dropna()
X_bike = bikes[["temp_c"]]
y_bike = bikes["rentals"]
bike_model = LinearRegression().fit(X_bike, y_bike)

xs = np.linspace(bikes["temp_c"].min(), bikes["temp_c"].max(), 80)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.scatter(bikes["temp_c"], bikes["rentals"], s=18, alpha=0.45, color="#3B82F6", edgecolors="none")
ax.plot(xs, bike_model.predict(pd.DataFrame({"temp_c": xs})), color=ACCENT)
ax.set_xlabel("temperature (°C)")
ax.set_ylabel("bike rentals")
ax.set_title("A real fitted line")
plt.show()

# %% [markdown]
# Look at the spread around the line. Low error does not mean perfect predictions;
# it means better than nearby lines.

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(1)

# %% [markdown]
# ## 🏆 Go further
#
# 1. **Beat the computer.** Find `w` and `b` with a smaller error than scikit-learn.
# 2. **Make it worse on purpose.** Pick a terrible line and explain which knob did the most damage.
# 3. **Outlier test.** Move one dot far upward. Which way does the best line tilt?
# 4. 🧸 **Little Kid Corner:** Put a string near toy cars on the floor. Move the string
#    until it is close to all the cars. That string is your model.
#
# ---
# **Next up:** Chapter 02 · *Lines That Decide* — where the same line stops predicting
# amounts and starts choosing sides.
