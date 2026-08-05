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
# ## 🎣 The Hook
#
# If you saved for **20 weeks**, how much money would be in your piggy bank?
#
# You already know how to answer. Draw a line through the dots and read the height at
# week 20.
#
# Congratulations. That line is a model. It turns a week number into a dollar guess.
#
# > 📖 **Grown-ups call this:** **linear regression** — a line used to predict a number.

# %%
weeks, dollars = allowance()
pd.DataFrame({"weeks saved": weeks, "dollars": dollars})

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Try this candidate line on four rounded rows:
#
# **dollars = 3 × weeks + 5**
#
# For each row, write the prediction, the mistake, and the squared mistake.

# %%
hand_y = np.round(dollars[:4])
hand = squared_error_table(weeks[:4], hand_y, w=3, b=5)
hand

# %%
print("Total squared mistake:", hand["mistake²"].sum())

# %% [markdown]
# Now work through the interactive workbook. Type your answer in each box and press
# **Check** — you will find out whether you were right, and why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(1)

# %% [markdown]
# Squaring does two jobs.
#
# - A +2 mistake and a -2 mistake both count as bad.
# - Big mistakes hurt much more than small ones.

# %% [markdown]
# ## 👀 See It
#
# The squared mistake is the **area** of the square. It is not a mystery word. It is a
# square on the graph.

# %%
fig, ax = plt.subplots(figsize=(7, 4.6))
regression_fit(weeks, dollars, 3, 5, ax=ax, show_errors=True, show_squares=True)
ax.set_xlabel("weeks saved")
ax.set_ylabel("dollars")
ax.set_title("Squared error means square area")
plt.show()

# %% [markdown]
# ## 🎛️ Play With It
#
# Change `w` and `b`. The first plot shows your line. The second plot shows where your
# line sits on the error hill.

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
# Now let the computer walk downhill. The gradient means: which way is downhill, and how
# steep is it?

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
# Here is the learning loop. Read it slowly. It is a loop, a prediction, two slopes, and
# two nudges.

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
# > The model **is two numbers**: `w` and `b`. That is the whole machine.

# %% [markdown]
# ## 💻 For Real
#
# scikit-learn finds the best line in one line of code.

# %%
model = LinearRegression().fit(weeks.reshape(-1, 1), dollars)
print("w =", round(model.coef_[0], 2))
print("b =", round(model.intercept_, 2))

# %% [markdown]
# Now use one real feature: temperature. We predict bike rentals from temperature alone.
# A line helps, but the real world has rain, holidays, seasons, and surprises.

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
# ## 🏆 Challenge
#
# 1. **Beat the computer.** Find `w` and `b` with a smaller error than scikit-learn.
# 2. **Make it worse on purpose.** Pick a terrible line and explain why it is terrible.
# 3. **Outlier test.** Move one dot far upward. Which way does the best line tilt?
# 4. 🧸 **Little Kid Corner:** Put a string near toy cars on the floor. Move the string
#    until it is close to all the cars. That string is your model.
#
# ---
# **Next up:** Chapter 02 · *Lines That Decide* — where a line stops predicting amounts
# and starts choosing sides.
