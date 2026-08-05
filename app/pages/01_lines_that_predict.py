"""Chapter 01 · Lines That Predict."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from kidsml import ui
from kidsml.datasets import allowance, load_table
from kidsml.linear import gradient_descent_line, mse_for_line, squared_error_table
from kidsml.plots import ACCENT, loss_surface, regression_fit

ui.page_setup(1)

# ---------------------------------------------------------------------------
ui.beat("hook")
st.markdown(
    """
Chapter 0 showed the whole course in one sentence: learn a rule from examples.
This time the answer is not **zeep** or **not zeep**. The answer is a number.

If you saved for **20 weeks**, how much money would be in your piggy bank? You
already have a plan: put a line near the dots, then read the line's height at
week 20.

That line is a model. It is a tiny machine: feed in a week number, get out a
dollar prediction.
"""
)
ui.jargon("linear regression", "A line used to predict a number.")

weeks, dollars = allowance()

# ---------------------------------------------------------------------------
ui.beat("byhand", "Four rows. One candidate line.")
st.markdown(
    """
Try the line **dollars = 3 × weeks + 5** on four rounded rows. For week 3, the
prediction is **3 × 3 + 5 = 14** dollars. The rounded real amount is 15 dollars,
so the mistake is **15 - 14 = 1**.

One mistake is not enough to judge the line. We need one score for all four rows,
so we square each mistake and add the squares.
"""
)
hand_y = np.round(dollars[:4])
hand = squared_error_table(weeks[:4], hand_y, w=3, b=5)
st.dataframe(hand, hide_index=True, use_container_width=True)
st.metric("Total squared mistake", f"{hand['mistake²'].sum():.2f}")
st.markdown(
    """
Why square instead of adding the raw mistakes? First, signs can hide errors:
**+2 + (-2) = 0**, even though the line missed twice. Squaring says
**2² + (-2)² = 4 + 4 = 8**, so both misses count.

Second, big mistakes get louder. A miss of 8 becomes **8² = 64**. A miss of 2
becomes **2² = 4**. The first miss is four times as far away, but it costs
sixteen times as much. That pressure pulls the best line away from giant misses.
"""
)

# ---------------------------------------------------------------------------
ui.beat("seeit")
st.markdown(
    """
The squared mistake is not a mystery word. On the graph, it is a real square.
A taller miss makes a taller square, and the area of that square is the number
in the table.
"""
)
fig, ax = ui.figure(7, 4.6)
regression_fit(weeks, dollars, 3, 5, ax=ax, show_errors=True, show_squares=True)
ax.set_xlabel("weeks saved")
ax.set_ylabel("dollars")
ax.set_title("The squared mistake is the area of the square")
ui.show(fig)
st.markdown(
    "Look at the red boxes: one small vertical miss makes a tiny area, while one tall miss would make a huge area."
)

# ---------------------------------------------------------------------------
ui.beat("play", "Move the line. Watch the hill map.")
st.markdown(
    """
A line has two knobs. **w** is dollars per week, so it tilts the line. **b** is
the starting height, so it slides the line up or down.

Every pair of knob settings gets its own average squared mistake. If we draw all
those scores as a map, the good settings form a low valley. Nearby lines make
nearby predictions, so the score changes smoothly instead of jumping around.
"""
)
col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    w = st.slider("w: dollars per week", 0.0, 6.0, 3.0, 0.1)
    b = st.slider("b: starting dollars", -5.0, 15.0, 5.0, 0.5)
    loss = mse_for_line(weeks, dollars, w, b)
    st.metric("Average squared mistake", f"{loss:.2f}")
    fig, ax = ui.figure(6, 4.5)
    regression_fit(weeks, dollars, w, b, ax=ax, show_errors=True, show_squares=False)
    ax.set_xlabel("weeks saved")
    ax.set_ylabel("dollars")
    ui.show(fig)
with col_b:
    W, B, Z = loss_surface(weeks, dollars, w_range=(0, 6), b_range=(-5, 15))
    fig, ax = ui.figure(6, 4.5)
    ax.contourf(W, B, Z, levels=24, cmap="viridis")
    ax.contour(W, B, Z, levels=12, colors="white", alpha=0.35, linewidths=0.7)
    ax.scatter([w], [b], s=110, c=ACCENT, edgecolors="white", zorder=5)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("You are here on the error hill")
    ui.show(fig)

st.markdown(
    """
Notice the valley shape on the right. Many terrible lines live up on the walls,
and the best line sits near the low floor.

A **gradient** is an arrow made from slopes. It says, "if you nudge **w** this
way and **b** that way, the loss rises fastest." To learn, the computer walks the
opposite way: downhill.
"""
)
steps = st.slider("Let the computer take this many downhill steps", 1, 120, 60)
path = gradient_descent_line(weeks, dollars, w=0, b=0, lr=0.01, steps=steps)
fig, ax = ui.figure(6, 4.5)
ax.contourf(W, B, Z, levels=24, cmap="viridis")
ax.plot(path["w"], path["b"], marker="o", markersize=2.5, color="white")
ax.scatter([path["w"][-1]], [path["b"][-1]], s=100, c=ACCENT, edgecolors="white")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_title("The computer walks down the valley")
ui.show(fig)
st.markdown("Follow the white dots: each step measures the slope, then nudges the two numbers toward lower error.")
ui.mermaid(
    """
flowchart LR
    A[Choose w and b] --> B[Predict dollars]
    B --> C[Measure squared mistakes]
    C --> D[Find downhill direction]
    D --> E[Nudge w and b]
    E --> B
""",
    height=260,
)
st.markdown("The loop is the whole training story: fit, measure, adjust, then try again.")
st.code(
    """for step in range(steps):
    pred = w * x + b
    loss = mean((pred - y) ** 2)
    w_slope = mean(2 * (pred - y) * x)
    b_slope = mean(2 * (pred - y))
    w = w - lr * w_slope
    b = b - lr * b_slope""",
    language="python",
)
ui.aha("The model **is two numbers**: w and b. Training is the loop that chooses them.")

# ---------------------------------------------------------------------------
ui.beat("forreal")
st.markdown(
    """
scikit-learn does the same job without showing every downhill step. It still
returns the same two things: a slope **w** and a starting height **b**.
"""
)
model = LinearRegression().fit(weeks.reshape(-1, 1), dollars)
st.code("LinearRegression().fit(weeks.reshape(-1, 1), dollars)", language="python")
st.write(f"scikit-learn finds w = **{model.coef_[0]:.2f}** and b = **{model.intercept_:.2f}**.")

st.markdown(
    """
Now use one real feature: temperature. A warmer day often means more bike
rentals, so a line can help. But real life also has rain, holidays, seasons, and
luck, so the dots will not sit neatly on the line.
"""
)
bikes = load_table("bikes").dropna()
X_bike = bikes[["temp_c"]]
y_bike = bikes["rentals"]
bike_model = LinearRegression().fit(X_bike, y_bike)
fig, ax = ui.figure(7, 4.6)
ax.scatter(bikes["temp_c"], bikes["rentals"], s=18, alpha=0.45, color="#3B82F6", edgecolors="none")
xs = np.linspace(bikes["temp_c"].min(), bikes["temp_c"].max(), 80)
ax.plot(xs, bike_model.predict(pd.DataFrame({"temp_c": xs})), color=ACCENT)
ax.set_xlabel("temperature (°C)")
ax.set_ylabel("bike rentals")
ax.set_title("Real bike rentals: temperature helps, but the world is messy")
ui.show(fig)
ui.careful("Look at the spread around the line. Low error does not mean perfect predictions; it means better than nearby lines.")

# ---------------------------------------------------------------------------
ui.beat("challenge")
st.markdown(
    """
1. **Beat the computer.** Find w and b with a smaller error than scikit-learn. What happens?
2. **Break it on purpose.** Move w and b until the error becomes huge. Which knob did the most damage?
3. **Outlier test.** Imagine dragging one dot far upward. Which way would the best line tilt, and why?
4. 🧸 **Little Kid Corner:** Put a string near some toy cars. Move the string until it sits
   close to all of them. That string is your model.
"""
)
ui.worksheet_link(1)
