"""Chapter 01 · Lines That Predict."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

from kidsml import lesson
from kidsml.datasets import allowance, load_table
from kidsml.linear import gradient_descent_line, mse_for_line, squared_error_table
from kidsml.lineanim import descent_gif_bytes
from kidsml.plots import ACCENT, COUNT_CMAP, loss_surface, regression_fit

lesson.begin(1)

weeks, dollars = allowance()
hand_y = np.round(dollars[:4])


@st.cache_data(show_spinner=False)
def descent_animation():
    return descent_gif_bytes()


@lesson.step("A line can answer how much", beat="hook")
def _():
    lesson.say(
        """
Ready? Same game, new scoreboard. Chapter 0 learned a rule from examples.
This time the answer is not **zeep** or **not zeep**. The answer is a number.

If you saved for **20 weeks**, how much money would be in your piggy bank? You
already have a plan: put a line near the dots, then read the line's height at
week 20.

That line is a **model**: a tiny guess machine. Feed in one number, weeks saved,
and it spits out one number, dollars.
"""
    )
    fig, ax = lesson.figure(6.6, 4.4)
    ax.scatter(weeks, dollars, s=55, color="#60A5FA", edgecolors="#171B26", linewidths=0.9)
    xs = np.linspace(weeks.min(), 20, 80)
    ax.plot(xs, 3 * xs + 5, color=ACCENT, label="candidate line")
    ax.scatter([20], [65], s=90, color=ACCENT, edgecolors="white", zorder=4)
    ax.vlines(20, dollars.min(), 65, color="#94A3B8", linestyle="--", linewidth=1.4)
    ax.set_xlabel("weeks saved")
    ax.set_ylabel("dollars")
    ax.set_title("Read a number from the line")
    ax.legend(loc="best", fontsize=9)
    lesson.show(fig)
    lesson.look_for("the green dot at week 20. The line turns one input number into one dollar guess.")
    lesson.jargon("linear regression", "A line used to predict a number.")


@lesson.step("Four rows. One candidate line", beat="byhand")
def _():
    lesson.say(
        """
Try the line **dollars = 3 × weeks + 5** on four rounded rows. For week 3, the
prediction is **3 × 3 + 5 = 14** dollars. The rounded real amount is 15 dollars,
so the mistake is **15 - 14 = 1**.
"""
    )
    hand = squared_error_table(weeks[:4], hand_y, w=3, b=5)
    st.dataframe(hand, hide_index=True, width="stretch")
    st.metric("Total squared mistake", f"{hand['mistake²'].sum():.2f}")
    lesson.look_for("the mistake column first, then the mistake² column. Signs vanish when we square.")


@lesson.step("Why the square matters", beat="byhand")
def _():
    lesson.say(
        """
        One mistake is not enough to judge the line. We need one score for all four rows,
        so we square each mistake and add the squares.

        Signs can hide errors: **+2 + (-2) = 0**, even though the line missed twice.
        Squaring says **2² + (-2)² = 4 + 4 = 8**, so both misses count.
        """
    )
    guess = lesson.predict(
        "On the graph, what will one squared mistake look like?",
        ["A vertical line", "An actual square area", "A curved arrow"],
        correct=1,
        why="The vertical miss becomes the square's side. Side × side gives the area, so a taller miss turns into a bigger red block!",
        key="ch01_square",
    )
    if guess is None:
        return

    lesson.aha(
        "Big mistakes get louder. A miss of 8 becomes **8² = 64**. A miss of 2 becomes **2² = 4**. "
        "The first miss is four times as far away, but it costs sixteen times as much."
    )
    lesson.say("The squared mistake is not a mystery word. On the graph, it is a real square.")
    fig, ax = lesson.figure(7, 4.6)
    regression_fit(weeks, dollars, 3, 5, ax=ax, show_errors=True, show_squares=True)
    ax.set_xlabel("weeks saved")
    ax.set_ylabel("dollars")
    ax.set_title("The squared mistake is the area of the square")
    lesson.show(fig)
    lesson.look_for("the red boxes: one small vertical miss makes a tiny area, while one tall miss would make a huge area.")


@lesson.step("Move the two knobs", beat="play")
def _():
    lesson.say(
        """
A line has two knobs. **w** is the **weight**: dollars per week, so it tilts the
line. **b** is the **bias**: the starting height, so it slides the line up or down.
"""
    )
    knobs, picture = lesson.controls()
    with knobs:
        w = st.slider("w: dollars per week", 0.0, 6.0, 3.0, 0.1, key="ch01_w")
        b = st.slider("b: starting dollars", -5.0, 15.0, 5.0, 0.5, key="ch01_b")
        loss = mse_for_line(weeks, dollars, w, b)
        st.metric("Average squared mistake", f"{loss:.2f}")
    with picture:
        fig, ax = lesson.figure(6, 4.5)
        regression_fit(weeks, dollars, w, b, ax=ax, show_errors=True, show_squares=False)
        ax.set_xlabel("weeks saved")
        ax.set_ylabel("dollars")
        lesson.show(fig)
    lesson.look_for("how w rotates the line, while b lifts the whole line without changing its tilt.")


@lesson.step("You are here on the hill", beat="play")
def _():
    guess = lesson.predict(
        "Every pair of w and b gets a score. Where do the best settings live on that map?",
        ["High on a wall", "In a low valley", "Scattered randomly"],
        correct=1,
        why="Tiny knob moves make tiny prediction moves, so the scores form a smooth valley you can walk down, not a bag of random pebbles!",
        key="ch01_hill",
    )
    if guess is None:
        return

    knobs, picture = lesson.controls()
    with knobs:
        w = st.slider("hill w: dollars per week", 0.0, 6.0, 3.0, 0.1, key="ch01_hill_w")
        b = st.slider("hill b: starting dollars", -5.0, 15.0, 5.0, 0.5, key="ch01_hill_b")
        st.metric("Average squared mistake", f"{mse_for_line(weeks, dollars, w, b):.2f}")
    with picture:
        W, B, Z = loss_surface(weeks, dollars, w_range=(0, 6), b_range=(-5, 15))
        fig, ax = lesson.figure(6, 4.5)
        ax.contourf(W, B, Z, levels=24, cmap=COUNT_CMAP)
        ax.contour(W, B, Z, levels=12, colors="white", alpha=0.35, linewidths=0.7)
        ax.scatter([w], [b], s=110, c=ACCENT, edgecolors="white", zorder=5)
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        ax.set_title("You are here on the error hill")
        lesson.show(fig)
    lesson.look_for("the dot that says where your current line lives, and the low green-blue valley it wants to reach. Every training step is one walk toward that valley.")


@lesson.step("Walking downhill", beat="play")
def _():
    lesson.say(
        """
A **gradient** is the arrow you get by putting those two slopes together. It points the way
that makes the average squared mistake climb fastest — nudge **w** this much and **b** that
much, and the mistake grows quicker than any other direction you could have gone.

Grown-ups call that mistake score the **loss**. And to learn, the computer walks the exact
opposite way: downhill. That whole downhill-walking trick is called **gradient descent**.
"""
    )
    st.image(descent_animation(), caption="Gradient descent walking down into the valley, one step at a time")
    lesson.look_for(
        "the dots landing one at a time as the walk feels its way down — a big first leap, "
        "then shorter and shorter nudges. Watch **w**, **b** and the **error** in the title "
        "shrink with every step. Nobody told the dot where the valley was; it found it by "
        "measuring the slope and moving."
    )
    lesson.say("Now grab the slider and set how many steps to take. Stop it early and the dot is caught mid-walk, still up on the slope.")
    steps = st.slider("Let the computer take this many downhill steps", 1, 120, 60, key="ch01_steps")
    path = gradient_descent_line(weeks, dollars, w=0, b=0, lr=0.01, steps=steps)
    W, B, Z = loss_surface(weeks, dollars, w_range=(0, 6), b_range=(-5, 15))
    fig, ax = lesson.figure(6, 4.5)
    ax.contourf(W, B, Z, levels=24, cmap=COUNT_CMAP)
    ax.plot(path["w"], path["b"], marker="o", markersize=2.5, color="white")
    ax.scatter([path["w"][-1]], [path["b"][-1]], s=100, c=ACCENT, edgecolors="white")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("The computer walks down the valley")
    lesson.show(fig)
    lesson.look_for("how few steps it takes to get most of the way down, and how the last steps barely move — the slope there is almost flat.")


@lesson.step("The training loop", beat="play")
def _():
    lesson.mermaid(
        """
flowchart TD
    A[Choose w and b] --> B[Predict dollars]
    B --> C[Measure squared mistakes]
    C --> D[Find downhill direction]
    D --> E[Nudge w and b]
    E --> B
""",
    )
    lesson.look_for("the loop: fit, measure, adjust, then try again.")
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
    lesson.aha("The model **is two numbers**: w and b. Training is the loop that chooses them.")


@lesson.step("scikit-learn hides the walking", beat="forreal")
def _():
    lesson.say(
        """
scikit-learn does the same job without showing every downhill step. It still
returns the same two things: a slope **w** and a starting height **b**.
"""
    )
    model = LinearRegression().fit(weeks.reshape(-1, 1), dollars)
    st.code("LinearRegression().fit(weeks.reshape(-1, 1), dollars)", language="python")
    st.write(f"scikit-learn finds w = **{model.coef_[0]:.2f}** and b = **{model.intercept_:.2f}**.")


@lesson.step("A real feature is messier", beat="forreal")
def _():
    bikes = load_table("bikes").dropna()
    lesson.say(
        """
Now use a real table: daily rentals from a city bike-share system. Each row is
one day. **temp_c** is the air temperature, and **rentals** is how many bikes
people checked out that day.

Temperature is the **feature** here: the input number the model gets to see. A
warmer day often means more rentals, but real life also has rain, holidays,
seasons and luck, so the dots will not sit neatly on the line.
"""
    )
    st.dataframe(bikes[["temp_c", "rentals"]].iloc[[0, len(bikes) // 3, 2 * len(bikes) // 3, -1]], hide_index=True, width="content")
    X_bike = bikes[["temp_c"]]
    y_bike = bikes["rentals"]
    bike_model = LinearRegression().fit(X_bike, y_bike)
    fig, ax = lesson.figure(7, 4.6)
    ax.scatter(bikes["temp_c"], bikes["rentals"], s=18, alpha=0.45, color="#3B82F6", edgecolors="none")
    xs = np.linspace(bikes["temp_c"].min(), bikes["temp_c"].max(), 80)
    ax.plot(xs, bike_model.predict(pd.DataFrame({"temp_c": xs})), color=ACCENT)
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("bike rentals")
    ax.set_title("Real bike rentals: temperature helps, but the world is messy")
    lesson.show(fig)
    lesson.look_for("the spread around the line. Low error means better than nearby lines, not perfect predictions.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Beat the computer.** Find w and b with a smaller error than scikit-learn. What happens?
2. **Break it on purpose.** Move w and b until the error becomes huge. Which knob did the most damage?
3. **Outlier test.** Imagine dragging one dot far upward. Which way would the best line tilt, and why?
"""
    )
    lesson.kid_corner(
        "Put a string near some toy cars. Move the string until it sits close to all of them. "
        "That string is your model."
    )


lesson.finish()
