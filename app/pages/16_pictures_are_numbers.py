"""Chapter 16 · Pictures Are Just Numbers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:  # pragma: no cover - the dependency is optional at display time
    st_canvas = None

from kidsml import ui
from kidsml.datasets import digits
from kidsml.plots import confusion_grid, image_strip, show_image
from kidsml import vision

ui.page_setup(16)


@st.cache_resource(show_spinner=False)
def cached_digit_model():
    return vision.train_digit_mlp(seed=0)


@st.cache_data(show_spinner=False)
def cached_digits():
    return digits()


X, y, images = cached_digits()
report = cached_digit_model()

# ---------------------------------------------------------------------------
ui.beat("hook", "A photo arrives as a spreadsheet.")

st.markdown(
    """
A computer has never seen anything. Not one thing.

When you show it a photo, it gets a spreadsheet. Here is proof. Squint at this
8 by 8 grid of numbers. Can you read the digit hiding in the numbers?
"""
)

example_index = 3
fig, ax = ui.figure(4.2, 4.2)
show_image(images[example_index], ax=ax, numbers=True, title="This is a digit 3")
ui.show(fig)

st.markdown(
    """
The picture **is** those numbers. The model from this chapter does not even get
the square. It receives the same 64 numbers as one long row.
"""
)
st.dataframe(vision.digit_as_flat_row(images[example_index]), hide_index=True, use_container_width=True)

ui.careful(
    "A plain MLP does not know that pixel 10 touches pixel 11. It gets a row of 64 values. "
    "Chapter 17 fixes that exact weakness."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Read a tiny picture before the model does.")

shape = np.array(
    [
        [0, 8, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 0, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 8, 8, 8, 0],
    ]
)
st.dataframe(pd.DataFrame(shape), hide_index=True)
st.markdown(
    """
Your eyes can read that as a **3** because the bright numbers make a shape.
A model with one weight per pixel would need:

- 64 weights for an 8×8 gray image
- 784 weights for a 28×28 gray image
- 3,000,000 weights for a 1000×1000 colour photo
"""
)
ui.jargon("pixel", "One little square in a picture. A gray pixel is one number. A colour pixel is three numbers: red, green, and blue.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Digits, and the ghosts of digits.")

first_examples = []
first_titles = []
for digit in range(10):
    idx = int(np.flatnonzero(y == digit)[0])
    first_examples.append(images[idx])
    first_titles.append(str(digit))
fig, _ = image_strip(first_examples, titles=first_titles, width=1.1)
ui.show(fig)

averages = vision.average_digit_images(images, y)
fig = vision.plot_small_images(averages, titles=[f"average {i}" for i in range(10)], width=1.35)
ui.show(fig)
ui.aha("The average 3 still looks a bit like a 3. The model gets no magic, only patterns in numbers.")

# ---------------------------------------------------------------------------
ui.beat("play", "Draw a digit and turn it into 64 numbers.")

st.markdown(
    """
Draw with white on the black square. The app crops your drawing, shrinks it to
8 by 8, and scales it to the same 0–16 numbers as the training digits.
"""
)

left, right = st.columns([1, 1], gap="large")
with left:
    if st_canvas is None:
        st.warning("The drawing canvas is not available here, so the demo uses a blank grid.")
        grid = np.zeros((8, 8), dtype=float)
    else:
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        grid = vision.canvas_to_digit_grid(canvas.image_data)

with right:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    show_image(grid, ax=axes[0], numbers=True, title="the 8×8 numbers")
    prediction, probabilities = vision.predict_digit_grid(report.model, grid)
    show_image(grid, ax=axes[1], title=f"model says {prediction}")
    fig.tight_layout()
    ui.show(fig)

st.markdown(f"### The model guesses: **{prediction}**")
st.bar_chart(vision.confidence_table(probabilities).set_index("digit"), height=260)
ui.careful(
    "The model has only ever seen neat centred digits. Draw yours off to one side, or make "
    "it tiny, and it can fall apart. That mismatch is a real machine-learning bug."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "A small neural net learns the digits.")

st.code(
    """
from sklearn.neural_network import MLPClassifier

X, y, images = digits()
X_train, X_test, y_train, y_test = train_test_split(X / 16, y, stratify=y)
model = MLPClassifier(hidden_layer_sizes=(48,), random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)
""",
    language="python",
)

st.metric("test accuracy", f"{report.accuracy:.1%}")
fig, ax = ui.figure(5.5, 4.6)
confusion_grid(report.confusion, labels=list(range(10)), ax=ax)
ui.show(fig)
st.markdown("In Chapter 09 you met this picture: rows are true answers, columns are guesses. The off-diagonal cells are the mix-ups.")

wrong = vision.misclassified_examples(report, limit=8)
if wrong:
    fig = vision.plot_small_images(
        [row[0] for row in wrong],
        titles=[f"true {row[1]} → {row[2]}" for row in wrong],
        width=1.3,
    )
    ui.show(fig)
    st.markdown("Do its confusions feel familiar? 4/9, 3/5, and 7/1 are hard for people too.")

weights = vision.first_layer_images(report.model, limit=12)
fig = vision.plot_small_images(weights, titles=[f"unit {i}" for i in range(len(weights))], width=1.15, vcenter=True)
ui.show(fig)
st.caption("First-layer weights are only semi-readable here, but some look like blurry strokes and digit parts.")

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. **Hardest digit.** Use the confusion matrix. Which row has the most mistakes?
2. **Find an easy mistake.** Pick a wrong image where you can read the digit right away.
3. **Blank a row.** In the notebook, set one row of pixels to 0. When does the prediction flip?
4. **Break it on purpose.** Draw a digit badly until the model changes its mind.
5. 🧸 **Little Kid Corner:** Make an 8 by 8 grid with coins or cereal pieces. Stand back. Can someone read the number?
"""
)

ui.worksheet_link(16)
