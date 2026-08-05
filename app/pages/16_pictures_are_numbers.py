"""Chapter 16 · Pictures Are Numbers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:  # pragma: no cover - the dependency is optional at display time
    st_canvas = None

from kidsml import lesson
from kidsml import vision
from kidsml.datasets import digits
from kidsml.plots import confusion_grid, image_strip, show_image

lesson.begin(16)


@st.cache_resource(show_spinner=False)
def cached_digit_model():
    return vision.train_digit_mlp(seed=0)


@st.cache_data(show_spinner=False)
def cached_digits():
    return digits()


@lesson.step("A photo arrives as a spreadsheet", beat="hook")
def _():
    lesson.say(
        """
A computer has never seen anything. Not one thing.

When you show it a photo, it gets a spreadsheet. Here is proof: squint at this 8 by 8 grid
of numbers and try to read the digit hiding inside.
"""
    )
    guess = lesson.predict(
        "Which digit is hiding in the grid?",
        ["1", "3", "8", "9"],
        correct=1,
        why="The bright numbers draw the top, middle, and bottom strokes of a 3.",
        key="ch16_read_grid",
    )
    if guess is None:
        return

    _, _, images = cached_digits()
    example_index = 3
    fig, ax = lesson.figure(4.2, 4.2)
    show_image(images[example_index], ax=ax, numbers=True, title="This is a digit 3")
    lesson.show(fig)
    lesson.look_for("the bright cells. Your eyes turn those numbers into strokes before the model sees anything.")


@lesson.step("The square becomes one row", beat="hook")
def _():
    _, _, images = cached_digits()
    example_index = 3
    lesson.say("The picture **is** those numbers. This chapter's model does not even get the square; it receives the same 64 numbers as one long row.")
    st.dataframe(vision.digit_as_flat_row(images[example_index]), hide_index=True, use_container_width=True)
    lesson.mermaid(
        """
graph LR
    A[Image] --> B[Grid of pixels]
    B --> C[Flat row of 64 numbers]
    C --> D[Model]
    D --> E[Ten digit scores]
""",
        height=240,
    )
    lesson.look_for("the flattening step. The model gets number 0 through number 63, then returns ten scores.")
    lesson.careful(
        "A plain MLP does not know that pixel 10 touches pixel 11. To it, moving from pixel 10 to pixel 11 is no more special than moving from pixel 10 to pixel 47. Chapter 17 fixes that weakness."
    )


@lesson.step("Read a tiny picture", beat="byhand")
def _():
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
    lesson.look_for("the 8s. They sketch the same top, middle, and bottom strokes your eyes read as a 3.")
    lesson.say("Your eyes can read that as a **3** because the bright numbers make a shape. A model with one weight per pixel has to attach a knob to every input number.")
    lesson.jargon("pixel", "One little square in a picture. A gray pixel is one number. A colour pixel is three numbers: red, green, and blue.")


@lesson.step("Pixels grow fast", beat="byhand")
def _():
    st.dataframe(
        pd.DataFrame(
            {
                "picture": ["8×8 gray image", "28×28 gray image", "1000×1000 colour photo"],
                "input weights": ["64", "784", "3,000,000"],
            }
        ),
        hide_index=True,
        use_container_width=False,
    )
    lesson.look_for("how fast the weight count grows as pictures get wider, taller, and gain colour channels.")
    lesson.aha("The jump is fast because pictures grow in two directions at once. Double the width and double the height, and you made four times as many pixels.")


@lesson.step("Digits have family resemblance", beat="seeit")
def _():
    _, y, images = cached_digits()
    first_examples = []
    first_titles = []
    for digit in range(10):
        idx = int(np.flatnonzero(y == digit)[0])
        first_examples.append(images[idx])
        first_titles.append(str(digit))
    fig, _ = image_strip(first_examples, titles=first_titles, width=1.1)
    lesson.show(fig)
    lesson.look_for("how different real handwriting is. The model has to learn the family resemblance, not memorize one perfect 3.")


@lesson.step("Average ghosts still look like digits", beat="seeit")
def _():
    _, y, images = cached_digits()
    averages = vision.average_digit_images(images, y)
    fig = vision.plot_small_images(averages, titles=[f"average {i}" for i in range(10)], width=1.35)
    lesson.show(fig)
    lesson.look_for("the average 3, 8, and 0. The useful signal is spread across many examples.")
    lesson.aha("The average 3 still looks a bit like a 3. The model gets patterns in numbers, not one magic row.")


@lesson.step("Draw a digit", beat="play")
def _():
    lesson.say("Draw with white on the black square. The app crops your drawing, shrinks it to 8 by 8, and scales it to the same 0–16 numbers as the training digits.")
    report = cached_digit_model()
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
                key="ch16_digit_canvas",
            )
            grid = vision.canvas_to_digit_grid(canvas.image_data)
    with right:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
        show_image(grid, ax=axes[0], numbers=True, title="the 8×8 numbers")
        prediction, probabilities = vision.predict_digit_grid(report.model, grid)
        show_image(grid, ax=axes[1], title=f"model says {prediction}")
        fig.tight_layout()
        lesson.show(fig)
    lesson.look_for("which cells light up after the crop and shrink. Off-centre drawings can lose the shape.")
    st.markdown(f"### The model guesses: **{prediction}**")
    st.bar_chart(vision.confidence_table(probabilities).set_index("digit"), height=260)


@lesson.step("Predict the mistakes", beat="forreal")
def _():
    guess = lesson.predict(
        "Will the model confuse some of the same digit pairs people confuse?",
        ["Yes, because both read the same strokes", "No, because it sees only numbers", "No, it never makes mistakes"],
        correct=0,
        why="A loopy 9 can look like a 4, a messy 5 can look like a 3, and a skinny 7 can look like a 1 to both humans and models.",
        key="ch16_confusion_prediction",
    )
    if guess is None:
        return
    lesson.say("Now we test that hunch on held-out digits the model did not train on.")


@lesson.step("The confusion map", beat="forreal")
def _():
    report = cached_digit_model()
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
    fig, ax = lesson.figure(5.5, 4.6)
    confusion_grid(report.confusion, labels=list(range(10)), ax=ax)
    lesson.show(fig)
    lesson.look_for("off-diagonal cells. Those are the exact pairs the model mixes up.")
    wrong = vision.misclassified_examples(report, limit=8)
    if wrong:
        fig = vision.plot_small_images([row[0] for row in wrong], titles=[f"true {row[1]} → {row[2]}" for row in wrong], width=1.3)
        lesson.show(fig)
        lesson.look_for("wrong examples that you can still read. The model's mistakes often rhyme with human mistakes.")


@lesson.step("The first layer learns clues", beat="forreal")
def _():
    report = cached_digit_model()
    weights = vision.first_layer_images(report.model, limit=12)
    fig = vision.plot_small_images(weights, titles=[f"unit {i}" for i in range(len(weights))], width=1.15, vcenter=True)
    lesson.show(fig)
    lesson.look_for("blurry strokes and digit parts. They are not full digits; they are small clues the network can combine.")
    lesson.careful("The model has only ever seen neat centred digits. Draw yours off to one side, or make it tiny, and it can fall apart.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Hardest digit.** Use the confusion matrix. Which row has the most mistakes?
2. **Find an easy mistake.** Pick a wrong image where you can read the digit right away.
3. **Blank a row.** In the notebook, set one row of pixels to 0. When does the prediction flip?
4. **Break it on purpose.** Draw a digit badly until the model changes its mind.
"""
    )
    lesson.kid_corner("Make an 8 by 8 grid with coins or cereal pieces. Stand back. Can someone read the number?")


lesson.finish()
