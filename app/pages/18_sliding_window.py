"""Chapter 18 · The Sliding Window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml import vision
from kidsml.datasets import digits, tiny_image
from kidsml.plots import show_image

lesson.begin(18)


@st.cache_resource(show_spinner="Training two tiny image models...")
def cached_vision_models():
    return vision.train_cnn_and_mlp(seed=0, train_size=6000, test_size=1000, epochs=2, allow_download=True)


@st.cache_data(show_spinner=False)
def cached_digit_image():
    _, y, images = digits()
    idx = int(np.flatnonzero(y == 3)[0])
    return images[idx] / 16.0


@lesson.step("Pictures need neighbours", beat="hook")
def _():
    lesson.say(
        """
Chapter 17's model had a weakness: it did not know that two pixels next to each other
belong together.

That is wrong for pictures. Nearby pixels team up to make strokes, corners, and edges. Here is the
fix: one small window slides across the image like a tiny inspector.
"""
    )
    lesson.say("That small grid of weights is called a **kernel**. The important trick is that the same kernel visits every spot.")
    lesson.mermaid(
        """
graph LR
    A[Image patch] --> B[Same 3x3 kernel]
    B --> C[Multiply and add]
    C --> D[One feature-map cell]
    D --> E[Slide right]
    E --> B
""",
        height=260,
    )
    lesson.look_for("the word same. One little grid visits the top-left corner, the middle, and the bottom-right corner.")
    lesson.kid_corner("Put a sticky note with a 3 by 3 hole over a picture. Peek through the hole, move it one square, and peek again.")


@lesson.step("One window by hand", beat="byhand")
def _():
    image = tiny_image()
    kernel = vision.KERNEL_PRESETS["vertical edge"]
    patch = image[:3, :3]
    left, mid, right = st.columns(3)
    with left:
        st.markdown("**5×5 image**")
        st.dataframe(pd.DataFrame(image.astype(int)), hide_index=True)
    with mid:
        st.markdown("**3×3 edge finder**")
        st.dataframe(pd.DataFrame(kernel.astype(int)), hide_index=True)
    with right:
        st.markdown("**First window**")
        st.dataframe(pd.DataFrame(patch.astype(int)), hide_index=True)
    lesson.say("For the first window: `0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 = 27`. Three bright 9s hit the +1 column, so the answer pops to 27!")
    lesson.jargon("convolution", "Slide a small grid of weights over a picture. Multiply what lines up, then add.")


@lesson.step("Now slide it everywhere", beat="byhand")
def _():
    image = tiny_image()
    kernel = vision.KERNEL_PRESETS["vertical edge"]
    output = vision.convolve2d_valid(image, kernel)
    lesson.say("Across the rows, the 3-high window can start at row 1, row 2, or row 3. Starting at row 4 would hang off the bottom like a tray sliding off a table.")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    show_image(image, ax=axes[0], numbers=True, title="image")
    show_image(kernel, ax=axes[1], numbers=True, title="kernel", cmap="coolwarm")
    show_image(output, ax=axes[2], numbers=True, title="3 by 3 output", cmap="magma")
    fig.tight_layout()
    lesson.show(fig)
    lesson.look_for("the 3 by 3 output. The window has 9 legal landing pads.")


@lesson.step("The edge lights up", beat="seeit")
def _():
    image = tiny_image()
    kernel = vision.KERNEL_PRESETS["vertical edge"]
    output = vision.convolve2d_valid(image, kernel)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    show_image(image, ax=axes[0], numbers=True, title="image")
    show_image(kernel, ax=axes[1], numbers=True, title="kernel", cmap="coolwarm")
    show_image(output, ax=axes[2], numbers=True, title="output", cmap="magma")
    fig.tight_layout()
    lesson.show(fig)
    lesson.look_for("the big output numbers. They flash where dark pixels crash into bright pixels.")
    lesson.aha("You detected an edge by hand, using the same multiply-and-add at every position!")


@lesson.step("Predict the blur kernel", beat="play")
def _():
    guess = lesson.predict(
        "What will a 3×3 kernel full of 1/9 values do to a picture?",
        ["Sharpen edges", "Blur by averaging neighbours", "Turn every pixel black"],
        correct=1,
        why="Each output cell becomes the average of its 3×3 neighbourhood, so sharp little jumps get smeared into nearby squares.",
        key="ch18_blur_kernel",
    )
    if guess is None:
        return
    live_image = vision.generated_pattern(28)
    fig, conv = vision.plot_kernel_demo(live_image, vision.KERNEL_PRESETS["blur"])
    lesson.show(fig)
    lesson.look_for("the softened stripes and diagonal. Averaging smears bright pixels into their neighbours.")
    st.caption(f"Raw output range: {conv.min():.2f} to {conv.max():.2f}")


@lesson.step("Edit the 3×3 window live", beat="play")
def _():
    lesson.say("Try the preset buttons, then change one number. The output jumps because it still obeys the same multiply-and-add rule you did by hand.")
    if "ch18_kernel" not in st.session_state:
        st.session_state["ch18_kernel"] = vision.KERNEL_PRESETS["vertical edge"].copy()

    buttons = st.columns(6)
    for col, name in zip(buttons, vision.KERNEL_PRESETS):
        if col.button(name, key=f"ch18_preset_{name}"):
            st.session_state["ch18_kernel"] = vision.KERNEL_PRESETS[name].copy()
            for r in range(3):
                for c in range(3):
                    st.session_state[f"ch18_kernel_{r}_{c}"] = float(st.session_state["ch18_kernel"][r, c])

    values = []
    for r in range(3):
        cols = st.columns(3)
        row = []
        for c in range(3):
            value = cols[c].number_input(
                f"k{r}{c}",
                value=float(st.session_state["ch18_kernel"][r, c]),
                step=0.25,
                key=f"ch18_kernel_{r}_{c}",
            )
            row.append(value)
        values.append(row)
    live_kernel = np.array(values, dtype=float)
    st.session_state["ch18_kernel"] = live_kernel

    which_image = st.radio("Try the kernel on:", ["a digit", "a bigger pattern"], horizontal=True, key="ch18_live_image")
    live_image = cached_digit_image() if which_image == "a digit" else vision.generated_pattern(28)
    fig, conv = vision.plot_kernel_demo(live_image, live_kernel)
    lesson.show(fig)
    lesson.look_for("which output spots turn bright. Those are the places your kernel matches the image patch and rings the bell.")
    st.caption(f"Kernel vibe check: raw output range {conv.min():.2f} to {conv.max():.2f}")


@lesson.step("Let the model choose kernels", beat="forreal")
def _():
    lesson.say(
        """
The kernels above were designed by a person. What if we let the model choose its own?

During training, the CNN twists the kernel numbers until useful patches light up.
"""
    )
    lesson.mermaid(
        """
graph LR
    A[Image] --> B[Convolution]
    B --> C[Squish]
    C --> D[Pool]
    D --> E[Classify]
""",
        height=220,
    )
    lesson.look_for("the stack: find small patterns, squish the scores, keep the strongest signals, then guess.")
    st.code(
        """
model = TinyCNN()
for image_batch, labels in train_loader:
    optimizer.zero_grad()
    guesses = model(image_batch)
    loss = cross_entropy(guesses, labels)
    loss.backward()
    optimizer.step()
""",
        language="python",
    )


@lesson.step("Predict which image model wins", beat="forreal")
def _():
    guess = lesson.predict(
        "Which model should do better on tiny clothing pictures?",
        ["The plain MLP", "The CNN with shared sliding windows", "They must tie"],
        correct=1,
        why="The CNN reuses a kernel everywhere, so one edge clue can fire in many locations.",
        key="ch18_cnn_wins",
    )
    if guess is None:
        return
    result = cached_vision_models()
    st.markdown(
        f"Using **{result.dataset_name}**: {result.train_size} training images, {result.test_size} test images, {result.epochs} epochs. First download is about 30 MB and then stays cached in `data/torchvision/`."
    )
    st.dataframe(vision.model_comparison_table(result), hide_index=True, width="stretch")
    lesson.look_for("the accuracy and parameter counts. The CNN shares small windows instead of buying a separate clue for every location.")
    lesson.aha("The same clue can be recognized wherever the object moved: sleeve edge, shoe edge, top-left edge, bottom-right edge!")


@lesson.step("The learned filters", beat="forreal")
def _():
    result = cached_vision_models()
    filters = vision.first_conv_filters(result)
    fig = vision.plot_small_images(filters, titles=[f"filter {i}" for i in range(len(filters))], width=1.1, vcenter=True)
    lesson.show(fig)
    lesson.look_for("tiny edge or blob detectors. These are the learned cousins of the kernels you edited.")
    maps = vision.feature_maps(result, limit=8)
    fig = vision.plot_small_images(maps, titles=[f"map {i}" for i in range(len(maps))], width=1.1)
    lesson.show(fig)
    lesson.look_for("bright spots. Each one marks where a filter lit up on one test image.")


@lesson.step("Where the CNN still struggles", beat="forreal")
def _():
    result = cached_vision_models()
    wrong = vision.cnn_wrong_examples(result, limit=6)
    if wrong:
        fig = vision.plot_small_images(
            [row[0] for row in wrong],
            titles=[f"{result.labels[row[1]]} → {result.labels[row[2]]}" for row in wrong],
            width=1.35,
        )
        lesson.show(fig)
        lesson.look_for("clothing classes that blur together in 28 by 28 gray pixels.")
    lesson.careful("Shirt, coat, and pullover can mash together even for humans in tiny gray pictures.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Diagonal hunter.** Design a 3×3 kernel that lights up on diagonal edges.
2. **Tiny champion.** In the notebook, reduce the CNN channels. What is the fewest that still beats the MLP?
3. **Upside down.** Flip a test image upside down and ask the CNN. It never saw that world.
4. **Compare to Chapter 17.** Is the CNN better because it has more weights, or because the weights are reused?
"""
    )
    lesson.kid_corner("Move a 3 by 3 Lego window over a drawing. Shout “edge!” whenever one side is empty and the other side is full.")


lesson.finish()
