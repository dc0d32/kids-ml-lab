"""Chapter 17 · The Sliding Window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import digits, tiny_image
from kidsml.plots import show_image
from kidsml import vision

ui.page_setup(17)


@st.cache_resource(show_spinner="Training two tiny image models...")
def cached_vision_models():
    return vision.train_cnn_and_mlp(seed=0, train_size=6000, test_size=1000, epochs=2, allow_download=True)


@st.cache_data(show_spinner=False)
def cached_digit_image():
    _, y, images = digits()
    idx = int(np.flatnonzero(y == 3)[0])
    return images[idx] / 16.0


# ---------------------------------------------------------------------------
ui.beat("hook", "Pictures need neighbours.")

st.markdown(
    """
Chapter 16's model had a weakness: it did not know that two pixels next to each
other belong together. Shuffle all 64 pixels the same way and it would learn about
as well.

That is wrong for pictures. Nearby pixels make strokes, corners, and edges.
Here is the fix. One small window slides across the image.
"""
)

ui.little_kid_corner(
    "Put a sticky note with a 3 by 3 hole over a picture. Look through the hole, move it one square, and look again. "
    "You are doing the sliding-window idea with paper."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Detect one edge with a pencil.")

image = tiny_image()
kernel = vision.KERNEL_PRESETS["vertical edge"]
output = vision.convolve2d_valid(image, kernel)
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

st.markdown(
    """
For the first window:

`0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 = 27`

Now slide one square at a time. A 5×5 image with a 3×3 window has 9 places to land.
"""
)
ui.jargon("convolution", "Slide a small grid of weights over a picture. Multiply what lines up, then add.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The edge lights up.")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
show_image(image, ax=axes[0], numbers=True, title="image")
show_image(kernel, ax=axes[1], numbers=True, title="kernel", cmap="coolwarm")
show_image(output, ax=axes[2], numbers=True, title="output", cmap="magma")
fig.tight_layout()
ui.show(fig)
ui.aha("The big numbers land where dark pixels become bright pixels. You detected an edge by hand.")

# ---------------------------------------------------------------------------
ui.beat("play", "Edit the 3×3 window live.")

st.markdown("Try the preset buttons, then change one number. A blur is a kernel full of 1/9 values.")
if "ch17_kernel" not in st.session_state:
    st.session_state["ch17_kernel"] = vision.KERNEL_PRESETS["vertical edge"].copy()

buttons = st.columns(6)
for col, name in zip(buttons, vision.KERNEL_PRESETS):
    if col.button(name, key=f"preset_{name}"):
        st.session_state["ch17_kernel"] = vision.KERNEL_PRESETS[name].copy()

values = []
for r in range(3):
    cols = st.columns(3)
    row = []
    for c in range(3):
        value = cols[c].number_input(
            f"k{r}{c}",
            value=float(st.session_state["ch17_kernel"][r, c]),
            step=0.25,
            key=f"kernel_{r}_{c}",
        )
        row.append(value)
    values.append(row)
live_kernel = np.array(values, dtype=float)
st.session_state["ch17_kernel"] = live_kernel

which_image = st.radio("Try the kernel on:", ["a digit", "a bigger pattern"], horizontal=True)
if which_image == "a digit":
    live_image = cached_digit_image()
else:
    live_image = vision.generated_pattern(28)

fig, conv = vision.plot_kernel_demo(live_image, live_kernel)
ui.show(fig)
st.caption(f"Raw output range: {conv.min():.2f} to {conv.max():.2f}")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Let the model choose its own kernels.")

st.markdown(
    """
The kernels above were designed by a person. What if we let the model choose its
own? That is the leap.
"""
)
st.code(
    """
model = TinyCNN()
for image_batch, labels in train_loader:
    guesses = model(image_batch)
    loss = cross_entropy(guesses, labels)
    loss.backward()
    optimizer.step()
""",
    language="python",
)

result = cached_vision_models()
st.markdown(
    f"Using **{result.dataset_name}**: {result.train_size} training images, {result.test_size} test images, "
    f"{result.epochs} epochs. First download is about 30 MB and is cached in `data/torchvision/`."
)
st.dataframe(vision.model_comparison_table(result), hide_index=True, use_container_width=True)

ui.aha(
    "The CNN reuses the same little window everywhere. It learns that an edge is an edge "
    "wherever it appears, instead of learning a separate edge detector for every spot."
)

filters = vision.first_conv_filters(result)
fig = vision.plot_small_images(filters, titles=[f"filter {i}" for i in range(len(filters))], width=1.1, vcenter=True)
ui.show(fig)
st.caption("Some learned filters look like tiny edge or blob detectors — the cousins of the kernels you edited.")

maps = vision.feature_maps(result, limit=8)
fig = vision.plot_small_images(maps, titles=[f"map {i}" for i in range(len(maps))], width=1.1)
ui.show(fig)
st.caption("Feature maps show where a filter lit up on one test image.")

wrong = vision.cnn_wrong_examples(result, limit=6)
if wrong:
    fig = vision.plot_small_images(
        [row[0] for row in wrong],
        titles=[f"{result.labels[row[1]]} → {result.labels[row[2]]}" for row in wrong],
        width=1.35,
    )
    ui.show(fig)
    ui.careful("Shirt, coat, and pullover can be hard even for humans in 28×28 gray pixels.")

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. **Diagonal hunter.** Design a 3×3 kernel that lights up on diagonal edges.
2. **Tiny champion.** In the notebook, reduce the CNN channels. What is the fewest that still beats the MLP?
3. **Upside down.** Flip a test image upside down and ask the CNN. It never saw that world.
4. **Compare to Chapter 14.** Is the CNN better because it has more weights, or because the weights are reused?
5. 🧸 **Little Kid Corner:** Move a 3 by 3 Lego window over a drawing. Shout “edge!” whenever one side is empty and the other side is full.
"""
)

ui.worksheet_link(17)
