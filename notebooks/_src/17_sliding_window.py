# %% [markdown]
# # Chapter 17 · The Sliding Window
#
# ### Convolutions by pencil, then a tiny CNN.
#
# *Part 4 · Seeing*
#
# ---
#
# Chapter 16's model had a weakness: it did not know that two pixels next to each other
# belong together. Shuffle all 64 pixels the same way and it would learn about as well.
#
# That is wrong for pictures. Nearby pixels make strokes, corners, and edges. The fix is
# to use one small window again and again.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kidsml import workbook
from kidsml.datasets import digits, tiny_image
from kidsml.plots import show_image, use_house_style
from kidsml import vision

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# Here is the fix. One small window slides across the image.
#
# ```mermaid
# graph LR
#     A[Image patch] --> B[Same 3x3 kernel]
#     B --> C[Multiply and add]
#     C --> D[One feature-map cell]
#     D --> E[Slide right]
#     E --> B
# ```
#
# Notice the word **same**. We do not invent a new edge detector for every spot. The same
# little grid visits the top-left corner, the middle, and the bottom-right corner.
#
# > 🧸 **Little Kid Corner** — Put a sticky note with a 3 by 3 hole over a picture.
# > Look through the hole, move it one square, and look again. You are doing the
# > sliding-window idea with paper.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# This image has a vertical edge: dark on the left, bright on the right.

# %%
image = tiny_image()
kernel = vision.KERNEL_PRESETS["vertical edge"]
patch = image[:3, :3]

pd.DataFrame(image.astype(int))

# %%
pd.DataFrame(kernel.astype(int))

# %% [markdown]
# For the first window, line up the two 3 by 3 grids. Multiply matching cells, then add.

# %%
pd.DataFrame(patch.astype(int))

# %%
first_answer = float((patch * kernel).sum())
print("first window answer:", first_answer)

# %% [markdown]
# The arithmetic is:
#
# `0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 = 27`
#
# Now slide one square at a time. Across the rows, the 3-high window can start at row 1,
# row 2, or row 3. Starting at row 4 would hang off the bottom. The columns work the same
# way, so the output is 3 rows by 3 columns: **9 places to land**.
#
# > 📖 **Grown-ups call this:** **convolution** means sliding a small grid of weights
# > over a picture. Multiply what lines up, then add.

# %% [markdown]
# ## 👀 See It
#
# Now do all 9 window positions with the same plain double loop.

# %%
output = vision.convolve2d_valid(image, kernel)
pd.DataFrame(output.astype(int))

# %%
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
show_image(image, ax=axes[0], numbers=True, title="image")
show_image(kernel, ax=axes[1], numbers=True, title="kernel", cmap="coolwarm")
show_image(output, ax=axes[2], numbers=True, title="output", cmap="magma")
fig.tight_layout()
plt.show()

# %% [markdown]
# > 💡 **Aha!** Look at the output grid. The big numbers land where dark pixels become
# > bright pixels. You detected an edge by hand, using the same multiply-and-add at every
# > position.

# %% [markdown]
# ## 🎛️ Play With It
#
# Edit the kernel below. A blur is a kernel full of `1/9` values because each output cell
# becomes the average of its 3×3 neighbourhood. Try `vertical edge`, `horizontal edge`,
# `blur`, `sharpen`, and your own numbers.

# %%
_, y_digits, digit_images = digits()
digit_index = int(np.flatnonzero(y_digits == 3)[0])
live_image = digit_images[digit_index] / 16.0

my_kernel = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=float,
)
fig, conv = vision.plot_kernel_demo(live_image, my_kernel)
plt.show()
print("raw output range:", round(float(conv.min()), 2), "to", round(float(conv.max()), 2))

# %%
pattern = vision.generated_pattern(28)
fig, conv = vision.plot_kernel_demo(pattern, vision.KERNEL_PRESETS["blur"])
plt.show()

# %% [markdown]
# ## 💻 For Real
#
# The kernels above were designed by a person. What if we let the model choose its own?
# That is the leap.
#
# During training, the CNN changes the kernel numbers until useful patches light up. It is
# still the same sliding-window game, but the edge finder is learned instead of hand-written.
#
# ```mermaid
# graph LR
#     A[Image] --> B[Convolution]
#     B --> C[Squish]
#     C --> D[Pool]
#     D --> E[Classify]
# ```
#
# Look at the stack: find small patterns, squish the scores, keep the strongest signals,
# then make the final guess.

# %%
result = vision.train_cnn_and_mlp(seed=0, train_size=6000, test_size=1000, epochs=2, allow_download=True)
print(result.dataset_name)
print("training seconds:", round(result.elapsed, 2))
vision.model_comparison_table(result)

# %% [markdown]
# If Fashion-MNIST is not cached, the first run downloads about 30 MB into
# `data/torchvision/`. If that fails, the helper falls back to sklearn's 8×8 digits and
# says so.
#
# The CNN reuses the same little window everywhere. That teaches it **an edge is an edge
# wherever it appears**: sleeve edge, shoe edge, top-left edge, bottom-right edge.
#
# This buys two things at once. Fewer parameters, because one kernel is shared across many
# positions. Better scores, because the same clue can be recognized wherever the object moved.

# %%
filters = vision.first_conv_filters(result)
fig = vision.plot_small_images(filters, titles=[f"filter {i}" for i in range(len(filters))], width=1.1, vcenter=True)
plt.show()

# %% [markdown]
# Look for tiny edge or blob detectors. These are the learned cousins of the kernels you edited.

# %%
maps = vision.feature_maps(result, limit=8)
fig = vision.plot_small_images(maps, titles=[f"map {i}" for i in range(len(maps))], width=1.1)
plt.show()

# %% [markdown]
# Bright spots show where a filter lit up on one test image. Same filter, many possible locations.

# %%
wrong = vision.cnn_wrong_examples(result, limit=6)
fig = vision.plot_small_images(
    [row[0] for row in wrong],
    titles=[f"{result.labels[row[1]]} → {result.labels[row[2]]}" for row in wrong],
    width=1.35,
)
plt.show()

# %% [markdown]
# Shirt, coat, and pullover can be hard even for humans in 28×28 gray pixels.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. **Diagonal hunter.** Design a 3×3 kernel that lights up on diagonal edges.
# 2. **Tiny champion.** Change the CNN channels in `kidsml/vision.py` in a copy of the function. What is the fewest that still beats the MLP?
# 3. **Upside down.** Flip a test image upside down and ask the CNN. It never saw that world.
# 4. **Compare to Chapter 14.** Is the CNN better because it has more weights, or because the weights are reused?
# 5. 🧸 **Little Kid Corner:** Move a 3 by 3 Lego window over a drawing. Shout “edge!” whenever one side is empty and the other side is full.

# %%
workbook.render(17)

# %% [markdown]
# ---
# **Next up:** Chapter 18 · *You Are Like Your Neighbors* — where the model learns by looking nearby instead of training.
