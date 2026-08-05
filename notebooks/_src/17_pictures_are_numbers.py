# %% [markdown]
# # Chapter 17 · Pictures Are Numbers
#
# ### Read a digit off a grid of numbers, then teach a net to.
#
# *Part 4 · Seeing*
#
# ---
#
# A computer has never seen anything. Not a cat. Not a basketball. Not one heroic snack
# photo.
#
# This dataset is the classic scikit-learn digits set, bundled in `kidsml`: 1,797 tiny
# handwritten digits, each an 8 by 8 gray image. Every little square is a pixel with a
# brightness from 0 (black) to 16 (white).
#
# When you show the computer one picture, it gets a spreadsheet. This notebook proves it,
# then shows the weakness that Chapter 18 will fix.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from kidsml import workbook
from kidsml.datasets import digits
from kidsml.plots import confusion_grid, image_strip, show_image, use_house_style
from kidsml import vision

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Squint at this 8 by 8 grid of numbers. Can you read the digit hiding inside?

# %%
X, y, images = digits()
example_index = 3
fig, ax = plt.subplots(figsize=(4.2, 4.2))
show_image(images[example_index], ax=ax, numbers=True, title="This is a digit 3")
plt.show()

# %% [markdown]
# The picture **is** those numbers. A plain MLP — a multilayer perceptron, or stack of
# neuron layers — does not get the square. It gets the same 64 pixel numbers zipped into
# one long row.
#
# ```mermaid
# graph LR
#     A[Image] --> B[Grid of pixels]
#     B --> C[Flat row of 64 numbers]
#     C --> D[Model]
#     D --> E[Ten digit scores]
# ```
#
# Follow the arrows. The model never receives the word "loop" or "top-left corner." It
# gets number 0, number 1, number 2, all the way to number 63. Then ten scores come back.

# %%
vision.digit_as_flat_row(images[example_index])

# %% [markdown]
# > ⚠️ **Careful** A plain MLP does not know that pixel 10 touches pixel 11. To it,
# > stepping from pixel 10 to pixel 11 is no more special than teleporting from pixel 10
# > to pixel 47.
# >
# > That matters because pictures are made from nearby pixels teaming up: strokes,
# > corners, holes, and edges. Chapter 18 bolts that neighbour map back on.

# %% [markdown]
# ## ✏️ Work it out
#
# Your eyes can read the bright numbers below as a digit. The model sees only numbers.

# %%
shape = np.array(
    [
        [0, 8, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 0, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 8, 8, 8, 0],
    ]
)
pd.DataFrame(shape)

# %% [markdown]
# A model with one weight per pixel has to bolt a knob onto every input number:
#
# - 64 weights for an 8×8 gray image
# - 784 weights for a 28×28 gray image
# - 3,000,000 weights for a 1000×1000 colour photo
#
# The jump is fast because pictures grow in two directions at once. Double the width and
# double the height, and you made four times as many pixels! Colour photos multiply again
# because every pixel has red, green, and blue.
#
# > 📖 **Grown-ups call this:** a **pixel** is one little square in a picture. A gray
# > pixel is one number. A colour pixel is three numbers: red, green, and blue.
#
# > 📖 **Grown-ups call this:** an **MLP** is a multilayer perceptron: a plain stack of
# > neuron layers, with no built-in idea of pixel neighbours.

# %% [markdown]
# ## 👀 Take a look
#
# Here are ten real examples, then the ghostly average image for each digit.

# %%
first_examples = []
first_titles = []
for digit in range(10):
    idx = int(np.flatnonzero(y == digit)[0])
    first_examples.append(images[idx])
    first_titles.append(str(digit))
fig, _ = image_strip(first_examples, titles=first_titles, width=1.1)
plt.show()

# %% [markdown]
# Look at how wobbly real handwriting is. The model has to learn the family resemblance,
# not memorize one museum-perfect 3.

# %%
averages = vision.average_digit_images(images, y)
fig = vision.plot_small_images(averages, titles=[f"average {i}" for i in range(10)], width=1.35)
plt.show()

# %% [markdown]
# > 💡 **Aha!** The average 3 still looks a bit like a 3. That means the useful signal is
# > spread across many examples, not hidden in one magic row. The model gets patterns in
# > numbers. That is the spark!

# %% [markdown]
# ## 🎛️ Your turn
#
# A notebook cannot use the drawing canvas from the app. In the app, your mouse draws
# thick white strokes on a black pad; the app crops the marks, shrinks them to 8 by 8,
# and converts brightness to the same 0–16 scale. Here you can hand-edit the 8 by 8
# numbers directly. Nudge a few values and run the cell again.

# %%
my_digit = np.array(
    [
        [0, 0, 8, 14, 14, 6, 0, 0],
        [0, 6, 16, 6, 8, 16, 2, 0],
        [0, 0, 0, 0, 10, 12, 0, 0],
        [0, 0, 0, 7, 16, 4, 0, 0],
        [0, 0, 0, 0, 12, 14, 1, 0],
        [0, 0, 0, 0, 1, 13, 8, 0],
        [0, 4, 12, 4, 7, 16, 7, 0],
        [0, 0, 7, 13, 13, 6, 0, 0],
    ],
    dtype=float,
)
fig, ax = plt.subplots(figsize=(4.2, 4.2))
show_image(my_digit, ax=ax, numbers=True, title="your editable digit")
plt.show()

# %%
report = vision.train_digit_mlp(seed=0)
prediction, probabilities = vision.predict_digit_grid(report.model, my_digit)
print("The model guesses:", prediction)
fig = vision.plot_confidences(probabilities)
plt.show()

# %% [markdown]
# > ⚠️ **Careful** The model has only ever seen neat centred digits. Move the bright
# > numbers to one side, or make them tiny, and watch the guess wobble apart. That
# > mismatch is a real machine-learning bug.

# %% [markdown]
# ## 💻 In real code
#
# Here is the whole training idea. `X / 16` scales pixel brightness from 0..16 down to
# 0..1, which is friendlier for the MLP. The real helper uses the same ingredients and a
# fixed seed, so the recipe stays steady.

# %%
X, y, images = digits()
X_train, X_test, y_train, y_test = train_test_split(X / 16, y, stratify=y, random_state=0)
model = MLPClassifier(hidden_layer_sizes=(48,), max_iter=200, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train)
print("test accuracy:", round(model.score(X_test, y_test), 3))

# %%
print("helper test accuracy:", round(report.accuracy, 3))
fig, ax = plt.subplots(figsize=(5.5, 4.6))
confusion_grid(report.confusion, labels=list(range(10)), ax=ax)
plt.show()

# %% [markdown]
# In Chapter 09 you met this picture: rows are true answers, columns are guesses. Look
# away from the diagonal. Those off-diagonal bright squares are the exact pairs the model mixes up.

# %%
wrong = vision.misclassified_examples(report, limit=8)
fig = vision.plot_small_images(
    [row[0] for row in wrong],
    titles=[f"true {row[1]} → {row[2]}" for row in wrong],
    width=1.3,
)
plt.show()

# %% [markdown]
# Do its confusions feel familiar? A loopy 9 can look like a 4, a messy 5 can look like a
# 3, and a skinny 7 can look like a 1. The model's mistakes often rhyme with human
# mistakes because both trip on the same strokes.

# %%
weights = vision.first_layer_images(report.model, limit=12)
fig = vision.plot_small_images(weights, titles=[f"unit {i}" for i in range(len(weights))], width=1.15, vcenter=True)
plt.show()

# %% [markdown]
# Look for blurry strokes and digit parts. They are not full digits; they are little clue
# tiles the network can snap together.

# %% [markdown]
# ## 🏆 Go further
#
# 1. **Hardest digit.** Use the confusion matrix. Which row has the most mistakes?
# 2. **Find an easy mistake.** Pick a wrong image where you can read the digit right away.
# 3. **Blank a row.** Set one row of `my_digit` to 0. When does the prediction flip?
# 4. **Side quest: break it on purpose.** Edit a digit badly until the model changes its mind.
# 5. 🧸 **Little Kid Corner:** Make an 8 by 8 grid with coins or cereal pieces. Stand back and squint. Can someone read the number?

# %%
workbook.render(17)

# %% [markdown]
# ---
# **Next up:** Chapter 18 · *The Sliding Window* — where the model learns that neighbours matter.
