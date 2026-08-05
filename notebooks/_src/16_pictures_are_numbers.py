# %% [markdown]
# # Chapter 16 · Pictures Are Just Numbers
#
# ### Read a digit off a grid of numbers, then teach a net to.
#
# *Part 4 · Seeing*
#
# ---
#
# A computer has never seen anything. Not one thing. When you show it a photo,
# it gets a spreadsheet. This notebook proves it.

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
# ## 🎣 The Hook
#
# Squint at this 8 by 8 grid of numbers. Can you read the digit hiding in it?

# %%
X, y, images = digits()
example_index = 3
fig, ax = plt.subplots(figsize=(4.2, 4.2))
show_image(images[example_index], ax=ax, numbers=True, title="This is a digit 3")
plt.show()

# %% [markdown]
# The picture **is** those numbers. A plain model does not get the square. It gets
# the same 64 numbers as one long row.

# %%
vision.digit_as_flat_row(images[example_index])

# %% [markdown]
# > ⚠️ **Careful** A plain MLP does not know that pixel 10 touches pixel 11. It gets
# > a row of 64 values. Chapter 17 fixes that exact weakness.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Your eyes can read the bright numbers below as a digit. The model will only see numbers.

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
# A model with one weight per pixel would need:
#
# - 64 weights for an 8×8 gray image
# - 784 weights for a 28×28 gray image
# - 3,000,000 weights for a 1000×1000 colour photo
#
# > 📖 **Grown-ups call this:** a **pixel** is one little square in a picture. A gray
# > pixel is one number. A colour pixel is three numbers: red, green, and blue.

# %% [markdown]
# ## 👀 See It
#
# Here are ten real examples, then the average image for each digit.

# %%
first_examples = []
first_titles = []
for digit in range(10):
    idx = int(np.flatnonzero(y == digit)[0])
    first_examples.append(images[idx])
    first_titles.append(str(digit))
fig, _ = image_strip(first_examples, titles=first_titles, width=1.1)
plt.show()

# %%
averages = vision.average_digit_images(images, y)
fig = vision.plot_small_images(averages, titles=[f"average {i}" for i in range(10)], width=1.35)
plt.show()

# %% [markdown]
# > 💡 **Aha!** The average 3 still looks a bit like a 3. The model gets no magic,
# > only patterns in numbers.

# %% [markdown]
# ## 🎛️ Play With It
#
# A notebook cannot use the drawing canvas from the app. Here you can hand-edit the
# 8 by 8 numbers directly. Change a few values and run the cell again.

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
# > numbers to one side, or make them tiny, and watch it fall apart. That mismatch is
# > a real machine-learning bug.

# %% [markdown]
# ## 💻 For Real
#
# Here is the whole training idea. The real helper uses the same ingredients and a fixed seed.

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
# In Chapter 09 you met this picture: rows are true answers, columns are guesses.
# The off-diagonal cells are the mix-ups.

# %%
wrong = vision.misclassified_examples(report, limit=8)
fig = vision.plot_small_images(
    [row[0] for row in wrong],
    titles=[f"true {row[1]} → {row[2]}" for row in wrong],
    width=1.3,
)
plt.show()

# %% [markdown]
# Do its confusions feel familiar? 4/9, 3/5, and 7/1 are hard for people too.

# %%
weights = vision.first_layer_images(report.model, limit=12)
fig = vision.plot_small_images(weights, titles=[f"unit {i}" for i in range(len(weights))], width=1.15, vcenter=True)
plt.show()

# %% [markdown]
# First-layer weights are only semi-readable here, but some look like blurry strokes and digit parts.

# %% [markdown]
# ## 🏆 Challenge
#
# 1. **Hardest digit.** Use the confusion matrix. Which row has the most mistakes?
# 2. **Find an easy mistake.** Pick a wrong image where you can read the digit right away.
# 3. **Blank a row.** Set one row of `my_digit` to 0. When does the prediction flip?
# 4. **Break it on purpose.** Edit a digit badly until the model changes its mind.
# 5. 🧸 **Little Kid Corner:** Make an 8 by 8 grid with coins or cereal pieces. Stand back. Can someone read the number?

# %%
workbook.render(16)

# %% [markdown]
# ---
# **Next up:** Chapter 17 · *The Sliding Window* — where the model learns that neighbours matter.
