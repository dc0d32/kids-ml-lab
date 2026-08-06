# %% [markdown]
# # Chapter 12 · One Neuron
#
# ### Chapter 2 plus Chapter 4, stacked into one circle.
#
# *Part 3 · Neural networks*
#
# ---
#
# This notebook is the same chapter as the app, but with the code showing.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kidsml.datasets import toy_shape, two_blobs_tiny, xor_exact
from kidsml.nn_numpy import Neuron
from kidsml.nnplots import neuron_surface_figure
from kidsml.plots import decision_boundary, draw_line, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Part 3 sounds like a new planet: **neural networks**. It is not. A neuron is Chapter 2's
# straight-line score bolted to Chapter 4's probability squish.
#
# Chapter 11 also showed why the squish matters: straight steps stacked on straight steps
# collapse back into one straight step. The squish is what keeps a neuron from being only a
# line wearing a costume.
#
# That matters because there is no missing spell. If you can read `w1*x1 + w2*x2 + b`,
# you can read the engine inside this circle. The circle wraps the score so a yes/no model
# can say “barely yes”, “very yes”, or “I am near the fence”.
#
# ```mermaid
# graph TD
#     X1[x₁] --> M[weights times inputs]
#     X2[x₂] --> M
#     M --> S((Σ + b))
#     S --> A[squish]
#     A --> Y[output from 0 to 1]
# ```
#
# Read every arrow as “this number flows into the next box.” `x₁` and `x₂` are the two
# input numbers. The weights multiply them, `Σ + b` adds the weighted pieces plus the
# bias, the squish clips the raw score to 0..1, and the output is the neuron's answer.
#
# > 💡 **Aha!** One neuron is `output = squish(w1*x1 + w2*x2 + b)`: Chapter 2 inside,
# > Chapter 4 outside.

# %% [markdown]
# ## ✏️ Work it out
#
# Use **w1 = 2**, **w2 = -1**, **b = 0.5**. We picked small numbers so every row can be
# checked by hand. First build the raw score `z`, then squish it.
#
# The raw score decides which side of the line the point is on. The squish keeps the same
# side, but turns “how far from the line?” into a 0-to-1 confidence gauge.

# %%
hand = pd.DataFrame(
    {
        'x1': [1, 0, 1],
        'x2': [0, 1, 2],
        'z working': ['2*1 + (-1)*0 + 0.5', '2*0 + (-1)*1 + 0.5', '2*1 + (-1)*2 + 0.5'],
        'z': [2.5, -0.5, 0.5],
        'sigmoid(z) approx': [0.92, 0.38, 0.62],
        'prediction': ['red', 'blue', 'red'],
    }
)
hand

# %% [markdown]
# Why not leave the raw score alone? For a class answer, `z = 19` and `z = 1900` both mean
# “red”, but a training rule needs a bounded target to compare with `0` and `1`. The squish
# clips the wild number into a soft confidence score without moving the fence.
#
# > 📖 **Grown-ups call this:** the **activation** is the squish function at the end of a
# > neuron.
#
# > ⚠️ **Careful:** If you double w1, w2, and b, every raw score doubles. The zero places
# > stay zero, so the boundary stays put. Far-away points become more confident; the line
# > does not move.

# %%
X_demo, _ = toy_shape('blobs', n=180, noise=0.22, seed=4)
demo_neuron = Neuron(w=np.array([2.0, -1.0]), b=0.5, activation='sigmoid')
neuron_surface_figure(demo_neuron, X_demo, steps=45, title='The squish turns the raw ramp into 0..1')

# %% [markdown]
# Look for the ramp flattening into a floor and ceiling. The middle fence stays in the same
# place.

# %% [markdown]
# ## 👀 Take a look
#
# `Neuron.raw(X)` shows the Chapter 2 score before the squish. A positive raw score lands
# on one side, a negative score lands on the other, and zero is the fence.
#
# These are toy blob coordinates from two clumps. We chose a few rows from both answers so
# the sign flips and the fence is visible. The neuron is not hiding a secret formula; it is
# computing `x1 + x2 - 7` and then sending that number through the squish.

# %%
X_tiny, y_tiny = two_blobs_tiny()
neuron = Neuron(w=np.array([1.0, 1.0]), b=-7.0, activation='sigmoid')
# A mix of both answers. The first five rows are all one class, so the score would
# never change sign and the fence would be invisible.
rows = [0, 3, 5, 7, 9]
pd.DataFrame({'x1': X_tiny[rows, 0], 'x2': X_tiny[rows, 1], 'raw z': neuron.raw(X_tiny[rows])})

# %% [markdown]
# The raw numbers are not probabilities yet. They are signed fence scores from the line
# machine.

# %% [markdown]
# ## 🎛️ Your turn
#
# The 3D surface is the neuron's output over the whole `x1, x2` plane. Height and colour
# both mean confidence: low is near 0, high is near 1. In the app you can rotate it; here
# the same object appears below.

# %%
X, y = toy_shape('blobs', n=180, noise=0.22, seed=4)
play_neuron = Neuron(w=np.array([2.0, -1.0]), b=0.5, activation='sigmoid')
fig, ax = plt.subplots(figsize=(5.6, 4.6))
decision_boundary(lambda G: play_neuron.forward(G), X, y, ax=ax, steps=180, shade_confidence=True, title='Boundary and confidence')
draw_line(play_neuron.w[0], play_neuron.w[1], play_neuron.b, ax=ax)
plt.show()

# %%
neuron_surface_figure(play_neuron, X, steps=45, title='Neuron output as a ramp')

# %% [markdown]
# Watch the green line first. That is where `z = 0`, so it is the same straight fence from
# Chapter 2. Now watch the colours and the 3D ramp: steeper weights make the answer change
# faster as you walk away from the fence. The ramp snaps upward!
#
# XOR is still the Chapter 3 wall. Opposite corners need the same colour, and a single
# straight boundary always cuts the square into two neighbouring chunks. One neuron has
# one boundary, so it cannot win.

# %%
X_xor, y_xor = xor_exact()
pd.DataFrame(
    {
        'x1': X_xor[:, 0],
        'x2': X_xor[:, 1],
        'truth': y_xor,
        'one neuron prediction': play_neuron.predict(X_xor),
    }
)

# %% [markdown]
# ## 💻 In real code
#
# Now we train the neuron instead of choosing its numbers by hand. Scikit-learn calls the
# same idea **logistic regression**: a line score, a sigmoid, and a training rule turning
# the knobs. Chapter 13 opens that training rule and shows how the knobs move.
#
# The learned numbers do not have to match exactly, because the two tools use different
# knob-moving recipes. What should match is the divider: it should point through the
# same gap in the blobs.

# %%
X_fit, y_fit = toy_shape('blobs', n=180, noise=0.18, seed=8)
mine = Neuron(w=np.zeros(2), b=0.0, activation='sigmoid')
losses = mine.fit(X_fit, y_fit, lr=0.8, epochs=900)
sk = LogisticRegression(C=1_000_000, solver='lbfgs').fit(X_fit, y_fit)

pd.DataFrame(
    {
        'model': ['our Neuron.fit', 'sklearn LogisticRegression'],
        'w1': [mine.w[0], sk.coef_[0, 0]],
        'w2': [mine.w[1], sk.coef_[0, 1]],
        'b': [mine.b, sk.intercept_[0]],
        'accuracy': [(mine.predict(X_fit) == y_fit).mean(), sk.score(X_fit, y_fit)],
    }
).round(3)

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.6))
decision_boundary(lambda G: mine.forward(G), X_fit, y_fit, ax=ax, steps=180, title='Our trained neuron')
plt.show()

# %% [markdown]
# Look at the gap between the two blob clouds. The trained neuron drove a straight divider
# through that gap, which is Chapter 2 plus Chapter 4 stacked together.
#
# ## 🏆 Go further
#
# Work through the interactive questions, then try these quests.

# %%
from kidsml import workbook

workbook.render(12)

# %% [markdown]
# 1. **Find perfect blob weights.** Use the app sliders until the blob mistakes hit zero.
#    Which knob mostly rotates the line?
# 2. **Say yes to everything.** Make almost the whole plane red. Which bias did it take?
# 3. **Make a shrug machine.** Set every learned number to zero. What output do you get?
# 4. **Feel the XOR wall.** Try to get zero XOR mistakes with one neuron, then explain the
#    Chapter 3 reason it cannot happen.
# 5. 🧸 **Little Kid Corner:** Draw a chalk line. Far from the line means a loud answer.
#    On the line means a shrug.
#
# ---
# **Next up:** Chapter 13 · *How a Neuron Learns* — the neuron moves its own sliders.
