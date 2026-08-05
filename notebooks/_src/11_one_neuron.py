# %% [markdown]
# # Chapter 11 · One Neuron
#
# ### It's Chapter 2 plus a squish. That's all.
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
from kidsml.nnplots import neuron_diagram, neuron_surface_figure
from kidsml.plots import decision_boundary, draw_line, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# People draw neural networks as terrifying webs of circles and arrows.
#
# Here is one circle. You already know what it does. The inside is Chapter 2. The squish
# is Chapter 4.

# %%
fig = neuron_diagram(weights=(2, -1), bias=0.5, activation='sigmoid')
plt.show()

# %% [markdown]
# `output = squish(w1*x1 + w2*x2 + b)`
#
# Point at the brackets: that is Chapter 2. Point at `squish`: that is Chapter 4.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Use **w1 = 2**, **w2 = -1**, **b = 0.5**.

# %%
hand = pd.DataFrame(
    {
        'x1': [1, 0, 1],
        'x2': [0, 1, 2],
        'z = 2*x1 - x2 + 0.5': [2.5, -0.5, 0.5],
        'sigmoid(z) approx': [0.92, 0.38, 0.62],
        'prediction': ['red', 'blue', 'red'],
    }
)
hand

# %% [markdown]
# Double w1, w2 and b. Every z doubles, but the place where z equals zero stays put.
# Confidence changes. The boundary does not.

# %% [markdown]
# ## 👀 See It
#
# `Neuron.raw(X)` is the old line score, before the squish.

# %%
X_tiny, y_tiny = two_blobs_tiny()
neuron = Neuron(w=np.array([1.0, 1.0]), b=-7.0, activation='sigmoid')
pd.DataFrame({'x1': X_tiny[:5, 0], 'x2': X_tiny[:5, 1], 'raw z': neuron.raw(X_tiny[:5])})

# %% [markdown]
# ## 🎛️ Play With It
#
# The 3D surface is the neuron's output over the whole plane. Rotate it in the app. In the
# notebook, the same object appears below.

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
# XOR is still a wall for one neuron.

# %%
X_xor, y_xor = xor_exact()
print('XOR predictions:', play_neuron.predict(X_xor).tolist())
print('XOR mistakes:', int((play_neuron.predict(X_xor) != y_xor).sum()))

# %% [markdown]
# ## 💻 For Real
#
# Train our neuron on blobs, then compare it with scikit-learn's logistic regression.

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
# You rediscovered logistic regression by stacking two chapters.
#
# ## 🏆 Challenge
#
# 1. Find weights by hand that classify the blobs perfectly.
# 2. Make a neuron that says yes for almost everything.
# 3. Make one that is maximally unsure everywhere.
# 4. Try to solve XOR with one neuron, then explain why it fights back.
# 5. 🧸 **Little Kid Corner:** Draw a chalk line. Far from the line means a loud answer.
#    On the line means a shrug.
#
# ---
# **Next up:** Chapter 12 · *How a Neuron Learns* — the neuron moves its own sliders.
