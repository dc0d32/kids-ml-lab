# %% [markdown]
# # Chapter 13 · Two Layers, Three Neurons
#
# ### Hidden neurons each draw a line — together they bend.
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

from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.nnplots import boundary_with_hidden, hidden_surfaces_figure, mlp_snapshot_training, model_from_snapshot, network_diagram
from kidsml.plots import decision_boundary, loss_curve, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# XOR is back. One neuron cannot do it. Three neurons wired in two layers can.

# %%
network_diagram([2, 3, 1], activation='tanh')
plt.show()

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Make two hidden features: OR-ish and AND-ish.

# %%
xor_table = pd.DataFrame(
    {'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1], 'OR-ish': [0, 1, 1, 1], 'AND-ish': [0, 0, 0, 1], 'XOR': [0, 1, 1, 0]}
)
xor_table

# %% [markdown]
# In the new space, `score = OR - 2*AND - 0.5` is positive only for the red XOR rows.
#
# The hidden layer did not bend anything by itself. It invented new features. That is the
# heart of this part.

# %% [markdown]
# ## 👀 See It
#
# Train `[2, 3, 1]` on XOR, then draw each hidden neuron's line and the final boundary.

# %%
X_xor, y_xor = xor_exact()
snaps = mlp_snapshot_training([2, 3, 1], X_xor, y_xor, lr=0.8, epochs=3000, every=150, activation='tanh', seed=2)
model = model_from_snapshot([2, 3, 1], snaps[-1], activation='tanh', seed=2)

fig, ax = plt.subplots(figsize=(5.4, 4.6))
boundary_with_hidden(model, X_xor, y_xor, ax=ax, title='Hidden lines plus final boundary', steps=180)
plt.show()

# %%
hidden_surfaces_figure(model, X_xor, steps=65)
plt.show()

# %% [markdown]
# The output neuron is doing Chapter 2 again, but its inputs are not x1 and x2. Its inputs
# are the three hidden reports: which side of line A, B, and C am I on?

# %% [markdown]
# ## 🎛️ Play With It
#
# Change the hidden size and activation in the app. Here are three sizes on XOR.

# %%
X_play, y_play = toy_shape('xor', n=180, noise=0.16, seed=3)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
for ax, hidden in zip(axes, [1, 2, 3]):
    m = MLP([2, hidden, 1], activation='tanh', seed=3)
    losses = m.fit(X_play, y_play, lr=0.6, epochs=900, record_every=5)
    decision_boundary(lambda G, model=m: model.predict_proba(G), X_play, y_play, ax=ax, steps=160, title=f'{hidden} hidden')
plt.show()

# %% [markdown]
# ## 💻 For Real
#
# More hidden neurons can help, but they can also wiggle around noisy dots.

# %%
X_over, y_over = toy_shape('moons', n=70, noise=0.32, seed=10)
small = MLP([2, 3, 1], activation='tanh', seed=1)
big = MLP([2, 8, 1], activation='tanh', seed=1)
small.fit(X_over, y_over, lr=0.6, epochs=1200, record_every=20)
big.fit(X_over, y_over, lr=0.6, epochs=2200, record_every=20)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
decision_boundary(lambda G: small.predict_proba(G), X_over, y_over, ax=axes[0], steps=160, title='3 hidden neurons')
decision_boundary(lambda G: big.predict_proba(G), X_over, y_over, ax=axes[1], steps=160, title='8 hidden neurons')
plt.show()

# %% [markdown]
# ## 🏆 Challenge
#
# 1. What is the smallest hidden layer that solves XOR?
# 2. What about spiral?
# 3. Can you set weights by hand using OR-ish and AND-ish?
# 4. Scrub training in the app and name what each hidden line learned.
# 5. 🧸 **Little Kid Corner:** Three friends make a team. Two notice where you stand.
#    The last friend listens and decides.
#
# ---
# **Next up:** Chapter 14 · *Deeper and Wider* — more layers, squishes, and over-studying.
