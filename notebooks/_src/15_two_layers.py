# %% [markdown]
# # Chapter 15 · Two Layers, Three Neurons
#
# ### Hidden neurons move the points so one line can win.
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
from IPython.display import Image

from kidsml import foldspace
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.nnplots import boundary_with_hidden, hidden_surfaces_figure, mlp_snapshot_training, model_from_snapshot
from kidsml.plots import COOL, PANEL, WARM, decision_boundary, loss_curve, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# XOR is back, the tiny checkerboard that keeps catching one-neuron models in the act.
#
# One neuron cannot solve it: Chapter 3 proved one straight line cannot put opposite
# corners together. The escape route was “invent better features.” A hidden layer does that
# for us, then the output neuron runs the Chapter 2 line trick on those new features.
#
# ```mermaid
# graph LR
#     X1[x1] --> H1[h1]
#     X1 --> H2[h2]
#     X1 --> H3[h3]
#     X2[x2] --> H1
#     X2 --> H2
#     X2 --> H3
#     H1 --> O[output neuron]
#     H2 --> O
#     H3 --> O
# ```
#
# Read left to right: two original inputs feed three hidden neurons, and those three
# reports feed one final neuron. `h₁`, `h₂`, and `h₃` mean hidden neuron 1, 2, and 3.
# Each one outputs a new number for the final neuron to read.
#
# > 📖 **Grown-ups call this:** a **hidden layer** is a layer between the inputs and the
# > final output. You can inspect its numbers, but they are not the final answer.

# %% [markdown]
# ## ✏️ Work it out
#
# We will make two hidden features by hand: **OR-ish** and **AND-ish**. This table is the
# whole XOR story shrunk to four dots. Here OR-ish is `h1` and AND-ish is `h2`:
# two fresh coordinates we invented from the same dot.
#
# In the original `x1, x2` square, the red points sit in opposite corners. No straight line
# can grab both red corners without also grabbing a blue one.

# %%
xor_table = pd.DataFrame(
    {'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1], 'OR-ish': [0, 1, 1, 1], 'AND-ish': [0, 0, 0, 1], 'XOR': [0, 1, 1, 0]}
)
xor_table

# %% [markdown]
# Those two new columns are more than numbers in a table. They *move the four dots*. Watch:
# each dot starts at its `(x1, x2)` spot and slides to its `(OR-ish, AND-ish)` spot from the
# table. Both blue corners stay put. But both red corners — `(0, 1)` and `(1, 0)` — head for
# the very same place, `(1, 0)`.

# %%
Image(data=foldspace.fold_gif_bytes())

# %% [markdown]
# Prefer a still picture? Here is the same thing frozen: before on the left, after on the
# right.

# %%
foldspace.fold_still_pair()

# %% [markdown]
# In the after picture, both red rows sit at `(h1, h2) = (1, 0)`, while the blue rows are
# `(0, 0)` and `(1, 1)`. Now `score = OR - 2*AND - 0.5` is positive only for red.
#
# The hidden layer never bent the output line. It **moved the points into a new space**
# where one straight line already works. Same ruler, better map — that is Chapter 3 escape
# route 1, done automatically.

# %% [markdown]
# ## 👀 Take a look
#
# Train `[2, 3, 1]` on XOR, then inspect the hidden coordinates before drawing the final
# boundary. The table columns `h1`, `h2`, and `h3` are the three hidden-neuron outputs:
# new coordinates for the same four XOR dots.

# %%
X_xor, y_xor = xor_exact()
snaps = mlp_snapshot_training([2, 3, 1], X_xor, y_xor, lr=0.8, epochs=3000, every=150, activation='tanh', seed=2)
model = model_from_snapshot([2, 3, 1], snaps[-1], activation='tanh', seed=2)

# %%
hidden = model.hidden_outputs(X_xor)
out = model.predict_proba(X_xor)
pd.DataFrame(
    {
        'x1': X_xor[:, 0],
        'x2': X_xor[:, 1],
        'h1': hidden[:, 0],
        'h2': hidden[:, 1],
        'h3': hidden[:, 2],
        'output': out,
        'XOR': y_xor,
    }
).round(3)

# %% [markdown]
# Look for rows with the same XOR answer. They are no longer trapped in opposite corners;
# the map has been folded before the output neuron makes one final cut. The table numbers
# become a hidden-space picture.

# %%
fig = plt.figure(figsize=(6.2, 5.0))
ax = fig.add_subplot(111, projection='3d')
colors = np.where(y_xor == 1, WARM, COOL)
ax.scatter(hidden[:, 0], hidden[:, 1], hidden[:, 2], c=colors, s=90, edgecolor=PANEL, linewidth=1.0)
ax.set_xlabel('h1 output')
ax.set_ylabel('h2 output')
ax.set_zlabel('h3 output')
ax.set_title('XOR points after the hidden layer')
plt.show()

# %%
hidden_surfaces_figure(model, X_xor, steps=65)

# %%
fig, ax = plt.subplots(figsize=(5.4, 4.6))
boundary_with_hidden(model, X_xor, y_xor, ax=ax, title='Hidden lines plus final boundary', steps=180)
plt.show()

# %% [markdown]
# Read these in order. The table shows the new hidden coordinates. The three coloured
# hidden lines send ramp readings. The final neuron adds those readings up, so the shaded
# boundary bends in the original picture.
#
# Why do they learn different lines? They start with small random differences. After that,
# each neuron receives slightly different gradients, so their jobs peel apart. If every
# hidden neuron started identical, they would tend to march in a crowd.

# %% [markdown]
# ## 🎛️ Your turn
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
# With one hidden neuron you are mostly back to one learned line. Add a few, and the model
# can invent several features before the final neuron decides.
#
# Look for the hidden lines first, then look at the shaded final boundary. The bend appears
# in original space because the output line is reading transformed hidden coordinates.

# %% [markdown]
# ## 💻 In real code
#
# Every hidden neuron adds one more bend the boundary can make. So the question is how many
# bends you actually want. Here are 3 bends versus 8 on the same noisy moons.

# %%
X_over, y_over = toy_shape('moons', n=70, noise=0.32, seed=10)
small = MLP([2, 3, 1], activation='tanh', seed=1)
big = MLP([2, 8, 1], activation='tanh', seed=1)
small.fit(X_over, y_over, lr=0.6, epochs=1200, record_every=20)
big.fit(X_over, y_over, lr=0.6, epochs=2200, record_every=20)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
decision_boundary(lambda G: small.predict_proba(G), X_over, y_over, ax=axes[0], steps=160, title='3 hidden neurons: 3 bends')
decision_boundary(lambda G: big.predict_proba(G), X_over, y_over, ax=axes[1], steps=160, title='8 hidden neurons: 8 bends')
plt.show()

# %% [markdown]
# Look at the right edge. The extra bends let it curl around single stray dots — that is the
# network fitting the noise instead of the pattern the two moons share. More bends can trace
# a real curve, and more bends can trace noise. Chapter 16 is about telling those apart.
#
# ## 🏆 Go further
#
# Work through the interactive questions, then try these quests.

# %%
from kidsml import workbook

workbook.render(15)

# %% [markdown]
# 1. **Smallest XOR solver.** What is the fewest hidden neurons that can solve XOR?
# 2. **Try spiral.** How many hidden neurons does it need before it starts curling the right way?
# 3. **Set XOR weights by hand.** Use OR-ish and AND-ish to beat training.
# 4. **Read the hidden lines.** Scrub training in the app and say what each hidden line learned.
# 5. 🧸 **Little Kid Corner:** Three friends make a team. Two notice where you stand. The
#    last friend listens and decides. Little reports become one decision.
#
# ---
# **Next up:** Chapter 16 · *Deeper and Wider* — more layers, squishes, and over-studying.
