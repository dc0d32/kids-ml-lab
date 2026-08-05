# %% [markdown]
# # Chapter 16 · Same Thing, in PyTorch
#
# ### Nothing magic — we check its gradients against ours, bolt for bolt.
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
import torch
from torch import nn

from kidsml import torch_bits as tb
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.plots import decision_boundary, loss_curve, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Everything so far was NumPy you could read top to bottom. PyTorch is the grown-up tool,
# but it is not a new kind of thinking.
#
# The promise is bigger: PyTorch calls a recipe-tracking array a tensor. When you call
# `backward()`, PyTorch walks that recipe backward and fills in gradients for every weight.
# Chapter 13, at framework speed!
#
# ```mermaid
# graph LR
#     A[NumPy arrays] --> B[torch tensors]
#     C[linear layer] --> D[nn.Linear]
#     E[manual gradients] --> F[loss.backward]
# ```
#
# The names changed, not the pieces. The proof later checks that the gradient numbers
# changed by almost nothing.

# %% [markdown]
# ## ✏️ Work it out
#
# Line up the two versions piece by piece. Our NumPy code stores arrays and calls methods
# we wrote. PyTorch stores tensors and modules that do the same jobs.
#
# A tensor is an array with a notebook attached. When `requires_grad` is on, the notebook
# records which operations made the tensor, so `backward()` can retrace them like
# footprints in wet mud.

# %%
pd.DataFrame(
    {
        'our name': ['a @ W + b', 'tanh', 'mse', 'nudge loop'],
        'PyTorch name': ['nn.Linear', 'nn.Tanh', 'nn.MSELoss', 'optimizer.step'],
    }
)

# %%
pt = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1), nn.Sigmoid()).double()
pt

# %% [markdown]
# Read the table as a translation dictionary. Nothing in PyTorch gets to skip the weighted
# sums, squishes, loss, or step.

# %% [markdown]
# Work these out on scrap paper, then type your answers in. You'll be told not only
# whether you were right, but why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(16)

# %% [markdown]
# ## 👀 Take a look
#
# > 📖 **Grown-ups call this:** a **tensor** is a NumPy-like array that can remember the
# > operations that created it, so gradients can be traced backward.
#
# `requires_grad` means “keep the recipe.” `backward()` walks the recipe backward and puts
# a gradient into each parameter's `.grad` bucket.
#
# Those buckets **accumulate**. PyTorch adds new gradients to whatever is already there
# because some advanced training loops add blame from several mini-batches before stepping.
# For our loop, old blame would be stale, so `zero_grad()` dumps the buckets before the next
# splash.

# %% [markdown]
# ## 🎛️ Your turn
#
# Now we settle the mystery question: is PyTorch doing the same backprop as our NumPy code?
#
# We copy our exact weights into a PyTorch model, run the same XOR points, and compare every
# gradient. If the largest difference is around one millionth or smaller, both systems are
# pointing the weights in the same direction.

# %%
X, y = xor_exact()
numpy_net = MLP([2, 3, 1], activation='tanh', seed=2)
torch_net = tb.mlp([2, 3, 1], activation='tanh', seed=0)
tb.copy_from_numpy(torch_net, numpy_net)

np_dWs, np_dbs, np_loss = numpy_net.gradients(X, y)
th_dWs, th_dbs, th_loss = tb.gradients(torch_net, X, y)

rows = []
for layer in range(len(np_dWs)):
    rows.append({'piece': f'W{layer}', 'largest difference': float(np.max(np.abs(np_dWs[layer] - th_dWs[layer])))})
    rows.append({'piece': f'b{layer}', 'largest difference': float(np.max(np.abs(np_dbs[layer] - th_dbs[layer])))})
proof = pd.DataFrame(rows)
proof

# %%
biggest = proof['largest difference'].max()
assert biggest < 1e-6
pd.DataFrame({'NumPy loss': [np_loss], 'PyTorch loss': [th_loss], 'biggest gradient difference': [biggest]})

# %% [markdown]
# Look down the difference column. The proof is not about hunches; matching gradients mean
# the next update step is the same step.

# %% [markdown]
# ## 💻 In real code
#
# Here is the PyTorch version training on a toy shape. The code is shorter because PyTorch
# handles the bookkeeping: storing parameters, tracing operations, and applying the step.
#
# At this toy size, NumPy is fine. PyTorch starts to matter when the model and data get
# much bigger, like the image chapters coming next.

# %%
X_train, y_train = toy_shape('moons', n=180, noise=0.18, seed=5)
torch_model = tb.mlp([2, 3, 1], activation='tanh', seed=5)
result = tb.train(torch_model, X_train, y_train, epochs=450, lr=0.25)
pd.DataFrame({'training seconds': [result['seconds']]})

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
decision_boundary(lambda G: tb.predict_proba(torch_model, G), X_train, y_train, ax=axes[0], steps=170, title='PyTorch boundary')
loss_curve(result['losses'], ax=axes[1], title='PyTorch loss curve')
plt.show()

# %% [markdown]
# Watch the loss curve and boundary together. This is the same Part 3 machine: forward
# pass, loss, backward gradients, downhill step. Same dance, bigger shoes.
#
# ## 🏆 Go further
#
# 1. **Add a layer.** Change `[2, 3, 1]` to `[2, 3, 3, 1]`.
# 2. **Forget zero_grad; sus move.** In a notebook, remove it and watch old blame pile up in the bucket.
# 3. **Try Adam.** Change `optimizer='adam'` in `torch_bits.train` and compare curves.
# 4. **Check again.** Copy weights from NumPy and make sure the gradient proof still passes.
# 5. 🧸 **Little Kid Corner:** PyTorch keeps footprints in the mud. Then it walks backward
#    to see which step made the splash.
#
# ---
# **Next up:** Chapter 17 · *Pictures Are Numbers* — every image becomes a grid a network can read.
