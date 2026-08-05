# %% [markdown]
# # Chapter 15 · Same Thing, in PyTorch
#
# ### Nothing magic — we check its gradients against ours.
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
# ## 🎣 The Hook
#
# Everything so far was NumPy you could read top to bottom. Real practitioners often use
# PyTorch. Before trusting it, we check that it agrees with us.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# The two networks line up piece by piece.

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
# ## 👀 See It
#
# > 📖 **Grown-ups call this:** a **tensor** — a NumPy-like array that remembers what made
# > it, so PyTorch can retrace the steps backward.
#
# `backward()` walks backward. `zero_grad()` clears old blame before the next step.

# %% [markdown]
# ## 🎛️ Play With It
#
# The proof: copy our NumPy weights into PyTorch and compare gradients.

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
print('NumPy loss:', np_loss)
print('PyTorch loss:', th_loss)
print('biggest gradient difference:', biggest)
assert biggest < 1e-6

# %% [markdown]
# Autograd is the nudge-and-blame idea done fast. There is no third thing hiding offstage.

# %% [markdown]
# ## 💻 For Real
#
# Train the PyTorch version on a toy shape.

# %%
X_train, y_train = toy_shape('moons', n=180, noise=0.18, seed=5)
torch_model = tb.mlp([2, 3, 1], activation='tanh', seed=5)
result = tb.train(torch_model, X_train, y_train, epochs=450, lr=0.25)
print(f"training seconds: {result['seconds']:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
decision_boundary(lambda G: tb.predict_proba(torch_model, G), X_train, y_train, ax=axes[0], steps=170, title='PyTorch boundary')
loss_curve(result['losses'], ax=axes[1], title='PyTorch loss curve')
plt.show()

# %% [markdown]
# At this toy size, NumPy is fine. PyTorch's advantage shows up when the data and models
# get much bigger.
#
# ## 🏆 Challenge
#
# 1. Change `[2, 3, 1]` to `[2, 3, 3, 1]`.
# 2. Break it by forgetting `zero_grad` and watch old blame pile up.
# 3. Swap SGD for Adam and compare loss curves.
# 4. Copy weights from NumPy again and make sure the proof still passes.
# 5. 🧸 **Little Kid Corner:** PyTorch keeps footprints. Then it walks backward to see who
#    stepped in the mud.
#
# ---
# **Next up:** Chapter 16 · *Pictures Are Just Numbers* — every image becomes a grid a network can read.
