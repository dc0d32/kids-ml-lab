# %% [markdown]
# # Chapter 12 · How a Neuron Learns
#
# ### Backprop by hand, then in 30 lines of NumPy.
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

from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP, Neuron, numeric_gradient
from kidsml.plots import decision_boundary, loss_curve, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# In Chapter 11 you moved the sliders. That was you doing the learning.
#
# Now the neuron moves its own sliders: measure how the loss changes, then step downhill.

# %% [markdown]
# ## ✏️ Do It By Hand
#
# One point: **x = (1, 2)**, answer **1**. Start with **w1 = 0, w2 = 0, b = 0**.

# %%
pd.DataFrame(
    [
        ['z', '0*1 + 0*2 + 0', 0.0],
        ['output', 'sigmoid(0)', 0.5],
        ['dL/dout', '2*(0.5 - 1)', -1.0],
        ['sigmoid slope', 'at z = 0', 0.25],
        ['dL/dz', '-1 * 0.25', -0.25],
        ['dw1', '-0.25 * 1', -0.25],
        ['dw2', '-0.25 * 2', -0.5],
        ['db', '-0.25', -0.25],
    ],
    columns=['piece', 'working', 'value'],
)

# %% [markdown]
# With **lr = 0.5**, the new weights are **w1 = 0.125, w2 = 0.25, b = 0.125**.

# %% [markdown]
# ## 👀 See It
#
# > 📖 **Grown-ups call this:** a **gradient** — what happens to loss if we nudge one
# > learned number upward.
#
# First we compute gradients the slow way: nudge, measure, divide. Then we compare with
# the fast blame-passing formula.

# %%
X_small = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 1.0]])
y_small = np.array([1.0, 0.0, 1.0])
model = MLP([2, 1], activation='sigmoid', seed=0)
model.Ws[0][:] = np.array([[0.2], [-0.1]])
model.bs[0][:] = 0.05
fast_W, fast_b, loss = model.gradients(X_small, y_small)
slow_W, slow_b = numeric_gradient(model, X_small, y_small)
proof = pd.DataFrame(
    {
        'piece': ['w1', 'w2', 'b'],
        'slow numeric gradient': [slow_W[0][0, 0], slow_W[0][1, 0], slow_b[0][0]],
        'fast backprop gradient': [fast_W[0][0, 0], fast_W[0][1, 0], fast_b[0][0]],
    }
)
proof

# %%
print('largest difference:', np.max(np.abs(proof.iloc[:, 1] - proof.iloc[:, 2])))

# %% [markdown]
# We did not ask you to trust the formula. We checked it.

# %% [markdown]
# ## 🎛️ Play With It
#
# Try different learning rates in the app. Here is one training run.

# %%
X, y = toy_shape('blobs', n=160, noise=0.25, seed=3)
rng = np.random.default_rng(2)
n = Neuron(w=rng.normal(0, 0.6, size=2), b=0.0, activation='sigmoid')
ws, losses = [], []
for _ in range(180):
    losses.append(n.step(X, y, lr=0.8))
    ws.append(n.w.copy())
ws = np.array(ws)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
decision_boundary(lambda G: n.forward(G), X, y, ax=axes[0], steps=180, title='Boundary after training')
loss_curve(losses, ax=axes[1], title='Loss while it learns')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(ws[:, 0], ws[:, 1], marker='o', markersize=2, color='#10B981')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('The weights walk across the loss valley')
plt.show()

# %% [markdown]
# ## 💻 For Real
#
# Downhill finds a bottom it can reach. On XOR, one neuron fails from different starts in
# different ways.

# %%
X_twist, y_twist = toy_shape('xor', n=160, noise=0.05, seed=5)
starts = []
for s in [1, 8]:
    rng = np.random.default_rng(s)
    n = Neuron(w=rng.normal(0, 1.0, size=2), b=0.0, activation='sigmoid')
    losses_s = n.fit(X_twist, y_twist, lr=0.7, epochs=500)
    starts.append({'start': s, 'final loss': losses_s[-1], 'mistakes': int((n.predict(X_twist) != y_twist).sum()), 'w1': n.w[0], 'w2': n.w[1], 'b': n.b})
pd.DataFrame(starts).round(3)

# %% [markdown]
# ## 🏆 Challenge
#
# 1. Find the largest learning rate that still works.
# 2. Find one where the loss goes down then blows up or wiggles.
# 3. Set lr to 0 and explain what happens.
# 4. Explain why the loss curve has bumps.
# 5. 🧸 **Little Kid Corner:** If your throw is short, toss harder next time. If it sails
#    over the fence, use a smaller correction.
#
# ---
# **Next up:** Chapter 13 · *Two Layers, Three Neurons* — hidden neurons invent features.
