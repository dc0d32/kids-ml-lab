# %% [markdown]
# # Chapter 14 · Deeper and Wider
#
# ### More layers, different squishes, and over-studying.
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
from kidsml.nn_numpy import MLP, mse
from kidsml.nnplots import network_diagram
from kidsml.plots import decision_boundary, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 The Hook
#
# You have the whole idea now. More layers are more of the same.

# %%
network_diagram([2, 5, 5, 1], activation='tanh')
plt.show()

# %% [markdown]
# ## ✏️ Do It By Hand
#
# Count the parameters in `[2, 5, 5, 1]`.

# %%
pd.DataFrame({'layer': ['2 → 5', '5 → 5', '5 → 1', 'biases'], 'count': [10, 25, 5, 11]})

# %% [markdown]
# 40 weights plus 11 biases means **51 parameters**.
#
# > 📖 **Grown-ups call this:** **parameters** — every adjustable number inside the model.

# %% [markdown]
# ## 👀 See It
#
# Activation zoo: same data, same architecture, different squish.

# %%
X_zoo, y_zoo = toy_shape('moons', n=180, noise=0.18, seed=4)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
for ax, act in zip(axes, ['sigmoid', 'tanh', 'relu']):
    m = MLP([2, 5, 1], activation=act, seed=2)
    m.fit(X_zoo, y_zoo, lr=0.5 if act != 'relu' else 0.08, epochs=1200, record_every=10)
    decision_boundary(lambda G, model=m: model.predict_proba(G), X_zoo, y_zoo, ax=ax, steps=150, title=act)
plt.show()

# %% [markdown]
# ReLU gives folded-paper pieces. Tanh and sigmoid give smoother bends. ReLU won a lot of
# real work because it is fast and its gradient stays alive more often.

# %% [markdown]
# ## 🎛️ Play With It
#
# Compare width and depth on spiral.

# %%
X_cmp, y_cmp = toy_shape('spiral', n=170, noise=0.18, seed=6)
archs = [[2, 3, 1], [2, 10, 1], [2, 5, 5, 1], [2, 5, 5, 5, 1]]
rows = []
models = []
for sizes in archs:
    m = MLP(sizes, activation='tanh', seed=3)
    losses = m.fit(X_cmp, y_cmp, lr=0.55, epochs=1600, record_every=20)
    rows.append({'network': ' → '.join(map(str, sizes)), 'parameters': m.n_parameters(), 'final loss': losses[-1]})
    models.append(m)
pd.DataFrame(rows).round(3)

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.7))
for ax, m in zip(axes, models):
    decision_boundary(lambda G, model=m: model.predict_proba(G), X_cmp, y_cmp, ax=ax, steps=130, title=m.describe())
plt.show()

# %% [markdown]
# On tiny toys, deeper is not automatically better. The real depth wins arrive when data
# has many little parts, like images.

# %% [markdown]
# ## 💻 For Real
#
# Overfitting: train loss keeps falling, while test loss turns around.

# %%
X, y_clean = toy_shape('spiral', n=70, noise=0.28, seed=1)
rng = np.random.default_rng(4)
y = y_clean.copy()
flips = rng.choice(len(y), size=14, replace=False)
y[flips] = 1 - y[flips]
X_test, y_test = toy_shape('spiral', n=240, noise=0.28, seed=99)

m = MLP([2, 16, 16, 1], activation='tanh', seed=5)
train_losses, test_losses = [], []
for e in range(2000):
    m.step(X, y, lr=0.5)
    if e % 25 == 0:
        train_losses.append(mse(m.forward(X), y.reshape(-1, 1)))
        test_losses.append(mse(m.forward(X_test), y_test.reshape(-1, 1)))
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
best = int(np.argmin(test_losses))

fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.plot(train_losses, label='train loss', color='#10B981')
ax.plot(test_losses, label='test loss', color='#EF4444')
ax.axvline(best, color='#94A3B8', linestyle='--')
ax.set_xlabel('checkpoint')
ax.set_ylabel('loss')
ax.set_title('Stop where test loss is best')
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.3))
decision_boundary(lambda G: m.predict_proba(G), X, y, ax=ax, steps=160, title='Boundary after over-studying')
plt.show()

# %% [markdown]
# The fixes are not magic words: stop early, keep weights small with weight decay, or get
# more data.
#
# ## 🏆 Challenge
#
# 1. Find the smallest network that solves spiral.
# 2. Make a network overfit as hard as possible.
# 3. Find a weight decay value that is clearly too strong.
# 4. Use more data and watch the train/test gap shrink.
# 5. 🧸 **Little Kid Corner:** Practice helps. Memorising one worksheet does not. New
#    questions tell the truth.
#
# ---
# **Next up:** Chapter 15 · *Same Thing, in PyTorch* — we check the framework against our own gradients.
