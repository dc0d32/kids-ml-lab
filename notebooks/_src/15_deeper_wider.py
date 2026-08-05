# %% [markdown]
# # Chapter 15 · Deeper and Wider
#
# ### More capacity helps patterns and can memorise noise.
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
from kidsml.plots import decision_boundary, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# You have the whole idea now: line scores, squishes, gradients, and hidden features.
#
# Deeper networks do not add a secret ingredient. They repeat the same move more times:
# make features, squish them, make new features from those features. That buys more
# flexible boundaries, and it also creates a new danger: memorising noise.
#
# ```mermaid
# graph LR
#     X[2 inputs] --> H1[5 neurons]
#     H1 --> H2[5 neurons]
#     H2 --> Y[1 output]
#     H1 -. wider .-> H1
#     H2 -. deeper .-> Y
# ```
#
# The diagram grew by adding more hidden neurons and another hidden layer. The arrows still
# carry numbers forward and gradients backward.

# %% [markdown]
# ## ✏️ Work it out
#
# Count the learnable numbers in `[2, 5, 5, 1]`. Every arrow is a weight, and every
# non-input neuron gets one bias.
#
# This is worth counting because capacity is not a mood word. It is a pile of adjustable
# numbers the network can use to fit the data.

# %%
pd.DataFrame({'layer': ['2 → 5', '5 → 5', '5 → 1', 'biases'], 'count': [10, 25, 5, 11]})

# %% [markdown]
# That is **40 weights + 11 biases = 51 parameters**. A bigger pile can fit more shapes,
# including shapes caused by bad luck.
#
# > 📖 **Grown-ups call this:** **parameters** are the weights and biases: every adjustable
# > number inside the model.

# %% [markdown]
# Work these out on scrap paper, then type your answers in. You'll be told not only
# whether you were right, but why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(15)

# %% [markdown]
# ## 👀 Take a look
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
# Look at the edges of the coloured regions. ReLU is a flat floor glued to a straight ramp,
# so many ReLUs make folded-paper boundaries with creases. Tanh and sigmoid are smooth
# S-curves, so their boundaries tend to bend more smoothly.
#
# Neither style is always best. The squish shape controls the kind of bends the network can
# build easily.

# %% [markdown]
# ## 🎛️ Your turn
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
# On tiny toy shapes, deeper is not automatically better. More capacity means more ways to
# curve around the points, but training still has to find useful curves.
#
# The big wins for depth show up later, when data has many reusable parts: edges inside
# images, sounds inside speech, or words inside sentences.

# %% [markdown]
# ## 💻 In real code
#
# Now watch overfitting. We give the network a small practice set and flip some labels, so
# some dots are lies.
#
# A high-capacity network can spend its extra wiggles chasing those lies. Train loss keeps
# falling because the practice dots look happier, while test loss rises because fresh dots
# want the calmer rule underneath.
#
# ```mermaid
# graph LR
#     A[broad pattern] --> B[practice loss falls]
#     B --> C[tiny wiggles]
#     C --> D[train loss lower]
#     C --> E[test loss higher]
# ```

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
# Look for the dashed line: early stopping works because broad patterns are often learned
# before noisy details. Weight decay helps in a different way. Small weights make gentler
# ramps, so the boundary has a harder time making sharp little detours around one weird dot.
#
# More data helps too: a single noisy point is less powerful when surrounded by many honest
# neighbours.
#
# ## 🏆 Go further
#
# 1. **Smallest spiral net.** Reduce the architecture until spiral breaks.
# 2. **Overfit hard.** Use few points, many neurons, and no weight decay.
# 3. **Too much calm.** Raise weight decay until the boundary becomes boring.
# 4. **More data.** Use more data and watch the train/test gap shrink.
# 5. 🧸 **Little Kid Corner:** Practice helps. Memorising one worksheet does not. New
#    questions tell the truth.
#
# ---
# **Next up:** Chapter 16 · *Same Thing, in PyTorch* — we check the framework against our own gradients.
