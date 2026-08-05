# %% [markdown]
# # Chapter 13 · How a Neuron Learns
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
# ## 🎣 Start here
#
# In Chapter 12 you grabbed the sliders yourself: try a number, watch the mistakes splat
# onto the graph, then try a better number.
#
# Now the neuron gets its own tiny steering wheel! The word **gradient** will show up a lot,
# so pin it down three ways: nudge a weight and see what loss does; read the slope of the
# loss hill; measure how much this weight matters for the mistake.
#
# ```mermaid
# graph LR
#     X[inputs and weights] --> Z[z score]
#     Z --> O[output]
#     O --> L[loss]
#     L -. blame .-> O
#     O -. blame .-> Z
#     Z -. blame .-> X
# ```
#
# The solid arrows zoom forward to make a prediction. The dotted arrows carry blame
# backward so each learned number knows which way to move.

# %% [markdown]
# ## ✏️ Work it out
#
# One point: **x = (1, 2)**, answer **1**. Start with **w1 = 0, w2 = 0, b = 0**.
#
# We will compute the first training step with every number on the table. The loss is
# squared error, so when the output is too low, `dL/dout` is negative: the arrow says push
# upward.

# %%
rows = pd.DataFrame(
    [
        ['z', '0*1 + 0*2 + 0', 0.0],
        ['output', 'sigmoid(0)', 0.5],
        ['loss', '(0.5 - 1)^2', 0.25],
        ['dL/dout', '2*(0.5 - 1)', -1.0],
        ['sigmoid slope', 'at z = 0', 0.25],
        ['dL/dz', '-1 * 0.25', -0.25],
        ['dw1', '-0.25 * x1 = -0.25 * 1', -0.25],
        ['dw2', '-0.25 * x2 = -0.25 * 2', -0.5],
        ['db', '-0.25 * 1', -0.25],
    ],
    columns=['piece', 'working', 'value'],
)
rows

# %% [markdown]
# ```mermaid
# graph LR
#     W[w1] -->|x1 = 1| Z[z]
#     Z -->|slope 0.25| O[output]
#     O -->|2(out-y) = -1| L[loss]
# ```
#
# The chain rule is this diagram read backward: `dL/dw1 = -1 * 0.25 * 1 = -0.25`.
# It is three “how much does this affect that?” numbers snapped together, one tug at a
# time.
#
# With **lr = 0.5**, subtract the gradient: `w1 = 0 - 0.5*(-0.25) = 0.125`,
# `w2 = 0 - 0.5*(-0.5) = 0.25`, and `b = 0 - 0.5*(-0.25) = 0.125`.
#
# > 💡 **Aha!** Subtracting the gradient walks downhill: if raising a weight raises loss,
# > subtract. If raising it lowers loss, the gradient is negative, and subtracting a
# > negative moves up. Weird sentence, correct move!

# %% [markdown]
# Work these out on scrap paper, then type your answers in. You'll be told not only
# whether you were right, but why the question was worth asking.

# %%
from kidsml import workbook

workbook.render(13)

# %% [markdown]
# ## 👀 Take a look
#
# > 📖 **Grown-ups call this:** a **gradient** is a number that says how the loss changes if
# > one learned number is nudged upward.
#
# First we measure the gradient the slow way: nudge one weight by a tiny amount, measure
# the loss change, and divide by the nudge size. That is the lab-bench check.
#
# Then we use the fast blame-passing formula. If the slow experiment and the fast formula
# match to many decimal places for every learned number, the formula is not a lucky story;
# it is computing the same slope from the other end of the tunnel.

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
max_difference = np.max(np.abs(proof.iloc[:, 1] - proof.iloc[:, 2]))
pd.DataFrame({'largest difference': [max_difference]})

# %% [markdown]
# The two routes found the same slopes. Tiny difference, huge payoff!

# %% [markdown]
# ## 🎛️ Your turn
#
# Try different learning rates in the app. Here is one training run, with the numbers taking a walk.

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

# %% [markdown]
# A too-large rate can explode because each bad jump lands on a new part of the hill. The
# next gradient is measured from that worse place, so the next jump can be even wilder
# instead of correcting the first miss. Boing, crash, boing.

# %%
fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(ws[:, 0], ws[:, 1], marker='o', markersize=2, color='#10B981')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('The weights walk across the loss valley')
plt.show()

# %% [markdown]
# ## 💻 In real code
#
# Here is the Chapter 12 wall with learning switched on. XOR still has the wrong shape for
# one neuron, so training can lower loss without solving the pattern.
#
# That is not a failure of gradients. The gradients are steering a model that owns one
# straight boundary, not a magic rubber fence. Chapter 14 changes the model, not the
# downhill idea.

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
# Both starts use the same rule. The final numbers differ because each start slides into a
# different best straight-line compromise.
#
# ## 🏆 Go further
#
# 1. **Find the biggest safe step, no cap.** Raise the learning rate until the loss starts bouncing off the walls.
# 2. **Break it later.** Find a rate where the first few steps improve, then the curve gets worse.
# 3. **Set lr to zero.** Explain why the map is not enough without a step.
# 4. **Explain the bumps.** The loss is measured after jumps, not drawn by a smooth pen.
# 5. 🧸 **Little Kid Corner:** If your throw is short, toss harder next time. If it sails
#    over the bucket, ease off. Too big a correction sends the beanbag flying past again.
#
# ---
# **Next up:** Chapter 14 · *Two Layers, Three Neurons* — hidden neurons invent features.
