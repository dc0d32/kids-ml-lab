# %% [markdown]
# # Chapter 14 · How a Neuron Learns
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
# In Chapter 13 you grabbed the sliders yourself: try a number, watch the mistakes splat
# onto the graph, then try a better number.
#
# Now the neuron gets its own tiny steering wheel. First it needs one mistake score to make
# smaller. Grown-ups call that score **loss**.
#
# Then it nudges each learned number upward a hair and asks, "Did loss rise or fall?" The
# answer is a **gradient**: how much that number matters for the loss, and which way points
# downhill. Passing those clues backward is called **backprop**.
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
# The solid arrows zoom forward to make a prediction. Grown-ups call that left-to-right
# trip the **forward pass**. The dotted arrows carry blame backward so each learned number
# knows which way to move.
#
# > 📖 **Grown-ups call this:** the **loss** is the single number for how bad the model's
# > answer was, and the **gradient** is which way to nudge each learned number to make the
# > loss smaller.

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
        ['x', 'the input point', '(1, 2)', 'x1 is 1 and x2 is 2'],
        ['y', 'the correct answer', '1', 'we want the output to rise toward 1'],
        ['z', 'raw score before squish', '0*1 + 0*2 + 0 = 0', 'w1*x1 + w2*x2 + b'],
        ['out', 'prediction after squish', 'sigmoid(0) = 0.5', 'zero becomes the unsure answer'],
        ['loss', 'mistake score', '(0.5 - 1)^2 = 0.25', 'squared error: prediction minus answer, squared'],
        ['dL/dout', 'loss tug on output', '2*(0.5 - 1) = -1', 'negative means raising output lowers loss'],
        ['sigmoid slope', 'output tug on z', '0.5*(1 - 0.5) = 0.25', 'the squish is this steep at z = 0'],
        ['dL/dz', 'loss tug on z', '-1*0.25 = -0.25', 'chain rule: multiply the two tugs'],
        ['dw1', 'loss tug on w1', '-0.25*x1 = -0.25*1 = -0.25', 'w1 matters through x1'],
        ['dw2', 'loss tug on w2', '-0.25*x2 = -0.25*2 = -0.5', 'x2 is bigger, so w2 gets a bigger tug'],
        ['db', 'loss tug on b', '-0.25*1 = -0.25', 'bias adds straight into z'],
    ],
    columns=['symbol', 'means', 'working', 'value clue'],
)
rows

# %% [markdown]
# ```mermaid
# graph LR
#     W[w1] -->|x1 = 1| Z[z]
#     Z -->|slope 0.25| O[output]
#     O -->|"2(out-y) = -1"| L[loss]
# ```
#
# When one number changes another number, and that one changes a third, multiply the little
# effects to get the whole tug. Grown-ups call that the **chain rule**.
#
# Read the diagram backward: `dL/dw1 = -1 * 0.25 * 1 = -0.25`. `dL/dw1` means “if w1 rises
# a tiny bit, what happens to loss?” It is three “how much does this affect that?” numbers
# snapped together, one tug at a time.
#
# With **lr = 0.5** — the learning rate, or step size — subtract `lr * gradient`:
# `w1 = 0 - 0.5*(-0.25) = 0 - (-0.125) = 0.125`,
# `w2 = 0 - 0.5*(-0.5) = 0 - (-0.25) = 0.25`, and
# `b = 0 - 0.5*(-0.25) = 0 - (-0.125) = 0.125`.
#
# > 💡 **Aha!** Subtracting the gradient walks downhill: if raising a weight raises loss,
# > subtract. If raising it lowers loss, the gradient is negative, and subtracting a
# > negative moves up. Weird sentence, correct move!

# %% [markdown]
# ## 👀 Take a look
#
# First we measure the gradient the slow way: nudge one weight by a tiny amount, measure
# the loss change, and divide by the nudge size. That is the lab-bench check.
#
# Then we use the fast blame-passing formula. If the slow experiment and the fast formula
# match to many decimal places for every learned number, the formula is not a lucky story;
# it is computing the same slope from the other end of the tunnel.
#
# Use three tiny points with both answers in the table, then compare the slow nudge test
# with the fast backward-blame calculation. `MLP` means multilayer perceptron: a stack of
# neuron layers; here `[2, 1]` is one tiny layer written in that format.

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
# Try different learning rates in the app. Here is one training run: the boundary it lands
# on, and the loss dropping as it learns.

# %%
X, y = toy_shape('blobs', n=160, noise=0.25, seed=3)
rng = np.random.default_rng(2)
n = Neuron(w=rng.normal(0, 0.6, size=2), b=0.0, activation='sigmoid')
losses = n.fit(X, y, lr=0.8, epochs=180)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
decision_boundary(lambda G: n.forward(G), X, y, ax=axes[0], steps=180, title='Boundary after training')
loss_curve(losses, ax=axes[1], title='Loss while it learns')
plt.show()

# %% [markdown]
# Now watch the weights themselves walk downhill. Two clean, far-apart blobs have no bottom
# to reach — the best weights just keep growing — so that walk would be a dull straight
# line. To *see* the walk we switch to two blobs that overlap, which gives the loss a real
# lowest point. We also hold the bias still, so the whole story is two weights and the loss
# is a valley we can draw. Darker is lower loss; the pink star is the bottom.
#
# Watch the same three learning rates: too small crawls, just right settles into the
# bottom, too big flies past it.

# %%
from kidsml.neuronlearn import draw_weight_walk

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
for ax, rate in zip(axes, [0.4, 2.5, 8.0]):
    draw_weight_walk(ax, lr=rate, seed=2)
    ax.set_title(f'lr = {rate}')
plt.show()

# %% [markdown]
# The small step (left) leaves the dots bunched near the start; it is still crawling. The
# middle step lands right on the star at the bottom. The big step (right) flings the walk
# past the star and up the far wall: each jump lands on a new part of the hill, so instead
# of correcting the miss it steps clean over the bottom it was aiming for.

# %% [markdown]
# ## 💻 In real code
#
# Here is the Chapter 13 wall with learning switched on. XOR still has the wrong shape for
# one neuron, so training can lower loss without solving the pattern.
#
# That is not a failure of gradients. The gradients are steering a model that owns one
# straight boundary, not a magic rubber fence. Chapter 15 changes the model, not the
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
# Both starts use the same rule. An **epoch** is one full pass through the practice data;
# each run below used 500 epochs. The final numbers differ because each start slides into a
# different best straight-line compromise.
#
# ## 🏆 Go further
#
# Work through the interactive questions, then try these quests.

# %%
from kidsml import workbook

workbook.render(14)

# %% [markdown]
# 1. **Find the biggest safe step, no cap.** Raise the learning rate until the loss starts bouncing off the walls.
# 2. **Break it later.** Find a rate where the first few steps improve, then the curve gets worse.
# 3. **Set lr to zero.** Explain why the map is not enough without a step.
# 4. **Explain the bumps.** The loss is measured after jumps, not drawn by a smooth pen.
# 5. 🧸 **Little Kid Corner:** If your throw is short, toss harder next time. If it sails
#    over the bucket, ease off. Too big a correction sends the beanbag flying past again.
#
# ---
# **Next up:** Chapter 15 · *Two Layers, Three Neurons* — hidden neurons invent features.
