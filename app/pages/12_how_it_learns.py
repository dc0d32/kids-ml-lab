"""Chapter 12 · How a Neuron Learns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP, Neuron, numeric_gradient
from kidsml.plots import decision_boundary, loss_curve

ui.page_setup(12)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
In Chapter 11 you moved the sliders. That was you doing the learning.

Now the neuron moves its own sliders. The trick is small: nudge a number, check whether
the loss got better, and step downhill.
"""
)

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown('One point: **x = (1, 2)**, answer **1**. Start with **w1 = 0, w2 = 0, b = 0**.')
rows = pd.DataFrame(
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
st.dataframe(rows, hide_index=True, use_container_width=False)
st.markdown('With **lr = 0.5**, the new weights are **w1 = 0.125, w2 = 0.25, b = 0.125**.')

# ---------------------------------------------------------------------------
ui.beat('seeit')
ui.jargon('gradient', 'A measurement of what happens to the loss if we nudge one learned number upward.')
st.markdown('First we measure it the slow way: nudge, measure, divide. Then we use blame-passing because it gives the same numbers much faster.')
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
st.dataframe(proof.round(12), hide_index=True)
st.success(f'The largest difference is {np.max(np.abs(proof.iloc[:, 1] - proof.iloc[:, 2])):.2e}. We checked the formula instead of asking you to trust it.')

# ---------------------------------------------------------------------------
ui.beat('play')

@st.cache_data(show_spinner=False)
def train_path(lr: float, seed: int, steps: int = 180):
    X, y = toy_shape('blobs', n=160, noise=0.25, seed=3)
    rng = np.random.default_rng(seed)
    n = Neuron(w=rng.normal(0, 0.6, size=2), b=0.0, activation='sigmoid')
    ws, bs, losses = [], [], []
    for _ in range(steps):
        losses.append(n.step(X, y, lr=lr))
        ws.append(n.w.copy())
        bs.append(n.b)
    return X, y, np.array(ws), np.array(bs), np.array(losses)

col_a, col_b, col_c = st.columns([0.7, 1.15, 1.15], gap='large')
with col_a:
    lr = st.slider('Learning rate', 0.0, 8.0, 0.8, 0.1)
    seed = st.slider('Random start', 0, 10, 2, 1)
X, y, ws, bs, losses = train_path(lr, seed)
current = Neuron(w=ws[-1], b=float(bs[-1]), activation='sigmoid')
with col_b:
    fig, ax = ui.figure(5.1, 4.3)
    decision_boundary(lambda G: current.forward(G), X, y, ax=ax, steps=180, title='Boundary after training')
    ui.show(fig)
with col_c:
    fig, ax = ui.figure(5.1, 4.3)
    loss_curve(losses, ax=ax, title='Loss while it learns')
    ui.show(fig)
st.markdown('Try a tiny learning rate, a middle one, and a huge one. You are choosing the size of each downhill step.')

fig, ax = ui.figure(5.8, 4.2)
ax.plot(ws[:, 0], ws[:, 1], color='#10B981', marker='o', markersize=2)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('The weights walk across the loss valley')
ui.show(fig)

ui.careful('Downhill finds a bottom it can reach from its starting place. On awkward problems, a different start can land in a different bottom.')

# ---------------------------------------------------------------------------
ui.beat('forreal')
X_twist, y_twist = toy_shape('xor', n=160, noise=0.05, seed=5)
starts = []
for s in [1, 8]:
    rng = np.random.default_rng(s)
    n = Neuron(w=rng.normal(0, 1.0, size=2), b=0.0, activation='sigmoid')
    losses_s = n.fit(X_twist, y_twist, lr=0.7, epochs=500)
    starts.append({'start': s, 'final loss': losses_s[-1], 'mistakes': int((n.predict(X_twist) != y_twist).sum()), 'w1': n.w[0], 'w2': n.w[1], 'b': n.b})
st.dataframe(pd.DataFrame(starts).round(3), hide_index=True)
st.markdown('Both starts use the same rule. XOR is the wrong shape for one neuron, so the two bottoms can look different and still fail.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Find the biggest safe step.** Raise the learning rate until the loss stops behaving.
2. **Break it later.** Find a rate where the first few steps improve, then the curve gets worse.
3. **Set lr to zero.** Explain why the map is not enough without a step.
4. **Explain the bumps.** The loss is measured after jumps, not drawn by a smooth pen.
5. 🧸 **Little Kid Corner:** If your throw is short, toss harder next time. If it sails over the fence, use a smaller correction.
"""
)
ui.worksheet_link(12)
