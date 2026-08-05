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
In Chapter 11 you moved the sliders. That was learning by hand: try a number, look at the
mistakes, try a better number.

Now the neuron moves its own sliders. The word **gradient** will show up a lot, so pin it
down three ways: nudge a weight and see what loss does; read the slope of the loss hill;
measure how much this weight matters for the mistake.
"""
)
ui.mermaid(
    """
graph LR
    X[inputs and weights] --> Z[z score]
    Z --> O[output]
    O --> L[loss]
    L -. blame .-> O
    O -. blame .-> Z
    Z -. blame .-> X
""",
    height=270,
)
st.markdown('The solid arrows make a prediction. The dotted arrows carry blame backward so each learned number knows which way to move.')

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown(
    """
One point: **x = (1, 2)**, answer **1**. Start with **w1 = 0, w2 = 0, b = 0**.

We will compute the first training step with every number showing. The loss is squared
error, so when the output is too low, `dL/dout` is negative.
"""
)
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
st.dataframe(rows, hide_index=True, use_container_width=True)
ui.mermaid(
    """
graph LR
    W[w1] -->|x1 = 1| Z[z]
    Z -->|slope 0.25| O[output]
    O -->|2(out-y) = -1| L[loss]
""",
    height=220,
)
st.markdown(
    """
The chain rule is this diagram read backward: `dL/dw1 = -1 * 0.25 * 1 = -0.25`.
It is three “how much does this affect that?” numbers multiplied together.

With **lr = 0.5**, subtract the gradient: `w1 = 0 - 0.5*(-0.25) = 0.125`,
`w2 = 0 - 0.5*(-0.5) = 0.25`, and `b = 0 - 0.5*(-0.25) = 0.125`.
"""
)
ui.aha('Subtracting the gradient goes downhill: if raising a weight raises loss, subtract. If raising it lowers loss, the gradient is negative, and subtracting a negative moves up.')

# ---------------------------------------------------------------------------
ui.beat('seeit')
ui.jargon('gradient', 'A number that says how the loss changes if one learned number is nudged upward.')
st.markdown(
    """
First we measure the gradient the slow way: nudge one weight by a tiny amount, measure the
loss change, and divide by the nudge size. That is an independent check.

Then we use the fast blame-passing formula. If the slow experiment and the fast formula
match to many decimal places for every learned number, the formula is not a lucky story;
it is computing the same slope.
"""
)
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
st.success(f'The largest difference is {np.max(np.abs(proof.iloc[:, 1] - proof.iloc[:, 2])):.2e}. The two routes found the same slopes.')

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
st.markdown(
    """
Try a tiny learning rate, a middle one, and a huge one. The learning rate multiplies every
downhill step.

A too-large rate can explode because each bad jump lands on a new part of the hill. The
next gradient is measured from that worse place, so the next jump can be even wilder
instead of correcting the first miss.
"""
)

fig, ax = ui.figure(5.8, 4.2)
ax.plot(ws[:, 0], ws[:, 1], color='#10B981', marker='o', markersize=2)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('The weights walk across the loss valley')
ui.show(fig)

ui.careful('Downhill finds a bottom it can reach from its starting place. On awkward problems, a different start can land in a different bottom.')

# ---------------------------------------------------------------------------
ui.beat('forreal')
st.markdown(
    """
Here is the limit from Chapter 11, now with learning turned on. XOR still has the wrong
shape for one neuron, so training can lower loss without solving the pattern.

That is not a failure of gradients. The gradients are steering a model that owns one
straight boundary. Chapter 13 changes the model, not the downhill idea.
"""
)
X_twist, y_twist = toy_shape('xor', n=160, noise=0.05, seed=5)
starts = []
for s in [1, 8]:
    rng = np.random.default_rng(s)
    n = Neuron(w=rng.normal(0, 1.0, size=2), b=0.0, activation='sigmoid')
    losses_s = n.fit(X_twist, y_twist, lr=0.7, epochs=500)
    starts.append({'start': s, 'final loss': losses_s[-1], 'mistakes': int((n.predict(X_twist) != y_twist).sum()), 'w1': n.w[0], 'w2': n.w[1], 'b': n.b})
st.dataframe(pd.DataFrame(starts).round(3), hide_index=True)
st.markdown('Both starts use the same rule. The final numbers differ because each start finds a different best straight-line compromise.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Find the biggest safe step.** Raise the learning rate until the loss stops behaving.
2. **Break it later.** Find a rate where the first few steps improve, then the curve gets worse.
3. **Set lr to zero.** Explain why the map is not enough without a step.
4. **Explain the bumps.** The loss is measured after jumps, not drawn by a smooth pen.
5. 🧸 **Little Kid Corner:** If your throw is short, toss harder next time. If it sails
   over the fence, use a smaller correction.
"""
)
ui.worksheet_link(12)
