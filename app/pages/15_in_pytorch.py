"""Chapter 15 · Same Thing, in PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import torch_bits as tb
from kidsml import ui
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.plots import decision_boundary, loss_curve

ui.page_setup(15)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
Everything so far was NumPy you could read top to bottom.

Real practitioners often use PyTorch. Before trusting it, we check that it gives the same
gradients as our own network.
"""
)

# ---------------------------------------------------------------------------
ui.beat('byhand')
col_a, col_b = st.columns(2)
with col_a:
    st.code(
        """# our NumPy network
net = MLP([2, 3, 1], activation='tanh')
out = net.forward(X)
loss = mse(out, y)
net.step(X, y, lr=0.2)""",
        language='python',
    )
with col_b:
    st.code(
        """# the PyTorch version
net = nn.Sequential(
    nn.Linear(2, 3), nn.Tanh(),
    nn.Linear(3, 1), nn.Sigmoid(),
)
loss = nn.MSELoss()(net(X), y)
loss.backward()
optimizer.step()""",
        language='python',
    )
st.dataframe(
    pd.DataFrame(
        {
            'our name': ['a @ W + b', 'tanh', 'mse', 'nudge loop'],
            'PyTorch name': ['nn.Linear', 'nn.Tanh', 'nn.MSELoss', 'optimizer.step'],
        }
    ),
    hide_index=True,
)

# ---------------------------------------------------------------------------
ui.beat('seeit')
ui.jargon('tensor', 'A NumPy-like array that remembers what made it, so PyTorch can retrace the steps backward.')
st.markdown('`requires_grad` means “keep the footprints.” `backward()` walks the footprints backward. `zero_grad()` clears old blame before the next step.')

# ---------------------------------------------------------------------------
ui.beat('play')

@st.cache_data(show_spinner=False)
def gradient_proof():
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
    biggest = max(r['largest difference'] for r in rows)
    assert biggest < 1e-6
    return pd.DataFrame(rows), float(np_loss), float(th_loss), biggest

proof, np_loss, th_loss, biggest = gradient_proof()
st.dataframe(proof, hide_index=True)
st.success(f'NumPy loss {np_loss:.12f} · PyTorch loss {th_loss:.12f} · biggest gradient difference {biggest:.2e}')
st.markdown('Autograd is the nudge-and-blame idea done fast. There is no third thing hiding offstage.')

# ---------------------------------------------------------------------------
ui.beat('forreal')

@st.cache_resource(show_spinner=False)
def train_torch(shape: str, hidden: int, lr: float, seed: int):
    X, y = toy_shape(shape, n=180, noise=0.18, seed=seed)
    model = tb.mlp([2, hidden, 1], activation='tanh', seed=seed)
    result = tb.train(model, X, y, epochs=450, lr=lr)
    return X, y, model, result['losses'], result['seconds']

shape = st.selectbox('Dataset', ['moons', 'xor', 'circles'], index=0)
hidden = st.slider('Hidden neurons', 2, 8, 3)
lr = st.slider('PyTorch learning rate', 0.02, 1.0, 0.25, 0.02)
X, y, torch_model, losses, seconds = train_torch(shape, hidden, lr, 5)
cols = st.columns(2)
with cols[0]:
    fig, ax = ui.figure(5.2, 4.3)
    decision_boundary(lambda G: tb.predict_proba(torch_model, G), X, y, ax=ax, steps=170, title='PyTorch boundary')
    ui.show(fig)
with cols[1]:
    fig, ax = ui.figure(5.2, 4.3)
    loss_curve(losses, ax=ax, title=f'PyTorch trained in {seconds:.2f}s')
    ui.show(fig)
st.markdown('At this toy size, NumPy is fine. PyTorch starts to matter when the model and data get much bigger, like image chapters.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Add a layer.** Change `[2, 3, 1]` to `[2, 3, 3, 1]`.
2. **Forget zero_grad.** In a notebook, remove it and watch old blame pile up.
3. **Try Adam.** Change `optimizer='adam'` in `torch_bits.train` and compare curves.
4. **Check again.** Copy weights from NumPy and make sure the gradient proof still passes.
5. 🧸 **Little Kid Corner:** PyTorch keeps footprints. Then it walks backward to see who stepped in the mud.
"""
)
ui.worksheet_link(15)
