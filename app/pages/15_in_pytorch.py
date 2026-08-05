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
Everything so far was NumPy you could read top to bottom. PyTorch is the grown-up tool,
but it is not a new kind of thinking.

The promise is bigger: tensors remember how they were made. If PyTorch remembers the
forward recipe, it can walk that recipe backward and fill in gradients for every weight.
Chapter 12, at framework speed.
"""
)
ui.mermaid(
    """
graph LR
    A[NumPy arrays] --> B[torch tensors]
    C[linear layer] --> D[nn.Linear]
    E[manual gradients] --> F[loss.backward]
""",
    height=230,
)
st.markdown('The names changed, not the pieces. The proof later checks that the gradient numbers changed by almost nothing.')

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown(
    """
Line up the two versions piece by piece. Our NumPy code stores arrays and calls methods we
wrote. PyTorch stores tensors and modules that do the same jobs.

A tensor is an array with a notebook attached. When `requires_grad` is on, the notebook
records which operations made the tensor, so `backward()` can retrace them.
"""
)
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
st.markdown('Read the table as a translation dictionary. Nothing in PyTorch gets to skip the weighted sums, squishes, loss, or step.')

# ---------------------------------------------------------------------------
ui.beat('seeit')
ui.jargon('tensor', 'A NumPy-like array that can remember the operations that created it, so gradients can be traced backward.')
st.markdown(
    """
`requires_grad` means “keep the recipe.” `backward()` walks the recipe backward and puts a
gradient into each parameter's `.grad` bucket.

Those buckets **accumulate**. PyTorch adds new gradients to whatever is already there
because some advanced training loops add blame from several mini-batches before stepping.
For our loop, old blame would be stale, so `zero_grad()` clears the buckets first.
"""
)

# ---------------------------------------------------------------------------
ui.beat('play')
st.markdown(
    """
Now we settle the mystery question: is PyTorch doing the same backprop as our NumPy code?

We copy our exact weights into a PyTorch model, run the same XOR points, and compare every
gradient. If the largest difference is around one millionth or smaller, both systems are
pointing the weights in the same direction.
"""
)

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
st.markdown('Look down the difference column. The proof is not about vibes; matching gradients mean the next update step is the same step.')

# ---------------------------------------------------------------------------
ui.beat('forreal')
st.markdown(
    """
Here is the PyTorch version training on a toy shape. The code is shorter because PyTorch
handles the bookkeeping: storing parameters, tracing operations, and applying the step.

At this toy size, NumPy is fine. PyTorch starts to matter when the model and data get much
bigger, like the image chapters coming next.
"""
)

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
st.markdown('Watch the loss curve and boundary together. This is the same Part 3 machine: forward pass, loss, backward gradients, downhill step.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Add a layer.** Change `[2, 3, 1]` to `[2, 3, 3, 1]`.
2. **Forget zero_grad.** In a notebook, remove it and watch old blame pile up.
3. **Try Adam.** Change `optimizer='adam'` in `torch_bits.train` and compare curves.
4. **Check again.** Copy weights from NumPy and make sure the gradient proof still passes.
5. 🧸 **Little Kid Corner:** PyTorch keeps footprints. Then it walks backward to see who
   stepped in the mud.
"""
)
ui.worksheet_link(15)
