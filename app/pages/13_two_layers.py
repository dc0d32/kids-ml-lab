"""Chapter 13 · Two Layers, Three Neurons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import toy_shape, xor_exact
from kidsml.nn_numpy import MLP
from kidsml.nnplots import boundary_with_hidden, hidden_surfaces_figure, mlp_snapshot_training, model_from_snapshot, network_diagram
from kidsml.plots import decision_boundary, loss_curve

ui.page_setup(13)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
XOR is back.

One neuron cannot do it. Two neurons sitting side by side still only vote with straight
lines. But three neurons wired in two layers can solve it, and you can see every piece.
"""
)
ui.show(network_diagram([2, 3, 1], activation='tanh'))

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown('Make two hidden features by hand: **OR-ish** and **AND-ish**. Then the output neuron uses those features.')
xor_table = pd.DataFrame(
    {'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1], 'OR-ish': [0, 1, 1, 1], 'AND-ish': [0, 0, 0, 1], 'XOR': [0, 1, 1, 0]}
)
st.dataframe(xor_table, hide_index=True, use_container_width=False)
st.markdown('In the new space, `score = OR - 2*AND - 0.5` is positive only for the red XOR rows.')
ui.aha('The hidden layer did not bend a line. It invented new features. That is Chapter 3 escape route 1, except the network invents the features for you.')

# ---------------------------------------------------------------------------
ui.beat('seeit')

@st.cache_data(show_spinner=False)
def trained_xor_snapshots():
    X, y = xor_exact()
    snaps = mlp_snapshot_training([2, 3, 1], X, y, lr=0.8, epochs=3000, every=150, activation='tanh', seed=2)
    return X, y, snaps

X_xor, y_xor, snaps = trained_xor_snapshots()
step_index = st.slider('Training step to inspect', 0, len(snaps) - 1, len(snaps) - 1, format='%d')
model = model_from_snapshot([2, 3, 1], snaps[step_index], activation='tanh', seed=2)
st.caption(f"showing step {snaps[step_index]['step']} · loss {snaps[step_index]['loss']:.4f}")
col_a, col_b = st.columns([1, 1], gap='large')
with col_a:
    fig, ax = ui.figure(5.2, 4.5)
    boundary_with_hidden(model, X_xor, y_xor, ax=ax, title='Hidden lines plus final boundary', steps=180)
    ui.show(fig)
with col_b:
    fig = hidden_surfaces_figure(model, X_xor, steps=65)
    ui.show(fig)
st.markdown('Each hidden neuron makes one ramp. The output neuron combines the three ramp readings into a bent boundary.')

# ---------------------------------------------------------------------------
ui.beat('play')

@st.cache_data(show_spinner=False)
def playground(shape: str, hidden: int, activation: str, lr: float, seed: int):
    X, y = toy_shape(shape, n=180, noise=0.16 if shape != 'spiral' else 0.22, seed=seed)
    m = MLP([2, hidden, 1], activation=activation, seed=seed)
    losses = m.fit(X, y, lr=lr, epochs=900, record_every=5)
    return X, y, m, np.array(losses)

cols = st.columns([0.8, 1.2, 1.2], gap='large')
with cols[0]:
    shape = st.selectbox('Dataset', ['xor', 'moons', 'circles', 'spiral'], index=0)
    hidden = st.slider('Hidden neurons', 1, 8, 3)
    activation = st.selectbox('Activation', ['tanh', 'sigmoid', 'relu'], index=0)
    lr = st.slider('Learning rate', 0.05, 1.5, 0.6, 0.05)
    seed = st.slider('Random seed', 0, 10, 3, 1)
X_play, y_play, play_model, play_losses = playground(shape, hidden, activation, lr, seed)
with cols[1]:
    fig, ax = ui.figure(5.0, 4.3)
    decision_boundary(lambda G: play_model.predict_proba(G), X_play, y_play, ax=ax, steps=180, title=play_model.describe())
    if hidden <= 8:
        from kidsml.nnplots import hidden_lines
        hidden_lines(ax, play_model, labels=False)
    ui.show(fig)
with cols[2]:
    fig, ax = ui.figure(5.0, 4.3)
    loss_curve(play_losses, ax=ax, title='Loss curve')
    ui.show(fig)
st.markdown('With one hidden neuron you are mostly back to one line. Add a few, and the model can invent several features.')

# ---------------------------------------------------------------------------
ui.beat('forreal')

@st.cache_data(show_spinner=False)
def overfit_pair():
    X, y = toy_shape('moons', n=70, noise=0.32, seed=10)
    small = MLP([2, 3, 1], activation='tanh', seed=1)
    big = MLP([2, 8, 1], activation='tanh', seed=1)
    small.fit(X, y, lr=0.6, epochs=1200, record_every=20)
    big.fit(X, y, lr=0.6, epochs=2200, record_every=20)
    return X, y, small, big

X_over, y_over, small, big = overfit_pair()
cols = st.columns(2)
for col, m, title in zip(cols, [small, big], ['3 hidden neurons: calmer', '8 hidden neurons: wobblier']):
    with col:
        fig, ax = ui.figure(5.1, 4.4)
        decision_boundary(lambda G, model=m: model.predict_proba(G), X_over, y_over, ax=ax, steps=180, title=title)
        ui.show(fig)
ui.careful('More hidden neurons give the network more ways to wiggle. That can help, and it can over-study noise. Chapter 14 is about that trade.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Smallest XOR solver.** What is the fewest hidden neurons that can solve XOR?
2. **Try spiral.** How many hidden neurons does it need before it looks decent?
3. **Set XOR weights by hand.** Use OR-ish and AND-ish to beat training.
4. **Watch the lines.** Scrub the training slider and say what each hidden line learned.
5. 🧸 **Little Kid Corner:** Three friends make a team. Two notice where you stand. The last friend listens and decides.
"""
)
ui.worksheet_link(13)
