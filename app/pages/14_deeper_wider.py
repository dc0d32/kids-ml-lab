"""Chapter 14 · Deeper and Wider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import toy_shape
from kidsml.nn_numpy import MLP, mse
from kidsml.plots import decision_boundary, loss_curve

ui.page_setup(14)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
You have the whole idea now: line scores, squishes, gradients, and hidden features.

Deeper networks do not add a secret ingredient. They repeat the same move more times:
make features, squish them, make new features from those features. That buys more
flexible boundaries, and it also creates a new danger: memorising noise.
"""
)
ui.mermaid(
    """
graph LR
    X[2 inputs] --> H1[5 neurons]
    H1 --> H2[5 neurons]
    H2 --> Y[1 output]
    H1 -. wider .-> H1
    H2 -. deeper .-> Y
""",
    height=260,
)
st.markdown('The diagram grew by adding more hidden neurons and another hidden layer. The arrows still carry numbers forward and gradients backward.')

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown(
    """
Count the learnable numbers in `[2, 5, 5, 1]`. Every arrow is a weight, and every non-input
neuron gets one bias.

This is worth counting because capacity is not a mood word. It is a pile of adjustable
numbers the network can use to fit the data.
"""
)
counts = pd.DataFrame(
    {'layer': ['2 → 5', '5 → 5', '5 → 1', 'biases'], 'count': [10, 25, 5, 11]}
)
st.dataframe(counts, hide_index=True, use_container_width=False)
st.markdown('That is **40 weights + 11 biases = 51 parameters**. A bigger pile can fit more shapes, including shapes caused by bad luck.')
ui.jargon('parameters', 'The weights and biases: every adjustable number inside the model.')

# ---------------------------------------------------------------------------
ui.beat('seeit')

@st.cache_data(show_spinner=False)
def activation_zoo():
    X, y = toy_shape('moons', n=180, noise=0.18, seed=4)
    models = []
    for act in ['sigmoid', 'tanh', 'relu']:
        m = MLP([2, 5, 1], activation=act, seed=2)
        m.fit(X, y, lr=0.5 if act != 'relu' else 0.08, epochs=1200, record_every=10)
        models.append((act, m))
    return X, y, models

X_zoo, y_zoo, models = activation_zoo()
cols = st.columns(3)
for col, (act, m) in zip(cols, models):
    with col:
        fig, ax = ui.figure(4.5, 4.0)
        decision_boundary(lambda G, model=m: model.predict_proba(G), X_zoo, y_zoo, ax=ax, steps=160, title=act)
        ui.show(fig)
st.markdown(
    """
Look at the edges of the coloured regions. ReLU is a flat floor glued to a straight ramp,
so many ReLUs make folded-paper boundaries with creases. Tanh and sigmoid are smooth
S-curves, so their boundaries tend to bend more smoothly.

Neither style is always best. The squish shape controls the kind of bends the network can
build easily.
"""
)

# ---------------------------------------------------------------------------
ui.beat('play')

@st.cache_data(show_spinner=False)
def compare_shapes():
    X, y = toy_shape('spiral', n=170, noise=0.18, seed=6)
    archs = [[2, 3, 1], [2, 10, 1], [2, 5, 5, 1], [2, 5, 5, 5, 1]]
    rows = []
    trained = []
    for sizes in archs:
        m = MLP(sizes, activation='tanh', seed=3)
        losses = m.fit(X, y, lr=0.55, epochs=1600, record_every=20)
        rows.append({'network': ' → '.join(map(str, sizes)), 'parameters': m.n_parameters(), 'final loss': losses[-1]})
        trained.append(m)
    return X, y, pd.DataFrame(rows), trained

X_cmp, y_cmp, table, trained = compare_shapes()
st.dataframe(table.round(3), hide_index=True, use_container_width=True)
cols = st.columns(4)
for col, m in zip(cols, trained):
    with col:
        fig, ax = ui.figure(3.8, 3.5)
        decision_boundary(lambda G, model=m: model.predict_proba(G), X_cmp, y_cmp, ax=ax, steps=140, title=m.describe())
        ui.show(fig)
st.markdown(
    """
On tiny toy shapes, deeper is not automatically better. More capacity means more ways to
curve around the points, but training still has to find useful curves.

The big wins for depth show up later, when data has many reusable parts: edges inside
images, sounds inside speech, or words inside sentences.
"""
)

# ---------------------------------------------------------------------------
ui.beat('forreal')
st.markdown(
    """
Now watch overfitting. We give the network a small practice set and flip some labels, so
some dots are lies.

A high-capacity network can spend its extra wiggles chasing those lies. Train loss keeps
falling because the practice dots look happier, while test loss rises because fresh dots
want the calmer rule underneath.
"""
)
ui.mermaid(
    """
graph LR
    A[broad pattern] --> B[practice loss falls]
    B --> C[tiny wiggles]
    C --> D[train loss lower]
    C --> E[test loss higher]
""",
    height=240,
)

@st.cache_data(show_spinner=False)
def overfit_story(weight_decay: float = 0.0, more_data: bool = False):
    n_train = 240 if more_data else 70
    X, y_clean = toy_shape('spiral', n=n_train, noise=0.28, seed=1)
    rng = np.random.default_rng(4)
    y = y_clean.copy()
    flips = rng.choice(len(y), size=max(1, len(y) // 5), replace=False)
    y[flips] = 1 - y[flips]
    X_test, y_test = toy_shape('spiral', n=240, noise=0.28, seed=99)
    m = MLP([2, 16, 16, 1], activation='tanh', seed=5)
    train_losses, test_losses = [], []
    for e in range(2000):
        m.step(X, y, lr=0.5, weight_decay=weight_decay)
        if e % 25 == 0:
            train_losses.append(mse(m.forward(X), y.reshape(-1, 1)))
            test_losses.append(mse(m.forward(X_test), y_test.reshape(-1, 1)))
    return X, y, X_test, y_test, m, np.array(train_losses), np.array(test_losses)

decay = st.slider('Weight decay', 0.0, 0.08, 0.0, 0.01)
more = st.checkbox('Use more practice questions', value=False)
X_train, y_train, X_test, y_test, over_model, train_losses, test_losses = overfit_story(decay, more)
best = int(np.argmin(test_losses))
cols = st.columns(2)
with cols[0]:
    fig, ax = ui.figure(5.2, 4.2)
    ax.plot(train_losses, label='train loss', color='#10B981')
    ax.plot(test_losses, label='test loss', color='#EF4444')
    ax.axvline(best, color='#94A3B8', linestyle='--')
    ax.text(best + 1, test_losses[best], 'best test moment', fontsize=9)
    ax.set_xlabel('checkpoint')
    ax.set_ylabel('loss')
    ax.set_title('Over-studying: train down, test back up')
    ax.legend()
    ui.show(fig)
with cols[1]:
    fig, ax = ui.figure(5.2, 4.2)
    decision_boundary(lambda G: over_model.predict_proba(G), X_train, y_train, ax=ax, steps=160, title='Boundary after training')
    ui.show(fig)
st.markdown(
    """
Look for the dashed line: early stopping works because broad patterns are often learned
before noisy details. Weight decay helps in a different way. Small weights make gentler
ramps, so the boundary has a harder time making sharp little detours around one weird dot.

More data helps too: a single noisy point is less powerful when surrounded by many honest
neighbours.
"""
)

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Smallest spiral net.** Reduce the architecture until spiral breaks.
2. **Overfit hard.** Use few points, many neurons, and no weight decay.
3. **Too much calm.** Raise weight decay until the boundary becomes boring.
4. **More data.** Turn on more practice questions and watch the loss gap shrink.
5. 🧸 **Little Kid Corner:** Practice helps. Memorising one worksheet does not. New
   questions tell the truth.
"""
)
ui.worksheet_link(14)
