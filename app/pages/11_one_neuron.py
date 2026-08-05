"""Chapter 11 · One Neuron."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from kidsml import ui
from kidsml.datasets import toy_shape, two_blobs_tiny, xor_exact
from kidsml.nn_numpy import ACTIVATION_BLURB, ACTIVATIONS, Neuron
from kidsml.nnplots import neuron_diagram, neuron_surface_figure
from kidsml.plots import decision_boundary, draw_line

ui.page_setup(11)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
People draw neural networks as scary webs of circles and arrows.

Here is one circle. You already know what it does. You built the inside in Chapter 2,
and you met the squish in Chapter 4.
"""
)
ui.show(neuron_diagram(weights=(2, -1), bias=0.5, activation='sigmoid'))
ui.aha('`output = squish(w1*x1 + w2*x2 + b)`. The brackets are Chapter 2. The squish is Chapter 4. That is one neuron.')

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown('Use **w1 = 2**, **w2 = -1**, **b = 0.5**. First compute `z`. Then look up a friendly sigmoid value.')
hand = pd.DataFrame(
    {
        'x1': [1, 0, 1],
        'x2': [0, 1, 2],
        'z': [2.5, -0.5, 0.5],
        'sigmoid(z) approx': [0.92, 0.38, 0.62],
        'prediction': ['red', 'blue', 'red'],
    }
)
st.dataframe(hand, hide_index=True, use_container_width=False)
ui.careful('Double w1, w2 and b. The zero line stays in the same place. The confidence gets steeper, but the boundary does not move.')

# ---------------------------------------------------------------------------
ui.beat('seeit')
st.markdown('`Neuron.raw(X)` is the old Chapter 2 score. The neuron does not hide it.')
X_tiny, y_tiny = two_blobs_tiny()
neuron = Neuron(w=np.array([1.0, 1.0]), b=-7.0, activation='sigmoid')
raw = neuron.raw(X_tiny[:5])
st.code('Neuron(w=[1, 1], b=-7).raw(X)   # x1 + x2 - 7', language='python')
st.dataframe(pd.DataFrame({'x1': X_tiny[:5, 0], 'x2': X_tiny[:5, 1], 'raw z': raw}), hide_index=True)

# ---------------------------------------------------------------------------
ui.beat('play')
col_controls, col_boundary, col_surface = st.columns([0.8, 1.2, 1.2], gap='large')
with col_controls:
    w1 = st.slider('w1', -6.0, 6.0, 2.0, 0.2)
    w2 = st.slider('w2', -6.0, 6.0, -1.0, 0.2)
    b = st.slider('b', -4.0, 4.0, 0.5, 0.1)
    activation = st.selectbox('Squish', list(ACTIVATIONS), index=list(ACTIVATIONS).index('sigmoid'))
    st.caption(ACTIVATION_BLURB[activation])
X, y = toy_shape('blobs', n=180, noise=0.22, seed=4)
play_neuron = Neuron(w=np.array([w1, w2]), b=b, activation=activation)
with col_boundary:
    fig, ax = ui.figure(5.0, 4.4)
    decision_boundary(lambda G: play_neuron.forward(G), X, y, ax=ax, steps=180, shade_confidence=True, title='Boundary and confidence')
    draw_line(w1, w2, b, ax=ax)
    ui.show(fig)
with col_surface:
    st.plotly_chart(neuron_surface_figure(play_neuron, X, steps=45, title='Rotate the ramp'), use_container_width=True)

st.markdown('The green boundary is where the ramp crosses the middle. Rotate the surface and the flat line becomes a cliff edge.')

X_xor, y_xor = xor_exact()
xor_misses = int((play_neuron.predict(X_xor) != y_xor).sum())
st.metric('XOR mistakes with this one neuron', xor_misses)
ui.careful('XOR is still waiting. A single smooth ramp cannot make opposite corners match.')

# ---------------------------------------------------------------------------
ui.beat('forreal')

@st.cache_data(show_spinner=False)
def fit_blob_models():
    X_fit, y_fit = toy_shape('blobs', n=180, noise=0.18, seed=8)
    mine = Neuron(w=np.zeros(2), b=0.0, activation='sigmoid')
    losses = mine.fit(X_fit, y_fit, lr=0.8, epochs=900)
    sk = LogisticRegression(C=1_000_000, solver='lbfgs').fit(X_fit, y_fit)
    return X_fit, y_fit, mine.w, mine.b, np.array(losses), sk.coef_[0], float(sk.intercept_[0])

X_fit, y_fit, w_mine, b_mine, losses, w_sklearn, b_sklearn = fit_blob_models()
learned = Neuron(w=w_mine, b=b_mine, activation='sigmoid')
compare = pd.DataFrame(
    {
        'model': ['our Neuron.fit', 'sklearn LogisticRegression'],
        'w1': [w_mine[0], w_sklearn[0]],
        'w2': [w_mine[1], w_sklearn[1]],
        'b': [b_mine, b_sklearn],
        'accuracy': [(learned.predict(X_fit) == y_fit).mean(), LogisticRegression(C=1_000_000, solver='lbfgs').fit(X_fit, y_fit).score(X_fit, y_fit)],
    }
)
st.dataframe(compare.round(3), hide_index=True, use_container_width=True)
fig, ax = ui.figure(5.8, 4.6)
decision_boundary(lambda G: learned.forward(G), X_fit, y_fit, ax=ax, steps=180, title='Our trained neuron')
ui.show(fig)
st.markdown('The scale can differ because the losses differ, but the learned divider points the same way. You rediscovered logistic regression by stacking two chapters.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Find perfect blob weights.** Use the sliders until the blob mistakes hit zero.
2. **Say yes to everything.** Make almost the whole plane red. Which bias did it take?
3. **Make a shrug machine.** Set every learned number to zero. What output do you get?
4. **Feel the XOR wall.** Try to get zero XOR mistakes with one neuron.
5. 🧸 **Little Kid Corner:** Draw a chalk line. Far from the line means a loud answer. On the line means a shrug.
"""
)
ui.worksheet_link(11)
