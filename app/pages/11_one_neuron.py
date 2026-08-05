"""Chapter 11 · One Neuron."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from kidsml import ui
from kidsml.datasets import toy_shape, two_blobs_tiny, xor_exact
from kidsml.nn_numpy import ACTIVATION_BLURB, ACTIVATIONS, Neuron
from kidsml.nnplots import neuron_surface_figure
from kidsml.plots import decision_boundary, draw_line

ui.page_setup(11)

# ---------------------------------------------------------------------------
ui.beat('hook')
st.markdown(
    """
Part 3 sounds like a new planet: **neural networks**. It is not. A neuron is the
straight-line score from Chapter 2, followed by the probability squish from Chapter 4.

That matters because there is no missing spell. If you can read `w1*x1 + w2*x2 + b`,
you can read the inside of this circle. The circle wraps the score so a yes/no model can
say “barely yes”, “very yes”, or “I am near the fence”.
"""
)
ui.mermaid(
    """
graph LR
    X1[x₁] --> M[weights times inputs]
    X2[x₂] --> M
    M --> S((Σ + b))
    S --> A[squish]
    A --> Y[output from 0 to 1]
""",
    height=260,
)
st.markdown(
    """
Read the diagram from left to right. The only new part is the squish at the end; the
weighted sum in the middle is the line machine you already built.
"""
)
ui.aha('One neuron is `output = squish(w1*x1 + w2*x2 + b)`: Chapter 2 inside, Chapter 4 outside.')

# ---------------------------------------------------------------------------
ui.beat('byhand')
st.markdown(
    """
Use **w1 = 2**, **w2 = -1**, **b = 0.5**. First build the raw score `z`, then squish it.

The raw score decides which side of the line the point is on. The squish keeps the same
side, but turns “how far from the line?” into a number between 0 and 1.
"""
)
hand = pd.DataFrame(
    {
        'x1': [1, 0, 1],
        'x2': [0, 1, 2],
        'z working': ['2*1 + (-1)*0 + 0.5', '2*0 + (-1)*1 + 0.5', '2*1 + (-1)*2 + 0.5'],
        'z': [2.5, -0.5, 0.5],
        'sigmoid(z) approx': [0.92, 0.38, 0.62],
        'prediction': ['red', 'blue', 'red'],
    }
)
st.dataframe(hand, hide_index=True, use_container_width=True)
st.markdown(
    """
Why not leave the raw score alone? For a class answer, `z = 19` and `z = 1900` both mean
“red”, but a training rule needs a bounded target to compare with `0` and `1`. The squish
makes a soft confidence score without moving the fence.
"""
)
ui.careful(
    'If you double w1, w2, and b, every raw score doubles. The zero places stay zero, so '
    'the boundary stays put. Far-away points become more confident; the line does not move.'
)

# ---------------------------------------------------------------------------
ui.beat('seeit')
st.markdown(
    """
`Neuron.raw(X)` shows the Chapter 2 score before the squish. A positive raw score lands
on one side, a negative score lands on the other, and zero is the fence.

Look at the first few blob points below. The neuron is not hiding a secret formula; it is
computing `x1 + x2 - 7` and then sending that number through the squish.
"""
)
X_tiny, y_tiny = two_blobs_tiny()
neuron = Neuron(w=np.array([1.0, 1.0]), b=-7.0, activation='sigmoid')
raw = neuron.raw(X_tiny[:5])
st.code('Neuron(w=[1, 1], b=-7).raw(X)   # x1 + x2 - 7', language='python')
st.dataframe(pd.DataFrame({'x1': X_tiny[:5, 0], 'x2': X_tiny[:5, 1], 'raw z': raw}), hide_index=True)
st.markdown('The raw numbers are not probabilities yet. They are signed distances from the line machine.')

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

st.markdown(
    """
Watch the green line first. That is where `z = 0`, so it is the same straight fence from
Chapter 2. Now watch the colours and the 3D ramp: steeper weights make the answer change
faster as you walk away from the fence.
"""
)

X_xor, y_xor = xor_exact()
xor_misses = int((play_neuron.predict(X_xor) != y_xor).sum())
st.metric('XOR mistakes with this one neuron', xor_misses)
ui.careful(
    'XOR is still the Chapter 3 wall. Opposite corners need the same colour, and a single '
    'straight boundary always cuts the square into two neighbouring chunks. One neuron has '
    'one boundary, so it cannot win.'
)

# ---------------------------------------------------------------------------
ui.beat('forreal')
st.markdown(
    """
Now we train the neuron instead of choosing its numbers by hand. Scikit-learn calls the
same idea **logistic regression**: a line score, a sigmoid, and a training rule.

The learned numbers do not have to match exactly, because the two training recipes use
different loss details. What should match is the divider: it should point through the
same gap in the blobs.
"""
)

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
st.markdown('Look at the gap between the two blob clouds. The trained neuron found a straight divider there, which is Chapter 2 plus Chapter 4 stacked together.')

# ---------------------------------------------------------------------------
ui.beat('challenge')
st.markdown(
    """
1. **Find perfect blob weights.** Use the sliders until the blob mistakes hit zero. Which
   knob mostly rotates the line?
2. **Say yes to everything.** Make almost the whole plane red. Which bias did it take?
3. **Make a shrug machine.** Set every learned number to zero. What output do you get?
4. **Feel the XOR wall.** Try to get zero XOR mistakes with one neuron, then explain the
   Chapter 3 reason it cannot happen.
5. 🧸 **Little Kid Corner:** Draw a chalk line. Far from the line means a loud answer. On
   the line means a shrug.
"""
)
ui.worksheet_link(11)
