"""Chapter 04 · Maybe, Probably, Definitely."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from kidsml import ui
from kidsml.datasets import load_table, toy_shape
from kidsml.linear import logistic_proba, sigmoid
from kidsml.nn_numpy import log_loss
from kidsml.plots import ACCENT, COOL, WARM, decision_boundary, draw_line

ui.page_setup(4)

# ---------------------------------------------------------------------------
ui.beat("hook")
st.markdown(
    """
Chapter 3 showed why a hard line sometimes needs help: some boundaries must
bend. But even when a straight boundary is good enough, Chapter 2's perceptron
has another problem.

It says red or blue and never wavers. A point sitting on the boundary should not
sound certain. It should say: **honestly, I have no idea**.

This chapter keeps the line score, then turns it into confidence.
"""
)
ui.jargon("logistic regression", "A straight-line score followed by an S-curve that turns it into a probability.")

# ---------------------------------------------------------------------------
ui.beat("byhand")
st.markdown(
    """
Start with the same raw line score **z**. Positive should lean red. Negative
should lean blue. A score near 0 should mean a shrug.

The S-curve we use is **sigmoid(z) = 1 / (1 + e^-z)**. At **z = 0**, the
arithmetic is **1 / (1 + e⁰) = 1 / (1 + 1) = 0.5**, so the boundary becomes the
50/50 place.
"""
)
z = np.array([-4, -2, -1, 0, 1, 2, 4], dtype=float)
p = sigmoid(z)
st.dataframe({"z": z, "sigmoid(z)": np.round(p, 3)}, hide_index=True, use_container_width=True)
st.markdown(
    """
Why this S-shape and not any other? It has the habits we need: every output is
between 0 and 1, 0 turns into exactly 0.5, and opposite scores balance out. For
example, **sigmoid(2) ≈ 0.88** and **sigmoid(-2) ≈ 0.12**.

It also has a training-friendly meaning: adding 1 to the score multiplies the
red-vs-blue odds by the same amount each time. That steady rule is why this
particular S-curve shows up everywhere.
"""
)
ui.mermaid(
    """
flowchart LR
    A[Point features] --> B[Raw score z]
    B --> C[Sigmoid S-curve]
    C --> D[Probability]
    D --> E{p >= 0.5?}
    E -->|yes| F[red]
    E -->|no| G[blue]
""",
    height=260,
)
st.markdown("Notice where the threshold sits: p = 0.5 happens exactly when the raw score z is 0.")

# ---------------------------------------------------------------------------
ui.beat("seeit")
st.markdown(
    """
The slider changes how sharply scores turn into probabilities. Large values make
the S-curve look like Chapter 2's hard step. Small values make the model shrug
for almost every score.
"""
)
w = st.slider("w: how decisive is the S-curve?", 0.2, 8.0, 1.0, 0.2)
xs = np.linspace(-6, 6, 300)
fig, ax = ui.figure(6.5, 4.4)
ax.plot(xs, sigmoid(w * xs), color=ACCENT)
ax.scatter(z, sigmoid(w * z), color=WARM, edgecolors="white", zorder=3)
ax.axhline(0.5, color="#94A3B8", linestyle="--")
ax.set_xlabel("line score z")
ax.set_ylabel("probability of red")
ax.set_title("The S-curve turns any score into maybe/probably/definitely")
ui.show(fig)
st.markdown("Look at the dashed 0.5 line. Scores near zero land near that line, which is the shrug zone.")

# ---------------------------------------------------------------------------
ui.beat("play")
st.markdown(
    """
The probability is curved, but the decision boundary stays straight. Why? The
model says red when **p ≥ 0.5**, and sigmoid reaches 0.5 exactly at **z = 0**.

The set of points with **z = w1·x1 + w2·x2 + b = 0** is the same straight line as
before. The new thing is the fade of confidence around it.
"""
)
X, y = toy_shape("blobs", n=220, noise=0.28, seed=6)
col_a, col_b = st.columns([1, 2], gap="large")
with col_a:
    w1 = st.slider("w1", -6.0, 6.0, 2.0, 0.1)
    w2 = st.slider("w2", -6.0, 6.0, 2.0, 0.1)
    b = st.slider("b", -4.0, 4.0, 0.0, 0.1)
    prob = logistic_proba(X, w1, w2, b)
    st.metric("Log loss", f"{log_loss(prob, y):.3f}")
with col_b:
    fig, ax = ui.figure(6, 5)
    decision_boundary(lambda G: logistic_proba(G, w1, w2, b), X, y, ax=ax, shade_confidence=True)
    draw_line(w1, w2, b, ax=ax)
    ax.set_title("The boundary is straight. The confidence fades near it.")
    ui.show(fig)
st.markdown(
    """
Notice that the black boundary is perfectly straight, even though the shading
changes smoothly. For a concrete score, use **w1 = 2**, **w2 = -1**, **b = 0.5**,
and point **(1, 3)**:

**z = 2(1) + (-1)(3) + 0.5 = -0.5**, so **sigmoid(-0.5) ≈ 0.38**. That means
"38% red," not a hard no.
"""
)

penalty = []
for pred, truth in [(0.9, 1), (0.6, 1), (0.1, 1), (0.99, 0), (0.5, 0)]:
    penalty.append({"predicted red": pred, "true answer": truth, "penalty": round(log_loss([pred], [truth]), 2)})
st.dataframe(penalty, hide_index=True, use_container_width=True)
st.markdown(
    """
Confidence is a promise. If the truth is red, predicting **0.6** gives penalty
about **-log(0.6) = 0.51**. Predicting **0.1** gives **-log(0.1) = 2.30**.

If the truth is blue and the model says **0.99 red**, the true class only got
probability **0.01**, so the penalty is **-log(0.01) = 4.61**. Being unsure and
wrong is forgivable. Being certain and wrong is expensive.
"""
)

# ---------------------------------------------------------------------------
ui.beat("forreal")
st.markdown(
    """
Now use real penguins. The model sees flipper length and weight, then estimates
how likely each penguin is to be Gentoo.

The circled penguin is the closest to 50/50. That is not failure; it is useful
honesty about a hard call.
"""
)
penguins = load_table("penguins").dropna(subset=["species", "flipper_length_mm", "weight_g"])
penguins = penguins.copy()
penguins["is_gentoo"] = (penguins["species"] == "Gentoo").astype(int)
features = ["flipper_length_mm", "weight_g"]
model = LogisticRegression().fit(penguins[features], penguins["is_gentoo"])
probs = model.predict_proba(penguins[features])[:, 1]
penguins["gentoo_probability"] = probs
penguins["wrong"] = (probs >= 0.5).astype(int) != penguins["is_gentoo"]
uncertain = penguins.iloc[np.argmin(np.abs(probs - 0.5))]

fig, ax = ui.figure(7, 5)
colors = np.where(penguins["is_gentoo"] == 1, WARM, COOL)
ax.scatter(penguins["flipper_length_mm"], penguins["weight_g"], c=colors, s=32, alpha=0.75, edgecolors="white")
ax.scatter([uncertain["flipper_length_mm"]], [uncertain["weight_g"]], s=180, facecolors="none", edgecolors=ACCENT, linewidths=2.5)
ax.set_xlabel("flipper length (mm)")
ax.set_ylabel("weight (g)")
ax.set_title("The circled penguin is the model's biggest shrug")
ui.show(fig)
st.write(f"Most uncertain penguin: **{uncertain['species']}**, probability Gentoo = **{uncertain['gentoo_probability']:.1%}**.")

sample = penguins.iloc[[0, len(penguins) // 2, int(np.argmin(np.abs(probs - 0.5))), -1]][["species", "gentoo_probability"]]
st.bar_chart(sample.set_index("species"))
score = accuracy_score(penguins["is_gentoo"], probs >= 0.5)
st.metric("Gentoo/not-Gentoo score", f"{score:.0%}")

# ---------------------------------------------------------------------------
ui.beat("challenge")
wrong = penguins[penguins["wrong"]]
wrong_text = "Try changing the features in a notebook if there are no confident mistakes here."
if len(wrong):
    bad = wrong.iloc[np.argmax(np.abs(wrong["gentoo_probability"] - 0.5))]
    wrong_text = f"One wrong penguin: {bad['species']} at {bad['gentoo_probability']:.1%} Gentoo."
st.markdown(
    f"""
1. **Find a confident mistake.** {wrong_text}
2. **Crank w.** Make the S-curve a cliff. Does confidence mean correctness?
3. **Find the shrug zone.** Move a point near the boundary and watch the probability approach 50%.
4. 🧸 **Little Kid Corner:** Stand near one side of a room and say "probably red team." Stand on
   the middle tape line and say "I do not know." That middle shrug is the new idea.
"""
)
ui.worksheet_link(4)
