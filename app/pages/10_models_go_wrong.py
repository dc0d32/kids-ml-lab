"""Chapter 10 · Where Models Go Wrong."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import realdata, ui
from kidsml.plots import WARM, confusion_grid, scatter_2d

ui.page_setup(10)


@st.cache_data(show_spinner=False)
def cached_threshold(threshold: float):
    return realdata.threshold_report(threshold)


@st.cache_data(show_spinner=False)
def cached_leakage():
    return realdata.leakage_scores()


@st.cache_data(show_spinner=False)
def cached_bias():
    return realdata.bias_report()


@st.cache_data(show_spinner=False)
def cached_far(span: float):
    return realdata.moons_out_of_world(span=span)


# ---------------------------------------------------------------------------
ui.beat("hook", "Four ways a score can lie.")

st.markdown(
    """
A model that is right **99%** of the time can be useless.

A model that scores brilliantly can be cheating without anybody noticing.

This chapter is the magic trick's secret. Once you know the trick, suspicious scores start to look suspicious. Good.
"""
)

ui.little_kid_corner(
    "If a smoke alarm never beeps, it is quiet almost all day. That does not make it a good smoke alarm. "
    "The important question is what happens on the one day with smoke."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "One confusion matrix, three numbers.")

st.markdown(
    """
Imagine 1000 people. The model says **sick** for 10 of them.

- 8 were really sick. Good catch.
- 2 were healthy. Scary false alarm.
- 40 sick people were missed.
- 950 healthy people were left alone.
"""
)

worked = pd.DataFrame(
    [[8, 40], [2, 950]],
    index=["really sick", "really healthy"],
    columns=["model said sick", "model said healthy"],
)
st.dataframe(worked, use_container_width=False)
metrics = realdata.metrics_from_counts(tp=8, fp=2, fn=40, tn=950)
st.markdown(
    f"Accuracy = (8 + 950) / 1000 = **{metrics['accuracy']:.1%}**. Looks great."
)
st.markdown(
    f"Of the people it flagged, 8 out of 10 really were sick: **{metrics['precision']:.0%}**."
)
st.markdown(
    f"Of the people who really were sick, it caught 8 out of 48: **{metrics['recall']:.0%}**. Ouch."
)
ui.jargon("precision", "Of the ones it flagged, how many really were?")
ui.jargon("recall", "Of the ones that really were, how many did it catch?")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Failure 1: the useless 99%.")

always = realdata.always_healthy_accuracy()
st.metric("Always say healthy", f"{always:.0%} accuracy")
st.markdown("It knows nothing. It missed every sick person. The score still looks shiny.")

threshold = st.slider("How worried must the model be before it says sick?", 0.00, 1.00, 0.50, 0.05)
report = cached_threshold(threshold)
fig, ax = ui.figure(5.5, 4.5)
confusion_grid(report["cm"], labels=["sick", "healthy"], ax=ax, title="Threshold trade-off")
ui.show(fig)

m = report["metrics"]
cols = st.columns(3)
cols[0].metric("accuracy", f"{m['accuracy']:.1%}")
cols[1].metric("precision", f"{m['precision']:.1%}")
cols[2].metric("recall", f"{m['recall']:.1%}")
st.markdown(
    "Move the slider. Catching more sick people usually means scaring more healthy people. "
    "For a smoke alarm, you may accept more false alarms. For a spam filter, eating real mail is painful. There is no universal right answer."
)

# ---------------------------------------------------------------------------
ui.beat("play", "Failures 2, 3, and 4.")

st.subheader("Failure 2 — the model cheated")
st.markdown("First we celebrate. Then we ask why the score is suspiciously perfect.")
st.dataframe(cached_leakage(), hide_index=True, use_container_width=True)
ui.jargon("leakage", "A column lets the answer sneak into the features, so the model is not learning the real pattern.")
st.markdown(
    "A hospital model once looked clever because it learned which scanner machine was used. "
    "The sickest patients used the portable scanner more often. The model learned the machine, not pneumonia."
)

st.subheader("Failure 3 — unfair copies make unfair models")
bias = cached_bias()
st.markdown("Here is a tiny hiring-style table. The old labels ask more from one group than the other.")
st.dataframe(bias["data"], hide_index=True, use_container_width=True)
st.metric("overall score against old labels", f"{bias['overall']:.1%}")
st.dataframe(bias["summary"], hide_index=True, use_container_width=True)
st.markdown("Same scores, different group:")
st.dataframe(bias["examples"], hide_index=True, use_container_width=True)
ui.careful(
    "The model is not being mean. It is copying. That is all it can do. If you copy from something unfair, "
    "you get something unfair, at scale and with a confident voice."
)

st.subheader("Failure 4 — confidently wrong, outside its world")
span = st.slider("How far outside the moon world should we look?", 3.0, 10.0, 8.0, 1.0)
far = cached_far(span)
fig, ax = ui.figure(6, 5)
img = ax.contourf(far["xx"], far["yy"], far["confidence"], levels=20, cmap="Blues", alpha=0.85)
scatter_2d(far["X"], far["y"], ax=ax, size=25)
ax.scatter([far["far"][0]], [far["far"][1]], marker="*", s=220, color=WARM, edgecolors="white", label="ten miles away")
ax.set_title("Confidence far outside the training data")
ax.legend(fontsize=9)
fig.colorbar(img, ax=ax, label="model confidence")
ui.show(fig)
st.markdown(
    f"At the star, the model says class {far['far_guess']} with **{far['far_confidence']:.0%} confidence**. "
    "It has no built-in idea of 'I have never seen anything like this.' Chapter 00 warned you: a model answers anyway."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "A blunt honesty checklist in code.")

st.code(
    """
print("baseline", baseline.score(X_test, y_test))
print("model", model.score(X_test, y_test))

# If this is perfect, look for leaked answer columns.
# If one score hides people, split the score by group.
# If the new point is far away, do not trust the confidence.
""".strip(),
    language="python",
)

st.markdown(
    """
The checklist is plain:

1. What is the baseline?
2. Does the score seem too good?
3. Does it work for everyone, or only on average?
4. What happens when it sees something new?
"""
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Build a model that scores 95% and is useless. Hint: make the answer lopsided.
2. Find the threshold where a spam filter starts eating real mail.
3. Take the Chapter 09 monsters model and find a monster it is confidently wrong about.
4. Make a suspiciously perfect model, then find the leaked column.
5. 🧸 **Little Kid Corner:** Make a pretend alarm that never rings. Count how many quiet minutes it gets right. Then ask what happens during toast smoke.
"""
)

ui.worksheet_link(10)
