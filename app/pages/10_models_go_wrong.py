"""Chapter 10 · Where Models Go Wrong."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import lesson, realdata
from kidsml.plots import WARM, confusion_grid, scatter_2d

lesson.begin(10)


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


@lesson.step("Four ways a score can lie", beat="hook")
def _():
    lesson.say(
        """
A model that is right **99%** of the time can still be a trophy made of fog.

A model that scores brilliantly can be cheating without anybody noticing. Once you know the
trick, suspicious scores start flashing like hazard lights.
"""
    )
    lesson.mermaid(
        """
graph TD
    A[Shiny score] --> B{What feels wrong?}
    B -->|rare answer| C[Useless accuracy]
    B -->|too perfect| D[Leakage]
    B -->|hurts one group| E[Copied unfair history]
    B -->|far away| F[Outside its world]
""",
        height=330,
    )
    lesson.look_for("the four exits from a shiny score. A score is the first clue, not the finish line.")
    lesson.kid_corner(
        "If a smoke alarm never beeps, it is quiet almost all day. That does not make it a good smoke alarm. "
        "The important question is what happens on the one day with smoke."
    )


@lesson.step("One matrix, three numbers", beat="byhand")
def _():
    lesson.say(
        """
Imagine 1000 people. The model says **sick** for 10 of them: 8 were really sick, 2 were
healthy, 40 sick people were missed, and 950 healthy people were left alone.
"""
    )
    worked = pd.DataFrame(
        [[8, 40], [2, 950]],
        index=["really sick", "really healthy"],
        columns=["model said sick", "model said healthy"],
    )
    st.dataframe(worked, width="content")
    lesson.say(
        """
This 2-by-2 box is a **confusion matrix**. It counts the four things that can happen:
caught sick people, missed sick people, false alarms, and correctly ignored healthy people.

Precision asks, "When the alarm rang, how often was it right?" Recall asks, "Of all the
sick people, how many did the alarm catch?"
"""
    )
    lesson.look_for("the 40 missed sick people. Accuracy can hide them inside the big healthy pile.")

    metrics = realdata.metrics_from_counts(tp=8, fp=2, fn=40, tn=950)
    st.markdown(f"Accuracy = (8 + 950) / 1000 = **{metrics['accuracy']:.1%}**. Looks great.")
    st.markdown(f"Precision = 8 / 10 = **{metrics['precision']:.0%}**.")
    st.markdown(f"Recall = 8 / 48 = **{metrics['recall']:.0%}**. Ouch.")
    lesson.jargon("precision", "Of the ones it flagged, how many really were?")
    lesson.jargon("recall", "Of the ones that really were, how many did it catch?")


@lesson.step("Predict the useless 99%", beat="seeit")
def _():
    guess = lesson.predict(
        "A disease appears in 1 out of 100 people. A model always says healthy. What is its accuracy?",
        ["0%", "50%", "99%"],
        correct=2,
        why="It catches the 99 healthy people and lets the one sick person walk past the alarm.",
        key="ch10_useless_99",
    )
    if guess is None:
        return

    always = realdata.always_healthy_accuracy()
    st.metric("Always say healthy", f"{always:.0%} accuracy")
    lesson.aha("This model knows nothing, misses every sick person, and still gets a shiny score. That is the trap!")


@lesson.step("Move the threshold", beat="seeit")
def _():
    lesson.say(
        """
The model has a hidden worry score for each person. The threshold is the line that says,
above here, call it sick.
"""
    )
    threshold = st.slider("How worried must the model be before it says sick?", 0.00, 1.00, 0.50, 0.05, key="ch10_threshold")
    report = cached_threshold(threshold)
    fig, ax = lesson.figure(5.5, 4.5)
    confusion_grid(report["cm"], labels=["sick", "healthy"], ax=ax, title="Threshold trade-off")
    lesson.show(fig)
    lesson.look_for("which mistake grows when the threshold moves: false alarms or missed sick people.")

    m = report["metrics"]
    cols = st.columns(3)
    cols[0].metric("accuracy", f"{m['accuracy']:.1%}")
    cols[1].metric("precision", f"{m['precision']:.1%}")
    cols[2].metric("recall", f"{m['recall']:.1%}")
    lesson.say("For a smoke alarm, false alarms are annoying but missed smoke is worse. For a spam filter, eating real mail is painful. Same slider, different stakes.")


@lesson.step("Failure 2: the model cheated", beat="play")
def _():
    guess = lesson.predict(
        "A messy real-data model scores 100%. What should you suspect first?",
        ["It became a genius", "The answer leaked into the features", "The baseline disappeared"],
        correct=1,
        why="Tiny toy worlds can be perfect. Messy real data that hits 100% makes the leaked-answer alarm clang first.",
        key="ch10_leakage_predict",
    )
    if guess is None:
        return

    st.dataframe(cached_leakage(), hide_index=True, width="stretch")
    lesson.look_for("the suspicious jump when leaked columns are allowed. That jump is evidence, and I believe the formal term is sus.")
    lesson.jargon("leakage", "A column lets the answer sneak into the features, so the model is not learning the real pattern.")
    lesson.say("A hospital model once learned which scanner machine was used. The machine was a shortcut; pneumonia was the real target.")


@lesson.step("Failure 3: unfair copies", beat="play")
def _():
    lesson.say(
        """
Here is a tiny hiring-style table. The old labels ask more from one group than the other.
The model does not know history is unfair. It only sees examples to copy.
"""
    )
    bias = cached_bias()
    st.dataframe(bias["data"], hide_index=True, width="stretch")
    st.metric("overall score against old labels", f"{bias['overall']:.1%}")
    lesson.look_for("the old labels. The model is copying them, not judging whether they were fair.")


@lesson.step("Averages can hide who gets hurt", beat="play")
def _():
    bias = cached_bias()
    st.dataframe(bias["summary"], hide_index=True, width="stretch")
    st.dataframe(bias["examples"], hide_index=True, width="stretch")
    lesson.look_for("people with the same useful scores but different groups. That is the crack where the average hides harm.")
    lesson.careful(
        "The model is not being mean. It is copying. If you copy from something unfair, "
        "you get something unfair, at scale and with a confident voice."
    )


@lesson.step("Failure 4: outside its world", beat="play")
def _():
    lesson.say("A model has no built-in new-planet alarm. Chapter 00 warned you: a model answers anyway.")
    span = st.slider("How far outside the moon world should we look?", 3.0, 10.0, 8.0, 1.0, key="ch10_far_span")
    far = cached_far(span)
    fig, ax = lesson.figure(6, 5)
    img = ax.contourf(far["xx"], far["yy"], far["confidence"], levels=20, cmap="Blues", alpha=0.85)
    scatter_2d(far["X"], far["y"], ax=ax, size=25)
    ax.scatter([far["far"][0]], [far["far"][1]], marker="*", s=220, color=WARM, edgecolors="white", label="ten miles away")
    ax.set_title("Confidence far outside the training data")
    ax.legend(fontsize=9)
    fig.colorbar(img, ax=ax, label="model confidence")
    lesson.show(fig)
    lesson.look_for("the star out in the dark. It is far from the training moons, but the model still sounds confident.")
    st.markdown(f"At the star, the model says class {far['far_guess']} with **{far['far_confidence']:.0%} confidence**.")


@lesson.step("The honesty checklist in code", beat="forreal")
def _():
    st.code(
        """
baseline_score = baseline.score(X_test, y_test)
model_score = model.score(X_test, y_test)

# If this is perfect, look for leaked answer columns.
# If one score hides people, split the score by group.
# If the new point is far away, do not trust the confidence.
""".strip(),
        language="python",
    )
    lesson.say(
        """
The checklist fits on a sticky note: What is the baseline? Does the score seem too good?
Does it work for everyone, or only on average? What happens when it sees something new?
"""
    )


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go find suspicious scores", beat="challenge")
def _():
    lesson.say(
        """
1. Build a model that scores 95% and is useless. Hint: make the answer lopsided.
2. Find the threshold where a spam filter starts eating real mail.
3. Take the Chapter 09 monsters model and find a monster it is confidently wrong about.
4. Make a suspiciously perfect model, then find the leaked column.
5. 🧸 **Little Kid Corner:** Make a pretend alarm that never rings. Count how many quiet minutes it gets right. Then ask what happens during toast smoke.
"""
    )


lesson.finish()
