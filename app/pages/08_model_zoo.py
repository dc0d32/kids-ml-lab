"""Chapter 08 · The Model Zoo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import lesson, ui
from kidsml.trees import (
    MODEL_PERSONALITIES,
    deep_tree_train_test,
    fold_scores,
    lopsided_baseline,
    penguin_leaderboard,
    plot_folds,
    plot_zoo,
    split_bounce_scores,
)

lesson.begin(8)


@st.cache_data(show_spinner=False)
def cached_zoo(shape, n, noise, seed):
    return plot_zoo(shape=shape, n=n, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def cached_leaderboard():
    return penguin_leaderboard()


@lesson.step("Which guesser should you use?", beat="hook")
def _():
    lesson.say(
        """
Open the model zoo gate. You now know logistic regression from Chapter 4, decision trees
from Chapter 5, random forests and boosting from Chapter 6, and support vector machines
from Chapter 7. Which one should you use?

The honest answer is: **try them and see**. But *see* is harder than it sounds, because a
model can look brilliant on the rows it studied and stumble on new rows.
"""
    )

    lesson.mermaid(
        """
graph LR
    A[all labelled rows] --> B[training rows]
    A --> C[hidden test rows]
    B --> D[train model]
    D --> E[predict hidden rows]
    C --> E
    E --> F[test score]
""",
        height=260,
    )
    lesson.look_for("the wall between training rows and hidden test rows. That wall makes the race honest.")
    lesson.kid_corner(
        "If you test a bike, a scooter, and skates, use the same hill for all three. "
        "A fair race needs fair rules."
    )


@lesson.step("Five practice tests", beat="byhand")
def _():
    lesson.say(
        """
One train/test split can be lucky. Cross-validation turns that one race into several smaller
races.

Cut 10 rows into 5 folds of 2 rows. Each round hides one fold as the test set and trains on
the other four folds.
"""
    )

    lesson.mermaid(
        """
graph TD
    A[10 rows] --> B[5 folds]
    B --> C[round 1: fold 1 tests]
    B --> D[round 2: fold 2 tests]
    B --> E[more rounds rotate]
    C --> F[average and spread]
    D --> F
    E --> F
""",
        height=280,
    )
    lesson.look_for("the rotation. Every row gets a turn being hidden.")

    fold_table = pd.DataFrame(
        {
            "round": [1, 2, 3, 4, 5],
            "test rows": ["1, 2", "3, 4", "5, 6", "7, 8", "9, 10"],
            "score": [0.80, 0.70, 0.90, 0.80, 0.60],
        }
    )
    st.dataframe(fold_table, hide_index=True, width="stretch")
    st.info("Average score = (0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 3.80 / 5 = 0.76")
    lesson.jargon("fold", "One chunk of rows held out for testing during one round of cross-validation.")
    lesson.jargon("cross-validation", "Take turns hiding different chunks, then report the average and spread.")


@lesson.step("Predict the model personality", beat="seeit")
def _():
    guess = lesson.predict(
        "On moon-shaped data, which model personality do you expect to do well?",
        ["A straight line", "A smooth curved road", "A model that always says the common answer"],
        correct=1,
        why="Moons curl like two banana slices. They need a curved boundary, and the zoo grid turns that hunch into evidence!",
        key="ch08_moons_personality",
    )
    if guess is None:
        return

    lesson.say(
        """
Now change the shape and see when your hunch stops being true. No personality wins every problem.

Decode the zoo labels first: `logistic` means logistic regression, `tree` means decision
tree, `forest` means random forest, `boosting` is Chapter 6's line of little tree fixes,
and `linear SVM` / `rbf SVM` are Chapter 7 roads. One guest is new: **kNN** asks nearby
training points to vote.
"""
    )
    lesson.jargon("k-nearest neighbors", "Store the training rows, then ask nearby points to vote on a new row.")

    knobs, picture = lesson.controls()
    with knobs:
        shape = ui.shape_picker(default="moons", key="ch08_zoo_shape")
        noise = ui.noise_slider(default=0.20, key="ch08_zoo_noise")
        n = ui.sample_slider(default=180, key="ch08_zoo_n")
        seed = ui.seed_slider(default=0, key="ch08_zoo_seed")
    fig = cached_zoo(shape, n, noise, seed)
    with picture:
        lesson.show(fig)
    lesson.look_for("both parts of each mini-plot: the boundary shape and the test score in the title.")


@lesson.step("The zoo has no permanent champion", beat="seeit")
def _():
    lesson.say(
        """
The model zoo is easier to read once you know each model's habit. The short labels are
name tags, not new chapters: `logistic` is logistic regression, `tree` is a decision tree,
and `rbf SVM` is the smooth-road SVM.
"""
    )

    st.dataframe(
        pd.DataFrame({"model": list(MODEL_PERSONALITIES), "personality": list(MODEL_PERSONALITIES.values())}),
        hide_index=True,
    )
    lesson.look_for("which personality sounds like a straight line, a box, a crowd, or a smooth island.")
    lesson.aha("No model wins on every shape. That is not a cop-out. It is the real state of the field.")


@lesson.step("A fake victory", beat="play")
def _():
    lesson.say(
        """
Scoring a model on its own training data is like taking a practice test after memorising the
answer key.

It may tell you the model stored the rows. It does not tell you whether it learned a pattern
that works on new rows.
"""
    )
    st.dataframe(pd.DataFrame([deep_tree_train_test()]), hide_index=True, width="stretch")
    lesson.look_for("the gap between train score and test score. The training score is the fake trophy, and the fake trophy has no aura.")
    lesson.careful(
        "Evaluating on the training data is a fake victory. A deep tree can score 100% there "
        "by memorising tiny boxes, then miss new points that do not land in those boxes."
    )


@lesson.step("One seed can bounce the score", beat="play")
def _():
    guess = lesson.predict(
        "Change only which rows land in the hidden test set. What happens to the score?",
        ["It stays exactly the same", "It bounces", "It always improves"],
        correct=1,
        why="Different hidden rows can be easy pebbles or slippery banana peels. One split is a shaky fact.",
        key="ch08_split_bounce",
    )
    if guess is None:
        return

    lesson.say("The model type and dataset stay the same; only the split seed changes.")
    test_size = st.slider("How much data goes in the test set?", 0.15, 0.50, 0.30, 0.05, key="ch08_test_size")
    bounce = split_bounce_scores(test_size=test_size, max_seed=10).set_index("seed")
    st.line_chart(bounce)
    lesson.look_for("seeds where the same model suddenly looks better or worse.")


@lesson.step("Average and spread", beat="play")
def _():
    lesson.say(
        """
Cross-validation exists because of the bounce. Instead of trusting one split, it rotates
through several hidden chunks and reports the average **and** the spread.
"""
    )

    scores = fold_scores()
    fig, ax = lesson.figure(7, 3.6)
    plot_folds(ax=ax)
    lesson.show(fig)
    lesson.look_for("folds that sit far from the average. That spread is part of the answer.")

    score_bits = " + ".join(f"{s:.2f}" for s in scores)
    st.metric("5-fold average", f"{scores.mean():.1%}", f"spread ±{scores.std():.1%}")
    st.caption(f"Fold average arithmetic: ({score_bits}) / 5 = {scores.mean():.2f}")


@lesson.step("Check the boring answer first", beat="play")
def _():
    baseline = lopsided_baseline()
    lesson.say(
        f"A useless model that always says the most common class scores **{baseline:.0%}** on a lopsided dataset."
    )
    st.metric("boring baseline", f"{baseline:.0%}", "always predicts the common class")
    st.dataframe(
        pd.DataFrame(
            {
                "guesser": ["always common class", "fancy model"],
                "score": [f"{baseline:.0%}", "92%"],
                "what it proves": ["the data is lopsided", "only 2 points above the floor"],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    lesson.look_for("how close the fancy score is to the boring floor. The applause starts after the baseline.")
    lesson.jargon("baseline", "A boring score from a model that does not learn, such as always predicting the most common class.")
    lesson.jargon("class imbalance", "When one answer pile is much bigger than another, so accuracy can look high without useful learning.")
    lesson.careful(
        "Check the baseline first, or you may celebrate a model that learned nothing. "
        "A fancy model has to beat the boring answer before it earns applause."
    )


@lesson.step("An honest penguin leaderboard", beat="forreal")
def _():
    lesson.say(
        """
Here is the penguin race with fair rules. Each row is one Palmer penguin, the columns are
measurements such as beak, flipper, island, and weight, and the target is species.

Every model gets the same five folds. The table reports mean plus spread, and the baseline
is included too.
"""
    )

    leaderboard = cached_leaderboard()
    st.dataframe(leaderboard, hide_index=True, width="stretch")
    lesson.look_for("two means that are closer than their spreads. That is not a crushing win.")
    lesson.say("Read `0.96 ± 0.03` as a small cloud of possible scores, not one magic number.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break the leaderboard", beat="challenge")
def _():
    lesson.say(
        """
1. Find a shape where logistic regression wins or ties.
2. Find a seed where a bad model looks good.
3. Make two models swap places by changing only the split seed.
4. 🧸 **Little Kid Corner:** Race three toys down the same ramp three times. Report the average and the wiggle.
"""
    )


lesson.finish()
