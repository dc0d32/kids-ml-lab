"""Chapter 05 · Twenty Questions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import load_table
from kidsml.plots import decision_boundary
from kidsml.trees import (
    creature_feature_names,
    creature_split_table,
    fit_creature_tree,
    fit_tree_shape,
    mushroom_tree,
    plot_decision_tree,
    shallow_mushroom_scores,
    tree_depth_scores,
)

ui.page_setup(5)


@st.cache_data(show_spinner=False)
def cached_creature_splits():
    return creature_split_table()


@st.cache_data(show_spinner=False)
def cached_depth_scores(shape, n, noise, seed):
    return tree_depth_scores(shape=shape, n=n, noise=noise, seed=seed)


@st.cache_data(show_spinner=False)
def cached_mushroom_scores():
    return shallow_mushroom_scores()


# ---------------------------------------------------------------------------
ui.beat("hook", "You already know this game.")

st.markdown(
    """
In Chapter 3, a straight line failed on circles and XOR. We had two escapes:
**invent new features**, or use a **bendy model**. This chapter is the first
bendy model: it bends by asking questions.

Think of **Twenty Questions** or **Guess Who**. You do not need one giant rule
at the start. You ask one useful yes/no question, split the pile, then ask a
new question inside each smaller pile.

That is why a tree is a flowchart. The clever part is not the drawing; it is
choosing which question makes the next step easiest.
"""
)

ui.mermaid(
    """
graph TD
    A[Start with one pile] --> B{Has wings?}
    B -->|yes| C{Has feathers?}
    B -->|no| D[mostly cannot fly]
    C -->|yes| E{Lives in water?}
    C -->|no| F[can fly]
""",
    height=270,
)

st.markdown(
    """
Notice the tree asks **one question at a time**. A row never answers every
question in the diagram; it walks one path until it reaches an answer.
"""
)

ui.little_kid_corner(
    "Put toy animals in a pile. Ask one yes/no question, like *does it have wings?* "
    "Move the yes toys left and the no toys right. Keep asking until each pile has one answer."
)

ui.jargon("decision tree", "A model that asks yes/no questions until it reaches an answer.")

# ---------------------------------------------------------------------------
ui.beat("byhand", "Ten creatures. Four possible first questions.")

st.markdown(
    """
Here are ten made-up creatures. We want to guess `can_fly`, but the tree is
only allowed to start with **one** column. It tries possible first questions
such as `has_wings` and `lives_in_water`.
"""
)

creatures = load_table("creatures")
st.dataframe(creatures, hide_index=True, use_container_width=True)

st.markdown(
    """
A bucket is **mixed** when different answers are still stuck together. Six
animals with 3 flyers and 3 non-flyers is very mixed. Four flyers and 0
non-flyers is clean.

The Gini mix score says: pick two random animals from the bucket. How likely
are you to be surprised by two different answers? For two answers, the score is
`1 - p_yes² - p_no²`.

Half-and-half gives `1 - (3/6)² - (3/6)² = 1 - 9/36 - 9/36 = 0.5`. A clean
bucket gives `1 - 1² - 0² = 0`. Lower means less mess left for the next question.
"""
)

splits = cached_creature_splits()
st.dataframe(splits, hide_index=True, use_container_width=True)

st.markdown(
    """
The weighted mix column counts both buckets. For `has_wings`, the yes bucket
has 6 animals and mix `1 - (4/6)² - (2/6)² = 0.444`, while the no bucket has 4
animals and mix `0`. So the split score is `(6×0.444 + 4×0) / 10 = 0.267`.
"""
)

winner = splits.iloc[0]["first question"]
st.success(f"The least-mixed first question is **{winner}**. Your pencil found it.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Now sklearn picks the same question.")

st.markdown(
    """
The computer is not guessing from vibes. It tries the same kind of split table,
picks the least-mixed question, and repeats that inside the new buckets.

Each split uses one column because a tree question has one job: send the row
left or right. Later questions can use different columns, but only after the
row has reached that branch.
"""
)

model, X_creatures, y_creatures = fit_creature_tree(max_depth=3)
fig, ax = plt.subplots(figsize=(11, 5))
plot_decision_tree(model, creature_feature_names(), ["cannot fly", "can fly"], ax=ax)
ui.show(fig)

st.markdown(
    """
Read the picture from top to bottom. At every box, the tree asks one yes/no
question and sends the row down exactly one branch.
"""
)

st.code("""from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)
model.fit(creature_questions, can_fly)""", language="python")

first = creature_feature_names()[int(model.tree_.feature[0])]
st.info(f"The computer's first split is **{first}** too.")

# ---------------------------------------------------------------------------
ui.beat("play", "A tree bends by making stairs.")

st.markdown(
    """
Chapter 3 asked for a bendy boundary. A tree can bend, but not like a smooth
rubber band. On a two-column picture, a question such as `x1 <= 0.4` makes a
vertical cut. A question such as `x2 <= -0.2` makes a horizontal cut.

That is why the boundary becomes a staircase. More depth means more questions,
and more questions mean more little rectangles.
"""
)

shape = ui.shape_picker(default="moons", key="ch05_shape", include=("moons", "circles", "xor", "spiral"))
noise = ui.noise_slider(default=0.20, key="ch05_noise")
n = ui.sample_slider(default=220, key="ch05_n")
seed = ui.seed_slider(default=1, key="ch05_seed")
depth = st.slider("How many questions deep?", 1, 20, 3, key="ch05_depth")

model, X_train, X_test, y_train, y_test = fit_tree_shape(shape, depth, n=n, noise=noise, seed=seed)
fig, ax = ui.figure(6.5, 5.2)
decision_boundary(model.predict, X_train, y_train, ax=ax, steps=180, shade_confidence=False, title=f"max_depth = {depth}")
ui.show(fig)

st.markdown(
    """
Notice the boundary is made only of horizontal and vertical cuts. The tree
cannot draw a diagonal line; it can only stack enough stair steps to fake one.
"""
)

col1, col2 = st.columns(2)
col1.metric("training accuracy", f"{model.score(X_train, y_train):.0%}")
col2.metric("test accuracy", f"{model.score(X_test, y_test):.0%}")

scores = cached_depth_scores(shape, n, noise, seed).set_index("max_depth")
st.line_chart(scores)

ui.aha(
    "A tree is **blocky-bendy**. Each question makes one straight cut, but a chain of "
    "small cuts can wrap around moons, circles, or XOR without inventing new features."
)
ui.careful(
    "A deep tree can score better on its own training dots by making tiny boxes around "
    "awkward points. That is memorising: it learns *this exact dot goes red* instead of "
    "learning a rule that helps on the next dot. Chapter 8 turns this worry into a fair test."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Mushrooms are a real table of words.")

st.markdown(
    """
Real tables often contain words. A column that says `smell = almond` cannot go
straight into a tree as a sentence, so we turn it into yes/no columns like
`smell_almond`, `smell_fishy`, and `smell_none`.

This is called one-hot encoding. It sounds fancy, but it gives the tree the
same kind of question it already knows how to ask: is this column 0 or 1?
"""
)

mush_depth = st.slider("Mushroom tree depth", 1, 8, 4, key="mush_depth")
_, _, _, _, _, scores, text = mushroom_tree(max_depth=mush_depth)
cols = st.columns(3)
cols[0].metric("train", f"{scores['train']:.1%}")
cols[1].metric("test", f"{scores['test']:.1%}")
cols[2].metric("top question", scores["top_question"].replace("_", " "))

st.dataframe(cached_mushroom_scores(), hide_index=True, use_container_width=True)
st.text(text[:2200])

st.markdown(
    """
Look at the top question before you read the whole printed tree. Smell sits
near the top, and real mushroom guides talk about smell too. That agreement is
a good sign: the model found a clue a human forager would recognise.
"""
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find the shallowest mushroom tree that stays above **95%** on the test set.
2. Find the tree depth where training and test accuracy are farthest apart.
3. Turn one mushroom rule into a sentence you could tell a person.
4. 🧸 **Little Kid Corner:** Play Guess Who with animals. What is the best first question?
"""
)

ui.worksheet_link(5)
