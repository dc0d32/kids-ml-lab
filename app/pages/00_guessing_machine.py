"""Chapter 00 · The Guessing Machine.

The whole of machine learning in one game: I have a secret rule, here are some
examples, work out the rule. You play it, then the computer plays it, and you compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from kidsml import ui
from kidsml.zeeps import (
    RULES,
    all_zeeps,
    encode,
    label_with,
    learning_curve,
    pretty,
)

ui.page_setup(0)

# ---------------------------------------------------------------------------
ui.beat("hook", "No maths yet. Just a game.")

st.markdown(
    """
I am thinking of a **secret rule**.

The rule decides whether a creature is a **zeep** or **not a zeep**. Every creature has
three things about it: a **shape**, a **colour**, and a **size**. That's all you get.

I won't tell you the rule. Instead I'll show you some creatures I've already sorted.
Your job: work out the rule from the examples.

That sentence — *work out the rule from the examples* — **is machine learning**.
There is nothing else. Everything in the other 24 chapters is a different way of doing
exactly this.
"""
)

ui.little_kid_corner(
    "Think of the game where someone says *hot* or *cold* while you look for a hidden "
    "toy. They never tell you where it is. You work it out from their hints. Here the "
    "hints are the examples."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Scrap paper out. Six examples, then some questions.")

col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    rule_name = st.selectbox(
        "Which secret rule shall I use?",
        list(RULES),
        index=1,
        format_func=lambda k: f"Secret rule #{list(RULES).index(k) + 1}",
        help="You can peek later. Try not to peek yet.",
    )
    n_examples = st.slider("How many examples do you want to see?", 3, 12, 6)
    shuffle = st.slider("Shuffle which examples you get", 0, 20, 0)

zeeps = all_zeeps()
labels = label_with(zeeps, rule_name)

rng = np.random.default_rng(shuffle)
order = rng.permutation(len(zeeps))
shown, hidden = order[:n_examples], order[n_examples : n_examples + 3]

with col_a:
    st.markdown("**The examples I've already sorted:**")
    example_table = pretty(zeeps.iloc[shown]).copy()
    example_table["zeep?"] = np.where(labels[shown], "✅ zeep", "❌ not a zeep")
    st.dataframe(example_table, hide_index=True, use_container_width=True)

with col_b:
    st.markdown("**Now you try. Are these three zeeps?**")
    quiz = pretty(zeeps.iloc[hidden]).copy()
    st.dataframe(quiz, hide_index=True, use_container_width=True)

    your_guesses = []
    for i, (_, row) in enumerate(quiz.iterrows()):
        guess = st.radio(
            f"A {row['size']} {row['colour']} {row['shape']}:",
            ["zeep", "not a zeep"],
            key=f"guess_{i}",
            horizontal=True,
        )
        your_guesses.append(guess == "zeep")

    reveal = st.button("Check my answers  ▶", type="primary")

truth = labels[hidden]

if reveal:
    your_score = int((np.array(your_guesses) == truth).sum())
    st.markdown(f"### You got **{your_score} out of 3**.")
    st.markdown(f"The secret rule was: **{RULES[rule_name]}**")

# ---------------------------------------------------------------------------
ui.beat("seeit", "Now the computer plays the same game.")

st.markdown(
    """
The computer gets **exactly what you got** — those same examples and nothing else.
It has never heard of shapes or colours or zeeps. All it sees is a table of numbers.

Here is literally what it sees:
"""
)

X_all = encode(zeeps)
peek = pd.DataFrame(X_all[shown], columns=["shape", "colour", "size"])
peek["answer"] = labels[shown].astype(int)

see_a, see_b = st.columns([1, 1], gap="large")
with see_a:
    st.markdown("**What you saw**")
    st.dataframe(example_table, hide_index=True, use_container_width=True)
with see_b:
    st.markdown("**What the computer saw**")
    st.dataframe(peek, hide_index=True, use_container_width=True)

ui.jargon(
    "features and labels",
    "The columns it uses to guess (shape, colour, size) are the **features**. "
    "The column it's trying to guess (zeep or not) is the **label**.",
)

model = DecisionTreeClassifier(random_state=0).fit(X_all[shown], labels[shown])
machine_guesses = model.predict(X_all[hidden]).astype(bool)

result = quiz.copy()
result["the computer says"] = np.where(machine_guesses, "zeep", "not a zeep")
if reveal:
    result["you said"] = ["zeep" if g else "not a zeep" for g in your_guesses]
    result["the truth"] = np.where(truth, "zeep", "not a zeep")

st.dataframe(result, hide_index=True, use_container_width=True)
machine_score = int((machine_guesses == truth).sum())

if reveal:
    your_score = int((np.array(your_guesses) == truth).sum())
    if your_score > machine_score:
        st.success(f"🏆 **You win, {your_score} to {machine_score}.**")
    elif your_score < machine_score:
        st.warning(f"🤖 **The computer wins, {machine_score} to {your_score}.**")
    else:
        st.info(f"🤝 **A tie, {your_score} each.**")

# ---------------------------------------------------------------------------
ui.beat("play", "The one thing that matters most.")

st.markdown(
    "Drag this and watch what happens. It is the single most important picture in the "
    "whole course."
)

n_train = st.slider("How many examples does the computer get to see?", 1, 17, 6, key="curve_n")

curve = learning_curve(rule_name, n_repeats=60)
chart = pd.DataFrame(
    {"examples seen": curve["n"], "how often it's right": curve["accuracy"]}
).set_index("examples seen")
st.line_chart(chart, height=280)

acc_now = float(curve["accuracy"][curve["n"] == n_train][0])
st.metric(f"With {n_train} example(s), the computer is right", f"{acc_now:.0%} of the time")

ui.aha(
    "Nobody made the computer smarter. Nobody changed its program. "
    "**The only thing that changed was how many examples it got to look at.**\n\n"
    "That is why people who work on AI spend most of their time thinking about data, "
    "not about code."
)

ui.careful(
    "Look at the left end of the graph. With one or two examples the computer is barely "
    "better than flipping a coin — but it still answers with total confidence. "
    "A model never says *I don't know*. It just guesses. Remember that in Chapter 10."
)

# ---------------------------------------------------------------------------
ui.beat("forreal", "The entire program is four lines.")

st.code(
    """
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()      # 1. pick a kind of guesser
model.fit(examples, answers)          # 2. show it the examples
model.predict(new_creatures)          # 3. ask it about new ones
""",
    language="python",
)

st.markdown(
    "That's it. That's a machine learning program. Every chapter from here changes "
    "**line 1** — a different kind of guesser — or changes what goes into **line 2**."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. **Beat the machine.** Set the examples slider to **3**. Can you still get all three
   right? Can the computer? Why is it so much harder for it than for you?

2. **Find the rule it can't learn.** Try secret rule #5 (*exactly one of: red, or big*).
   Watch the graph. Why does this one need so many more examples than "it is red"?

3. **Break it on purpose.** Set the shuffle slider until the six examples you get are
   *all* zeeps. Now what can the computer possibly learn? What would you learn?

4. 🧸 **Little Kid Corner:** Play this with a real person. Think of a secret rule about
   the things in your room — *anything blue*, or *anything you can eat*. Point at five
   things and say yes or no for each. See how many they need before they get it.
"""
)

ui.worksheet_link(0)
