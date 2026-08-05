"""Chapter 00 · The Guessing Machine.

The whole of machine learning in one game: I have a secret rule, here are some
examples, work out the rule. You play it, then the computer plays it, and you compare.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from kidsml import lesson
from kidsml.zeeps import (
    RULES,
    all_zeeps,
    encode,
    label_with,
    learning_curve,
    pretty,
    quiz_examples,
    teaching_examples,
)

lesson.begin(0)

ZEEPS = all_zeeps()
X_ALL = encode(ZEEPS)


def deal(rule: str, n_examples: int, shuffle: int):
    """Pick which creatures they get to see, and which they get quizzed on.

    Dealt on purpose rather than at random. Some rules are true of only three creatures
    out of eighteen, so a random handful can come back with no zeeps at all — leaving a
    reader with nothing to spot and every reason to feel stupid about it.
    """
    labels = label_with(ZEEPS, rule)
    shown = teaching_examples(rule, n=n_examples, seed=shuffle)
    hidden = quiz_examples(rule, shown, n=3, seed=shuffle)
    return labels, shown, hidden


# ---------------------------------------------------------------------------


@lesson.step("I'm thinking of a rule", beat="hook")
def _():
    lesson.say(
        """
Boom: I have a **secret rule**. It decides whether a creature is a **zeep** or not.

Every creature has three things about it: a **shape**, a **colour** and a **size**.
That's all you get. I won't tell you the rule — you work it out from examples I've
already sorted.

By the end of this tiny game, you will race a computer and see the whole course in
miniature. No robot fog machine required.
"""
    )

    st.dataframe(pretty(ZEEPS).head(6), hide_index=True, width="content")
    st.caption("Six of the eighteen creatures in this world. Tiny universe, big side quest.")

    lesson.say(
        """
That sentence — *work out the rule from the examples* — **is machine learning**.
There isn't a second thing. Every other chapter is a different way of doing exactly this.
"""
    )

    lesson.kid_corner(
        "Think of the game where someone says *hot* or *cold* while you hunt for a hidden "
        "toy. They never tell you where it is. You work it out from the hints. Here, the "
        "hints are the examples."
    )


@lesson.step("Your turn to be the machine", beat="byhand")
def _():
    lesson.say("Here are six creatures I've sorted. Study them, then guess my rule.")

    labels, shown, _ = deal("big_square", 6, 0)

    examples = pretty(ZEEPS.iloc[shown]).copy()
    examples["zeep?"] = np.where(labels[shown], "✅ zeep", "❌ no")
    st.dataframe(examples, hide_index=True, width="content")

    lesson.look_for(
        "two rows that are nearly the same but got different answers. Those are the rows "
        "that give the rule away."
    )

    guess = lesson.predict(
        "What do you think my rule is?",
        ["It is red", "It is a square", "It is big", "It is big AND a square"],
        correct=3,
        why=(
            "Both switches have to click on. The big blue square is a zeep, so "
            "*it is red* crashes. The small red square is not, so *it is a square* crashes. "
            "The rule survives when both pieces survive together!"
        ),
        key="rule",
    )

    if guess is not None:
        lesson.say("Hold that thought. The computer is about to try the same thing.")


@lesson.step("What the computer sees", beat="seeit")
def _():
    labels, shown, _ = deal("big_square", 6, 0)

    lesson.say(
        """
The computer gets **exactly what you got**: those same six examples, nothing else.

But it has never heard of shapes or colours. `circle` became 0, `square` became 1,
`triangle` became 2. It has no idea what those mean, and it doesn't need to.
"""
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**What you saw**")
        seen = pretty(ZEEPS.iloc[shown]).copy()
        seen["zeep?"] = np.where(labels[shown], "✅", "❌")
        st.dataframe(seen, hide_index=True, width="stretch")
    with right:
        st.markdown("**What the computer saw**")
        raw = pd.DataFrame(X_ALL[shown], columns=["shape", "colour", "size"])
        raw["answer"] = labels[shown].astype(int)
        st.dataframe(raw, hide_index=True, width="stretch")

    lesson.jargon(
        "features and labels",
        "The columns it guesses <i>from</i> are the features. The column it guesses "
        "<i>at</i> is the label.",
    )
    lesson.jargon(
        "model",
        "The guesser after it has learned from examples. In code below, it is the "
        "thing named <code>model</code>.",
    )


@lesson.step("Race the machine", beat="seeit")
def _():
    labels, shown, hidden = deal("big_square", 6, 0)
    quiz = pretty(ZEEPS.iloc[hidden])
    truth = labels[hidden]

    lesson.say("Three creatures neither of you has seen. You answer first, then we compare.")
    st.dataframe(quiz, hide_index=True, width="content")

    yours = []
    for i, (_, row) in enumerate(quiz.iterrows()):
        pick = st.radio(
            f"A {row['size']} {row['colour']} {row['shape']}",
            ["zeep", "not a zeep"],
            key=f"race_{i}",
            index=None,
            horizontal=True,
        )
        yours.append(pick)

    if None in yours:
        st.caption("Answer all three to see the result.")
        return

    model = DecisionTreeClassifier(random_state=0).fit(X_ALL[shown], labels[shown])
    machine = model.predict(X_ALL[hidden]).astype(bool)

    table = quiz.copy()
    table["you said"] = yours
    table["computer said"] = np.where(machine, "zeep", "not a zeep")
    table["truth"] = np.where(truth, "zeep", "not a zeep")
    st.dataframe(table, hide_index=True, width="content")

    you_score = sum(1 for got, real in zip(yours, truth) if (got == "zeep") == real)
    machine_score = int((machine == truth).sum())

    a, b = st.columns(2)
    a.metric("You", f"{you_score} / 3")
    b.metric("The computer", f"{machine_score} / 3")

    if you_score > machine_score:
        st.success("🏆 You win. You needed fewer examples than it did.")
    elif you_score < machine_score:
        st.warning("🤖 The computer wins this round.")
    else:
        st.info("🤝 A tie.")


@lesson.step("The one thing that matters", beat="play")
def _():
    lesson.say("The most important picture in the whole course. Guess before you drag.")

    guess = lesson.predict(
        "The program never changes. Only the number of examples does. "
        "What happens to how often it's right?",
        [
            "Nothing — the program is the same",
            "It gets better, then levels off",
            "It gets worse with too many examples",
        ],
        correct=1,
        why="The code stays bolted to the table across this whole graph. The only thing pouring in is more data!",
        key="curve",
    )
    if guess is None:
        return

    knobs, picture = lesson.controls()
    with knobs:
        rule = st.selectbox(
            "Secret rule", list(RULES), index=0,
            format_func=lambda k: RULES[k], key="curve_rule",
        )
        n_train = st.slider("Examples the computer sees", 1, 17, 6, key="curve_n")

    curve = learning_curve(rule, n_repeats=60)
    with picture:
        chart = pd.DataFrame(
            {"examples seen": curve["n"], "how often it's right": curve["accuracy"]}
        ).set_index("examples seen")
        st.line_chart(chart, height=300)

    accuracy = float(curve["accuracy"][curve["n"] == n_train][0])
    st.metric(f"With {n_train} example(s), right", f"{accuracy:.0%} of the time")

    lesson.look_for(
        "the far left of the line. With one or two examples it's barely better than "
        "flipping a coin — and it still answers with total confidence."
    )

    lesson.aha(
        "Nobody made the computer smarter. Nobody changed its program. "
        "**The only thing that changed was how many examples it saw.**\n\n"
        "That's why people who work on AI spend most of their time thinking about data, "
        "not about code."
    )


@lesson.step("Some rules are harder than others", beat="play")
def _():
    lesson.say(
        """
Not every rule is equally easy to spot. Here are all five, raced against each other.
"""
    )

    fig, ax = lesson.figure(7, 4.2)
    for rule in RULES:
        curve = learning_curve(rule, n_repeats=40)
        ax.plot(curve["n"], curve["accuracy"], marker="o", markersize=3, label=RULES[rule])
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1.4)
    ax.text(1, 0.515, "pure guessing", color="#94A3B8", fontsize=9)
    ax.set_xlabel("examples seen")
    ax.set_ylabel("how often it is right")
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    lesson.show(fig)

    lesson.look_for("the lowest line. That's *exactly one of: red / big*.")

    lesson.say(
        """
That rule is hard because **no single column helps you**. Colour alone tells you nothing.
Size alone tells you nothing. You have to look at two columns *together*.

Hold onto that. It's the whole reason Chapter 03 has to exist.
"""
    )

    lesson.careful(
        "A model never says *I don't know*. It always answers, even with almost nothing "
        "to go on. Remember that in Chapter 10."
    )


@lesson.step("The whole program", beat="forreal")
def _():
    lesson.say("This is a complete machine learning program. Three lines.")

    st.code(
        """
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()      # 1. pick a kind of guesser
model.fit(examples, answers)          # 2. show it the examples
model.predict(new_creatures)          # 3. ask it about new ones
""",
        language="python",
    )

    lesson.say(
        "Every chapter from here changes **line 1** — a different kind of guesser — or "
        "changes what goes into **line 2**. That's the entire course."
    )


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go break it", beat="challenge")
def _():
    lesson.say(
        """
1. **Beat the machine.** Go back and set the examples slider to **3**. Can you still get
   all three right? Can it? Why is it so much harder for the computer than for you?

2. **Starve it.** Find a rule and a number of examples where the computer does *worse*
   than flipping a coin. What has it latched onto?

3. **Invent a rule.** Open `kidsml/zeeps.py`, add one to `RULES` and `_RULE_FUNCS`, and
   see where it lands on the graph. Can you invent one it can never learn from 17
   examples?
"""
    )

    lesson.kid_corner(
        "Play this away from the laptop. Think of a secret rule about things in your "
        "room — *anything blue*, or *anything you can eat*. Point at five things and say "
        "yes or no for each, explaining nothing. Then ask them to guess the sixth. "
        "How many did they need? Now swap."
    )


lesson.finish()
