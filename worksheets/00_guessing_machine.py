"""Chapter 00 workbook · The Guessing Machine.

Questions are answered on screen — type in a box, press check, find out why the
question was worth asking.
"""

import pandas as pd

from kidsml.workbook import Question, Workbook

EXAMPLES = pd.DataFrame(
    [
        ["square", "red", "big", "✅ zeep"],
        ["circle", "red", "big", "❌ no"],
        ["square", "blue", "big", "✅ zeep"],
        ["square", "red", "small", "❌ no"],
        ["triangle", "green", "big", "❌ no"],
        ["square", "green", "small", "❌ no"],
    ],
    columns=["shape", "colour", "size", "zeep?"],
)

QUIZ = pd.DataFrame(
    [
        ["square", "green", "big"],
        ["circle", "blue", "small"],
        ["triangle", "red", "big"],
    ],
    columns=["shape", "colour", "size"],
)

XOR_TABLE = pd.DataFrame(
    [
        ["circle", "red", "big"],
        ["circle", "red", "small"],
        ["square", "blue", "big"],
        ["square", "blue", "small"],
    ],
    columns=["shape", "colour", "size"],
)

WORKBOOK = Workbook(
    chapter=0,
    title="Workbook · Find the rule",
    intro=(
        "I have a secret rule that decides whether a creature is a **zeep**. "
        "Here are six creatures I have already sorted. Work out the rule."
    ),
    questions=[
        Question(
            prompt="Look at these six. What is my secret rule? (Say it in a few words.)",
            kind="text",
            answer=["big and square", "square and big", "it is big and a square",
                    "big square", "a big square"],
            table=EXAMPLES,
            hint=(
                "Rows 1 and 3 are both zeeps but they are different colours — so colour "
                "can't be part of it. Row 4 is a big... no wait, row 4 is a *small* "
                "square, and it is not a zeep. What does that rule out?"
            ),
            why=(
                "The rule is **big AND square**. Both halves have to be true. Row 3 is "
                "what kills the guess *it is red*, and row 4 is what kills the guess "
                "*it is a square*. Every example you get either kills a guess or it "
                "doesn't — and that is the only thing data ever does."
            ),
        ),
        Question(
            prompt="Using your rule: is a **big green square** a zeep?",
            kind="choice",
            choices=["zeep", "not a zeep"],
            answer="zeep",
            table=QUIZ,
            why="Big ✓ and square ✓, so yes — even though you never saw a green zeep.",
        ),
        Question(
            prompt="Is a **small blue circle** a zeep?",
            kind="choice",
            choices=["zeep", "not a zeep"],
            answer="not a zeep",
            why="Not big, not a square. It fails both halves.",
        ),
        Question(
            prompt="Is a **big red triangle** a zeep?",
            kind="choice",
            choices=["zeep", "not a zeep"],
            answer="not a zeep",
            why=(
                "Big ✓ but not a square ✗. This is the one people get wrong, because "
                "*big* was true for so many of the zeeps that it starts to feel like the "
                "whole rule. Models fall for this too — it's called latching onto the "
                "wrong feature."
            ),
        ),
        Question(
            prompt=(
                "Suppose I had only shown you rows **1, 3 and 5**. Could you still have "
                "found the rule? What else might you have guessed instead?"
            ),
            kind="open",
            why=(
                "No. Rows 1, 3 and 5 are also perfectly consistent with *it is big*, "
                "*it is a square*, *it is not green*, and *it is not a triangle*. With "
                "too few examples, many different rules survive, and the learner has to "
                "pick one at random. That is exactly what the left-hand end of the graph "
                "in this chapter looks like."
            ),
        ),
        Question(
            prompt=(
                "Suppose **all six** examples had been ✅ zeep. What rule would you "
                "guess then?"
            ),
            kind="open",
            why=(
                "\"Everything is a zeep.\" It's the only thing the evidence supports — "
                "and it's a real failure mode. A model trained on data where every "
                "answer is the same will happily learn to always give that answer, and "
                "will look 100% accurate on its own examples while being useless."
            ),
        ),
        Question(
            prompt=(
                "Here's a harder rule: *a creature is a zeep if **exactly one** of these "
                "is true — it is red, or it is big.* How many of these four are zeeps?"
            ),
            kind="number",
            answer=2,
            table=XOR_TABLE,
            hint="Go row by row. Count how many of {red, big} are true. Exactly one → zeep.",
            why=(
                "Two. Red+big = both true → no. Red+small = exactly one → **zeep**. "
                "Blue+big = exactly one → **zeep**. Blue+small = neither → no.\n\n"
                "This rule is genuinely harder because **no single column tells you "
                "anything on its own**. Colour alone is useless. Size alone is useless. "
                "You have to look at two columns *together*. That's why its curve in the "
                "chapter stays low the longest — and it is the whole reason Chapter 03 "
                "has to exist."
            ),
        ),
        Question(
            prompt=(
                "The big one. The computer used the **exact same program** whether it saw "
                "2 examples or 17 — not one line of code changed. So what actually made "
                "it better?"
            ),
            kind="text",
            answer=["more examples", "more data", "examples", "data", "more example",
                    "the number of examples", "seeing more examples"],
            why=(
                "More examples. That's it. This is the single most important idea in the "
                "course, and it's why people who work on AI spend most of their time "
                "thinking about data rather than about code."
            ),
        ),
    ],
    kid_corner=(
        "Play this with someone in your family, away from the laptop.\n\n"
        "1. Think of a secret rule about things in your room — *anything blue*, or "
        "*anything you can eat*, or *anything smaller than your hand*.\n"
        "2. Point at five things one at a time and say **yes** or **no** for each. "
        "Don't explain anything.\n"
        "3. Now point at a sixth thing and ask them to guess.\n"
        "4. Keep going until they get three in a row right.\n\n"
        "How many examples did they need? Now swap, and see how many *you* need."
    ),
)
