"""Chapter 01 workbook · Lines That Predict."""

import pandas as pd

from kidsml.workbook import Question, Workbook

PIGGY_BANK = pd.DataFrame(
    [
        [1, 8],
        [2, 11],
        [3, 15],
        [4, 17],
    ],
    columns=["weeks saved", "real dollars"],
)

COMPARE_LINE = pd.DataFrame(
    [
        [1, 8],
        [2, 11],
        [3, 15],
        [4, 17],
    ],
    columns=["weeks saved", "real dollars"],
)

WORKBOOK = Workbook(
    chapter=1,
    title="Workbook · Measure a line's mistakes",
    intro=(
        "A line is a tiny prediction machine. Here you test one piggy-bank line, "
        "add up how wrong it is, then compare it with another line."
    ),
    questions=[
        Question(
            prompt=(
                "Use **dollars = 3 × weeks + 5** on these four rounded rows. "
                "What is the **total squared mistake**?"
            ),
            kind="number",
            answer=1,
            tolerance=0.01,
            table=PIGGY_BANK,
            hint="Predictions are 8, 11, 14, 17. Subtract prediction from real dollars, then square each mistake.",
            why=(
                "The predictions are 8, 11, 14, and 17. The mistakes are 0, 0, 1, and 0, "
                "so the squared mistakes are 0, 0, 1, and 0. Total = **1**. One score lets you "
                "compare a whole line instead of arguing about four separate rows."
            ),
        ),
        Question(
            prompt="A model misses by **+2** on one point and **-2** on another. If you add the raw mistakes, what total do you get?",
            kind="number",
            answer=0,
            tolerance=0.01,
            hint="The signs matter when you add raw mistakes: +2 plus -2.",
            why=(
                "+2 + -2 = **0**, even though the model was wrong twice. That is the danger of raw "
                "errors: opposite signs can cancel and make a bad line look perfect."
            ),
        ),
        Question(
            prompt="Now square those two mistakes first. What is **2² + (-2)²**?",
            kind="number",
            answer=8,
            tolerance=0.01,
            hint="Both squares are positive.",
            why=(
                "2² = 4 and (-2)² = 4, so the total is **8**. Squaring keeps both mistakes visible, "
                "and it also makes bigger misses grow faster than small misses."
            ),
        ),
        Question(
            prompt="Which total feels more honest for those two misses: raw total 0, or squared total 8? Why?",
            kind="open",
            hint="Think about whether the model was actually perfect.",
            why=(
                "The squared total is more honest here. A raw total of 0 sounds like no error, but the line "
                "missed both points. Squaring answers the question you really care about: how much wrongness is left?"
            ),
        ),
        Question(
            prompt=(
                "Try a second candidate line: **dollars = 4 × weeks + 1**. "
                "What is its total squared mistake on the same four rows?"
            ),
            kind="number",
            answer=17,
            tolerance=0.01,
            table=COMPARE_LINE,
            hint="Predictions are 5, 9, 13, 17. The mistakes are real minus prediction.",
            why=(
                "The predictions are 5, 9, 13, and 17. The mistakes are 3, 2, 2, and 0, "
                "so the squared mistakes are 9, 4, 4, and 0. Total = **17**. The smaller total wins, "
                "so the first line fits these rows better."
            ),
        ),
        Question(
            prompt="Did **4 × weeks + 1** beat **3 × weeks + 5** on these rows?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            why=(
                "No. The first line scored 1, and the second line scored 17. Lower squared mistake wins because "
                "it means less total wrongness across the same data."
            ),
        ),
        Question(
            prompt="After training, the whole line is two numbers: **w** and **b**. What does **w** control on the graph?",
            kind="text",
            answer=["the tilt", "tilt", "slope", "the slope", "how steep it is", "steepness"],
            hint="Change w in the app and watch whether the line rotates or slides.",
            why=(
                "**w** controls the tilt, or slope. It says how many dollars the prediction changes when weeks "
                "goes up by 1. The model is not a mystery box here; it is a line with two knobs."
            ),
        ),
        Question(
            prompt="What does **b** control?",
            kind="text",
            answer=["up and down", "the height", "height", "starting dollars", "start", "intercept", "the intercept", "slides up or down"],
            hint="Set weeks to 0 in dollars = w × weeks + b.",
            why=(
                "**b** slides the line up or down. If weeks = 0, the formula becomes dollars = b, so b is the "
                "starting height before the slope adds anything."
            ),
        ),
    ],
    kid_corner=(
        "Put four toy cars in a row, not perfectly straight. Lay a string near them.\n\n"
        "1. Move the string until it is close to all the cars.\n"
        "2. Point to where the next car might go.\n"
        "3. Move one car far away. What happens to the string?\n\n"
        "That far-away car is an outlier. It can pull the best line toward it."
    ),
)
