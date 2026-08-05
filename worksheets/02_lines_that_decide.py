"""Chapter 02 workbook · Lines That Decide."""

import pandas as pd

from kidsml.workbook import Question, Workbook

TINY_POINTS = pd.DataFrame(
    [
        [1, 1, "blue"],
        [2, 1, "blue"],
        [1, 2, "blue"],
        [2, 3, "blue"],
        [3, 2, "blue"],
    ],
    columns=["x1", "x2", "truth"],
)

BIAS_POINTS = pd.DataFrame(
    [
        [1, 4, "blue"],
        [2, 5, "blue"],
        [1, 1, "red"],
        [2, 2, "red"],
    ],
    columns=["x1", "x2", "class"],
)

WORKBOOK = Workbook(
    chapter=2,
    title="Workbook · Let a line choose sides",
    intro=(
        "A perceptron makes a raw score first. Then it uses the sign: positive means red, "
        "negative means blue."
    ),
    questions=[
        Question(
            prompt=(
                "Use **score = x1 + x2 - 8**. For these five points, how many scores are positive, "
                "so the line guesses **red**?"
            ),
            kind="number",
            answer=0,
            tolerance=0.01,
            table=TINY_POINTS,
            hint="Compute x1 + x2 - 8 for each row. Positive means red.",
            why=(
                "The scores are -6, -5, -5, -3, and -3. All are negative, so every guess is blue. "
                "The model makes a number first, then turns the sign into a class."
            ),
        ),
        Question(
            prompt="Start with a bad line: **w = (1, 1), b = -20**. What score does point **(6, 5)** get?",
            kind="number",
            answer=-9,
            tolerance=0.01,
            hint="Use w1×x1 + w2×x2 + b = 1×6 + 1×5 - 20.",
            why=(
                "The score is 6 + 5 - 20 = **-9**. It is negative, so this bad line puts the point on the blue side."
            ),
        ),
        Question(
            prompt="Point **(6, 5)** is really red. With score -9, what did the bad line guess?",
            kind="choice",
            choices=["red", "blue"],
            answer="blue",
            why=(
                "Negative scores mean blue, so the line guessed blue. The truth is red, which triggers one perceptron update."
            ),
        ),
        Question(
            prompt="Because that red point was missed, add its first coordinate to w1. New **w1** = ?",
            kind="number",
            answer=7,
            tolerance=0.01,
            hint="Old w1 was 1. Add x1 = 6.",
            why="New w1 is **7**. The update is small enough to do by hand: add the missed red point to the weights.",
        ),
        Question(
            prompt="Add the second coordinate to w2. New **w2** = ?",
            kind="number",
            answer=6,
            tolerance=0.01,
            hint="Old w2 was 1. Add x2 = 5.",
            why="New w2 is **6**. Training is a nudge, not magic: the weights move toward the red point.",
        ),
        Question(
            prompt="For this missed red point, add 1 to the bias. New **b** = ?",
            kind="number",
            answer=-19,
            tolerance=0.01,
            hint="Start at -20 and add 1.",
            why="New b is **-19**. The bias lets the line slide as the perceptron learns.",
        ),
        Question(
            prompt="Can one straight line that must pass through **(0, 0)** separate these red and blue points?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            table=BIAS_POINTS,
            hint="The blue points sit above the red points, but a line pinned to the origin has very little room to slide between them.",
            why=(
                "No. The bias **b** lets a line slide. Without it, every line is pinned to (0, 0), and this clean split is impossible."
            ),
        ),
        Question(
            prompt="A perceptron stops changing when it gets every training point right. What might happen if the red and blue piles overlap?",
            kind="open",
            hint="Ask whether a perfect straight divider exists.",
            why=(
                "It may keep changing because perfection is impossible. That is an honest limit of the perceptron, not a coding bug."
            ),
        ),
    ],
    kid_corner=(
        "Make two piles of toys: red team and blue team. Put a pencil between them.\n\n"
        "1. Move the pencil so all red toys are on one side.\n"
        "2. Move one toy into the other pile.\n"
        "3. Can the pencil still make everyone happy?\n\n"
        "The moved toy creates overlap. A straight divider works only when the piles can be cleanly separated."
    ),
)
