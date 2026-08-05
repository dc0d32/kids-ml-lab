"""Chapter 02 workbook · Lines That Decide."""

import pandas as pd

from kidsml.workbook import Question, Workbook

# Five of the ten dogs from the chapter — a mix of both answers, so counting them
# actually tells you something. Five puppies in a row would not.
TINY_POINTS = pd.DataFrame(
    [
        [1, 1, "puppy"],
        [2, 3, "puppy"],
        [6, 5, "grown dog"],
        [8, 6, "grown dog"],
        [7, 8, "grown dog"],
    ],
    columns=["how tall (x1)", "how heavy (x2)", "really a"],
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
        "Ten dogs, measured two ways: how tall in hand-spans, how heavy in bags of sugar. "
        "The line scores each dog, then reads the sign — positive means grown dog, "
        "negative means puppy."
    ),
    questions=[
        Question(
            prompt=(
                "Use **score = x1 + x2 − 8**. For these five dogs, how many scores come out "
                "positive — that is, how many does the line call a **grown dog**?"
            ),
            kind="number",
            answer=3,
            tolerance=0.01,
            table=TINY_POINTS,
            hint="Work out x1 + x2 − 8 for each row. Positive means grown dog.",
            why=(
                "The scores are −6, −3, +3, +6 and +7, so three come out positive — and those "
                "three are exactly the grown dogs. This one guessed line got all five right.\n\n"
                "That is the whole perceptron rhythm: two coordinates funnel into one score, "
                "then the sign snaps that score into an answer. Notice the size of the score "
                "does no work at all. +3 and +7 mean the same thing."
            ),
        ),
        Question(
            prompt="Start with a bad line: **w = (1, 1), b = -20**. What score does point **(6, 5)** get?",
            kind="number",
            answer=-9,
            tolerance=0.01,
            hint="Use w1×x1 + w2×x2 + b = 1×6 + 1×5 - 20.",
            why=(
                "The score is **1(6) + 1(5) − 20 = −9**. Negative, so this bad line files a grown dog "
                "under puppy. That wrong sign is the alarm bell that starts training."
            ),
        ),
        Question(
            prompt="The dog at **(6, 5)** is really a grown dog. With a score of −9, what did the bad line guess?",
            kind="choice",
            choices=["grown dog", "puppy"],
            answer="puppy",
            why=(
                "Negative scores mean puppy, so the line guessed puppy. It is really a grown dog, so the update has one job: raise "
                "this point's future score."
            ),
        ),
        Question(
            prompt="Because that grown dog was missed, add its first coordinate to w1. New **w1** = ?",
            kind="number",
            answer=7,
            tolerance=0.01,
            hint="Old w1 was 1. Add x1 = 6.",
            why=(
                "New w1 is **1 + 6 = 7**. Adding a missed red point to the weights raises that point's score "
                "next time, shoving the boundary toward the red side where it belongs."
            ),
        ),
        Question(
            prompt="Add the second coordinate to w2. New **w2** = ?",
            kind="number",
            answer=6,
            tolerance=0.01,
            hint="Old w2 was 1. Add x2 = 5.",
            why=(
                "New w2 is **1 + 5 = 6**. The update is a nudge in the direction of the missed point, not magic dust. "
                "For (6, 5), both weights need to tug harder toward that red region."
            ),
        ),
        Question(
            prompt="For this missed red point, add 1 to the bias. New **b** = ?",
            kind="number",
            answer=-19,
            tolerance=0.01,
            hint="Start at -20 and add 1.",
            why=(
                "New b is **-20 + 1 = -19**. Bias changes every score by the same amount, so geometrically it "
                "slides the whole line like a ruler across the desk without turning it."
            ),
        ),
        Question(
            prompt="Can one straight line that must pass through **(0, 0)** separate these red and blue points?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            table=BIAS_POINTS,
            hint="The blue points sit above the red points, but a line pinned to the origin has very little room to slide between them.",
            why=(
                "No. The bias **b** lets a line slide into the gap between piles. Without b, every boundary is "
                "pinned to (0, 0), so the line can spin but cannot scoot into the clean space between these rows."
            ),
        ),
        Question(
            prompt="A perceptron stops changing when it gets every training point right. What might happen if the red and blue piles overlap?",
            kind="open",
            hint="Ask whether a perfect straight divider exists.",
            why=(
                "It may keep changing because perfection is impossible. One update patches one mistake, but with overlap "
                "that patch can tear open another mistake somewhere else. That is a limit of the model, not a coding bug."
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
