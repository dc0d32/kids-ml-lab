"""Chapter 03 workbook · When a Ruler Isn't Enough."""

import pandas as pd

from kidsml.workbook import Question, Workbook

XOR_TABLE = pd.DataFrame(
    [
        [0, 0, "blue"],
        [0, 1, "red"],
        [1, 0, "red"],
        [1, 1, "blue"],
    ],
    columns=["x1", "x2", "answer"],
)

RADIUS_ROWS = pd.DataFrame(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [2, 0],
        [0, 2],
    ],
    columns=["x1", "x2"],
)

WORKBOOK = Workbook(
    chapter=3,
    title="Workbook · Feel the ruler fail",
    intro=(
        "Some patterns cannot be split by one straight line. XOR is the tiny proof: four dots, "
        "four answers, no ruler line works. For circles, we invent x3 as a new height so a "
        "flat cut can turn into a curved boundary."
    ),
    questions=[
        Question(
            prompt="Look at XOR. Can one straight line put both red points on one side and both blue points on the other?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            table=XOR_TABLE,
            hint="Opposite corners match. Try to keep both red corners together without catching a blue corner.",
            why=(
                "No. You can get close, but one corner always pops onto the wrong side. The next questions turn that feeling "
                "into a proof, so the failure does not feel like a bad drawing."
            ),
        ),
        Question(
            prompt="A line score is **w1×x1 + w2×x2 + b**. Which expression belongs to blue point **(1, 1)**?",
            kind="choice",
            choices=["b", "w1 + b", "w2 + b", "w1 + w2 + b"],
            answer="w1 + w2 + b",
            why=(
                "Plug in x1 = 1 and x2 = 1: **w1(1) + w2(1) + b = w1 + w2 + b**. "
                "Because this corner is blue, a perfect line would need that score below 0, safely on the blue side."
            ),
        ),
        Question(
            prompt="Which expression belongs to red point **(1, 0)**?",
            kind="choice",
            choices=["b", "w1 + b", "w2 + b", "w1 + w2 + b"],
            answer="w1 + b",
            why=(
                "Plug in x1 = 1 and x2 = 0: **w1(1) + w2(0) + b = w1 + b**. This point is red, "
                "so a perfect line would need the score above 0, safely on the red side."
            ),
        ),
        Question(
            prompt="Add the two red-row needs: **w1 + b > 0** and **w2 + b > 0**. What do you get?",
            kind="choice",
            choices=["w1 + w2 + 2b > 0", "w1 + w2 + 2b < 0", "b < 0"],
            answer="w1 + w2 + 2b > 0",
            why=(
                "Adding the left sides gives **w1 + w2 + 2b**. Two numbers each bigger than 0 add to another number "
                "bigger than 0, so the red corners demand **w1 + w2 + 2b > 0**."
            ),
        ),
        Question(
            prompt="Add the two blue-row needs: **b < 0** and **w1 + w2 + b < 0**. What do you get?",
            kind="choice",
            choices=["w1 + w2 + 2b > 0", "w1 + w2 + 2b < 0", "w1 + b > 0"],
            answer="w1 + w2 + 2b < 0",
            why=(
                "Adding those blue inequalities gives the same left side, **w1 + w2 + 2b**. But both blue scores "
                "sit below 0, so their sum must also sit below 0."
            ),
        ),
        Question(
            prompt="Why is that impossible?",
            kind="open",
            hint="Compare the two answers you got from the red rows and the blue rows.",
            why=(
                "The red corners require **w1 + w2 + 2b > 0**. The blue corners require **w1 + w2 + 2b < 0**. "
                "That is one number being pulled through two different doors. The assumed perfect line cannot exist!"
            ),
        ),
        Question(
            prompt="For circles, invent **x3 = x1² + x2²**. What is x3 for point **(2, 0)**?",
            kind="number",
            answer=4,
            tolerance=0.01,
            table=RADIUS_ROWS,
            hint="Square each coordinate, then add.",
            why=(
                "2² + 0² = **4**. The new feature measures distance from the middle, squared. Middle points stay low, "
                "ring points rise, and a flat cut in the lifted space casts a circle when you look back down."
            ),
        ),
        Question(
            prompt="The stripes shape changes class again and again as x1 moves left to right. What kind of feature might help?",
            kind="choice",
            choices=["a feature that grows forever", "a feature that repeats", "a feature that ignores x1"],
            answer="a feature that repeats",
            hint="Stripes come back again and again.",
            why=(
                "Stripes repeat: the same kind of region comes back as x1 moves. A repeating feature such as sine "
                "or cosine gives the model a coordinate that marches with the pattern instead of growing forever."
            ),
        ),
    ],
    kid_corner=(
        "Put a donut-shaped ring and a button on a table. Try to separate the ring from the button with one straight piece of string.\n\n"
        "Now lift the ring pieces higher than the button. Could a flat book separate high from low?\n\n"
        "Lifting adds a new direction. That is the physical version of adding a feature."
    ),
)
