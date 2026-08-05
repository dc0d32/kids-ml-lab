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
        "four answers, no ruler line works."
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
                "No. Many near-misses are possible, and feeling that failure matters. The proof in the next questions shows why the near-miss cannot become perfect."
            ),
        ),
        Question(
            prompt="A line score is **w1×x1 + w2×x2 + b**. Which expression belongs to blue point **(1, 1)**?",
            kind="choice",
            choices=["b", "w1 + b", "w2 + b", "w1 + w2 + b"],
            answer="w1 + w2 + b",
            why="Plug in x1 = 1 and x2 = 1. The score becomes **w1 + w2 + b**, and blue would need it to be less than 0.",
        ),
        Question(
            prompt="Which expression belongs to red point **(1, 0)**?",
            kind="choice",
            choices=["b", "w1 + b", "w2 + b", "w1 + w2 + b"],
            answer="w1 + b",
            why="Plug in x1 = 1 and x2 = 0. The score becomes **w1 + b**, and red would need it to be greater than 0.",
        ),
        Question(
            prompt="Add the two red-row needs: **w1 + b > 0** and **w2 + b > 0**. What do you get?",
            kind="choice",
            choices=["w1 + w2 + 2b > 0", "w1 + w2 + 2b < 0", "b < 0"],
            answer="w1 + w2 + 2b > 0",
            why="Adding the left sides gives **w1 + w2 + 2b**. Adding two things bigger than 0 gives something bigger than 0.",
        ),
        Question(
            prompt="Add the two blue-row needs: **b < 0** and **w1 + w2 + b < 0**. What do you get?",
            kind="choice",
            choices=["w1 + w2 + 2b > 0", "w1 + w2 + 2b < 0", "w1 + b > 0"],
            answer="w1 + w2 + 2b < 0",
            why="Adding those blue inequalities gives **w1 + w2 + 2b < 0**. Now the same number has two opposite demands.",
        ),
        Question(
            prompt="Why is that impossible?",
            kind="open",
            hint="Compare the two answers you got from the red rows and the blue rows.",
            why=(
                "The same number cannot be both greater than 0 and less than 0. That is the clean proof that XOR is not linearly separable."
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
                "2² + 0² = **4**. The new feature measures distance from the middle, squared. A straight model can work if the features are cleverer."
            ),
        ),
        Question(
            prompt="The stripes shape changes class again and again as x1 moves left to right. What kind of feature might help?",
            kind="choice",
            choices=["a feature that grows forever", "a feature that repeats", "a feature that ignores x1"],
            answer="a feature that repeats",
            hint="Stripes come back again and again.",
            why=(
                "Stripes are periodic, so a repeating feature such as sine or cosine can line them up in a space where a straight model has a chance."
            ),
        ),
    ],
    kid_corner=(
        "Put a donut-shaped ring and a button on a table. Try to separate the ring from the button with one straight piece of string.\n\n"
        "Now lift the ring pieces higher than the button. Could a flat book separate high from low?\n\n"
        "Lifting adds a new direction. That is the physical version of adding a feature."
    ),
)
