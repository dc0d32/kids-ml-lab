"""Chapter 20 workbook · Squishing Dimensions."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [["A", 1, 2], ["B", 2, 2], ["C", 3, 3], ["D", 4, 3]],
    columns=["point", "x", "y"],
)

WORKBOOK = Workbook(
    chapter=20,
    title="Workbook · Pick the best shadow",
    intro="Four points lie nearly along a sideways line. If you can keep one number, which shadow keeps more of the story?",
    questions=[
        Question(
            prompt="What is the average x value?",
            kind="number",
            answer=2.5,
            table=POINTS,
            why=(
                "(1 + 2 + 3 + 4) / 4 = 2.5. Spread is measured around the middle because "
                "we want to know how much the shadow still separates the points."
            ),
        ),
        Question(
            prompt="Using x only, the squared distances from 2.5 are 2.25, 0.25, 0.25, 2.25. What is their total?",
            kind="number",
            answer=5,
            why=(
                "That total is the x-shadow spread. A larger spread means the points stayed more different "
                "from each other after we kept only x."
            ),
        ),
        Question(
            prompt="What is the average y value?",
            kind="number",
            answer=2.5,
            why=(
                "The y values are 2, 2, 3, 3, so the middle is (2 + 2 + 3 + 3) / 4 = 2.5. "
                "Now we can measure how much the y shadow spreads around that middle."
            ),
        ),
        Question(
            prompt="Using y only, four squared distances are all 0.25. What is their total?",
            kind="number",
            answer=1,
            why="The y-shadow is less spread out, so it kept less information about which point is which.",
        ),
        Question(
            prompt="If you could keep only one axis, which would you keep?",
            kind="choice",
            choices=["x", "y"],
            answer="x",
            why=(
                "The x shadow has spread 5, while the y shadow has spread 1. PCA hunts for the spread-out shadow "
                "because spread keeps more information about which point is which."
            ),
        ),
        Question(
            prompt="If you had to describe each classmate with one number, what might you choose, and what would you lose?",
            kind="open",
            why="Any one number throws information away. Height loses hobbies. Age loses personality. Compression is powerful, but it always has a cost.",
        ),
    ],
    kid_corner="Hold your hand near a wall with a flashlight. Turn your hand. Some shadows look like a hand; some look like a blob. PCA is the game of picking the useful shadow.",
)
