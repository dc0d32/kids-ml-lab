"""Chapter 19 workbook · Sorting Without Labels."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [["P1", 1, 1], ["P2", 1, 2], ["P3", 2, 1], ["P4", 7, 7], ["P5", 8, 7], ["P6", 7, 8]],
    columns=["point", "x", "y"],
)

WORKBOOK = Workbook(
    chapter=19,
    title="Workbook · Move the centres",
    intro="Start with one centre at **(0, 0)** and one at **(10, 10)**. Do one full k-means round.",
    questions=[
        Question(
            prompt="P1 is at (1, 1). Which starting centre is closer?",
            kind="choice",
            choices=["(0, 0)", "(10, 10)"],
            answer="(0, 0)",
            table=POINTS,
            why="P1 is 2 squared-steps from (0, 0) and 162 from (10, 10). It joins the left centre.",
        ),
        Question(
            prompt="How many points join the left centre?",
            kind="number",
            answer=3,
            why="P1, P2, and P3 are the small clump near (0, 0).",
        ),
        Question(
            prompt="The left group is (1,1), (1,2), (2,1). What is its new x coordinate?",
            kind="number",
            answer=4 / 3,
            tolerance=0.02,
            hint="Average the x values: (1 + 1 + 2) / 3.",
            why="A centre moves to the average of its members. Here that is 4/3, about 1.33.",
        ),
        Question(
            prompt="What is the new y coordinate of the left centre?",
            kind="number",
            answer=4 / 3,
            tolerance=0.02,
            why="The y values are also 1, 2, and 1, so the y average is 4/3 too.",
        ),
        Question(
            prompt="After the centres move, what happens on round two?",
            kind="choice",
            choices=["the groups change", "nothing changes", "all points swap sides"],
            answer="nothing changes",
            why="The same three points are still closest to the left centre, and the same three are closest to the right. The algorithm has converged.",
        ),
        Question(
            prompt="If k equals the number of points, what happens to the total distance to each centre?",
            kind="choice",
            choices=["it becomes zero", "it becomes huge", "it cannot be measured"],
            answer="it becomes zero",
            why="Each point can become its own centre. That makes the score perfect but useless, because it found no helpful piles.",
        ),
    ],
    kid_corner="Sort laundry into piles. First toss clothes near a few basket spots. Then move each basket to the middle of its pile. Keep doing those two moves until the baskets stop moving.",
)
