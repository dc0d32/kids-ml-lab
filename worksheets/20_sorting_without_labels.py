"""Chapter 20 workbook · Sorting Without Labels."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [["P1", 1, 1], ["P2", 1, 2], ["P3", 2, 1], ["P4", 7, 7], ["P5", 8, 7], ["P6", 7, 8]],
    columns=["point", "x", "y"],
)

PIXELS = pd.DataFrame(
    [["leaf pixel", 30, 160, 45], ["sky pixel", 80, 140, 240], ["flower pixel", 220, 60, 120]],
    columns=["pixel", "red", "green", "blue"],
)

WORKBOOK = Workbook(
    chapter=20,
    title="Workbook · Move the centres",
    intro="Start with one centre at **(0, 0)** and one at **(10, 10)**. Do one full k-means round. A cluster is a pile; its centroid is the middle.",
    questions=[
        Question(
            prompt="P1 is at (1, 1). Which starting centre is closer?",
            kind="choice",
            choices=["(0, 0)", "(10, 10)"],
            answer="(0, 0)",
            table=POINTS,
            why=(
                "P1 is 2 squared-steps from (0, 0): 1² + 1². It is 162 squared-steps from (10, 10): "
                "9² + 9². It joins the left centre because that ruler distance is smaller."
            ),
        ),
        Question(
            prompt="How many points join the left centre?",
            kind="number",
            answer=3,
            why=(
                "P1, P2, and P3 are the small clump near (0, 0). k-means does not know the word clump; "
                "it gets there by asking which centre is closest."
            ),
        ),
        Question(
            prompt="The left group is (1,1), (1,2), (2,1). What is its new x coordinate?",
            kind="number",
            answer=4 / 3,
            tolerance=0.02,
            hint="Average the x values: (1 + 1 + 2) / 3.",
            why="A centre moves to the average of its members. Here that is 4/3, about 1.33, so the centre scoots toward its little pile.",
        ),
        Question(
            prompt="What is the new y coordinate of the left centre?",
            kind="number",
            answer=4 / 3,
            tolerance=0.02,
            why="The y values are also 1, 2, and 1, so the y average is 4/3 too. The centre lands at the middle of the same tiny triangle.",
        ),
        Question(
            prompt="The right group is (7,7), (8,7), (7,8). What is its new x coordinate?",
            kind="number",
            answer=22 / 3,
            tolerance=0.02,
            hint="Average the x values: (7 + 8 + 7) / 3.",
            why="The right centre moves to the middle of its three points. The x average is 22/3, about 7.33, matching the by-hand round.",
        ),
        Question(
            prompt="After the centres move, what happens on round two?",
            kind="choice",
            choices=["the groups change", "nothing changes", "all points swap sides"],
            answer="nothing changes",
            why=(
                "The same three points are still closest to the left centre, and the same three are closest to the right. "
                "The centres would move to the same averages again, so the algorithm has converged."
            ),
        ),
        Question(
            prompt="If k equals the number of points, what happens to the total squared distance to each centre?",
            kind="choice",
            choices=["it becomes zero", "it becomes huge", "it cannot be measured"],
            answer="it becomes zero",
            why="Each point can become its own centre. That makes the score perfect but useless, because it found no helpful piles.",
        ),
        Question(
            prompt="In the photo demo, what is one pixel treated as?",
            kind="choice",
            choices=["a point with red, green, and blue coordinates", "a whole cluster name", "a future label"],
            answer="a point with red, green, and blue coordinates",
            table=PIXELS,
            why=(
                "Colour quantization treats each pixel as a point in 3D colour space: (red amount, green amount, blue amount). "
                "K-means finds colour centroids, then repaints each pixel with its nearest centre colour."
            ),
        ),
    ],
    kid_corner="Sort laundry into piles. First toss clothes near a few basket spots. Then move each basket to the middle of its pile. Keep doing those two moves until the baskets stop moving.",
)
