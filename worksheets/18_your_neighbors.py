"""Chapter 18 workbook · You Are Like Your Neighbors."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [
        ["A", 3, 4, "red"],
        ["B", 6, 0, "blue"],
        ["C", 0, 8, "blue"],
        ["D", 9, 12, "red"],
        ["E", 16, 0, "red"],
    ],
    columns=["point", "x", "y", "label"],
)

WORKBOOK = Workbook(
    chapter=18,
    title="Workbook · Ask the neighbours",
    intro="The new point is at **(0, 0)**. Work out who it copies.",
    questions=[
        Question(
            prompt="What is the distance from the new point to A at (3, 4)?",
            kind="number",
            answer=5,
            table=POINTS,
            hint="3² + 4² = 25. Which number squared is 25?",
            why="This is the 3-4-5 triangle. kNN is built from this distance calculation over and over.",
        ),
        Question(
            prompt="Which point is closest?",
            kind="choice",
            choices=["A", "B", "C", "D", "E"],
            answer="A",
            why="A is 5 away. The others are 6, 8, 15, and 16 away.",
        ),
        Question(
            prompt="With k = 1, what label does the new point get?",
            kind="choice",
            choices=["red", "blue"],
            answer="red",
            why="With k = 1, the closest point gets the whole vote. A is red.",
        ),
        Question(
            prompt="With k = 3, the nearest labels are red, blue, blue. What wins?",
            kind="choice",
            choices=["red", "blue", "tie"],
            answer="blue",
            why="Two blue votes beat one red vote. Changing k changed the answer.",
        ),
        Question(
            prompt="With k = 5, what label wins?",
            kind="choice",
            choices=["red", "blue", "tie"],
            answer="red",
            why="All five points vote: three red and two blue. A bigger neighbourhood can flip the decision again.",
        ),
        Question(
            prompt="Why can we sort by dx² + dy² without taking the square root?",
            kind="open",
            why="Square roots keep the same order. If 25 is smaller than 36, then √25 is smaller than √36. So the squared distances are enough for finding nearest neighbours.",
        ),
        Question(
            prompt="Why is an even k risky in a two-team vote?",
            kind="open",
            why="Even k can split the votes exactly in half. Then the model needs an extra tie-break rule, and that rule may feel arbitrary.",
        ),
    ],
    kid_corner="Put five toys on the floor and give each toy a team colour. Drop a sock somewhere. The sock joins the team of the closest toy, or the closest three toys vote.",
)
