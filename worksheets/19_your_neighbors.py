"""Chapter 19 workbook · You Are Like Your Neighbors."""

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
    chapter=19,
    title="Workbook · Ask the neighbours",
    intro=(
        "kNN does no training. The new point is at **(0, 0)**, so it asks the nearest old labelled points to vote. "
        "Distance is the ruler; scaling makes different columns use fair rulers."
    ),
    questions=[
        Question(
            prompt="What is the distance from the new point to A at (3, 4)?",
            kind="number",
            answer=5,
            table=POINTS,
            hint="3² + 4² = 25. Which number squared is 25?",
            why=(
                "From (0, 0) to (3, 4), the distance is √(3² + 4²) = √(9 + 16) = √25 = 5. "
                "kNN is built from this distance calculation over and over."
            ),
        ),
        Question(
            prompt="Which point is closest?",
            kind="choice",
            choices=["A", "B", "C", "D", "E"],
            answer="A",
            why=(
                "A is 5 away. The others are 6, 8, 15, and 16 away. The closest old point "
                "gets the first chance to tug the new point's label."
            ),
        ),
        Question(
            prompt="With k = 1, what label does the new point get?",
            kind="choice",
            choices=["red", "blue"],
            answer="red",
            why=(
                "With k = 1, the closest point gets the whole vote. A is red. This is powerful but risky, "
                "because one noisy neighbour can grab the whole microphone."
            ),
        ),
        Question(
            prompt="With k = 3, the nearest labels are red, blue, blue. What wins?",
            kind="choice",
            choices=["red", "blue", "tie"],
            answer="blue",
            why=(
                "Two blue votes beat one red vote. Changing k changed the answer because the model listened "
                "to a small crowd instead of one nearest point."
            ),
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
        Question(
            prompt="Penguin body mass is measured in grams, while beaks are measured in millimetres. Why scale before kNN?",
            kind="choice",
            choices=["so grams do not drown out beak lengths", "so penguins become pictures", "so k becomes even"],
            answer="so grams do not drown out beak lengths",
            why="kNN adds feature differences into one distance. A 500-gram body-mass difference can swamp a 5-millimetre beak difference unless scaling puts the columns on fair rulers.",
        ),
    ],
    kid_corner="Put five toys on the floor and give each toy a team colour. Drop a sock somewhere. The sock joins the team of the closest toy, or the closest three toys vote.",
)
