"""Chapter 16 workbook · Pictures Are Just Numbers."""

import pandas as pd

from kidsml.workbook import Question, Workbook

TINY_THREE = pd.DataFrame(
    [
        [0, 8, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 0, 8, 8, 0],
        [0, 0, 0, 8, 0],
        [0, 8, 8, 8, 0],
    ]
)

PHOTO_COUNTS = pd.DataFrame(
    [
        ["8×8 gray digit", "8 × 8", "?"],
        ["28×28 gray digit", "28 × 28", "?"],
        ["1000×1000 colour photo", "1000 × 1000 × 3", "?"],
    ],
    columns=["image", "numbers", "total"],
)

WORKBOOK = Workbook(
    chapter=16,
    title="Workbook · Read the numbers",
    intro=(
        "Pictures in this chapter are grids of numbers. Use scrap paper for any counting, "
        "then type your answers here."
    ),
    questions=[
        Question(
            prompt="What digit does this 5×5 grid look like?",
            kind="number",
            answer=3,
            table=TINY_THREE,
            why=(
                "It looks like a 3 because the high numbers make the stroke. The picture is not hidden behind the numbers; "
                "the numbers are the picture."
            ),
        ),
        Question(
            prompt="How many numbers are in an 8×8 gray image?",
            kind="number",
            answer=64,
            table=PHOTO_COUNTS,
            why="8 rows times 8 columns is 64. That is small enough to show as one row for a model.",
        ),
        Question(
            prompt="How many numbers are in a 28×28 gray image?",
            kind="number",
            answer=784,
            why="28 × 28 = 784. A tiny-looking image already has hundreds of inputs.",
        ),
        Question(
            prompt="How many numbers are in a 1000×1000 colour photo?",
            kind="number",
            answer=3000000,
            tolerance=0.5,
            why=(
                "One million pixels, and each colour pixel has red, green, and blue. "
                "That makes 3,000,000 numbers before the model has learned anything."
            ),
        ),
        Question(
            prompt="If a model has one weight per pixel-number, how many weights does that colour photo need?",
            kind="number",
            answer=3000000,
            tolerance=0.5,
            why=(
                "One weight for each input number means 3,000,000 weights. That is why image models need careful ideas, "
                "not only more layers."
            ),
        ),
        Question(
            prompt="Why can a plain MLP miss something important when the 64 pixels are flattened into one row?",
            kind="open",
            why=(
                "Flattening throws away the map shape. The values are still there, but the model is not told which pixels were neighbours. "
                "Chapter 17 puts that neighbour idea back."
            ),
        ),
        Question(
            prompt="Look at a confusion matrix. If row 9, column 4 has a big number, what happened many times?",
            kind="choice",
            choices=["real 9s were guessed as 4s", "real 4s were guessed as 9s", "the model saw no 9s"],
            answer="real 9s were guessed as 4s",
            why=(
                "Rows are true answers and columns are guesses, like in Chapter 09. Off-diagonal cells are the model's mix-ups."
            ),
        ),
    ],
    kid_corner=(
        "Make an 8 by 8 grid with coins, cereal pieces, or sticky notes. Use more pieces where the line of the number should be. "
        "Stand back and squint. If someone can read it, they are reading the numbers as a picture."
    ),
)
