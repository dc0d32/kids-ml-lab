"""Chapter 17 workbook · The Sliding Window."""

import pandas as pd

from kidsml.workbook import Question, Workbook

IMAGE = pd.DataFrame(
    [
        [0, 0, 9, 9, 9],
        [0, 0, 9, 9, 9],
        [0, 0, 9, 9, 9],
        [0, 0, 9, 9, 9],
        [0, 0, 9, 9, 9],
    ]
)

KERNEL = pd.DataFrame(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]
)

FIRST_PATCH = pd.DataFrame(
    [
        [0, 0, 9],
        [0, 0, 9],
        [0, 0, 9],
    ]
)

WORKBOOK = Workbook(
    chapter=17,
    title="Workbook · Slide the window",
    intro=(
        "A 3×3 kernel looks at one tiny patch at a time. Multiply matching cells, add them, "
        "then slide one square."
    ),
    questions=[
        Question(
            prompt="For this first patch and kernel, what number goes in the first output cell?",
            kind="number",
            answer=27,
            table=FIRST_PATCH,
            hint="Only the right column of the kernel has +1 values. Add the three 9s.",
            why=(
                "The sum is 0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 + 0·(-1) + 0·0 + 9·1 = 27. "
                "The window sees dark values on the left and bright values on the right, so this vertical-edge kernel fires hard."
            ),
        ),
        Question(
            prompt="Work out the full 3×3 output for the 5×5 image. Press Check when you have your grid on scrap paper.",
            kind="open",
            table=IMAGE,
            why=(
                "The full output is [[27, 27, 0], [27, 27, 0], [27, 27, 0]]. The large values sit where the vertical edge is. "
                "The rightmost windows are all bright, so left and right cancel."
            ),
        ),
        Question(
            prompt="If you transpose the kernel so the -1 row is on top and the +1 row is on bottom, what kind of edge does it detect?",
            kind="choice",
            choices=["horizontal edges", "vertical edges", "colour names"],
            answer="horizontal edges",
            table=KERNEL,
            why=(
                "The transposed kernel compares top versus bottom instead of left versus right, so it looks for horizontal changes."
            ),
        ),
        Question(
            prompt="A 5×5 image with a 3×3 kernel gives how many output rows?",
            kind="number",
            answer=3,
            why=(
                "The 3-high window can start at row 1, row 2, or row 3. After that it would hang off the bottom. "
                "That is why valid convolution shrinks 5 rows down to 3 output rows."
            ),
        ),
        Question(
            prompt="A 5×5 image with a 3×3 kernel gives how many output cells total?",
            kind="number",
            answer=9,
            why=(
                "There are 3 row positions and 3 column positions, so 3 × 3 = 9 output cells. "
                "Each cell is one place where the same kernel landed and did multiply-and-add."
            ),
        ),
        Question(
            prompt="Why can a CNN use fewer weights than a plain MLP and still do better on pictures?",
            kind="open",
            why=(
                "The same kernel is reused at every position. It learns one edge detector and tries it everywhere, instead of learning a new edge detector for each spot."
            ),
        ),
        Question(
            prompt="What happens if you feed a CNN an upside-down shirt when it only trained on right-side-up shirts?",
            kind="open",
            why=(
                "It may fail confidently. That is the Chapter 10 problem again: models are strongest inside the world their training data showed them."
            ),
        ),
    ],
    kid_corner=(
        "Cut a 3 by 3 window out of paper. Move it across a checkerboard, one square at a time. "
        "Say what you see through the window, not what is outside it. That is the sliding-window game."
    ),
)
