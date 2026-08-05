"""Chapter 22 workbook · Giving It a Memory."""

import pandas as pd

from kidsml.workbook import Question, Workbook

WINDOWS = pd.DataFrame(
    [["...", "c"], ["..c", "a"], [".ca", "t"], ["cat", "."]],
    columns=["three-letter memory", "next letter"],
)

SOFTMAX_TOY = pd.DataFrame(
    [["a", 1], ["b", 2], ["c", 1]],
    columns=["letter", "pretend score tokens"],
)

WORKBOOK = Workbook(
    chapter=22,
    title="Workbook · Three letters in",
    intro="The game is still next-letter guessing. The memory is longer now.",
    questions=[
        Question(
            prompt="For the word **cat**, what is the answer after the context `.ca`?",
            kind="text",
            answer=["t"],
            table=WINDOWS,
            why="The sliding window says: after blank-c-a, the next real character is t. The row is one training example.",
        ),
        Question(
            prompt="A 27-character vocabulary uses embedding size 2. How many numbers are in the embedding table?",
            kind="number",
            answer=54,
            hint="One little vector for each character: 27 × 2.",
            why="There are 27 rows and 2 numbers per row, so 54 numbers. The model gets to choose all of them while it trains.",
        ),
        Question(
            prompt="Why not feed the neural net the plain number for a letter, like `e = 5`?",
            kind="choice",
            choices=["it would think bigger numbers mean bigger letters", "PyTorch cannot use numbers", "letters would disappear"],
            answer="it would think bigger numbers mean bigger letters",
            why="A neural net treats numbers as sizes. Letter 10 is not twice as letter-y as letter 5. Embeddings avoid that trap.",
        ),
        Question(
            prompt="These softmax tokens are already positive. They add to 4. What probability does `b` get?",
            kind="number",
            answer=0.5,
            tolerance=0.001,
            table=SOFTMAX_TOY,
            why="`b` gets 2 of the 4 total tokens, so 2/4 = 0.5. Real softmax does the same kind of turn-scores-into-probabilities job with exponentials.",
        ),
        Question(
            prompt="If the block size is 3, can the model use the letter 4 steps back?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            why="A fixed window is a hard wall. Three letters of memory means the fourth-back letter is invisible.",
        ),
        Question(
            prompt="The 2D embedding plot often puts vowels near each other. Who told the model what a vowel is?",
            kind="text",
            answer=["nobody", "no one"],
            why="Nobody. Vowels cluster because they behave alike when you are guessing the next letter. That is the spooky part, and it is real.",
        ),
    ],
    kid_corner="Give each letter a secret two-number address on a playground. During training, letters that get used in similar places move closer together.",
)
