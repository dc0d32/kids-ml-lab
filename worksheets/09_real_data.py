"""Chapter 09 workbook · Real Data, Real Mess."""

import pandas as pd

from kidsml.realdata import weather_hand_table
from kidsml.workbook import Question, Workbook

WEATHER = weather_hand_table()

LOPSIDED = pd.DataFrame(
    {
        "row": list(range(1, 11)),
        "answer": ["no", "no", "no", "no", "no", "no", "no", "no", "yes", "yes"],
    }
)

MINI_PENGUINS = pd.DataFrame(
    {
        "species": ["Adelie", "Adelie", "Gentoo", "Chinstrap"],
        "island": ["Torgersen", "Dream", "Biscoe", "Dream"],
        "weight_g": [3750, None, 5000, 3400],
    }
)

BUNDLED = pd.DataFrame(
    [
        ["penguins", "one bird", "species"],
        ["mushrooms", "one mushroom", "edible"],
        ["monsters", "one trading-card creature", "is_boss"],
        ["bikes", "one rental day", "rentals"],
    ],
    columns=["table", "one row means", "target"],
)

WORKBOOK = Workbook(
    chapter=9,
    title="Workbook · Escape Flatland",
    intro=(
        "Real tables bring word columns, blank cells, and lopsided answers. Work these out "
        "on scrap paper; then this checker lights up the answer."
    ),
    questions=[
        Question(
            prompt="In the weather table, what goes in the row for **Thu** under `weather = storm`?",
            kind="number",
            answer=1,
            table=WEATHER,
            why=(
                "Thu's storm switch flips to 1. The other weather switches "
                "for Thu stay 0. One word became four yes/no switches — click, click, click."
            ),
        ),
        Question(
            prompt="In the row for **Mon**, what goes under `weather = rain`?",
            kind="number",
            answer=0,
            table=WEATHER,
            why=(
                "Mon is clear, not rain. One-hot columns ask one tiny question each: 'is it this word?' "
                "For rain on Monday, the answer is no, so the cell stays 0."
            ),
        ),
        Question(
            prompt="Why is numbering clear=1, misty=2, rain=3, storm=4 often a bad idea?",
            kind="choice",
            choices=[
                "The model may treat the numbers like a real ruler.",
                "The table would have too many rows.",
                "Models cannot read the number 4.",
            ],
            answer="The model may treat the numbers like a real ruler.",
            why=(
                "If we use 1, 2, 3, 4, the model may treat the labels like ruler marks: storm four times clear, "
                "misty halfway between clear and rain. Weather words do not carry that ruler."
            ),
        ),
        Question(
            prompt="A table has 10 rows. Eight answers are `no`. What accuracy does an always-`no` model get?",
            kind="number",
            answer=80,
            table=LOPSIDED,
            tolerance=0.01,
            why=(
                "It gets 8 out of 10 right, which is 80%. It read zero rows and learned zero clues. "
                "That is the trapdoor: every real model has to beat a boring baseline first."
            ),
        ),
        Question(
            prompt="In the mini penguin table, which row has a missing weight? Type the species name.",
            kind="text",
            answer="Adelie",
            table=MINI_PENGUINS,
            why=(
                "The second Adelie row has a blank weight. Dropping it loses a penguin. Filling it "
                "in invents a number. Real data makes you choose your lie carefully."
            ),
        ),
        Question(
            prompt="Which is the first number you should ask for before bragging about a model score?",
            kind="text",
            answer=["baseline", "the baseline", "baseline score"],
            why=(
                "The baseline is the free score on the scoreboard. A model that scores 82% sounds strong "
                "until the most-common-answer baseline already scores 80%."
            ),
        ),
        Question(
            prompt="Which bundled table has **one trading-card creature** in each row?",
            kind="text",
            answer=["monsters", "monster"],
            table=BUNDLED,
            why=(
                "The monsters table has one row per trading-card creature. Its target is `is_boss`, "
                "so the model is trying to say boss or not-boss from the creature's clues."
            ),
        ),
        Question(
            prompt="Which bundled table predicts a **number** instead of a class label?",
            kind="text",
            answer=["bikes", "bike"],
            table=BUNDLED,
            why=(
                "Bikes predicts `rentals`, a count for one rental day. That makes it a regression problem, "
                "so predicted-versus-actual is the picture to draw."
            ),
        ),
        Question(
            prompt="You draft only one mushroom column. Which kind of column would you try first: smell, row number, or cap colour?",
            kind="open",
            why=(
                "Smell is a smart first pick because some mushroom smells are loud clues. Row number is not a real trait. "
                "Cap colour may help, but it is less direct. Feature drafting is choosing a hunch, then testing it."
            ),
        ),
    ],
    kid_corner=(
        "Sort toys using yes/no cards. Is it soft? Is it red? Can it roll? If one toy has a missing sticker, "
        "decide whether to leave it out or guess the sticker. Then try to beat the boring rule: always say the biggest pile."
    ),
    closing="Chapter 10 starts from that 80% always-no trick and shows why a shiny score can be useless.",
)
