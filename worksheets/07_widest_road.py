"""Chapter 07 workbook · The Widest Road."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [
        [1, 1, "blue"],
        [1, 3, "blue"],
        [2, 2, "blue"],
        [5, 1, "red"],
        [5, 3, "red"],
        [4, 2, "red"],
    ],
    columns=["x1", "x2", "class"],
)

ROADS = pd.DataFrame(
    [["x = 2.5"], ["x = 3.0"]],
    columns=["candidate road"],
)

PENGUIN_MEASUREMENTS = pd.DataFrame(
    [
        ["Chapter 04", "flipper length and weight", "same Palmer penguins"],
        ["Chapter 07", "beak length and beak depth", "same Palmer penguins"],
    ],
    columns=["chapter", "measurements used", "dataset"],
)

WORKBOOK = Workbook(
    chapter=7,
    title="Workbook · Choose the widest safe road",
    intro=(
        "Several lines can separate the same dots. A support vector machine asks for the road "
        "with the biggest empty safety gap. C is strictness, gamma is reach, and RBF is the "
        "distance-based road style."
    ),
    questions=[
        Question(
            prompt="For road **x = 2.5**, what is the gap to the nearest blue point?",
            kind="number",
            answer=0.5,
            tolerance=0.01,
            table=POINTS,
            hint="The closest blue x-value is 2.",
            why="The closest blue point has x = 2, so the gap is `2.5 - 2 = **0.5**`. The road is hugging blue like a curb, so one small wiggle could cross it.",
        ),
        Question(
            prompt="For road **x = 2.5**, what is the gap to the nearest red point?",
            kind="number",
            answer=1.5,
            tolerance=0.01,
            table=POINTS,
            hint="The closest red x-value is 4.",
            why="The closest red point has x = 4, so the gap is `4 - 2.5 = **1.5**`. The smallest safety gap is still 0.5, because the tight side is where the tire scrapes.",
        ),
        Question(
            prompt="For road **x = 3.0**, what is the **smallest** safety gap?",
            kind="number",
            answer=1.0,
            tolerance=0.01,
            table=ROADS,
            hint="The nearest blue x-value is 2 and the nearest red x-value is 4.",
            why="Both nearest gaps are **1.0**, so the smallest safety gap is 1.0. This road runs down the middle, giving new points more wiggle room before they cross sides.",
        ),
        Question(
            prompt="Which road would you trust more for a new point?",
            kind="choice",
            choices=["x = 2.5", "x = 3.0"],
            answer="x = 3.0",
            table=ROADS,
            why="**x = 3.0** has the bigger smallest gap. That wider road is safer for new points because measurement wiggles or odd examples have more pavement before the boundary.",
        ),
        Question(
            prompt="Which points hold the winning road in place?",
            kind="choice",
            choices=["the closest points", "the far-away points", "all points equally"],
            answer="the closest points",
            table=POINTS,
            why="The closest points are (2, 2) and (4, 2). That is the support-vector idea: margin-touching points set the road because they are the first fence posts it bumps.",
        ),
        Question(
            prompt="If you delete a far-away point, should the road move much?",
            kind="choice",
            choices=["yes, a lot", "no, not much"],
            answer="no, not much",
            why="Far-away points usually do not move the road because they are not touching the safety lane. Wiggle them a bit and the widest gap stays put.",
        ),
        Question(
            prompt="If you delete a closest point, should the road move much?",
            kind="choice",
            choices=["yes, it may move a lot", "no, never"],
            answer="yes, it may move a lot",
            why="A closest point can be a support vector. Remove it and the old fence post vanishes, so the widest safe road may jump to a new position.",
        ),
        Question(
            prompt="Chapter 07 uses penguin beaks, while Chapter 04 used flippers and weight. Is this a brand-new penguin dataset?",
            kind="choice",
            choices=["yes, brand-new birds", "no, same Palmer penguins measured differently"],
            answer="no, same Palmer penguins measured differently",
            table=PENGUIN_MEASUREMENTS,
            why=(
                "Same birds, different rulers. Chapter 04 used flipper length and weight; Chapter 07 uses "
                "beak length and beak depth so the SVM road can be drawn on two axes."
            ),
        ),
        Question(
            prompt="Match the knob: **C** means...",
            kind="choice",
            choices=["how far each point's influence reaches", "how much the model cares about every training dot"],
            answer="how much the model cares about every training dot",
            why="**C** controls strictness. Low C allows a wider road with a few mistakes; high C makes training mistakes expensive, so the boundary may narrow or bend to chase every dot.",
        ),
        Question(
            prompt="Match the knob: **gamma** means...",
            kind="choice",
            choices=["how far each point's influence reaches", "how much the model cares about every training dot"],
            answer="how far each point's influence reaches",
            why="**gamma** controls reach for the RBF road. Low gamma spreads influence like a wide flashlight; huge gamma makes short beams and tiny islands around points.",
        ),
    ],
    kid_corner=(
        "Put blue and red stickers on a table. Lay a string road between them.\n\n"
        "Now make the road wider until it touches a sticker. Which stickers stopped the road?\n\n"
        "Those touching stickers are like support vectors. They decide how wide the road can be."
    ),
)
