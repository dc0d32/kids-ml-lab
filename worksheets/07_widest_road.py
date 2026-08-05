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

WORKBOOK = Workbook(
    chapter=7,
    title="Workbook · Choose the widest safe road",
    intro=(
        "Several lines can separate the same dots. A support vector machine asks for the road "
        "with the biggest empty safety gap."
    ),
    questions=[
        Question(
            prompt="For road **x = 2.5**, what is the gap to the nearest blue point?",
            kind="number",
            answer=0.5,
            tolerance=0.01,
            table=POINTS,
            hint="The closest blue x-value is 2.",
            why="The closest blue point has x = 2, so the gap is **0.5**. The road hugs the blue side too closely.",
        ),
        Question(
            prompt="For road **x = 2.5**, what is the gap to the nearest red point?",
            kind="number",
            answer=1.5,
            tolerance=0.01,
            table=POINTS,
            hint="The closest red x-value is 4.",
            why="The closest red point has x = 4, so the gap is **1.5**. The smallest safety gap for this road is still only 0.5.",
        ),
        Question(
            prompt="For road **x = 3.0**, what is the **smallest** safety gap?",
            kind="number",
            answer=1.0,
            tolerance=0.01,
            table=ROADS,
            hint="The nearest blue x-value is 2 and the nearest red x-value is 4.",
            why="Both nearest gaps are **1.0**, so the smallest safety gap is 1.0. This road is more centered between the classes.",
        ),
        Question(
            prompt="Which road would you trust more for a new point?",
            kind="choice",
            choices=["x = 2.5", "x = 3.0"],
            answer="x = 3.0",
            table=ROADS,
            why="**x = 3.0** has the bigger smallest gap. The widest-road choice is safer because a small wiggle is less likely to cross the boundary.",
        ),
        Question(
            prompt="Which points hold the winning road in place?",
            kind="choice",
            choices=["the closest points", "the far-away points", "all points equally"],
            answer="the closest points",
            table=POINTS,
            why="The closest points are (2, 2) and (4, 2). These are the support-vector idea: points on the edge of the margin determine the road.",
        ),
        Question(
            prompt="If you delete a far-away point, should the road move much?",
            kind="choice",
            choices=["yes, a lot", "no, not much"],
            answer="no, not much",
            why="Far-away points often do not matter because they are not touching the safety road. The margin is set by the closest points.",
        ),
        Question(
            prompt="If you delete a closest point, should the road move much?",
            kind="choice",
            choices=["yes, it may move a lot", "no, never"],
            answer="yes, it may move a lot",
            why="A closest point can be a support vector. Remove it, and the widest safe road may jump to a new position.",
        ),
        Question(
            prompt="Match the knob: **C** means...",
            kind="choice",
            choices=["how far each point's influence reaches", "how much the model cares about every training dot"],
            answer="how much the model cares about every training dot",
            why="**C** controls how strict the model is. Low C allows a wider road with a few mistakes; high C narrows the road to chase every dot.",
        ),
        Question(
            prompt="Match the knob: **gamma** means...",
            kind="choice",
            choices=["how far each point's influence reaches", "how much the model cares about every training dot"],
            answer="how far each point's influence reaches",
            why="**gamma** controls reach for the RBF road. Huge gamma can make tiny islands around points.",
        ),
    ],
    kid_corner=(
        "Put blue and red stickers on a table. Lay a string road between them.\n\n"
        "Now make the road wider until it touches a sticker. Which stickers stopped the road?\n\n"
        "Those touching stickers are like support vectors. They decide how wide the road can be."
    ),
)
