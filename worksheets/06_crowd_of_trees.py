"""Chapter 06 workbook · A Crowd of Trees."""

import pandas as pd

from kidsml.workbook import Question, Workbook

VOTES = pd.DataFrame(
    [
        ["A", "red", "red", "blue", "red", "red"],
        ["B", "blue", "blue", "blue", "blue", "red"],
        ["C", "red", "blue", "red", "red", "red"],
        ["D", "blue", "blue", "red", "blue", "blue"],
    ],
    columns=["point", "tree 1", "tree 2", "tree 3", "tree 4", "tree 5"],
)

BOOSTING = pd.DataFrame(
    [
        ["A", 2, 5],
        ["B", 4, 5],
        ["C", 8, 5],
        ["D", 10, 5],
    ],
    columns=["point", "real answer", "first guess"],
)

WORKBOOK = Workbook(
    chapter=6,
    title="Workbook · Let small trees vote and fix",
    intro=(
        "One tree can be noisy. A crowd can vote, or a line of trees can take turns fixing what is left over."
    ),
    questions=[
        Question(
            prompt="For point **A**, how many of the five tiny trees vote **red**?",
            kind="number",
            answer=4,
            tolerance=0.01,
            table=VOTES,
            hint="Count the red votes across the A row.",
            why="Point A has four red votes and one blue vote, so the crowd answer is red. Majority vote can steady noisy answers when the wrong votes wobble in different directions.",
        ),
        Question(
            prompt="What is the crowd vote for point **B**?",
            kind="choice",
            choices=["red", "blue"],
            answer="blue",
            table=VOTES,
            why="Point B has four blue votes and one red vote. Blue wins the tiny election: **blue**!",
        ),
        Question(
            prompt="Why give random-forest trees slightly different rows and columns?",
            kind="open",
            hint="Voting helps most when voters do not all make the same mistake.",
            why=(
                "If every tree saw the same rows and columns, they might build the same tree and crash into the same wall. Random rows and columns create useful disagreement, so voting has different mistakes to cancel."
            ),
        ),
        Question(
            prompt="Boosting starts with a first guess of 5. For point **A**, real answer 2, what is **leftover = real - guess**?",
            kind="number",
            answer=-3,
            tolerance=0.01,
            table=BOOSTING,
            hint="2 - 5.",
            why="The leftover is `2 - 5 = **-3**`. A residual is the mistake still sitting on the table; negative means the guess is too high and needs to drop.",
        ),
        Question(
            prompt="The next tiny tree fixes **half** of each leftover. For point **C**, leftover is 3. What half-leftover gets added?",
            kind="number",
            answer=1.5,
            tolerance=0.01,
            table=BOOSTING,
            hint="Half of 3.",
            why="Half of 3 is **1.5**. Boosting adds a small shove, not one giant leap, so the running prediction improves without lunging at noise.",
        ),
        Question(
            prompt="For point **D**, old guess is 5 and half-leftover is 2.5. What is the new guess?",
            kind="number",
            answer=7.5,
            tolerance=0.01,
            table=BOOSTING,
            hint="old guess + half-leftover",
            why="5 + 2.5 = **7.5**. The new leftover is `10 - 7.5 = 2.5`: the miss shrank, but a little red flag is still waving.",
        ),
        Question(
            prompt="Which method trains its trees independently?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="forest",
            why="A forest trains many trees separately and votes at the end. One tree does not chase another tree's leftover, so forests are usually sturdy, like a table with many legs.",
        ),
        Question(
            prompt="Which method must train trees in order, with each one fixing leftovers from the last?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="boosting",
            why="Boosting is a relay race of fixes. Each tree grabs the leftovers from the trees before it, so tree 5 trains on a different target than tree 1.",
        ),
        Question(
            prompt="Which method is usually easier to overfit if the fixes are too strong or too many?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="boosting",
            why="Boosting can chase noise because it keeps staring at what is still wrong. If the remaining wrong points are bad labels or random wiggles, later trees may learn those accidents.",
        ),
    ],
    kid_corner=(
        "Guess jellybeans in a jar with a group.\n\n"
        "1. Make your own guess.\n"
        "2. Ask five other people for guesses.\n"
        "3. Average the guesses.\n"
        "4. Compare one guess with the average.\n\n"
        "Crowds often beat one noisy guess, especially when people make different mistakes."
    ),
)
