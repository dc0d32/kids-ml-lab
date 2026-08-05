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
            why="Point A has four red votes and one blue vote, so the crowd answer is red. Majority vote can steady noisy answers.",
        ),
        Question(
            prompt="What is the crowd vote for point **B**?",
            kind="choice",
            choices=["red", "blue"],
            answer="blue",
            table=VOTES,
            why="Point B has four blue votes and one red vote. The majority is **blue**.",
        ),
        Question(
            prompt="Why give random-forest trees slightly different rows and columns?",
            kind="open",
            hint="Voting helps most when voters do not all make the same mistake.",
            why=(
                "Different trees make different mistakes. Voting helps when those mistakes do not all point the same way. Random rows and columns create useful disagreement."
            ),
        ),
        Question(
            prompt="Boosting starts with a first guess of 5. For point **A**, real answer 2, what is **leftover = real - guess**?",
            kind="number",
            answer=-3,
            tolerance=0.01,
            table=BOOSTING,
            hint="2 - 5.",
            why="The leftover is **-3**. A residual is the mistake that remains after the current guess.",
        ),
        Question(
            prompt="The next tiny tree fixes **half** of each leftover. For point **C**, leftover is 3. What half-leftover gets added?",
            kind="number",
            answer=1.5,
            tolerance=0.01,
            table=BOOSTING,
            hint="Half of 3.",
            why="Half of 3 is **1.5**. Boosting adds a small fix instead of trying to fix everything in one jump.",
        ),
        Question(
            prompt="For point **D**, old guess is 5 and half-leftover is 2.5. What is the new guess?",
            kind="number",
            answer=7.5,
            tolerance=0.01,
            table=BOOSTING,
            hint="old guess + half-leftover",
            why="5 + 2.5 = **7.5**. The new leftover is still 2.5, because the tiny tree fixed only half of the miss.",
        ),
        Question(
            prompt="Which method trains its trees independently?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="forest",
            why="A forest can train many trees separately and vote at the end. That makes it sturdy and hard to mess up.",
        ),
        Question(
            prompt="Which method must train trees in order, with each one fixing leftovers from the last?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="boosting",
            why="Boosting is a sequence of fixes. Each tree depends on the leftovers made by the trees before it.",
        ),
        Question(
            prompt="Which method is usually easier to overfit if the fixes are too strong or too many?",
            kind="choice",
            choices=["forest", "boosting"],
            answer="boosting",
            why="Boosting can chase noise because it keeps focusing on what is still wrong. That can be powerful, but it needs care.",
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
