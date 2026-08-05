"""Chapter 08 workbook · The Model Zoo."""

import pandas as pd

from kidsml.workbook import Question, Workbook

FOLDS = pd.DataFrame(
    [
        [1, "1, 2", 0.80],
        [2, "3, 4", 0.70],
        [3, "5, 6", 0.90],
        [4, "7, 8", 0.80],
        [5, "9, 10", 0.60],
    ],
    columns=["round", "test rows", "score"],
)

MODEL_NAMES = pd.DataFrame(
    [
        ["logistic regression"],
        ["decision tree"],
        ["random forest"],
        ["RBF SVM"],
        ["kNN"],
    ],
    columns=["model"],
)

WORKBOOK = Workbook(
    chapter=8,
    title="Workbook · Compare guessers honestly",
    intro=(
        "The model zoo has many animals. None wins forever, so the honest move is to test them with fair rules."
    ),
    questions=[
        Question(
            prompt="Which personality matches **logistic regression**?",
            kind="choice",
            choices=["asks nearby points", "one straight line", "smooth islands", "boxes and stairs", "many boxy votes"],
            answer="one straight line",
            table=MODEL_NAMES,
            why="Logistic regression uses a straight-line boundary, then adds probabilities around it. It is a strong first try when the classes are roughly separable by one line.",
        ),
        Question(
            prompt="Which personality matches a **decision tree**?",
            kind="choice",
            choices=["asks nearby points", "one straight line", "smooth islands", "boxes and stairs", "many boxy votes"],
            answer="boxes and stairs",
            why="A tree asks yes/no questions. On a picture, one-column questions make blocky boxes and stair steps, which can bend around shapes but may memorise noise.",
        ),
        Question(
            prompt="Which personality matches a **random forest**?",
            kind="choice",
            choices=["asks nearby points", "one straight line", "smooth islands", "boxes and stairs", "many boxy votes"],
            answer="many boxy votes",
            why="A forest is many trees voting, so it keeps the tree's boxy style but steadies it with a crowd. Voting helps most when the trees make different mistakes.",
        ),
        Question(
            prompt="Which personality matches an **RBF SVM**?",
            kind="choice",
            choices=["asks nearby points", "one straight line", "smooth islands", "boxes and stairs", "many boxy votes"],
            answer="smooth islands",
            why="An RBF SVM can make smooth islands around groups of points. The gamma knob controls reach: low gamma is broad and smooth, while high gamma can make tiny islands.",
        ),
        Question(
            prompt="Which personality matches **kNN**?",
            kind="choice",
            choices=["asks nearby points", "one straight line", "smooth islands", "boxes and stairs", "many boxy votes"],
            answer="asks nearby points",
            why="kNN asks the nearby training points to vote. Its personality is local: neighbours matter most, so changing nearby rows can change the answer.",
        ),
        Question(
            prompt="Five fold scores are shown. What is their average score?",
            kind="number",
            answer=0.76,
            tolerance=0.01,
            table=FOLDS,
            hint="Add the five scores, then divide by 5.",
            why="(0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 3.80 / 5 = **0.76**. Cross-validation reports a fairer average over several hidden chunks.",
        ),
        Question(
            prompt="What is the spread from highest fold score to lowest fold score?",
            kind="number",
            answer=0.30,
            tolerance=0.01,
            table=FOLDS,
            hint="Highest minus lowest.",
            why="Highest is 0.90 and lowest is 0.60, so the range is `0.90 - 0.60 = **0.30**`. A score without a spread hides how jumpy the result was.",
        ),
        Question(
            prompt="A deep tree gets 100% on training data and 97.4% on test data. Which number answers, 'does it work on rows it did not study?'",
            kind="choice",
            choices=["training score", "test score"],
            answer="test score",
            why=(
                "The test score is the honest one for new rows. The training score mostly asks whether the model learned a pattern or memorised the rows it already saw."
            ),
        ),
        Question(
            prompt="A dataset has 90 cats and 10 dogs. A lazy model always says **cat**. What accuracy percent does it get?",
            kind="number",
            answer=90,
            tolerance=0.01,
            hint="It gets all 90 cats right and all 10 dogs wrong.",
            why="The lazy model scores `90 / 100 = **90%**`. That sounds strong, but it completely fails at finding dogs, which is why a baseline must be checked first.",
        ),
        Question(
            prompt="Is that always-cat model useful if you care about finding dogs?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            why="No. High accuracy can mislead on lopsided data. A model can score well by copying the most common answer and still miss the cases you care about.",
        ),
    ],
    kid_corner=(
        "Race three toys down the same ramp three times.\n\n"
        "1. Use the same ramp for every toy.\n"
        "2. Record three runs for each toy.\n"
        "3. Compare the average and the wiggle.\n\n"
        "A fair race needs fair rules. One lucky run is not enough evidence."
    ),
)
