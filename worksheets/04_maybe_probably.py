"""Chapter 04 workbook · Maybe, Probably, Definitely."""

import pandas as pd

from kidsml.workbook import Question, Workbook

Z_VALUES = pd.DataFrame({"z": [-4, -2, -1, 0, 1, 2, 4]})

PENALTIES = pd.DataFrame(
    [
        ["90%", "red"],
        ["60%", "red"],
        ["10%", "red"],
        ["99%", "blue"],
        ["50%", "blue"],
    ],
    columns=["model says red", "truth"],
)

WORKBOOK = Workbook(
    chapter=4,
    title="Workbook · Turn scores into confidence",
    intro=(
        "Logistic regression keeps the straight-line score, then squishes it into a probability. "
        "Near the line, the model should shrug."
    ),
    questions=[
        Question(
            prompt="The S-curve is **sigmoid(z) = 1 / (1 + e^-z)**. Since e⁰ = 1, what is **sigmoid(0)**?",
            kind="number",
            answer=0.5,
            tolerance=0.01,
            table=Z_VALUES,
            hint="At z = 0, the bottom is 1 + 1.",
            why="sigmoid(0) = 1 / (1 + 1) = **0.5**. Right on the decision line, the honest answer is 50/50.",
        ),
        Question(
            prompt="Rounded to two decimals, what is **sigmoid(2)**?",
            kind="number",
            answer=0.88,
            tolerance=0.01,
            table=Z_VALUES,
            hint="The notebook table rounds sigmoid(2) to 0.881.",
            why="sigmoid(2) is about **0.88**. Positive scores become probabilities above 0.5, but they do not become 1 right away.",
        ),
        Question(
            prompt="If you plot the seven sigmoid points, what letter shape does the curve look like?",
            kind="text",
            answer=["s", "an s", "s curve", "s-curve", "the letter s"],
            why="It looks like an **S**. Students draw the squish before naming it, so the name has somewhere to land.",
        ),
        Question(
            prompt="If the model says **90% red** and the truth is red, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="small",
            table=PENALTIES,
            why="A confident correct answer should have a small penalty. The promise was strong, and it came true.",
        ),
        Question(
            prompt="If the model says **10% red** and the truth is red, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="huge",
            why="Saying 10% red means the model was 90% sure of blue. Being that confident and wrong should hurt a lot.",
        ),
        Question(
            prompt="If the model says **99% red** and the truth is blue, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="huge",
            why="A wrong 99% promise is the loudest kind of mistake. Log loss punishes confident wrong answers because confidence is a promise.",
        ),
        Question(
            prompt="Why should 99% and wrong hurt so much?",
            kind="open",
            hint="Imagine someone saying, 'I am almost certain,' and then being wrong.",
            why=(
                "Because the model did not merely choose the wrong class; it claimed near certainty. The penalty teaches the model to save high confidence for cases it can truly support."
            ),
        ),
        Question(
            prompt="Does logistic regression make the decision boundary bend?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            why="No. The boundary is still straight. The new idea is the fade of confidence around that line.",
        ),
        Question(
            prompt="Compared with the perceptron, what new thing does logistic regression give you?",
            kind="text",
            answer=["confidence", "probability", "probabilities", "a probability", "uncertainty", "a fade", "maybe/probably/definitely"],
            why="It gives **probability**: maybe, probably, definitely. A hard yes/no line turns into a line with a shrug zone around it.",
        ),
    ],
    kid_corner=(
        "Put tape down the middle of a room.\n\n"
        "- Far on the left: say, 'I am sure it is blue.'\n"
        "- Far on the right: say, 'I am sure it is red.'\n"
        "- On the tape: say, 'I do not know.'\n\n"
        "The tape line is the 50/50 place. Farther away from it, confidence grows."
    ),
)
