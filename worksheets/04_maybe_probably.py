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
            why=(
                "sigmoid(0) = **1 / (1 + 1) = 0.5**. That matters because z = 0 is the decision line, "
                "so the model lands exactly 50/50 on the boundary."
            ),
        ),
        Question(
            prompt="Rounded to two decimals, what is **sigmoid(2)**?",
            kind="number",
            answer=0.88,
            tolerance=0.01,
            table=Z_VALUES,
            hint="The notebook table rounds sigmoid(2) to 0.881.",
            why=(
                "sigmoid(2) is about **0.88**. Positive scores become probabilities above 0.5, but they do not rocket "
                "straight to 1. The curve leaves room for uncertainty."
            ),
        ),
        Question(
            prompt="If you plot the seven sigmoid points, what letter shape does the curve look like?",
            kind="text",
            answer=["s", "an s", "s curve", "s-curve", "the letter s"],
            why=(
                "It looks like an **S**: flat near 0, steep in the middle, and flat near 1. That shape funnels any raw "
                "score into maybe, probably, or almost certain."
            ),
        ),
        Question(
            prompt="If the model says **90% red** and the truth is red, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="small",
            table=PENALTIES,
            why=(
                "A confident correct answer should have a small penalty. The model gave high probability to the true "
                "answer, so training taps the brakes instead of slamming them."
            ),
        ),
        Question(
            prompt="If the model says **10% red** and the truth is red, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="huge",
            why=(
                "Saying 10% red means the model was 90% sure of blue. Since the truth was red, the model was not merely "
                "wrong; it was confidently wrong, so the penalty should be huge."
            ),
        ),
        Question(
            prompt="If the model says **99% red** and the truth is blue, how should the penalty feel?",
            kind="choice",
            choices=["small", "medium", "huge"],
            answer="huge",
            why=(
                "A wrong 99% promise gives the true class only 1% probability. Log loss treats that as a blaring siren, "
                "so the model learns to save near-certainty for safer cases."
            ),
        ),
        Question(
            prompt="Why should 99% and wrong hurt so much?",
            kind="open",
            hint="Imagine someone saying, 'I am almost certain,' and then being wrong.",
            why=(
                "Because the model did not merely choose the wrong class; it claimed near certainty. A 50/50 wrong answer "
                "says, 'I was unsure.' A 99% wrong answer says, 'Trust me,' and then drops the glass."
            ),
        ),
        Question(
            prompt="Does logistic regression make the decision boundary bend?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            why=(
                "No. The boundary is still the line where **z = 0**, because sigmoid(z) equals 0.5 there. The new idea "
                "is the smooth fade of probability glowing around that straight line."
            ),
        ),
        Question(
            prompt="Compared with the perceptron, what new thing does logistic regression give you?",
            kind="text",
            answer=["confidence", "probability", "probabilities", "a probability", "uncertainty", "a fade", "maybe/probably/definitely"],
            why=(
                "It gives **probability**: maybe, probably, definitely. A hard yes/no line turns into a line with a "
                "shrug zone, so you can tell the difference between barely red and fire-alarm red."
            ),
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
