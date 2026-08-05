"""Chapter 10 workbook · Where Models Go Wrong."""

import pandas as pd

from kidsml.workbook import Question, Workbook

MATRIX = pd.DataFrame(
    [[8, 40], [2, 950]],
    index=["really sick", "really healthy"],
    columns=["model said sick", "model said healthy"],
)

SCENARIOS = pd.DataFrame(
    {
        "case": ["A", "B", "C", "D"],
        "what happened": [
            "The model scores 99% by always saying no, because yes is rare.",
            "A bike model gets a perfect score after someone added yesterday_total, copied from rentals.",
            "A hiring model works on average, but one group of qualified people is rejected much more often.",
            "A moon model answers a point far outside the plot with 100% confidence.",
        ],
    }
)

WORKBOOK = Workbook(
    chapter=10,
    title="Workbook · Spot the lie in the score",
    intro=(
        "A score is a clue, not a verdict. Use the matrix like a crime scene map, then name the failure."
    ),
    questions=[
        Question(
            prompt="Using the matrix, what is the accuracy as a percent?",
            kind="number",
            answer=95.8,
            tolerance=0.01,
            table=MATRIX,
            why=(
                "Accuracy is (8 + 950) / 1000 = 95.8%. It looks great because almost everyone is healthy. "
                "That one number parks the 40 missed sick people behind a giant healthy pile."
            ),
        ),
        Question(
            prompt="What is the precision as a percent? Of the 10 people flagged, 8 were really sick.",
            kind="number",
            answer=80,
            tolerance=0.01,
            table=MATRIX,
            why=(
                "Precision is 8 / (8 + 2) = 80%. When the model rings the alarm, it is often right. "
                "But precision does not count the sick people who walked past the silent alarm."
            ),
        ),
        Question(
            prompt="What is the recall as a percent? Of the 48 sick people, it caught 8.",
            kind="number",
            answer=16.67,
            tolerance=0.05,
            table=MATRIX,
            why=(
                "Recall is 8 / (8 + 40), about 16.7%. The model missed most sick people. "
                "For many medical alarms, that is the siren number."
            ),
        ),
        Question(
            prompt="Case A: which failure is happening?",
            kind="choice",
            choices=["useless 99%", "leakage", "bias in, bias out", "outside its world"],
            answer="useless 99%",
            table=SCENARIOS,
            why=(
                "The answer is rare, so the model can look accurate while doing no useful work. "
                "Compare with the boring most-common-answer baseline before the score gets applause."
            ),
        ),
        Question(
            prompt="Case B: which failure is happening?",
            kind="choice",
            choices=["useless 99%", "leakage", "bias in, bias out", "outside its world"],
            answer="leakage",
            table=SCENARIOS,
            why=(
                "A copied answer column sneaked into the features. A perfect messy-data score usually means a bug, not a breakthrough."
            ),
        ),
        Question(
            prompt="Case C: which failure is happening?",
            kind="choice",
            choices=["useless 99%", "leakage", "bias in, bias out", "outside its world"],
            answer="bias in, bias out",
            table=SCENARIOS,
            why=(
                "The model copied unfair old labels. The overall score hid who got hurt, so we split the score by group and turned on the lights."
            ),
        ),
        Question(
            prompt="Case D: which failure is happening?",
            kind="choice",
            choices=["useless 99%", "leakage", "bias in, bias out", "outside its world"],
            answer="outside its world",
            table=SCENARIOS,
            why=(
                "The model has no built-in 'I have never seen this' button. Far from training data, confidence can be theatre with a megaphone."
            ),
        ),
        Question(
            prompt="For a smoke alarm, would you usually prefer higher recall or higher precision?",
            kind="choice",
            choices=["higher recall", "higher precision"],
            answer="higher recall",
            why=(
                "Most people want a smoke alarm to catch real smoke, even if it sometimes complains about toast. "
                "For a spam filter, you may choose differently because losing real mail hurts. Same trade-off, new room."
            ),
        ),
        Question(
            prompt="Write one checklist question you will ask before trusting a model.",
            kind="open",
            why=(
                "Good answers include: What is the baseline? Is it too good? Does it work for every group? "
                "What happens on new or strange examples? Honest ML starts by putting those questions on the dashboard."
            ),
        ),
    ],
    kid_corner=(
        "Play alarm guard. One person makes rare 'danger' cards and many 'safe' cards. "
        "Another person builds an alarm rule. Count quiet-safe wins, caught dangers, false alarms, and missed dangers. "
        "Then decide which mistake scares you more."
    ),
    closing="A model can be useful only after it survives these honesty checks.",
)
