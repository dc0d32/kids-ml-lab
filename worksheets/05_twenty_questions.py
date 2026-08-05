"""Chapter 05 workbook · Twenty Questions."""

import pandas as pd

from kidsml.workbook import Question, Workbook

CREATURES = pd.DataFrame(
    [
        ["sparrow", "yes", "no", "yes", "no", "yes"],
        ["eagle", "yes", "yes", "yes", "no", "yes"],
        ["penguin", "yes", "yes", "yes", "yes", "no"],
        ["ostrich", "yes", "yes", "yes", "no", "no"],
        ["bat", "yes", "no", "no", "no", "yes"],
        ["bumblebee", "yes", "no", "no", "no", "yes"],
        ["cat", "no", "no", "no", "no", "no"],
        ["elephant", "no", "yes", "no", "no", "no"],
        ["dolphin", "no", "yes", "no", "yes", "no"],
        ["goldfish", "no", "no", "no", "yes", "no"],
    ],
    columns=["creature", "has_wings", "bigger_than_cat", "has_feathers", "lives_in_water", "can_fly"],
)

SPLITS = pd.DataFrame(
    [
        ["has_wings", "4 fly, 2 do not", "0 fly, 4 do not", 0.267],
        ["lives_in_water", "0 fly, 3 do not", "4 fly, 3 do not", 0.343],
        ["bigger_than_cat", "1 fly, 4 do not", "3 fly, 2 do not", 0.400],
        ["has_feathers", "2 fly, 2 do not", "2 fly, 4 do not", 0.467],
    ],
    columns=["first question", "yes bucket", "no bucket", "weighted mix"],
)

WORKBOOK = Workbook(
    chapter=5,
    title="Workbook · Pick the first question",
    intro=(
        "A decision tree plays Twenty Questions. It tries possible questions and chooses the one "
        "that makes the cleanest buckets."
    ),
    questions=[
        Question(
            prompt="For the question **has_wings**, how many creatures in the **yes** bucket can fly?",
            kind="number",
            answer=4,
            tolerance=0.01,
            table=CREATURES,
            hint="Look only at rows where has_wings is yes, then count can_fly = yes.",
            why="The winged flyers are sparrow, eagle, bat, and bumblebee: **4**. Penguin and ostrich ride in the same winged bucket but stay grounded, so the bucket helps and still has crumbs of mess.",
        ),
        Question(
            prompt="For **has_wings**, how many creatures in the **yes** bucket do **not** fly?",
            kind="number",
            answer=2,
            tolerance=0.01,
            table=CREATURES,
            hint="Penguin and ostrich both have wings but do not fly.",
            why="There are **2** winged non-flyers: penguin and ostrich. Both answers are still sharing one bucket, so that branch needs another yes/no flashlight.",
        ),
        Question(
            prompt="For **has_wings**, how many creatures in the **no** bucket can fly?",
            kind="number",
            answer=0,
            tolerance=0.01,
            table=CREATURES,
            hint="The no-wings rows are cat, elephant, dolphin, and goldfish.",
            why="The no bucket has **0** flyers and 4 non-flyers. That bucket is clean glass: mix 0, the kind of stopping place a tree loves.",
        ),
        Question(
            prompt="A bucket has 4 yes and 0 no. Using **mix = 1 - p_yes² - p_no²**, what is the mix?",
            kind="number",
            answer=0,
            tolerance=0.01,
            hint="p_yes = 1 and p_no = 0.",
            why="1 - 1² - 0² = **0**. Every row points the same way, so the bucket has no swirl left and the tree can stop asking there.",
        ),
        Question(
            prompt="A bucket has 3 yes and 3 no. What is the mix?",
            kind="number",
            answer=0.5,
            tolerance=0.01,
            hint="Both fractions are 3/6 = 0.5.",
            why="1 - (3/6)² - (3/6)² = 1 - 0.25 - 0.25 = **0.5**. Half-and-half is maximum soup for two answers: reach in blind and either label can pop out.",
        ),
        Question(
            prompt="For the **has_wings** yes bucket, compute **1 - (4/6)² - (2/6)²**. Round to two decimals.",
            kind="number",
            answer=0.44,
            tolerance=0.01,
            hint="The exact answer is 4/9.",
            why="The yes bucket mix is **4/9 ≈ 0.44**. The no bucket is 0, so the weighted split score is `(6×0.44 + 4×0) / 10 ≈ 0.27`. One messy tray, one clean tray, weighted by size.",
        ),
        Question(
            prompt="Which first question has the lowest weighted mix in this table?",
            kind="choice",
            choices=["has_wings", "lives_in_water", "bigger_than_cat", "has_feathers"],
            answer="has_wings",
            table=SPLITS,
            why=(
                "**has_wings** wins on this tiny dataset. A tree is not magic; it tests candidate questions, scores the bucket slosh, and grabs the split with less cleanup left."
            ),
        ),
        Question(
            prompt="Why is the lowest-mix question a good first question?",
            kind="open",
            hint="Think about how much work remains after the first split.",
            why=(
                "Cleaner buckets give the next questions a shorter mop job. The first split matters because every later branch starts from those piles, so a poor first split makes the whole flowchart sweat."
            ),
        ),
        Question(
            prompt="A depth-1 tree asks one question. A depth-20 tree can ask question after question. Which is more likely to memorise last year's test answers?",
            kind="choice",
            choices=["depth-1 tree", "depth-20 tree"],
            answer="depth-20 tree",
            why="The depth-20 tree can carve tiny fences around training points. That can become memorising: *this exact dot is red*, even when the dot is noise and teaches no useful rule.",
        ),
        Question(
            prompt="What picture clue would show that a tree is memorising?",
            kind="open",
            hint="Look for a boundary that wraps tiny boxes around individual dots.",
            why=(
                "A memorising tree often has tiny boxes wrapped around individual training points, plus training accuracy much higher than test accuracy. It flexes on studied rows, then wobbles on new ones."
            ),
        ),
    ],
    kid_corner=(
        "Play animal Guess Who away from the laptop.\n\n"
        "1. Pick 8 animals.\n"
        "2. Choose a secret answer, like **can fly**.\n"
        "3. Ask yes/no questions to split the animals.\n"
        "4. Which first question helped most?\n\n"
        "The best first question is the one that makes the cleanest piles."
    ),
)
