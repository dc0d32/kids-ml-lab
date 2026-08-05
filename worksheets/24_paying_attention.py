"""Chapter 24 workbook · Paying Attention."""

import pandas as pd

from kidsml.workbook import Question, Workbook

SCORES = pd.DataFrame(
    [["letter A", 1, 10], ["letter B", 2, 20], ["letter C", 1, 30]],
    columns=["earlier spot", "attention tokens", "value"],
)

MASK = pd.DataFrame(
    [
        ["row 1", "look", "blocked", "blocked", "blocked"],
        ["row 2", "look", "look", "blocked", "blocked"],
        ["row 3", "look", "look", "look", "blocked"],
        ["row 4", "look", "look", "look", "look"],
    ],
    columns=["position", "col 1", "col 2", "col 3", "col 4"],
)

WORKBOOK = Workbook(
    chapter=24,
    title="Workbook · Which earlier letter matters?",
    intro="Attention is still next-letter guessing. Now each spot chooses what to look back at.",
    questions=[
        Question(
            prompt="The attention tokens add to 4. What weight does letter B get?",
            kind="number",
            answer=0.5,
            tolerance=0.001,
            table=SCORES,
            why="B has 2 of the 4 tokens, so it gets weight 0.5. Real attention gets the tokens from query-key matches, then softmax turns them into weights.",
        ),
        Question(
            prompt="Using those weights, what weighted average value do you get? (A=10, B=20, C=30.)",
            kind="number",
            answer=20,
            tolerance=0.001,
            hint="Weights are 1/4, 2/4, 1/4.",
            table=SCORES,
            why="10×0.25 + 20×0.5 + 30×0.25 = 20. Attention mixes the value numbers using the attention weights.",
        ),
        Question(
            prompt="In a 4×4 causal mask, is row 2 allowed to look at column 4?",
            kind="choice",
            choices=["yes", "no"],
            answer="no",
            table=MASK,
            why="Column 4 is in the future for row 2. Generation only works if each position looks backward, never forward.",
        ),
        Question(
            prompt="What would happen if we removed the causal mask during training?",
            kind="choice",
            choices=["the model could cheat", "the model would forget spaces", "the vocabulary would shrink"],
            answer="the model could cheat",
            why="It could peek at the answers on its right. The loss would look amazing, but the model would be useless when generating, because the future is not there yet.",
        ),
        Question(
            prompt="The T in GPT stands for what?",
            kind="text",
            answer=["transformer"],
            why="GPT means Generative Pre-trained Transformer. Your version is tiny, but the code shape is the same idea.",
        ),
        Question(
            prompt="Why can attention use a clue that is 15 characters back when a block-size-3 MLP cannot?",
            kind="open",
            why="Attention lets each position look across the whole block of earlier characters. The fixed-window MLP only receives its last three characters and nothing else.",
        ),
    ],
    kid_corner="Imagine reading a page with a flashlight. For each new letter, you point the flashlight at earlier letters that seem useful. The mask is the rule that says: no shining it into the future.",
)
