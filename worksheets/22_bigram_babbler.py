"""Chapter 22 workbook · The Bigram Babbler."""

import pandas as pd

from kidsml.workbook import Question, Workbook

TINY_WORDS = pd.DataFrame(
    [
        [".", "m", "mae begins"],
        ["m", "a", "mae"],
        ["a", "e", "mae"],
        ["e", ".", "mae ends"],
        [".", "m", "mia begins"],
        ["m", "i", "mia"],
        ["i", "a", "mia"],
        ["a", ".", "mia ends"],
        [".", "m", "mo begins"],
        ["m", "o", "mo"],
        ["o", ".", "mo ends"],
    ],
    columns=["from", "to", "where you saw it"],
)

ROW_COUNTS = pd.DataFrame(
    [["a → .", 2], ["a → n", 3], ["a → r", 5]],
    columns=["next letter", "tallies"],
)

WORKBOOK = Workbook(
    chapter=22,
    title="Workbook · Count, divide, roll",
    intro="This one has no neural network. You are the tally chart. A corpus is the text pile, the vocabulary is its allowed characters, and `.` is the special start/stop blank.",
    questions=[
        Question(
            prompt="In the tiny corpus **mae, mia, mo**, how many times does the pair `. → m` appear?",
            kind="number",
            answer=3,
            table=TINY_WORDS,
            why="Each word gets a starting blank `.`, and all three words roll from that blank into m. That drops three tally marks in the `. → m` box.",
        ),
        Question(
            prompt="How many times does `a → .` appear in those same words?",
            kind="number",
            answer=1,
            why="Only **mia** ends with a. **mae** has a → e, and **mo** has no a. The ending blank is a real next character, so `a → .` gets one tally.",
        ),
        Question(
            prompt="This row has 10 tallies total. What probability should `a → r` get?",
            kind="number",
            answer=0.5,
            tolerance=0.001,
            table=ROW_COUNTS,
            hint="Probability = tallies for that choice ÷ tallies in the row.",
            why="`a → r` has 5 tallies out of 10, so its probability is 5/10 = 0.5. The row acts like a die with uneven sides.",
        ),
        Question(
            prompt="Using that same row, a random roll lands at **7 out of 10**. Walk along `.`, then `n`, then `r`. Which letter do you pick?",
            kind="text",
            answer=["r"],
            table=ROW_COUNTS,
            why="The first 2 rolls land on `.`, rolls 3-5 land on `n`, and rolls 6-10 land on `r`. Roll 7 lands inside the `r` chunk.",
        ),
        Question(
            prompt="Why do we add one fake tally to every box before dividing?",
            kind="choice",
            choices=["so no pair is impossible", "so the table is bigger", "so every name is real"],
            answer="so no pair is impossible",
            why="A zero probability says a pair can never happen. Then one surprise pair would create infinite surprise. A tiny fake tally keeps every door unlocked a crack.",
        ),
        Question(
            prompt="Why does this babbler make pronounceable nonsense instead of brilliant names every time?",
            kind="open",
            why="It only looks one letter back. If it has seen `a`, it knows nothing about whether the word started `ma`, `ka`, or `bra`. Chapter 23 gives it a longer memory window.",
        ),
        Question(
            prompt="What is the vocabulary in this tiny character model?",
            kind="choice",
            choices=["the characters it may write, including `.`", "the list of real names", "the hidden test score"],
            answer="the characters it may write, including `.`",
            why="The vocabulary is the allowed character list. In this chapter it includes letters plus the blank `.` that marks start and stop.",
        ),
    ],
    kid_corner="Put letter cards in bowls. After an `m`, put lots of `a` cards, some `i` cards, and a few weird cards. Draw one card. That is sampling.",
)
