"""Chapter 25 workbook · So What Now?"""

from kidsml.workbook import Question, Workbook

WORKBOOK = Workbook(
    chapter=25,
    title="Workbook · Pick the right tool",
    intro="No new algorithm. This is the victory lap and the safety check.",
    questions=[
        Question(
            prompt="You want to sort your own photos into rough groups without labels. Which chapter helps most?",
            kind="choice",
            choices=["Chapter 20 k-means", "Chapter 04 probabilities", "Chapter 22 bigrams"],
            answer="Chapter 20 k-means",
            why="No labels means you need an unsupervised tool. K-means groups things by closeness.",
        ),
        Question(
            prompt="Autocomplete is closest to which Part 6 game?",
            kind="choice",
            choices=["guess the next letter", "draw a wider road", "average five folds"],
            answer="guess the next letter",
            why="Autocomplete and chatbots both lean on next-token prediction. Bigger models use pieces of words, but the training target is the same kind of thing.",
        ),
        Question(
            prompt="A chatbot gives a confident answer about your homework. What should you remember?",
            kind="choice",
            choices=["check things that matter", "confidence means truth", "long answers are always right"],
            answer="check things that matter",
            why="Chapter 24 trained a model to produce likely-looking text, not guaranteed-true text. Looking right and being right are different targets.",
        ),
        Question(
            prompt="Why is bias not a spooky extra bug, but a data problem you already understand?",
            kind="open",
            why="A model copies patterns from its training data. If the data is lopsided, missing people, or unfair, the model can copy that too. Chapter 10 already gave you that map.",
        ),
        Question(
            prompt="Name one weekend project you would actually want to build, and which chapter it starts from.",
            kind="open",
            why="The best next project is personal and small: your photos, your writing, your game, your music. Pick one idea and make the first version tiny.",
        ),
    ],
    kid_corner="You built a box of tools. A hammer is great for nails and silly for soup. Being smart about AI starts with asking: what job is this tool for?",
)
