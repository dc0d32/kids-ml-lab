"""Chapter 25 · So What Now?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import generate_ui, lesson
from kidsml.datasets import load_corpus
from kidsml.langmodels import generate_transformer, train_transformer_language_model
from kidsml.plots import ACCENT, COOL, INK, MUTED, PANEL, WARM

lesson.begin(25)


@st.cache_resource(show_spinner="Warming up your Chapter 24 model...")
def ch24_transformer():
    text = (load_corpus("rhymes") + "\n" + load_corpus("fables")).lower()
    return train_transformer_language_model(text, block_size=32, embed_dim=48, n_heads=4, n_layers=1, steps=900, batch_size=64, seed=4)


def course_map_figure():
    fig, ax = lesson.figure(10, 6)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.8, 4.8)
    ax.axis("off")

    parts = [
        ("Start", 0, 4.0, [0]),
        ("Classical", 1, 3.2, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("Messy data", 2, 2.4, [10, 11]),
        ("Neural nets", 3, 3.55, [12, 13, 14, 15, 16, 17]),
        ("Seeing", 4, 2.4, [18, 19]),
        ("No labels", 5, 2.4, [20, 21]),
        ("Making things up", 6, 3.2, [22, 23, 24, 25]),
    ]

    for name, x, y, chapters in parts:
        ax.text(x, 4.55, name, ha="center", va="center", fontsize=10, weight="bold")
        for j, number in enumerate(chapters):
            yy = y - 0.35 * j
            colour = ACCENT if number in {1, 2, 13, 15, 19, 24} else COOL
            if number in {20, 21}:
                colour = WARM
            if number in {0, 25}:
                colour = MUTED
            ax.scatter([x], [yy], s=420, c=colour, edgecolors=PANEL, linewidths=1.4, zorder=3)
            ax.text(x, yy, f"{number:02d}", ha="center", va="center", color=INK, weight="bold", fontsize=9)

    main_path = [(1, 3.2), (1, 2.85), (3, 3.2), (3, 2.5), (4, 2.05), (6, 2.5)]
    for (x1, y1), (x2, y2) in zip(main_path, main_path[1:]):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2.2})
    ax.text(3.2, 0.15, "green path: one neuron → layers → vision → Transformer", color=ACCENT, ha="center", fontsize=10)
    ax.text(5.1, 0.55, "red branch: learning without labels", color=WARM, ha="center", fontsize=10)
    return fig


@lesson.step("Look what you built", beat="hook")
def _():
    lesson.say(
        """
You opened with a secret-rule game. You ended by training a tiny Transformer.

That is not a toy achievement. It is the map of modern machine learning, built brick by brick with small numbers you could touch. Nice work!
"""
    )
    lesson.kid_corner("You did not get one magic wand. You filled a backpack with tools. Now you can reach in and pick the tool for the job.")
    lesson.say("You have personally built this backpack of tools. Shake it and it rattles:")
    built = [
        ["linear model", "a line that turns numbers into a score"],
        ["perceptron", "a hand-trained yes/no neuron"],
        ["decision trees", "question ladders that split the data"],
        ["ensembles", "crowds of models voting or correcting each other"],
        ["SVM", "the widest safe street between classes"],
        ["backprop", "the blame-passing move checked by numerical gradients"],
        ["CNN", "shared sliding-window filters for pictures"],
        ["k-means", "moving centres that make unlabeled piles"],
        ["PCA", "best-shadow squishing that keeps spread"],
        ["bigram generator", "one-letter-back counting"],
        ["embedding MLP", "three-letter memory with learned letter addresses"],
        ["Transformer", "attention choosing which earlier clues to use"],
    ]
    st.dataframe(pd.DataFrame(built, columns=["tool", "remember it as"]), hide_index=True, width="stretch")
    lesson.look_for("how many different tools are not neural networks. Modern ML is a toolbox, not one spell.")
    fig = course_map_figure()
    lesson.show(fig)
    lesson.look_for("the green line to Chapter 24, and the red branch that detoured through clustering and PCA.")
    st.caption("The whole map, and it is lowkey enormous: green is the neural-network road. Red is learning without answer labels.")
    guess = lesson.predict(
        "Which road carried you from one neuron to the Transformer?",
        ["The red no-labels branch", "The green neural-network road", "The classical-model loop"],
        correct=1,
        why="The green road carries you from a line, to layers, to vision, to attention.",
        key="ch25_course_map",
    )
    if guess is None:
        return
    lesson.aha("You followed the green road all the way from a line to attention, then met the red no-labels tools on the side road.")


@lesson.step("Where this is already in your life", beat="play")
def _():
    lesson.say("Pick a real-world thing and snap it back to chapters you have touched. Big chat systems usually guess **tokens** — chunks of text — while ours used letters so the gears stayed visible.")
    uses = {
        "Recommendations": "Chapters 08 and 20: find things near things you already like, or group people and items by pattern.",
        "Photo search": "Chapters 18, 19, 21: pictures are numbers, CNNs spot patterns, PCA can shrink them.",
        "Autocomplete": "Chapters 22-24: guess the next letter or text piece over and over.",
        "Spam filters": "Chapters 04, 09, 11: probability, model choice, and checking failure modes.",
        "Voice assistants": "Chapters 15, 19, 24: layers, sliding windows for sound-like patterns, and language models for text.",
        "Game AI": "Chapters 00, 05, 06, 15: rules from examples, trees, crowds, and small neural nets.",
        "Translation": "Chapter 24: attention helps connect words far apart across languages.",
    }
    choice = st.selectbox("Pick a real-world thing", list(uses), key="ch25_real_use")
    st.info(uses[choice])


@lesson.step("The honest limits", beat="forreal")
def _():
    limits = {
        "Hallucination": "A language model was trained to produce likely-looking text. Looking right and being right are different targets, so important answers need checking.",
        "Bias": "A model copies patterns from its data. If the examples are lopsided, missing people, or unfair, that shape can come along for the ride. Chapter 11 gave you the warning lights.",
        "Confidently wrong": "Out-of-distribution inputs can still get confident answers. Chapter 00 showed confidence from too few clues; Chapter 11 showed the edge of the map.",
        "No world inside": "A language model knows patterns in text. It does not have a lived-in world the way you do.",
    }
    choice = st.selectbox("Pick a limit to inspect", list(limits), key="ch25_limit")
    lesson.say(f"**{choice}.** {limits[choice]}")
    lesson.careful("Confidently wrong is still wrong. That is not a shame bell; it is a safety signal. Chapter 00 showed how thin evidence can fool you, and Chapter 11 showed models answering outside what they understood.")


@lesson.step("Make your own hallucination", beat="forreal")
def _():
    guess = lesson.predict(
        "Your Chapter 24 model never learned facts. If we start a factual-looking sentence, what should we expect?",
        ["A checked fact", "Likely-looking text", "Silence until it knows"],
        correct=1,
        why="It practised next-character likelihood, not truth. That is why fluent text can still be false, even when it sounds smooth.",
        key="ch25_hallucination",
    )
    if guess is None:
        return
    bundle = ch24_transformer()
    prompt = st.text_input("Start with a factual-looking prompt", value="the moon is made of ", key="ch25_hallucination_prompt")
    temperature = st.slider("Temperature", 0.05, 1.8, 0.9, 0.05, key="ch25_hallucination_temp")
    roll = generate_ui.regenerate(label="🎲 Generate again", key="ch25_hallucination")
    made = generate_transformer(bundle, start=prompt, temperature=temperature, length=180, seed=7 + roll)
    st.text_area("Your tiny model continues", made, height=140)
    lesson.say("The model rolls dice for each letter, so a fresh answer on every press is the model working, not a bug.")
    lesson.look_for("how it keeps the shape of text without checking whether the sentence is true.")
    lesson.say(
        """
Being smart about AI:

- Check things that matter.
- Treat it as a tool, not an oracle.
- Keep private stuff out of the machine.
- Using it to learn is great. Using it to dodge learning is a bad trade.
"""
    )
    lesson.aha("You now understand more about how these systems work than most people using them every day. That is real leverage!")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("A pile of weekend projects", beat="challenge")
def _():
    lesson.say(
        "Pick one that sounds fun enough to break on purpose. A ⭐ marks a good first one. "
        "Every project says which chapters it uses and where the data comes from — all of it "
        "is stuff you can collect yourself or a set already bundled in this course."
    )
    lesson.say(
        """
**🍿 Start here — an afternoon each (chapters 00-09)**

- ⭐ **Secret-rule game.** You invent a yes/no rule; a friend guesses it from your examples. *Ch 00. Data: make it up on paper.*
- ⭐ **Sweet or salty?** Rate 15 snacks on a couple of numbers, predict the label. *Ch 01, 05. Data: rate them yourself.*
- **Penguin species guesser.** Name the species from body measurements. *Ch 05, 08. Data: bundled `penguins`.*
- **Will it fly?** Build a decision tree by hand for made-up creatures. *Ch 05. Data: bundled `creatures`.*
- **Monster boss detector.** Is this card a boss, from its stats? *Ch 05, 06. Data: bundled `monsters`.*
"""
    )
    lesson.say(
        """
**🖼️ Play with pictures (chapters 18-19)**

- ⭐ **Handwritten-digit reader.** Train on the digits set, then feed it your own scribbles. *Ch 18, 19. Data: bundled digits.*
- ⭐ **Your own symbol alphabet.** Invent 5 symbols, draw each 10 times, train a reader. *Ch 18, 19. Data: draw them in a paint app, save small.*
- **Fashion sorter.** Tell a shirt from a sneaker. *Ch 19. Data: Fashion-MNIST, already cached in Ch 19.*
- **Filter explorer.** Slide a 3x3 filter over your photo and watch edges pop. *Ch 19. Data: any photo you took.*
- **Confusion detective.** Which two digits does your model mix up most? *Ch 18. Data: bundled digits.*
"""
    )
    lesson.say(
        """
**💬 Play with words (chapters 22-24)**

- ⭐ **Babbler trained on you.** Feed it your own writing; it will sound weirdly like you. *Ch 22-24. Data: paste your text into a file.*
- ⭐ **Rhyme machine.** Train the tiny Transformer on rhymes and read the wobbles. *Ch 24. Data: bundled `rhymes`.*
- **Fantasy team-name generator.** Tune temperature, keep the best accidents. *Ch 22-23. Data: bundled `names` or a list you type.*
- **Pet-name inventor.** Feed it 100 real pet names, generate new ones. *Ch 22-23. Data: type the list.*
- **Attention peek.** Generate a line, then find which earlier letters it leaned on. *Ch 24. Data: bundled `rhymes`/`fables`.*
"""
    )
    lesson.say(
        """
**🔍 Find patterns with no answer key (chapters 08, 20, 21)**

- ⭐ **Music mood clusters.** Group your songs by features you pick, like tempo and loudness. *Ch 20. Data: rate 30 of your songs.*
- ⭐ **Penguin islands.** Cluster penguins with the species label hidden, then peek. *Ch 20. Data: bundled `penguins`.*
- **Photo sorter for your room.** Cluster your own pictures into piles. *Ch 20, 21. Data: your phone photos, shrunk small.*
- **Squish-and-see.** PCA your symbol images down to 2D and hunt for clumps. *Ch 21. Data: bundled digits or your own symbols.*
- **Odd one out.** Run k-means, then find the point farthest from every centre. *Ch 08, 20. Data: any small table you collect.*
"""
    )
    lesson.say(
        """
**🔬 Be a scientist about it (chapters 09, 11)**

- ⭐ **Predict your bus.** Log arrivals for two weeks, then test honestly on days you held out. *Ch 01, 09, 11. Data: log it yourself.*
- ⭐ **Bias detective.** Build a lopsided dataset on purpose and catch the failure. *Ch 10, 11. Data: make a slanted table.*
- **Mushroom safety, honestly.** High accuracy is not enough when a mistake is deadly. *Ch 05, 10, 11. Data: bundled `mushrooms`.*
- **Data-leakage trap.** Sneak the answer into a feature and watch the score look too good. *Ch 11. Data: any bundled table.*
- **Break your own model.** Find the input that fools it most, then explain why. *Ch 11. Data: your own project's model.*
"""
    )
    lesson.say(
        """
**🛠️ Build something someone else can use**

- ⭐ **Rock-paper-scissors bot** that learns your habits and starts beating you. *Ch 00, 05. Data: your own moves as you play.*
- ⭐ **"Is this spam?" filter** for a club or family chat. *Ch 04, 09. Data: label 50 of your own messages.*
- **Bike-day advisor.** Guess the rental count from the forecast. *Ch 01, 10. Data: bundled `bikes`.*
- **Flashcard picker** that guesses which card you are about to miss. *Ch 08, 10. Data: log your own right/wrong.*
- **Name-my-pet app.** Type a vibe, get invented names back. *Ch 22-23. Data: bundled `names`.*
"""
    )
    lesson.kid_corner("Pick one tool from the backpack and use it on something in your own room.")


@lesson.step("What to learn next", beat="challenge")
def _():
    next_steps = {
        "algebra": "Algebra makes model formulas easier to move around.",
        "vectors": "Vectors make data feel natural.",
        "derivatives": "Derivatives unlock the grown-up version of backprop.",
        "a GPU": "A GPU changes the scale: bigger batches, bigger models, more experiments before dinner.",
    }
    choice = st.selectbox("What sounds useful next?", list(next_steps), key="ch25_next_topic")
    lesson.say(f"{next_steps[choice]} The secret is not missing; the dials are larger, and now you know what they do!")


lesson.finish()
