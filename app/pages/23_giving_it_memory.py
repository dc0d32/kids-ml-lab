"""Chapter 23 · Giving It a Memory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.datasets import load_words
from kidsml.langmodels import embedding_points, random_nll, sample_mlp, train_mlp_language_model
from kidsml.plots import ACCENT, COOL, MUTED, WARM, loss_curve
from kidsml.text import STOP, bigram_counts, bigram_nll, counts_to_probs, letter_groups, sample_bigram

lesson.begin(23)


@st.cache_resource(show_spinner="Training the tiny name model...")
def trained_2d_model():
    return train_mlp_language_model(load_words("names"), block_size=3, embed_dim=2, hidden=96, n_words=8000, steps=1200, lr=0.03, seed=1)


@st.cache_resource(show_spinner="Training context-size models...")
def trained_context_models():
    models = {}
    for block_size in [1, 3, 5]:
        models[block_size] = train_mlp_language_model(
            load_words("names"), block_size=block_size, embed_dim=8, hidden=96, n_words=8000, steps=1000, lr=0.02, seed=10 + block_size
        )
    return models


def plot_embeddings(bundle):
    points = embedding_points(bundle)
    groups = letter_groups(bundle.vocab)
    colours = []
    for group in groups:
        if group == "vowel":
            colours.append(WARM)
        elif group == "blank":
            colours.append(MUTED)
        else:
            colours.append(COOL)
    fig, ax = lesson.figure(6, 5)
    ax.scatter(points[:, 0], points[:, 1], c=colours, s=130, edgecolors="white", linewidths=1.0)
    for i, ch in enumerate(bundle.vocab.chars):
        ax.text(points[i, 0], points[i, 1], ch, ha="center", va="center", fontsize=10, weight="bold")
    ax.set_xlabel("embedding number 1")
    ax.set_ylabel("embedding number 2")
    ax.set_title("Letters after training")
    return fig


@lesson.step("The game has not changed", beat="hook")
def _():
    lesson.say(
        """
Chapter 22 could peek **one** letter back. Now hand it **three** letters of memory.

That one extra handful moves us from tally chart to neural network. Same mission, bigger backpack: **guess the next letter**.
"""
    )
    lesson.kid_corner(
        "Instead of asking the kid beside you, ask the last three kids in line what letters they are holding. Then slap down your next-card guess."
    )
    lesson.say("Slide a three-letter window across `cat`, and four training examples pop out. The blank `.` still marks the starting gate and the finish line.")
    windows = pd.DataFrame(
        [["...", "c"], ["..c", "a"], [".ca", "t"], ["cat", "."]],
        columns=["input memory", "answer"],
    )
    st.dataframe(windows, hide_index=True, width="content")
    guess = lesson.predict(
        "When the input memory is `.ca`, what answer is the model training toward?",
        ["c", "a", "t", "."],
        correct=2,
        why="The window slides across `.cat.` like a little scanner. After `.ca`, the next character is `t`.",
        key="ch23_window",
    )
    if guess is None:
        return
    lesson.jargon("embedding", "A tiny number address for one character. The model gets to move those addresses while it learns.")
    lesson.jargon("softmax", "A probability machine: feed in scores, get probabilities that add to 1. Bigger scores get bigger slices.")


@lesson.step("The pipeline in one row", beat="seeit")
def _():
    lesson.say("Three letters go in. Each becomes a small vector. The vectors get glued together, squeezed through a `tanh` hidden layer, and turned into 27 softmax probabilities.")
    pipeline = pd.DataFrame(
        [
            ["input", "`.ca`", "three remembered letters"],
            ["embedding table", "three tiny vectors", "numbers the model learned"],
            ["hidden layer", "tanh", "mix the clues"],
            ["output", "27 probabilities", "one for each next character"],
        ],
        columns=["stage", "what it holds", "why it matters"],
    )
    st.dataframe(pipeline, hide_index=True, width="stretch")
    lesson.look_for("where the three letters become numbers. That is the new learned machinery clicking into place.")


@lesson.step("The vowels find each other", beat="seeit")
def _():
    guess = lesson.predict(
        "The model has never been told what a vowel is. Look at the letter map — what do you expect?",
        ["Letters scattered randomly", "Vowels sitting near each other", "Alphabetical order"],
        correct=1,
        why="Vowels do similar jobs when the model guesses the next letter, so training tends to park them near each other.",
        key="ch23_vowels",
    )
    if guess is None:
        return
    bundle2 = trained_2d_model()
    fig = plot_embeddings(bundle2)
    lesson.show(fig)
    lesson.look_for("a, e, i, o, u. Nobody handed the model vowel labels; training tugged them into a similar neighbourhood.")
    lesson.aha("The vowels land near each other! Nobody told the model what a vowel is. It grouped them because they behave alike when guessing the next letter.")
    st.caption("This is a real training run, not a stored picture. The vowels landed together on every seed we tried; the exact shape still wiggles.")


@lesson.step("Memory length changes the names", beat="play")
def _():
    lesson.say("Now slide the memory wall shorter or longer and listen to the names change shape.")
    models = trained_context_models()
    knobs, picture = lesson.controls()
    with knobs:
        block_size = st.select_slider("How many letters can it remember?", options=[1, 3, 5], value=3, key="ch23_block_size")
        temperature = st.slider("Temperature", 0.05, 2.0, 0.85, 0.05, key="ch23_temp")
        seed = st.slider("Random seed", 0, 99, 4, key="ch23_seed")
        starter = st.text_input("Start a name with these letters", value="ma", key="ch23_starter")
    chosen = models[block_size]
    with picture:
        samples = [sample_mlp(chosen, start=starter, temperature=temperature, seed=seed + i, max_len=18) for i in range(10)]
        st.write(" · ".join(samples))
        st.metric("Held-out surprise", f"{chosen.test_loss:.3f}")
        lesson.look_for("whether the 1-letter model drops starts that the 3-letter or 5-letter model can still hold.")


@lesson.step("Watch training settle", beat="play")
def _():
    lesson.say("Training is the model grinding hidden words down from surprising to less surprising. The curve bumps because it learns from small batches, not the whole mountain at once.")
    models = trained_context_models()
    block_size = st.select_slider("Which memory length?", options=[1, 3, 5], value=3, key="ch23_curve_block")
    fig, ax = lesson.figure(6.5, 3.2)
    loss_curve(models[block_size].losses, ax=ax, title=f"Training loss, block size {block_size}", ylabel="surprise")
    lesson.show(fig)
    lesson.look_for("the overall downhill slide, not every pebble-sized wiggle.")
    compare = pd.DataFrame({"memory": ["1 letter", "3 letters", "5 letters"], "held-out surprise": [models[1].test_loss, models[3].test_loss, models[5].test_loss]})
    st.dataframe(compare, hide_index=True, width="content")


@lesson.step("Does memory beat counting?", beat="forreal")
def _():
    guess = lesson.predict(
        "On hidden names, which model should have the lowest surprise?",
        ["Random guessing", "The Chapter 22 bigram", "The three-letter MLP"],
        correct=2,
        why="The MLP can hold three letters of context in its hand, so it beats the one-letter tally chart here.",
        key="ch23_loss_compare",
    )
    if guess is None:
        return
    models = trained_context_models()
    main = models[3]
    vocab = main.vocab
    counts = bigram_counts(main.train_words, vocab)
    probs = counts_to_probs(counts, smoothing=1.0)
    bigram_loss = bigram_nll(main.test_words, probs, vocab)
    random_loss = random_nll(vocab)
    fig, ax = lesson.figure(6, 3.5)
    ax.bar(["random", "bigram", "MLP"], [random_loss, bigram_loss, main.test_loss], color=[MUTED, COOL, ACCENT])
    ax.set_ylabel("average surprise (lower is better)")
    ax.set_title("Three-letter memory helps")
    lesson.show(fig)
    lesson.look_for("the MLP bar. Memory beats counting one letter back!")
    st.dataframe(
        pd.DataFrame(
            {
                "model": ["random guessing", "Chapter 22 bigram", "three-letter MLP"],
                "held-out surprise": [random_loss, bigram_loss, main.test_loss],
            }
        ),
        hide_index=True,
        width="content",
    )


@lesson.step("Hear the difference", beat="forreal")
def _():
    lesson.say("Same seed, same temperature, two different memories. Listen for the longer echo.")
    models = trained_context_models()
    main = models[3]
    vocab = main.vocab
    counts = bigram_counts(main.train_words, vocab)
    probs = counts_to_probs(counts, smoothing=1.0)
    temperature = st.slider("Temperature", 0.05, 2.0, 0.85, 0.05, key="ch23_compare_temp")
    seed = st.slider("Random seed", 0, 99, 4, key="ch23_compare_seed")
    rng = np.random.default_rng(seed)
    bigram_names = [sample_bigram(probs, vocab, rng=rng, temperature=temperature, max_len=18) for _ in range(8)]
    mlp_names = [sample_mlp(main, temperature=temperature, seed=seed + i, max_len=18) for i in range(8)]
    st.markdown("**Bigram:** " + " · ".join(bigram_names))
    st.markdown("**MLP:** " + " · ".join(mlp_names))
    lesson.look_for("places where the MLP keeps a name habit alive for more than one letter.")


@lesson.step("The whole program", beat="forreal")
def _():
    lesson.say("A fixed-window neural network is still the same input-output game with extra gears: letters in, next letter out.")
    st.code(
        """
X, y = make_context_dataset(words, vocab, block_size=3)
model = ContextMLP(vocab_size=27, block_size=3)
# three letters in, next letter out
""",
        language="python",
    )
    lesson.look_for("`block_size=3`. That number is the memory wall you can bump into.")
    lesson.careful("A block size of 3 is a hard wall. The fourth-back letter is invisible, even if it is waving a flag. Chapter 24 fixes that in a different way.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go stretch its memory", beat="challenge")
def _():
    lesson.say(
        """
1. Find a starter that the MLP finishes like a real name.
2. Compare block size 1 and 5 using the same starter.
3. Lower the temperature until the model gets boring. What name habit does it overuse?
4. 🧸 **Little Kid Corner:** Play telephone, but each person may ask only the last three people what they heard.
"""
    )


lesson.finish()
