"""Chapter 22 · Giving It a Memory."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import load_words
from kidsml.langmodels import embedding_points, random_nll, sample_mlp, train_mlp_language_model
from kidsml.plots import ACCENT, COOL, MUTED, WARM, loss_curve
from kidsml.text import CharVocab, STOP, bigram_counts, bigram_nll, counts_to_probs, letter_groups, sample_bigram

ui.page_setup(22)


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
    fig, ax = ui.figure(6, 5)
    ax.scatter(points[:, 0], points[:, 1], c=colours, s=130, edgecolors="white", linewidths=1.0)
    for i, ch in enumerate(bundle.vocab.chars):
        ax.text(points[i, 0], points[i, 1], ch, ha="center", va="center", fontsize=10, weight="bold")
    ax.set_xlabel("embedding number 1")
    ax.set_ylabel("embedding number 2")
    ax.set_title("Letters after training")
    return fig


# ---------------------------------------------------------------------------
ui.beat("hook", "The game has not changed.")

st.markdown(
    """
Chapter 21 could see **one** letter back. Let's give it **three**.

That one change takes us from a tally chart to an actual neural network. The job is still
the same: **guess the next letter**.
"""
)

ui.little_kid_corner(
    "Instead of asking only the kid beside you, ask the last three kids in line what letters they are holding. Then guess the next card."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Slide a three-letter window.")

windows = pd.DataFrame(
    [["...", "c"], ["..c", "a"], [".ca", "t"], ["cat", "."]],
    columns=["input memory", "answer"],
)
st.dataframe(windows, hide_index=True, use_container_width=False)
st.markdown("The word `cat` becomes four training examples. The blank `.` still means start or stop.")

ui.jargon("embedding", "A tiny list of numbers for one character. The model gets to choose those numbers while it learns.")
ui.jargon("softmax", "A way to turn any list of scores into probabilities that add to 1. Bigger scores get more probability.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The pipeline in one row.")

st.markdown(
    """
Three letters in → each becomes a small vector → glue the vectors together → hidden layer
with `tanh` → 27 scores → softmax probabilities.
"""
)

pipeline = pd.DataFrame(
    [
        ["input", "`.ca`", "three remembered letters"],
        ["embedding table", "three tiny vectors", "numbers the model learned"],
        ["hidden layer", "tanh", "mix the clues"],
        ["output", "27 probabilities", "one for each next character"],
    ],
    columns=["stage", "what it holds", "why it matters"],
)
st.dataframe(pipeline, hide_index=True, use_container_width=True)

bundle2 = trained_2d_model()
fig = plot_embeddings(bundle2)
ui.show(fig)

ui.aha(
    "The vowels often land near each other. Nobody told the model what a vowel is. It grouped them because they behave alike when guessing the next letter."
)
st.caption("This is a real training run. It is usually clear with this seed, but not every random start makes a perfect picture.")

# ---------------------------------------------------------------------------
ui.beat("play", "Memory length changes the names.")

models = trained_context_models()
block_size = st.select_slider("How many letters can it remember?", options=[1, 3, 5], value=3)
temperature = st.slider("Temperature", 0.05, 2.0, 0.85, 0.05)
seed = st.slider("Random seed", 0, 99, 4)
starter = st.text_input("Start a name with these letters", value="ma")
chosen = models[block_size]

samples = []
for i in range(10):
    samples.append(sample_mlp(chosen, start=starter, temperature=temperature, seed=seed + i, max_len=18))
st.write(" · ".join(samples))
st.metric("Held-out surprise", f"{chosen.test_loss:.3f}")

fig, ax = ui.figure(6.5, 3.2)
loss_curve(chosen.losses, ax=ax, title=f"Training loss, block size {block_size}", ylabel="surprise")
ui.show(fig)

compare = pd.DataFrame(
    {"memory": ["1 letter", "3 letters", "5 letters"], "held-out surprise": [models[1].test_loss, models[3].test_loss, models[5].test_loss]}
)
st.dataframe(compare, hide_index=True, use_container_width=False)

# ---------------------------------------------------------------------------
ui.beat("forreal", "Does memory beat counting?")

main = models[3]
vocab = main.vocab
counts = bigram_counts(main.train_words, vocab)
probs = counts_to_probs(counts, smoothing=1.0)
bigram_loss = bigram_nll(main.test_words, probs, vocab)
random_loss = random_nll(vocab)

fig, ax = ui.figure(6, 3.5)
ax.bar(["random", "bigram", "MLP"], [random_loss, bigram_loss, main.test_loss], color=[MUTED, COOL, ACCENT])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("Three-letter memory helps")
ui.show(fig)

rng = np.random.default_rng(seed)
bigram_names = []
for _ in range(8):
    bigram_names.append(sample_bigram(probs, vocab, rng=rng, temperature=temperature, max_len=18))
mlp_names = []
for i in range(8):
    mlp_names.append(sample_mlp(main, temperature=temperature, seed=seed + i, max_len=18))

st.markdown("**Bigram:** " + " · ".join(bigram_names))
st.markdown("**MLP:** " + " · ".join(mlp_names))

st.code(
    """
X, y = make_context_dataset(words, vocab, block_size=3)
model = ContextMLP(vocab_size=27, block_size=3)
# three letters in, next letter out
""",
    language="python",
)

ui.careful(
    "A block size of 3 is a hard wall. The fourth-back letter is invisible. Chapter 23 fixes that in a different way."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find a starter that the MLP finishes like a real name.
2. Compare block size 1 and 5 using the same starter.
3. Lower the temperature until the model gets boring. What name habit does it overuse?
4. 🧸 **Little Kid Corner:** Play telephone, but each person may ask only the last three people what they heard.
"""
)

ui.worksheet_link(22)
