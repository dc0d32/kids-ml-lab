"""Chapter 21 · The Bigram Babbler."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import load_corpus, load_words
from kidsml.langmodels import random_nll, sample_bigram_trace, top_letters
from kidsml.plots import heatmap
from kidsml.text import CharVocab, STOP, bigram_counts, bigram_nll, counts_to_probs, sample_bigram, train_test_split_words

ui.page_setup(21)


@st.cache_data(show_spinner=False)
def bigram_bundle(corpus: str):
    if corpus == "names":
        words = load_words("names")
    else:
        words = load_words(corpus)
    vocab = CharVocab.from_words(words)
    train, test = train_test_split_words(words, frac=0.9, seed=0)
    counts = bigram_counts(train, vocab)
    probs = counts_to_probs(counts, smoothing=1.0)
    return words, vocab, train, test, counts, probs


def label(ch: str) -> str:
    if ch == STOP:
        return "."
    if ch == " ":
        return "sp"
    return ch


# ---------------------------------------------------------------------------
ui.beat("hook", "The secret is smaller than it sounds.")

st.markdown(
    """
You want to know how ChatGPT works. Here is the honest answer: **it guesses the next
letter. Over and over.**

The next three chapters are three ways of guessing. This first one is so small you could do
it with a tally chart. Nothing is learned here. We only count letter pairs.
"""
)

ui.little_kid_corner(
    "Line up letter cards from a name. Every time two cards touch, put a tally mark in that box. Later, draw the next card from the busiest boxes."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Tally marks before code.")

st.markdown(
    """
Every name secretly starts and ends with the blank character **`.`**.

`mia` is really `.mia.`. That lets the tally chart learn what starts names and what ends
names.
"""
)

tiny = pd.DataFrame(
    [[".mia.", ". → m, m → i, i → a, a → ."], [".mo.", ". → m, m → o, o → ."], [".mae.", ". → m, m → a, a → e, e → ."]],
    columns=["padded word", "pairs you tally"],
)
st.dataframe(tiny, hide_index=True, use_container_width=True)
st.info("For these three words, the `. → m` box gets 3 tally marks. The `m → a`, `m → i`, and `m → o` boxes get 1 each.")

ui.jargon("bigram", "A pair of touching characters. `m → a` is one bigram.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The whole tally chart.")

words, vocab, train, test, counts, probs = bigram_bundle("names")
fig, ax = ui.figure(7, 6)
heatmap(counts, xlabels=[label(c) for c in vocab.chars], ylabels=[label(c) for c in vocab.chars], ax=ax, title="Name bigram tallies")
ax.set_xlabel("next letter")
ax.set_ylabel("letter before it")
ui.show(fig)

st.markdown("A bright square means: **that pair happened a lot**. No magic. Tally marks.")

row_col, col_col = st.columns(2)
start_row = pd.DataFrame(
    {"starts names": [label(vocab.itos[i]) for i in np.argsort(counts[vocab.stoi[STOP]])[::-1][:8]],
     "tallies": [int(counts[vocab.stoi[STOP], i]) for i in np.argsort(counts[vocab.stoi[STOP]])[::-1][:8]]}
)
end_col = pd.DataFrame(
    {"ends names": [label(vocab.itos[i]) for i in np.argsort(counts[:, vocab.stoi[STOP]])[::-1][:8]],
     "tallies": [int(counts[i, vocab.stoi[STOP]]) for i in np.argsort(counts[:, vocab.stoi[STOP]])[::-1][:8]]}
)
with row_col:
    st.markdown("**The row for `.`:** letters that start names")
    st.dataframe(start_row, hide_index=True)
with col_col:
    st.markdown("**The column for `.`:** letters that end names")
    st.dataframe(end_col, hide_index=True)

ui.aha("Those two lists are different. The chart discovered a real fact about names by counting.")

# ---------------------------------------------------------------------------
ui.beat("play", "Pick a row. Roll the uneven die.")

corpus = st.selectbox("Corpus", ["names", "rhymes", "fables"], index=0)
words, vocab, train, test, counts, probs = bigram_bundle(corpus)
letters = list(vocab.chars)
def format_letter(c):
    return label(c)

picked = st.selectbox("After this character, what tends to come next?", letters, index=letters.index("q") if "q" in letters else 0, format_func=format_letter)
row = probs[vocab.stoi[picked]]
names, values = top_letters(row, vocab, n=min(10, len(vocab)))
fig, ax = ui.figure(7, 3.5)
ax.bar([label(c) for c in names], values, color="#3B82F6")
ax.set_ylim(0, max(values) * 1.15)
ax.set_ylabel("probability")
ax.set_title(f"After {label(picked)!r}, the likely next characters")
ui.show(fig)

if picked == "q" and "u" in vocab.stoi:
    st.caption("This is the famous q → u habit. The model was not told spelling rules. It counted them.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    temperature = st.slider("Temperature", 0.05, 2.0, 0.9, 0.05)
with col_b:
    seed = st.slider("Random seed", 0, 99, 4)
with col_c:
    n_samples = st.slider("How many inventions?", 5, 15, 10)

rng = np.random.default_rng(seed)
samples = []
for _ in range(n_samples):
    samples.append(sample_bigram(probs, vocab, rng=rng, temperature=temperature, max_len=18))
st.write(" · ".join(s if s else "(blank)" for s in samples))
st.caption("Low temperature is boring and safe. High temperature is weird and adventurous.")

trace_word, trace = sample_bigram_trace(probs, vocab, seed=seed, temperature=temperature, max_len=14)
st.markdown(f"**Trace one invention:** `{trace_word or '(blank)'}`")
trace_rows = []
for step in trace:
    trace_rows.append({"after": label(step["after"]), "picked": label(step["picked"]), "probability": round(step["probability"], 3)})
st.dataframe(pd.DataFrame(trace_rows), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
ui.beat("forreal", "A score for surprise.")

loss = bigram_nll(test, probs, vocab)
random_loss = random_nll(vocab)
st.metric("Bigram surprise on hidden words", f"{loss:.3f}")
st.metric("Knows-nothing random surprise", f"{random_loss:.3f}")

fig, ax = ui.figure(5.5, 3.5)
ax.bar(["random", "bigram"], [random_loss, loss], color=["#94A3B8", "#10B981"])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("Counting beats knowing nothing")
ui.show(fig)

st.code(
    """
counts = bigram_counts(train_words, vocab)
probs = counts_to_probs(counts, smoothing=1)
loss = bigram_nll(hidden_words, probs, vocab)
""",
    language="python",
)

ui.careful(
    "The babbler sounds almost name-like, but it only looks one letter back. It has no idea what happened three letters ago. That wall is Chapter 22."
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find the most predictable next letter. Try `q`, `x`, and `.`.
2. Make the babbler produce a real name by luck.
3. Set temperature near 0. Why does it keep making the same safe choices?
4. 🧸 **Little Kid Corner:** Make an uneven die with paper scraps. More scraps for common letters. Draw, write, repeat.
"""
)

ui.worksheet_link(21)
