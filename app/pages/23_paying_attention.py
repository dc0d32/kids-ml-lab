"""Chapter 23 · Paying Attention."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from kidsml import ui
from kidsml.datasets import load_corpus
from kidsml.langmodels import (
    attention_snapshot,
    generate_transformer,
    random_nll,
    sample_stream_bigram,
    sample_text_mlp,
    stream_bigram_counts,
    stream_bigram_nll,
    train_text_mlp,
    train_transformer_language_model,
)
from kidsml.plots import ACCENT, COOL, MUTED, WARM, heatmap, loss_curve
from kidsml.text import counts_to_probs

ui.page_setup(23)


@st.cache_resource(show_spinner="Training the tiny Transformer...")
def trained_transformer():
    text = (load_corpus("rhymes") + "\n" + load_corpus("fables")).lower()
    return train_transformer_language_model(text, block_size=32, embed_dim=48, n_heads=4, n_layers=1, steps=900, batch_size=64, seed=4)


@st.cache_resource(show_spinner="Training the comparison MLP...")
def trained_text_mlp():
    text = (load_corpus("rhymes") + "\n" + load_corpus("fables")).lower()
    return train_text_mlp(text, block_size=3, embed_dim=8, hidden=80, steps=700, batch_size=256, seed=2)


def shown_char(ch: str) -> str:
    if ch == "\n":
        return "↵"
    if ch == " ":
        return "␠"
    return ch


# ---------------------------------------------------------------------------
ui.beat("hook", "This is the one.")

st.markdown(
    """
The **T** in GPT stands for **Transformer**. Yours will be tiny, and it will read nursery
rhymes and fables, but it is the same idea.

The game still has not changed: **guess the next letter**.
"""
)

ui.little_kid_corner(
    "When you guess the next word in a story, you look back at the useful bits. Maybe the clue is nearby. Maybe it was two sentences ago."
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Which old letters should I look at?")

st.markdown(
    """
Before the mechanism, here is the question:

**I am about to guess the next letter. Which earlier letters should I look at?**

In `the cat sat on the m`, the useful clue for the next letter might be far back. A fixed
window of three cannot reach it. Attention can look anywhere inside its block.
"""
)

attention_toy = pd.DataFrame(
    [["earlier t", 1, 10], ["earlier h", 2, 20], ["earlier e", 1, 30]],
    columns=["place", "attention weight tokens", "value number"],
)
st.dataframe(attention_toy, hide_index=True, use_container_width=False)
st.info("Weights 1/4, 2/4, 1/4 give a weighted average of 20. Attention mixes values using learned weights.")

ui.jargon("query, key, value", "A position holds up a question, earlier positions wear labels, and the model copies more content from labels that match the question.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "The mask stops cheating.")

mask = np.tril(np.ones((8, 8)))
fig, ax = ui.figure(4.8, 4.2)
heatmap(mask, xlabels=list(range(1, 9)), ylabels=list(range(1, 9)), ax=ax, title="Causal mask: white cells are blocked")
ax.set_xlabel("place it wants to look")
ax.set_ylabel("place making a guess")
ui.show(fig)

ui.careful(
    "A position may only look backward. If it could see forward, it would peek at the answer. That is data leakage from Chapter 10 in a new costume."
)

st.code(
    """
scores = query @ key.T / sqrt(head_size)
scores = scores.masked_fill(future_cells, -1e9)
weights = softmax(scores)
out = weights @ value
""",
    language="python",
)

# ---------------------------------------------------------------------------
ui.beat("play", "Generate text and inspect its attention.")

bundle = trained_transformer()
start = st.text_input("Starting phrase", value="the ")
temperature = st.slider("Temperature", 0.05, 1.8, 0.9, 0.05)
seed = st.slider("Random seed", 0, 99, 5)
length = st.slider("How many new characters?", 80, 260, 180, 20)
made = generate_transformer(bundle, start=start, temperature=temperature, length=length, seed=seed)
st.text_area("Tiny Transformer says", made, height=150)

chars, heads = attention_snapshot(bundle, made)
head = st.selectbox("Attention head", list(range(heads.shape[0])), format_func=lambda h: f"head {h + 1}")
position = st.slider("Generated position to inspect", 0, len(chars) - 1, len(chars) - 1)
labels = [shown_char(c) for c in chars]
fig, ax = ui.figure(7, 5)
heatmap(heads[head], xlabels=labels, ylabels=labels, ax=ax, title=f"Attention map, head {head + 1}")
ax.axhline(position - 0.5, color="white", linewidth=1.2)
ax.axhline(position + 0.5, color="white", linewidth=1.2)
ax.set_xlabel("looked-at earlier character")
ax.set_ylabel("character doing the looking")
ui.show(fig)

row = heads[head, position]
order = np.argsort(row)[::-1][:5]
leans = []
for i in order:
    leans.append(f"{shown_char(chars[int(i)])} ({row[int(i)]:.2f})")
st.caption("This position leaned most on: " + ", ".join(leans))
st.caption("Tiny heads often attend to nearby letters or spaces. Do not force a story onto every square.")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Same idea, more scale.")

mlp = trained_text_mlp()
counts = stream_bigram_counts(bundle.train_text, bundle.vocab)
probs = counts_to_probs(counts, smoothing=1.0)
random_loss = random_nll(bundle.vocab)
bigram_loss = stream_bigram_nll(bundle.test_text, probs, bundle.vocab)
mlp_loss = mlp.test_loss
transformer_loss = bundle.test_loss

fig, ax = ui.figure(7, 3.6)
ax.bar(["random", "bigram", "MLP", "Transformer"], [random_loss, bigram_loss, mlp_loss, transformer_loss], color=[MUTED, COOL, WARM, ACCENT])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("The Part 6 ladder")
ui.show(fig)

st.markdown("**Bigram sample**")
st.text(sample_stream_bigram(probs, bundle.vocab, start="the ", temperature=temperature, length=150, seed=seed))
st.markdown("**Fixed-window MLP sample**")
st.text(sample_text_mlp(mlp, start="the ", temperature=temperature, length=150, seed=seed))
st.markdown("**Tiny Transformer sample**")
st.text(made[:220])

st.metric("Tiny Transformer parameters", f"{bundle.model.n_parameters():,}")
st.markdown(
    """
GPT-class models use the same code shape, then turn every dial far up: hundreds of billions
of parameters, huge text collections, many GPUs, and long training runs. Same idea. About a
billion times more of everything.
"""
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Try temperature 0.1 and 1.6. Which failure do you prefer?
2. Start with a phrase from a nursery rhyme, then one you invented.
3. Find a line that is almost real English.
4. Ask a grown-up what would happen if you removed the mask. Hint: the loss would look suspiciously good.
5. 🧸 **Little Kid Corner:** Tell a story one letter at a time. Each turn, point at earlier letters you used as clues.
"""
)

ui.worksheet_link(23)
