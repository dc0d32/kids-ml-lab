"""Chapter 24 · Paying Attention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
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
from kidsml.plots import ACCENT, COOL, MUTED, WARM, heatmap
from kidsml.text import counts_to_probs

lesson.begin(24)


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
        return "sp"
    return ch


def comparison_numbers():
    bundle = trained_transformer()
    mlp = trained_text_mlp()
    counts = stream_bigram_counts(bundle.train_text, bundle.vocab)
    probs = counts_to_probs(counts, smoothing=1.0)
    return bundle, mlp, probs, random_nll(bundle.vocab), stream_bigram_nll(bundle.test_text, probs, bundle.vocab), mlp.test_loss, bundle.test_loss


@lesson.step("This is the one", beat="hook")
def _():
    lesson.say(
        """
Here comes the shiny machine. The **T** in GPT stands for **Transformer**. Yours is tiny, and it reads nursery rhymes and fables, but the idea is the same.

The game still has not changed: **guess the next letter**. New engine, same race!
"""
    )
    lesson.kid_corner(
        "When you guess the next word in a story, your eyes zip back to useful clues. Maybe the clue is next door. Maybe it is two sentences back waving a tiny flag."
    )


@lesson.step("Which old letters should I look at?", beat="byhand")
def _():
    lesson.say(
        """
Before the mechanism, grab the real question: **I am about to guess the next letter. Which earlier letters should I look at?**

In `the cat sat on the m`, the useful clue may be far back, sitting on `cat`. A fixed window of three cannot reach that far.
"""
    )
    attention_toy = pd.DataFrame(
        [["earlier t", 1, 10], ["earlier h", 2, 20], ["earlier e", 1, 30]],
        columns=["place", "attention weight tokens", "value number"],
    )
    st.dataframe(attention_toy, hide_index=True, width="content")
    st.info("Weights 1/4, 2/4, 1/4 make 10×1/4 + 20×2/4 + 30×1/4 = 20. Attention mixes value numbers using learned weights.")
    lesson.jargon("query, key, value", "One position holds up a question, earlier positions wear labels, and the model copies more content from labels that match the question.")


@lesson.step("The mask stops cheating", beat="seeit")
def _():
    lesson.say("A mask is a cover sheet for the score table. A **causal** mask blocks every future square, because future letters would hand over the answer.")
    guess = lesson.predict(
        "What happens if a position is allowed to peek at future letters while training?",
        ["It learns honestly", "It cheats and the score looks too good", "It forgets older letters"],
        correct=1,
        why="The future contains the answer. That is data leakage from Chapter 10 wearing a new costume.",
        key="ch24_mask",
    )
    if guess is None:
        return
    mask = np.tril(np.ones((8, 8)))
    fig, ax = lesson.figure(4.8, 4.2)
    heatmap(mask, xlabels=list(range(1, 9)), ylabels=list(range(1, 9)), ax=ax, title="Causal mask: bright cells are allowed")
    ax.set_xlabel("place it wants to look")
    ax.set_ylabel("place making a guess")
    lesson.show(fig)
    lesson.look_for("the dark upper triangle. Those are future places sealed behind glass.")
    lesson.careful("A position may only look backward. If it could see forward, it would peek at the answer key.")


@lesson.step("The four attention lines", beat="seeit")
def _():
    lesson.say("Here is the mechanism with the panels off. Scores become weights; weights mix values; the mask blocks future peeking.")
    st.code(
        """
scores = query @ key.T / sqrt(head_size)
scores = scores.masked_fill(future_cells, -1e9)
weights = softmax(scores)
out = weights @ value
""",
        language="python",
    )
    lesson.look_for("the mask line. It squashes future-looking scores before softmax can turn them into attention.")


@lesson.step("Inspect live attention", beat="play")
def _():
    lesson.say("A trained tiny Transformer has several attention heads. Pick one row and watch which earlier characters it leaned on.")
    bundle = trained_transformer()
    made = generate_transformer(bundle, start="the ", temperature=0.9, length=160, seed=5)
    chars, heads = attention_snapshot(bundle, made)
    labels = [shown_char(c) for c in chars]
    knobs, picture = lesson.controls()
    with knobs:
        head = st.selectbox("Attention head", list(range(heads.shape[0])), format_func=lambda h: f"head {h + 1}", key="ch24_head")
        position = st.slider("Generated position to inspect", 0, len(chars) - 1, len(chars) - 1, key="ch24_position")
    with picture:
        fig, ax = lesson.figure(7, 5)
        heatmap(heads[head], xlabels=labels, ylabels=labels, ax=ax, title=f"Attention map, head {head + 1}")
        ax.axhline(position - 0.5, color="white", linewidth=1.2)
        ax.axhline(position + 0.5, color="white", linewidth=1.2)
        ax.set_xlabel("looked-at earlier character")
        ax.set_ylabel("character doing the looking")
        lesson.show(fig)
        lesson.look_for("the highlighted row. Bright squares are earlier characters this position grabbed as clues.")
    row = heads[head, position]
    order = np.argsort(row)[::-1][:5]
    leans = [f"{shown_char(chars[int(i)])} ({row[int(i)]:.2f})" for i in order]
    st.caption("This position leaned most on: " + ", ".join(leans))
    st.caption("Tiny heads often attend to nearby letters or spaces. Do not force a story onto every square; that is attention glazing in a lab coat.")


@lesson.step("Generate with the Transformer", beat="play")
def _():
    lesson.say("Now run the model as a text machine. Temperature still controls safe choices versus weird sparks.")
    bundle = trained_transformer()
    start = st.text_input("Starting phrase", value="the ", key="ch24_start")
    temperature = st.slider("Temperature", 0.05, 1.8, 0.9, 0.05, key="ch24_temp")
    seed = st.slider("Random seed", 0, 99, 5, key="ch24_seed")
    length = st.slider("How many new characters?", 80, 260, 180, 20, key="ch24_length")
    made = generate_transformer(bundle, start=start, temperature=temperature, length=length, seed=seed)
    st.text_area("Tiny Transformer says", made, height=150, key="ch24_text")
    lesson.look_for("phrases that almost sound like a rhyme or fable, then wobble off the sidewalk.")


@lesson.step("The Part 6 ladder", beat="forreal")
def _():
    guess = lesson.predict(
        "Which model do you expect to have the lowest surprise on hidden text?",
        ["Bigram", "Fixed-window MLP", "Tiny Transformer"],
        correct=2,
        why="Attention can reach any earlier position inside the block, so it wins this tiny ladder.",
        key="ch24_ladder",
    )
    if guess is None:
        return
    _, _, _, random_loss, bigram_loss, mlp_loss, transformer_loss = comparison_numbers()
    fig, ax = lesson.figure(7, 3.6)
    ax.bar(["random", "bigram", "MLP", "Transformer"], [random_loss, bigram_loss, mlp_loss, transformer_loss], color=[MUTED, COOL, WARM, ACCENT])
    ax.set_ylabel("average surprise (lower is better)")
    ax.set_title("The Part 6 ladder")
    lesson.show(fig)
    lesson.look_for("the bars stepping down as the models get more ways to use context!")
    st.dataframe(
        pd.DataFrame(
            {
                "model": ["random", "bigram", "fixed-window MLP", "Tiny Transformer"],
                "held-out surprise": [random_loss, bigram_loss, mlp_loss, transformer_loss],
            }
        ),
        hide_index=True,
        width="content",
    )


@lesson.step("Same prompt, three machines", beat="forreal")
def _():
    lesson.say("Send the same prompt through the last three machines and listen to the echoes.")
    bundle, mlp, probs, _, _, _, _ = comparison_numbers()
    temperature = st.slider("Temperature", 0.05, 1.8, 0.9, 0.05, key="ch24_compare_temp")
    seed = st.slider("Random seed", 0, 99, 5, key="ch24_compare_seed")
    made = generate_transformer(bundle, start="the ", temperature=temperature, length=180, seed=seed)
    st.markdown("**Bigram sample**")
    st.text(sample_stream_bigram(probs, bundle.vocab, start="the ", temperature=temperature, length=150, seed=seed))
    st.markdown("**Fixed-window MLP sample**")
    st.text(sample_text_mlp(mlp, start="the ", temperature=temperature, length=150, seed=seed))
    st.markdown("**Tiny Transformer sample**")
    st.text(made[:220])
    lesson.look_for("which sample holds the shape of words and spaces for longest.")


@lesson.step("Same idea, more scale", beat="forreal")
def _():
    bundle = trained_transformer()
    st.metric("Tiny Transformer parameters", f"{bundle.model.n_parameters():,}")
    dial = st.selectbox(
        "Which dial would a GPT-class model turn up?",
        ["parameters", "text collection", "GPUs", "training time"],
        key="ch24_scale_dial",
    )
    st.info(f"Yes: {dial}. Real systems turn all of these up at once.")
    lesson.say(
        """
GPT-class models use the same code shape, then crank every dial far up: hundreds of billions of parameters, huge text collections, many GPUs, and long training runs.

Same idea. About a billion times more of everything — mountain-sized!
"""
    )


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go inspect heads", beat="challenge")
def _():
    lesson.say(
        """
1. Try temperature 0.1 and 1.6. Which failure do you prefer?
2. Start with a phrase from a nursery rhyme, then one you invented.
3. Find a line that is almost real English.
4. Ask a grown-up what would happen if you removed the mask. Hint: the loss would look suspiciously good.
5. 🧸 **Little Kid Corner:** Tell a story one letter at a time. Each turn, point at earlier letters you used as clues.
"""
    )


lesson.finish()
