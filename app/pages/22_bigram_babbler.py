"""Chapter 22 · The Bigram Babbler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from kidsml import lesson
from kidsml.datasets import load_words
from kidsml.langmodels import random_nll, sample_bigram_trace, top_letters
from kidsml.plots import heatmap
from kidsml.text import STOP, CharVocab, bigram_counts, bigram_nll, counts_to_probs, sample_bigram, train_test_split_words

lesson.begin(22)


@st.cache_data(show_spinner=False)
def bigram_bundle(corpus: str):
    words = load_words("names") if corpus == "names" else load_words(corpus)
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


def format_letter(ch: str) -> str:
    return label(ch)


@lesson.step("The secret is smaller than it sounds", beat="hook")
def _():
    lesson.say(
        """
You want to know how ChatGPT works. Peek behind the shiny robot curtain: **it guesses the next letter, scoots forward, then guesses again.**

The next three chapters build three guessers. This first one fits on a kitchen-table tally chart. Tiny! Loudly useful!
"""
    )
    lesson.kid_corner(
        "Line up letter cards from a name like dominoes. Every time two cards touch, drop a tally mark in that box. Later, draw the next card from the busiest boxes."
    )


@lesson.step("Tally marks before code", beat="byhand")
def _():
    lesson.say("Names get bumper pads: a blank **`.`** at the start and another at the finish. So `mia` enters the tally machine as `.mia.`.")
    tiny = pd.DataFrame(
        [[".mia.", ". → m, m → i, i → a, a → ."], [".mo.", ". → m, m → o, o → ."], [".mae.", ". → m, m → a, a → e, e → ."]],
        columns=["padded word", "pairs you tally"],
    )
    st.dataframe(tiny, hide_index=True, width="stretch")
    guess = lesson.predict(
        "For these three words, which tally box gets the most marks?",
        [". → m", "m → a", "a → ."],
        correct=0,
        why="All three padded words roll out of the start blank into `m`, so the `. → m` box gets 3 marks.",
        key="ch21_tally_box",
    )
    if guess is None:
        return
    st.info("The `m → a`, `m → i`, and `m → o` boxes get 1 each. The chart is already taking shape!")
    lesson.jargon("bigram", "A pair of touching characters. `m → a` is one bigram.")


@lesson.step("The whole tally chart", beat="seeit")
def _():
    lesson.say("A bright square is a footprint: **that pair happened a lot**. No magic wand. Tally marks piled up.")
    _, vocab, _, _, counts, _ = bigram_bundle("names")
    fig, ax = lesson.figure(7, 6)
    heatmap(counts, xlabels=[label(c) for c in vocab.chars], ylabels=[label(c) for c in vocab.chars], ax=ax, title="Name bigram tallies")
    ax.set_xlabel("next letter")
    ax.set_ylabel("letter before it")
    lesson.show(fig)
    lesson.look_for("the row and column for `.`. Starts of names and ends of names leave different tracks.")

    row_col, col_col = st.columns(2)
    start_order = np.argsort(counts[vocab.stoi[STOP]])[::-1][:8]
    end_order = np.argsort(counts[:, vocab.stoi[STOP]])[::-1][:8]
    start_row = pd.DataFrame({"starts names": [label(vocab.itos[i]) for i in start_order], "tallies": [int(counts[vocab.stoi[STOP], i]) for i in start_order]})
    end_col = pd.DataFrame({"ends names": [label(vocab.itos[i]) for i in end_order], "tallies": [int(counts[i, vocab.stoi[STOP]]) for i in end_order]})
    with row_col:
        st.markdown("**The row for `.`:** letters that start names")
        st.dataframe(start_row, hide_index=True)
    with col_col:
        st.markdown("**The column for `.`:** letters that end names")
        st.dataframe(end_col, hide_index=True)
    lesson.aha("Those two lists are different! The chart discovered a real fact about names by counting footsteps.")


@lesson.step("What follows q?", beat="play")
def _():
    lesson.say("Grab one row of the tally chart. That row becomes an uneven die for the next letter.")
    guess = lesson.predict(
        "Before you see the row: after `q`, what do you expect the busiest next letter to be?",
        ["a", "u", "the stop dot"],
        correct=1,
        why="This is the famous q → u habit. The model was not handed spelling rules; it bumped into them by counting.",
        key="ch21_q_next",
    )
    if guess is None:
        return

    corpus = st.selectbox("Corpus", ["names", "rhymes", "fables"], index=0, key="ch21_row_corpus")
    _, vocab, _, _, _, probs = bigram_bundle(corpus)
    letters = list(vocab.chars)
    default = letters.index("q") if "q" in letters else 0
    picked = st.selectbox("After this character, what tends to come next?", letters, index=default, format_func=format_letter, key="ch21_picked_letter")
    row = probs[vocab.stoi[picked]]
    names, values = top_letters(row, vocab, n=min(10, len(vocab)))
    fig, ax = lesson.figure(7, 3.5)
    ax.bar([label(c) for c in names], values, color="#3B82F6")
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel("probability")
    ax.set_title(f"After {label(picked)!r}, the likely next characters")
    lesson.show(fig)
    lesson.look_for("the tallest bar. The babbler grabs choices from this row, not from a grammar rulebook.")


@lesson.step("Temperature changes the dice", beat="play")
def _():
    lesson.say("Temperature changes how daring the uneven die feels. Low temperature hugs the busiest boxes; high temperature lets odd little boxes jump into the game.")
    guess = lesson.predict(
        "What will very low temperature do to the invented names?",
        ["Make safer, more repeated choices", "Make wilder spellings", "Make every letter equally likely"],
        correct=0,
        why="Low temperature sharpens the row like a spotlight, so the same likely letters win again and again.",
        key="ch21_temperature",
    )
    if guess is None:
        return
    knobs, picture = lesson.controls()
    with knobs:
        corpus = st.selectbox("Corpus", ["names", "rhymes", "fables"], index=0, key="ch21_temp_corpus")
        temperature = st.slider("Temperature", 0.05, 2.0, 0.9, 0.05, key="ch21_temp")
        seed = st.slider("Random seed", 0, 99, 4, key="ch21_temp_seed")
    with picture:
        _, vocab, _, _, _, probs = bigram_bundle(corpus)
        word, trace = sample_bigram_trace(probs, vocab, seed=seed, temperature=temperature, max_len=14)
        st.markdown(f"**One invention:** `{word or '(blank)'}`")
        trace_rows = [{"after": label(step["after"]), "picked": label(step["picked"]), "probability": round(step["probability"], 3)} for step in trace]
        st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")
        lesson.look_for("probabilities near 1.0 when the temperature is low, and riskier picks when it is high.")


@lesson.step("Press the babble button", beat="play")
def _():
    lesson.say("Now let the chart roll its uneven die over and over. It keeps clattering until it hits the stop dot.")
    corpus = st.selectbox("Corpus", ["names", "rhymes", "fables"], index=0, key="ch21_gen_corpus")
    temperature = st.slider("Temperature", 0.05, 2.0, 0.9, 0.05, key="ch21_gen_temp")
    seed = st.slider("Random seed", 0, 99, 4, key="ch21_gen_seed")
    n_samples = st.slider("How many inventions?", 5, 15, 10, key="ch21_gen_count")
    if st.button("Generate inventions", key="ch21_generate"):
        st.session_state["ch21_generated"] = True
    if not st.session_state.get("ch21_generated", False):
        st.caption("Press the button when you are ready to hear the babbler. Lowkey, it is a letter carnival.")
        return
    _, vocab, _, _, _, probs = bigram_bundle(corpus)
    rng = np.random.default_rng(seed)
    samples = [sample_bigram(probs, vocab, rng=rng, temperature=temperature, max_len=18) for _ in range(n_samples)]
    st.write(" · ".join(s if s else "(blank)" for s in samples))
    lesson.look_for("names that almost work. The tally chart is launching pronounceable little accidents.")


@lesson.step("A score for surprise", beat="forreal")
def _():
    lesson.say("A model earns a better score when hidden words surprise it less. Lower bars mean fewer tiny shocks.")
    _, vocab, _, test, _, probs = bigram_bundle("names")
    loss = bigram_nll(test, probs, vocab)
    random_loss = random_nll(vocab)
    fig, ax = lesson.figure(5.5, 3.5)
    ax.bar(["random", "bigram"], [random_loss, loss], color=["#94A3B8", "#10B981"])
    ax.set_ylabel("average surprise (lower is better)")
    ax.set_title("Counting beats knowing nothing")
    lesson.show(fig)
    lesson.look_for("the bigram bar sitting lower than the random bar. Counting pairs helped!")
    a, b = st.columns(2)
    a.metric("Bigram surprise on hidden words", f"{loss:.3f}")
    b.metric("Knows-nothing random surprise", f"{random_loss:.3f}")


@lesson.step("The whole program", beat="forreal")
def _():
    lesson.say("The whole model snaps into three lines: count pairs, turn counts into probabilities, score hidden words.")
    st.code(
        """
counts = bigram_counts(train_words, vocab)
probs = counts_to_probs(counts, smoothing=1)
loss = bigram_nll(hidden_words, probs, vocab)
""",
        language="python",
    )
    lesson.look_for("the middle line. The tally counts become the uneven dice that make the babbler move.")
    lesson.careful("The babbler sounds almost name-like, but it only looks one letter back. It has no idea what happened three letters ago. That wall is waiting in Chapter 23.")


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go make nonsense", beat="challenge")
def _():
    lesson.say(
        """
1. Find the most predictable next letter. Try `q`, `x`, and `.`.
2. Make the babbler produce a real name by luck.
3. Set temperature near 0. Why does it keep making the same safe choices?
4. 🧸 **Little Kid Corner:** Make an uneven die with paper scraps. More scraps for common letters. Draw, write, repeat.
"""
    )


lesson.finish()
