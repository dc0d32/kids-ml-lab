# %% [markdown]
# # Chapter 22 · The Bigram Babbler
#
# ### Count letter pairs, roll a die, invent words.
#
# *Part 6 · Making things up*
#
# ---
#
# This notebook is the same chapter as the app, with the code showing.

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kidsml.datasets import load_words
from kidsml.langmodels import random_nll, sample_bigram_trace, top_letters
from kidsml.plots import heatmap, use_house_style
from kidsml.text import CharVocab, STOP, bigram_counts, bigram_nll, counts_to_probs, sample_bigram, train_test_split_words

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# You want to know how ChatGPT works. Peek behind the shiny robot curtain: **it guesses
# the next letter, scoots forward, then guesses again.**
#
# The next three chapters build three guessers. This first one fits on a kitchen-table
# tally chart. Tiny! Loudly useful!
#
# Nothing is trained here. We count letter pairs and let the counts steer the dice.
#
# > 🧸 **Little Kid Corner** — Line up letter cards from a name like dominoes. Every time
# > two cards touch, drop a tally mark in that box. Later, draw the next card from the
# > busiest boxes.

# %% [markdown]
# ## ✏️ Work it out
#
# Names get bumper pads: a blank **`.`** at the start and another at the finish.
#
# `mia` enters the tally machine as `.mia.`. That lets the chart learn what starts names
# and what ends names.

# %%
tiny = pd.DataFrame(
    [[".mia.", ". → m, m → i, i → a, a → ."], [".mo.", ". → m, m → o, o → ."], [".mae.", ". → m, m → a, a → e, e → ."]],
    columns=["padded word", "pairs you tally"],
)
tiny

# %%
print("For these three words, the '. → m' box gets 3 tally marks.")
print("The m → a, m → i, and m → o boxes get 1 each.")

# %% [markdown]
# > 📖 **Grown-ups call this:** A **bigram** is a pair of touching characters. `m → a` is
# > one bigram.

# %% [markdown]
# ## 👀 Take a look
#
# Build the full tally chart on first names. Now the little boxes become a heat map.

# %%
words = load_words("names")
vocab = CharVocab.from_words(words)
train, test = train_test_split_words(words, frac=0.9, seed=0)
counts = bigram_counts(train, vocab)
# Add one tiny fake tally to every box before dividing. That keeps unseen pairs rare
# instead of impossible.
probs = counts_to_probs(counts, smoothing=1.0)

print(len(words), "names")
print(len(vocab), "characters, including the blank")

# %%
fig, ax = plt.subplots(figsize=(7, 6))
heatmap(counts, xlabels=vocab.chars, ylabels=vocab.chars, ax=ax, title="Name bigram tallies")
ax.set_xlabel("next letter")
ax.set_ylabel("letter before it")
plt.show()

# %% [markdown]
# A bright square is a footprint: **that pair happened a lot**. No magic wand. Tally
# marks piled up.
#
# The row for `.` means letters that start names. The column for `.` means letters that end
# names. They leave different tracks.

# %%
start_order = np.argsort(counts[vocab.stoi[STOP]])[::-1][:8]
end_order = np.argsort(counts[:, vocab.stoi[STOP]])[::-1][:8]

pd.DataFrame(
    {
        "starts names": [vocab.itos[int(i)] for i in start_order],
        "start tallies": [int(counts[vocab.stoi[STOP], int(i)]) for i in start_order],
        "ends names": [vocab.itos[int(i)] for i in end_order],
        "end tallies": [int(counts[int(i), vocab.stoi[STOP]]) for i in end_order],
    }
)

# %% [markdown]
# > 💡 **Aha!** The start list and end list are different! The chart discovered a real
# > fact about names by counting footsteps.

# %% [markdown]
# ## 🎛️ Your turn
#
# Grab one row. Here is the `q` row, where spelling habits leave a giant clue.

# %%
row = probs[vocab.stoi["q"]]
letters, values = top_letters(row, vocab, n=10)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(letters, values)
ax.set_ylabel("probability")
ax.set_title("After q, what comes next?")
plt.show()

# %%
rng = np.random.default_rng(4)
for temperature in [0.25, 0.9, 1.6]:
    samples = []
    for _ in range(10):
        samples.append(sample_bigram(probs, vocab, rng=rng, temperature=temperature, max_len=18))
    print("temperature", temperature, "→", ", ".join(samples))

# %%
word, trace = sample_bigram_trace(probs, vocab, seed=7, temperature=0.9, max_len=12)
print("made:", word)
pd.DataFrame(
    [{"after": step["after"], "picked": step["picked"], "probability": round(step["probability"], 3)} for step in trace]
)

# %% [markdown]
# Low temperature hugs safe choices. High temperature flings open the weird door.

# %% [markdown]
# ## 💻 In real code
#
# The score is average surprise on hidden names. Lower means fewer tiny shocks.

# %%
loss = bigram_nll(test, probs, vocab)
random_loss = random_nll(vocab)
print("knows-nothing random surprise:", round(random_loss, 3))
print("bigram surprise:", round(loss, 3))

# %%
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(["random", "bigram"], [random_loss, loss], color=["#94A3B8", "#10B981"])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("Counting beats knowing nothing")
plt.show()

# %%
counts = bigram_counts(train, vocab)
# `smoothing=1` adds one fake tally to every box before dividing, so an unseen pair is
# rare instead of impossible.
probs = counts_to_probs(counts, smoothing=1)
loss = bigram_nll(test, probs, vocab)
loss

# %% [markdown]
# > ⚠️ **Careful** The babbler sounds almost name-like, but it only looks one letter back.
# > It has no idea what happened three letters ago. That wall is waiting in Chapter 23.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find the most predictable next letter. Try `q`, `x`, and `.`.
# 2. Make the babbler produce a real name by luck.
# 3. Set temperature near 0. Why does it keep marching back to the same safe choices?
# 4. 🧸 **Little Kid Corner:** Make an uneven die with paper scraps. More scraps for common
#    letters. Draw, write, repeat.

# %%
from kidsml import workbook
workbook.render(22)

# %% [markdown]
# ---
# **Next up:** Chapter 23 · *Giving It a Memory* — the same guessing game, with three
# letters of memory.
