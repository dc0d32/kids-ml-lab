# %% [markdown]
# # Chapter 24 · Paying Attention
#
# ### A tiny Transformer, with its attention shown live.
#
# *Part 6 · Making things up*
#
# ---
#
# Here comes the shiny machine. The **T** in GPT stands for **Transformer**.

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from kidsml.plots import ACCENT, COOL, MUTED, WARM, heatmap, loss_curve, use_house_style
from kidsml.text import counts_to_probs

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Yours is tiny, and it reads nursery rhymes and fables, but the idea is the same.
#
# The game still has not changed: **guess the next letter**. Chapter 22 counted one letter
# back. Chapter 23 used a fixed memory window. Chapter 24 lets each spot choose which
# earlier spots matter. New engine, same race!
#
# > 🧸 **Little Kid Corner** — When you guess the next word in a story, your eyes zip back
# > to useful clues. Maybe the clue is next door. Maybe it is two sentences back waving a
# > tiny flag.

# %% [markdown]
# ## ✏️ Work it out
#
# Before the mechanism, grab the real question:
#
# **I am about to guess the next letter. Which earlier letters should I look at?**
#
# In `the cat sat on the m`, the useful clue may be far back, sitting on `cat`. A fixed
# window of three cannot reach that far. Attention can look anywhere inside its block.
#
# Think of query/key/value like sticky notes:
# - the current spot writes a **query**: what clue do I need?
# - earlier spots wear **keys**: what clue am I?
# - their **values** are the content the model can mix in.

# %%
attention_toy = pd.DataFrame(
    [["earlier t", 1, 10], ["earlier h", 2, 20], ["earlier e", 1, 30]],
    columns=["place", "key-match chips", "value number"],
)
attention_toy

# %%
weights = np.array([1, 2, 1]) / 4
values = np.array([10, 20, 30])
print("weighted average =", float((weights * values).sum()))

# %% [markdown]
# > 📖 **Grown-ups call this:** **Query, key, value** means: one position holds up a
# > question, earlier positions wear labels, and the model copies more content from labels
# > that match the question. Softmax turns key-match scores into the weights that do the
# > copying.

# %% [markdown]
# ## 👀 Take a look
#
# A mask is a cover sheet for the score table. A **causal** mask blocks every future
# square, because future letters would hand over the answer. That is data leakage from
# Chapter 11 wearing a new costume.

# %%
mask = np.tril(np.ones((8, 8)))
fig, ax = plt.subplots(figsize=(4.8, 4.2))
heatmap(mask, xlabels=list(range(1, 9)), ylabels=list(range(1, 9)), ax=ax, title="Causal mask: bright cells are allowed")
ax.set_xlabel("place it wants to look")
ax.set_ylabel("place making a guess")
plt.show()

# %%
print("scores = query @ key.T / sqrt(head_size)")
print("scores = scores.masked_fill(future_cells, -1e9)")
print("weights = softmax(scores)")
print("out = weights @ value")

# %% [markdown]
# The scaling by `sqrt(head_size)` keeps the numbers from getting huge. It is a technical
# stabiliser, not a new idea: a seatbelt for the scores.

# %% [markdown]
# ## 🎛️ Your turn
#
# Train a tiny Transformer on about 17KB of rhymes and fables. Small text, real machinery.
# An **attention head** is one separate clue-lookback machine; several heads can look for
# different habits at the same time.

# %%
text = (load_corpus("rhymes") + "\n" + load_corpus("fables")).lower()
bundle = train_transformer_language_model(text, block_size=32, embed_dim=48, n_heads=4, n_layers=1, steps=900, batch_size=64, seed=4)
print("characters read:", len(text))
print("parameters:", bundle.model.n_parameters())
print("held-out surprise:", round(bundle.test_loss, 3))

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
loss_curve(bundle.losses, ax=ax, title="Tiny Transformer training", ylabel="surprise")
plt.show()

# %%
made = generate_transformer(bundle, start="the ", temperature=0.9, length=220, seed=5)
print(made)

# %%
def shown_char(ch):
    if ch == "\n":
        return "↵"
    if ch == " ":
        return "sp"
    return ch

chars, heads = attention_snapshot(bundle, made)
labels = [shown_char(c) for c in chars]
head = 0
fig, ax = plt.subplots(figsize=(7, 5))
heatmap(heads[head], xlabels=labels, ylabels=labels, ax=ax, title="Attention map, head 1")
ax.set_xlabel("looked-at earlier character")
ax.set_ylabel("character doing the looking")
plt.show()

# %%
position = len(chars) - 1
row = heads[head, position]
order = np.argsort(row)[::-1][:5]
for i in order:
    print("looked at", repr(chars[int(i)]), "with weight", round(float(row[int(i)]), 3))

# %% [markdown]
# Tiny heads often attend to nearby letters or spaces. Do not force a story onto every
# square; that is attention glazing in a lab coat.

# %% [markdown]
# ## 💻 In real code
#
# Compare the whole Part 6 ladder on the same held-out text. Watch the bars step down!

# %%
mlp = train_text_mlp(text, block_size=3, embed_dim=8, hidden=80, steps=700, batch_size=256, seed=2)
counts = stream_bigram_counts(bundle.train_text, bundle.vocab)
probs = counts_to_probs(counts, smoothing=1.0)

random_loss = random_nll(bundle.vocab)
bigram_loss = stream_bigram_nll(bundle.test_text, probs, bundle.vocab)
mlp_loss = mlp.test_loss
transformer_loss = bundle.test_loss

pd.DataFrame(
    {"model": ["random", "bigram", "MLP", "Transformer"], "held-out surprise": [random_loss, bigram_loss, mlp_loss, transformer_loss]}
)

# %%
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(["random", "bigram", "MLP", "Transformer"], [random_loss, bigram_loss, mlp_loss, transformer_loss], color=[MUTED, COOL, WARM, ACCENT])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("The Part 6 ladder")
plt.show()

# %%
print("BIGRAM")
print(sample_stream_bigram(probs, bundle.vocab, start="the ", temperature=0.9, length=150, seed=5))
print("\nMLP")
print(sample_text_mlp(mlp, start="the ", temperature=0.9, length=150, seed=5))
print("\nTRANSFORMER")
print(made[:220])

# %% [markdown]
# GPT-class models use the same code shape, then crank every dial far up: hundreds of
# billions of parameters, huge text collections, many GPUs, and long training runs. Same
# idea. About a billion times more of everything — mountain-sized!

# %% [markdown]
# ## 🏆 Go further
#
# 1. Try temperature 0.1 and 1.6. Which failure do you prefer?
# 2. Start with a phrase from a nursery rhyme, then one you invented.
# 3. Find a line that is almost real English, then watch where it wobbles.
# 4. Train with the mask removed and watch the loss look suspiciously good while generation
#    falls apart.
# 5. 🧸 **Little Kid Corner:** Tell a story one letter at a time. Each turn, point at
#    earlier letters you used as clues.

# %%
from kidsml import workbook
workbook.render(24)

# %% [markdown]
# ---
# **Next up:** Chapter 25 · *So What Now?* — the victory lap and the honest limits.
