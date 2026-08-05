# %% [markdown]
# # Chapter 23 · Giving It a Memory
#
# ### It discovers vowels on its own. Nobody told it.
#
# *Part 6 · Making things up*
#
# ---
#
# Chapter 22 counted one letter back. This chapter gives the guesser three letters of memory.

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kidsml.datasets import load_words
from kidsml.langmodels import embedding_points, random_nll, sample_mlp, train_mlp_language_model
from kidsml.plots import ACCENT, COOL, MUTED, WARM, loss_curve, use_house_style
from kidsml.text import bigram_counts, bigram_nll, counts_to_probs, letter_groups, sample_bigram

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# Chapter 22 could see **one** letter back. Let's give it **three**.
#
# That one change takes us from a tally chart to an actual neural network. The job is still
# the same: **guess the next letter**.
#
# > 🧸 **Little Kid Corner** — Instead of asking only the kid beside you, ask the last
# > three kids in line what letters they are holding. Then guess the next card.

# %% [markdown]
# ## ✏️ Work it out
#
# Slide a three-letter window over `cat`.

# %%
windows = pd.DataFrame(
    [["...", "c"], ["..c", "a"], [".ca", "t"], ["cat", "."]],
    columns=["input memory", "answer"],
)
windows

# %% [markdown]
# > 📖 **Grown-ups call this:** An **embedding** is a tiny list of numbers for one
# > character. The model gets to choose those numbers while it learns.
#
# > 📖 **Grown-ups call this:** **Softmax** turns any list of scores into probabilities
# > that add to 1. Bigger scores get more probability.

# %% [markdown]
# ## 👀 Take a look
#
# The pipeline is readable now:
#
# Three letters in → each becomes a small vector → glue the vectors together → hidden layer
# with `tanh` → 27 scores → softmax probabilities.

# %%
pipeline = pd.DataFrame(
    [
        ["input", "`.ca`", "three remembered letters"],
        ["embedding table", "three tiny vectors", "numbers the model learned"],
        ["hidden layer", "tanh", "mix the clues"],
        ["output", "27 probabilities", "one for each next character"],
    ],
    columns=["stage", "what it holds", "why it matters"],
)
pipeline

# %% [markdown]
# Train a model with **2D embeddings** so we can draw the letters.

# %%
words = load_words("names")
bundle2 = train_mlp_language_model(words, block_size=3, embed_dim=2, hidden=96, n_words=8000, steps=1200, lr=0.03, seed=1)
print("held-out surprise:", round(bundle2.test_loss, 3))
print("parameters:", bundle2.model.n_parameters())

# %%
points = embedding_points(bundle2)
groups = letter_groups(bundle2.vocab)
colours = []
for group in groups:
    if group == "vowel":
        colours.append(WARM)
    elif group == "blank":
        colours.append(MUTED)
    else:
        colours.append(COOL)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(points[:, 0], points[:, 1], c=colours, s=130, edgecolors="white", linewidths=1.0)
for i, ch in enumerate(bundle2.vocab.chars):
    ax.text(points[i, 0], points[i, 1], ch, ha="center", va="center", fontsize=10, weight="bold")
ax.set_xlabel("embedding number 1")
ax.set_ylabel("embedding number 2")
ax.set_title("Letters after training")
plt.show()

# %% [markdown]
# > 💡 **Aha!** The vowels often land near each other. Nobody told the model what a vowel
# > is. It grouped them because they behave alike when guessing the next letter.
#
# This is a real training run. It is usually clear with this seed, but not every random
# start makes a perfect picture.

# %% [markdown]
# ## 🎛️ Your turn
#
# Train three fixed-window models. Only the memory length changes.

# %%
models = {}
for block_size in [1, 3, 5]:
    models[block_size] = train_mlp_language_model(
        words, block_size=block_size, embed_dim=8, hidden=96, n_words=8000, steps=1000, lr=0.02, seed=10 + block_size
    )

pd.DataFrame(
    {"memory": ["1 letter", "3 letters", "5 letters"], "held-out surprise": [models[1].test_loss, models[3].test_loss, models[5].test_loss]}
)

# %%
for block_size in [1, 3, 5]:
    names = []
    for i in range(8):
        names.append(sample_mlp(models[block_size], start="ma", temperature=0.85, seed=i, max_len=18))
    print("memory", block_size, "→", ", ".join(names))

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
loss_curve(models[3].losses, ax=ax, title="Training loss, block size 3", ylabel="surprise")
plt.show()

# %% [markdown]
# ## 💻 In real code
#
# Does the neural net beat the Chapter 22 bigram on the same held-out words?

# %%
main = models[3]
vocab = main.vocab
counts = bigram_counts(main.train_words, vocab)
probs = counts_to_probs(counts, smoothing=1.0)
bigram_loss = bigram_nll(main.test_words, probs, vocab)
random_loss = random_nll(vocab)

print("random:", round(random_loss, 3))
print("bigram:", round(bigram_loss, 3))
print("MLP:", round(main.test_loss, 3))

# %%
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(["random", "bigram", "MLP"], [random_loss, bigram_loss, main.test_loss], color=[MUTED, COOL, ACCENT])
ax.set_ylabel("average surprise (lower is better)")
ax.set_title("Three-letter memory helps")
plt.show()

# %%
rng = np.random.default_rng(4)
bigram_names = []
for _ in range(8):
    bigram_names.append(sample_bigram(probs, vocab, rng=rng, temperature=0.85, max_len=18))
mlp_names = []
for i in range(8):
    mlp_names.append(sample_mlp(main, temperature=0.85, seed=i, max_len=18))

print("bigram:", ", ".join(bigram_names))
print("MLP:   ", ", ".join(mlp_names))

# %%
from kidsml.text import make_context_dataset
from kidsml.langmodels import ContextMLP

X, y = make_context_dataset(words[:5], vocab, block_size=3)
model = ContextMLP(vocab_size=len(vocab), block_size=3)
print(X.shape, y.shape)

# %% [markdown]
# > ⚠️ **Careful** A block size of 3 is a hard wall. The fourth-back letter is invisible.
# > Chapter 24 fixes that in a different way.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Find a starter that the MLP finishes like a real name.
# 2. Compare block size 1 and 5 using the same starter.
# 3. Lower the temperature until the model gets boring. What name habit does it overuse?
# 4. 🧸 **Little Kid Corner:** Play telephone, but each person may ask only the last three
#    people what they heard.

# %%
from kidsml import workbook
workbook.render(23)

# %% [markdown]
# ---
# **Next up:** Chapter 24 · *Paying Attention* — the T in GPT, built tiny.
