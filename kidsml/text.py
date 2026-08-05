"""Character-level text tools for the "making things up" chapters.

Everything here works on *letters*, not words. A model that predicts the next letter is
small enough to train on a laptop in a minute, and its mistakes are funny rather than
boring — which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The blank/stop character. Every word secretly starts and ends with it, which is how
# the model knows where words begin and end.
STOP = "."


@dataclass
class CharVocab:
    """A two-way lookup between characters and numbers.

    Computers only do arithmetic, so every letter gets a number. ``a`` might be 1,
    ``b`` might be 2, and the blank ``.`` is always 0.
    """

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        """Build a vocabulary from whatever characters actually appear in ``text``."""
        found = sorted(set(text) - {STOP})
        return cls(chars=(STOP, *found))

    @classmethod
    def from_words(cls, words) -> "CharVocab":
        return cls.from_text("".join(words))

    def __post_init__(self):
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        """Text -> array of numbers."""
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        """Array of numbers -> text."""
        return "".join(self.itos[int(i)] for i in np.asarray(ids).ravel())


# ---------------------------------------------------------------------------
# Counting pairs (chapter 21)
# ---------------------------------------------------------------------------


def bigram_counts(words, vocab: CharVocab) -> np.ndarray:
    """Count how often each letter follows each other letter.

    Returns an ``(V, V)`` table where ``counts[a, b]`` is "how many times ``b`` came
    straight after ``a``". Nothing is learned here — we just tally, like tally marks
    on a whiteboard.
    """
    counts = np.zeros((len(vocab), len(vocab)), dtype=np.int64)
    for word in words:
        padded = STOP + word + STOP
        for a, b in zip(padded, padded[1:]):
            counts[vocab.stoi[a], vocab.stoi[b]] += 1
    return counts


def counts_to_probs(counts: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Turn tallies into probabilities, each row adding up to 1.

    ``smoothing`` adds a fake tally mark everywhere so nothing is ever *impossible* —
    otherwise one unseen pair would make the model insist it can never happen.
    """
    p = counts.astype(float) + smoothing
    return p / p.sum(axis=1, keepdims=True)


def sample_next(probs_row: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> int:
    """Pick a letter at random, but weighted by the probabilities.

    ``temperature`` below 1 makes the model play it safe (boring but sensible);
    above 1 makes it take risks (weird, sometimes wonderful).
    """
    p = np.asarray(probs_row, dtype=float).clip(1e-12)
    if temperature != 1.0:
        p = p ** (1.0 / max(temperature, 1e-3))
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def sample_bigram(probs: np.ndarray, vocab: CharVocab, rng: np.random.Generator | None = None,
                  temperature: float = 1.0, max_len: int = 20) -> str:
    """Invent one word: start at the blank, keep drawing letters until the blank returns."""
    rng = rng or np.random.default_rng()
    out, current = [], vocab.stoi[STOP]
    for _ in range(max_len):
        current = sample_next(probs[current], rng, temperature)
        if current == vocab.stoi[STOP]:
            break
        out.append(current)
    return vocab.decode(out)


def bigram_nll(words, probs: np.ndarray, vocab: CharVocab) -> float:
    """How surprised the model is by real words. Lower is better.

    This is the same "loss" number the neural models report, so chapter 22 can say
    honestly whether it beat chapter 21.
    """
    total, count = 0.0, 0
    for word in words:
        padded = STOP + word + STOP
        for a, b in zip(padded, padded[1:]):
            total += -np.log(probs[vocab.stoi[a], vocab.stoi[b]])
            count += 1
    return float(total / max(count, 1))


# ---------------------------------------------------------------------------
# Context windows (chapters 22 and 23)
# ---------------------------------------------------------------------------


def make_context_dataset(words, vocab: CharVocab, block_size: int = 3):
    """Slide a window over each word to build ``(context -> next letter)`` examples.

    With ``block_size=3`` the word ``cat`` produces:

    ==========  ======
    context     answer
    ==========  ======
    ``...``     c
    ``..c``     a
    ``.ca``     t
    ``cat``     .
    ==========  ======

    Returns ``(X, y)`` with ``X`` of shape ``(n, block_size)``.
    """
    xs, ys = [], []
    for word in words:
        context = [vocab.stoi[STOP]] * block_size
        for ch in word + STOP:
            ix = vocab.stoi[ch]
            xs.append(context.copy())
            ys.append(ix)
            context = context[1:] + [ix]
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def make_stream_dataset(text: str, vocab: CharVocab, block_size: int = 32):
    """Chop a long piece of text into overlapping windows for the Transformer chapter.

    Returns ``(X, Y)`` where ``Y`` is ``X`` shifted one step left: at every position the
    model tries to guess the very next character.
    """
    ids = vocab.encode(text)
    n = len(ids) - block_size - 1
    if n <= 0:
        raise ValueError("text is shorter than one block")
    X = np.stack([ids[i : i + block_size] for i in range(n)])
    Y = np.stack([ids[i + 1 : i + 1 + block_size] for i in range(n)])
    return X, Y


def train_test_split_words(words, frac: float = 0.9, seed: int = 0):
    """Hide some words from the model so we can find out if it really learned anything."""
    rng = np.random.default_rng(seed)
    words = list(words)
    rng.shuffle(words)
    cut = int(len(words) * frac)
    return words[:cut], words[cut:]


# ---------------------------------------------------------------------------
# Handy groupings for pictures
# ---------------------------------------------------------------------------

VOWELS = set("aeiou")


def letter_groups(vocab: CharVocab) -> list[str]:
    """Label each vocabulary entry as vowel / consonant / blank.

    Chapter 22 colours the learned embedding plot with this — and the vowels turn out
    to have grouped themselves together, without anyone ever mentioning vowels.
    """
    groups = []
    for c in vocab.chars:
        if c == STOP:
            groups.append("blank")
        elif c in VOWELS:
            groups.append("vowel")
        else:
            groups.append("consonant")
    return groups
