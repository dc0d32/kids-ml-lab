"""Tiny character language models for Part 6.

The chapters in Part 6 all play the same game: guess the next character. This file keeps
that shared code in one place so the app pages and notebooks cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from kidsml.text import CharVocab, STOP, counts_to_probs, train_test_split_words


torch.set_num_threads(1)


@dataclass
class MLPBundle:
    """A trained fixed-window character model plus the numbers worth showing."""

    model: nn.Module
    vocab: CharVocab
    block_size: int
    losses: list[float]
    test_loss: float
    train_loss: float
    train_words: list[str] | None = None
    test_words: list[str] | None = None
    train_text: str = ""
    test_text: str = ""


@dataclass
class TransformerBundle:
    """A trained tiny Transformer and its held-out score."""

    model: nn.Module
    vocab: CharVocab
    block_size: int
    losses: list[float]
    test_loss: float
    train_text: str
    test_text: str


class ContextMLP(nn.Module):
    """A Bengio-style character model: context letters in, next-letter scores out."""

    def __init__(self, vocab_size: int, block_size: int = 3, embed_dim: int = 2, hidden: int = 80):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(block_size * embed_dim, hidden)
        self.output = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        flat = emb.reshape(emb.shape[0], -1)
        h = torch.tanh(self.hidden(flat))
        return self.output(h)

    def n_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total


class CausalSelfAttention(nn.Module):
    """One masked attention layer. Each position may look only backward."""

    def __init__(self, embed_dim: int, n_heads: int, block_size: int):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must divide evenly by n_heads")
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, steps, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(width, dim=2)

        q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        blocked = self.mask[:steps, :steps] == 0
        scores = scores.masked_fill(blocked, -1e9)
        weights = F.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.proj(out), weights


class TransformerBlock(nn.Module):
    """Attention, then a small per-position neural net."""

    def __init__(self, embed_dim: int, n_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        attn_out, weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, weights


class TinyTransformerLM(nn.Module):
    """A tiny GPT-shaped character model."""

    def __init__(self, vocab_size: int, block_size: int = 32, embed_dim: int = 48, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(TransformerBlock(embed_dim, n_heads, block_size))
        self.ln = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None, return_attention: bool = False):
        batch, steps = x.shape
        positions = torch.arange(steps, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)[None, :, :]
        attentions = []
        for block in self.blocks:
            h, weights = block(h)
            attentions.append(weights)
        logits = self.output(self.ln(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if return_attention:
            return logits, loss, attentions
        return logits, loss

    def n_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total


def random_nll(vocab: CharVocab) -> float:
    """Loss for a model that knows nothing and guesses every character equally."""
    return float(math.log(len(vocab)))


def _seed_everything(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _torch_xy(X, y):
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def _mlp_loss(model: ContextMLP, X, y, batch_size: int = 4096) -> float:
    model.eval()
    x, target = _torch_xy(X, y)
    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size]
            yb = target[start : start + batch_size]
            loss = F.cross_entropy(model(xb), yb, reduction="sum")
            total += float(loss.item())
            seen += int(len(xb))
    return total / max(seen, 1)


def train_mlp_language_model(
    words,
    block_size: int = 3,
    embed_dim: int = 2,
    hidden: int = 80,
    n_words: int = 6000,
    steps: int = 900,
    batch_size: int = 256,
    lr: float = 0.03,
    seed: int = 1,
) -> MLPBundle:
    """Train the fixed-window model on names."""
    from kidsml.text import make_context_dataset

    rng = _seed_everything(seed)
    vocab = CharVocab.from_words(words)
    words = list(words)
    order = rng.permutation(len(words))
    words = [words[int(i)] for i in order[: min(n_words, len(words))]]
    train_words, test_words = train_test_split_words(words, frac=0.9, seed=seed)
    X_train, y_train = make_context_dataset(train_words, vocab, block_size)
    X_test, y_test = make_context_dataset(test_words, vocab, block_size)

    model = ContextMLP(len(vocab), block_size=block_size, embed_dim=embed_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_train, y_train_t = _torch_xy(X_train, y_train)
    losses = []

    for step in range(steps):
        idx = torch.randint(0, len(x_train), (batch_size,))
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits, y_train_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 60) == 0 or step == steps - 1:
            losses.append(float(loss.item()))

    return MLPBundle(
        model=model.eval(),
        vocab=vocab,
        block_size=block_size,
        losses=losses,
        train_loss=_mlp_loss(model, X_train, y_train),
        test_loss=_mlp_loss(model, X_test, y_test),
        train_words=train_words,
        test_words=test_words,
    )


def make_text_context_dataset(text: str, vocab: CharVocab, block_size: int = 3):
    """Continuous text version of a fixed context window."""
    ids = vocab.encode(text)
    n = len(ids) - block_size
    X = np.empty((n, block_size), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        X[i] = ids[i : i + block_size]
        y[i] = ids[i + block_size]
    return X, y


def train_text_mlp(
    text: str,
    block_size: int = 3,
    embed_dim: int = 8,
    hidden: int = 80,
    steps: int = 700,
    batch_size: int = 256,
    lr: float = 0.02,
    seed: int = 2,
) -> MLPBundle:
    """Train a fixed-window model on a stream of characters."""
    rng = _seed_everything(seed)
    text = text.lower()
    vocab = CharVocab.from_text(text)
    cut = int(len(text) * 0.9)
    train_text = text[:cut]
    test_text = text[cut - block_size :]
    X_train, y_train = make_text_context_dataset(train_text, vocab, block_size)
    X_test, y_test = make_text_context_dataset(test_text, vocab, block_size)

    model = ContextMLP(len(vocab), block_size=block_size, embed_dim=embed_dim, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_train, y_train_t = _torch_xy(X_train, y_train)
    losses = []

    for step in range(steps):
        idx_np = rng.integers(0, len(x_train), size=batch_size)
        idx = torch.tensor(idx_np, dtype=torch.long)
        loss = F.cross_entropy(model(x_train[idx]), y_train_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            losses.append(float(loss.item()))

    return MLPBundle(
        model=model.eval(),
        vocab=vocab,
        block_size=block_size,
        losses=losses,
        train_loss=_mlp_loss(model, X_train, y_train),
        test_loss=_mlp_loss(model, X_test, y_test),
        train_text=train_text,
        test_text=test_text,
    )


def sample_mlp(bundle: MLPBundle, start: str = "", temperature: float = 1.0, max_len: int = 20, seed: int = 0) -> str:
    """Generate a name-like word from a fixed-window model."""
    rng = _seed_everything(seed)
    model = bundle.model.eval()
    vocab = bundle.vocab
    context = [vocab.stoi[STOP]] * bundle.block_size
    out = []

    for ch in start.lower():
        if ch in vocab.stoi and ch != STOP:
            ix = vocab.stoi[ch]
            out.append(ix)
            context = context[1:] + [ix]

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)[0].numpy()
            next_ix = _sample_logits(logits, rng, temperature)
            if next_ix == vocab.stoi[STOP]:
                break
            out.append(next_ix)
            context = context[1:] + [next_ix]
    return vocab.decode(out)


def sample_text_mlp(bundle: MLPBundle, start: str = "the ", temperature: float = 1.0, length: int = 180, seed: int = 0) -> str:
    """Generate continuous text from the fixed-window model."""
    rng = _seed_everything(seed)
    vocab = bundle.vocab
    start = _clean_start(start.lower(), vocab)
    context = [vocab.stoi[start[-1]]] * bundle.block_size
    for ch in start[-bundle.block_size :]:
        context = context[1:] + [vocab.stoi[ch]]
    out = [vocab.stoi[ch] for ch in start]

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([context], dtype=torch.long)
            logits = bundle.model(x)[0].numpy()
            next_ix = _sample_logits(logits, rng, temperature)
            out.append(next_ix)
            context = context[1:] + [next_ix]
    return vocab.decode(out)


def embedding_points(bundle: MLPBundle):
    """The learned letter vectors as a NumPy array."""
    return bundle.model.embedding.weight.detach().numpy().copy()


def _sample_logits(logits, rng: np.random.Generator, temperature: float) -> int:
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-3)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def sample_bigram_trace(probs: np.ndarray, vocab: CharVocab, seed: int = 0, temperature: float = 1.0, max_len: int = 16):
    """Generate one word and keep the probability row used at each step."""
    rng = np.random.default_rng(seed)
    current = vocab.stoi[STOP]
    letters = []
    rows = []
    for _ in range(max_len):
        row = probs[current]
        adjusted = row.astype(float).clip(1e-12)
        adjusted = adjusted ** (1.0 / max(temperature, 1e-3))
        adjusted = adjusted / adjusted.sum()
        next_ix = int(rng.choice(len(adjusted), p=adjusted))
        rows.append(
            {
                "after": vocab.itos[current],
                "picked": vocab.itos[next_ix],
                "probability": float(adjusted[next_ix]),
                "row": adjusted.copy(),
            }
        )
        current = next_ix
        if current == vocab.stoi[STOP]:
            break
        letters.append(current)
    return vocab.decode(letters), rows


def top_letters(row, vocab: CharVocab, n: int = 8):
    """The most likely next characters in one probability row."""
    order = np.argsort(row)[::-1][:n]
    letters = []
    probs = []
    for ix in order:
        letters.append(vocab.itos[int(ix)])
        probs.append(float(row[int(ix)]))
    return letters, probs


def stream_bigram_counts(text: str, vocab: CharVocab) -> np.ndarray:
    """Bigram counts for continuous text, including spaces and newlines."""
    ids = vocab.encode(text)
    counts = np.zeros((len(vocab), len(vocab)), dtype=np.int64)
    for a, b in zip(ids, ids[1:]):
        counts[int(a), int(b)] += 1
    return counts


def stream_bigram_nll(text: str, probs: np.ndarray, vocab: CharVocab) -> float:
    """Held-out next-character loss for a continuous-text bigram model."""
    ids = vocab.encode(text)
    total = 0.0
    count = 0
    for a, b in zip(ids, ids[1:]):
        total += -math.log(float(probs[int(a), int(b)]))
        count += 1
    return total / max(count, 1)


def sample_stream_bigram(probs: np.ndarray, vocab: CharVocab, start: str = "the ", temperature: float = 1.0, length: int = 180, seed: int = 0) -> str:
    """Generate fixed-length text from continuous bigram probabilities."""
    rng = np.random.default_rng(seed)
    start = _clean_start(start.lower(), vocab)
    out = [vocab.stoi[ch] for ch in start]
    current = out[-1]
    for _ in range(length):
        row = probs[current]
        adjusted = row.astype(float).clip(1e-12)
        adjusted = adjusted ** (1.0 / max(temperature, 1e-3))
        adjusted = adjusted / adjusted.sum()
        current = int(rng.choice(len(adjusted), p=adjusted))
        out.append(current)
    return vocab.decode(out)


def train_transformer_language_model(
    text: str,
    block_size: int = 32,
    embed_dim: int = 48,
    n_heads: int = 4,
    n_layers: int = 1,
    steps: int = 900,
    batch_size: int = 64,
    lr: float = 0.003,
    seed: int = 3,
) -> TransformerBundle:
    """Train a tiny GPT-shaped model on a short character stream."""
    rng = _seed_everything(seed)
    text = text.lower()
    vocab = CharVocab.from_text(text)
    cut = int(len(text) * 0.9)
    train_text = text[:cut]
    test_text = text[cut - block_size - 1 :]
    train_ids = vocab.encode(train_text)
    test_ids = vocab.encode(test_text)

    model = TinyTransformerLM(len(vocab), block_size=block_size, embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    for step in range(steps):
        xb, yb = _stream_batch(train_ids, block_size, batch_size, rng)
        logits, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 60) == 0 or step == steps - 1:
            losses.append(float(loss.item()))

    return TransformerBundle(
        model=model.eval(),
        vocab=vocab,
        block_size=block_size,
        losses=losses,
        test_loss=transformer_nll(model, test_ids, block_size),
        train_text=train_text,
        test_text=test_text,
    )


def _stream_batch(ids, block_size: int, batch_size: int, rng: np.random.Generator):
    max_start = len(ids) - block_size - 1
    starts = rng.integers(0, max_start, size=batch_size)
    X = np.empty((batch_size, block_size), dtype=np.int64)
    Y = np.empty((batch_size, block_size), dtype=np.int64)
    for row, start in enumerate(starts):
        X[row] = ids[start : start + block_size]
        Y[row] = ids[start + 1 : start + 1 + block_size]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def transformer_nll(model: TinyTransformerLM, ids, block_size: int, batch_size: int = 128) -> float:
    """Average next-character loss over every held-out block."""
    model.eval()
    n = len(ids) - block_size - 1
    total = 0.0
    tokens = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            rows = end - start
            X = np.empty((rows, block_size), dtype=np.int64)
            Y = np.empty((rows, block_size), dtype=np.int64)
            for row, i in enumerate(range(start, end)):
                X[row] = ids[i : i + block_size]
                Y[row] = ids[i + 1 : i + 1 + block_size]
            xb = torch.tensor(X, dtype=torch.long)
            yb = torch.tensor(Y, dtype=torch.long)
            _, loss = model(xb, yb)
            total += float(loss.item()) * rows * block_size
            tokens += rows * block_size
    return total / max(tokens, 1)


def generate_transformer(bundle: TransformerBundle, start: str = "the ", temperature: float = 0.9, length: int = 220, seed: int = 0) -> str:
    """Generate text from the tiny Transformer."""
    rng = _seed_everything(seed)
    vocab = bundle.vocab
    text = _clean_start(start.lower(), vocab)
    ids = [vocab.stoi[ch] for ch in text]

    with torch.no_grad():
        for _ in range(length):
            current = ids[-bundle.block_size :]
            x = torch.tensor([current], dtype=torch.long)
            logits, _ = bundle.model(x)
            next_ix = _sample_logits(logits[0, -1].numpy(), rng, temperature)
            ids.append(next_ix)
    return vocab.decode(ids)


def attention_snapshot(bundle: TransformerBundle, text: str):
    """Return the last block of text and the attention weights for it."""
    vocab = bundle.vocab
    clean = _clean_start(text.lower(), vocab)
    ids = [vocab.stoi[ch] for ch in clean][-bundle.block_size :]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, _, attentions = bundle.model(x, return_attention=True)
    heads = attentions[-1][0].detach().numpy()
    chars = [vocab.itos[int(i)] for i in ids]
    return chars, heads


def _clean_start(start: str, vocab: CharVocab) -> str:
    cleaned = ""
    for ch in start:
        if ch in vocab.stoi:
            cleaned += ch
    if cleaned:
        return cleaned
    if " " in vocab.stoi:
        return " "
    return vocab.chars[0]


def loss_bar_data(names, losses):
    """Tiny helper for consistent leaderboard tables."""
    rows = []
    for name, loss in zip(names, losses):
        rows.append({"model": name, "loss": float(loss)})
    return rows
