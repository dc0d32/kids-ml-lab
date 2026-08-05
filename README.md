# 🧪 Kids ML Lab

An interactive, hands-on crash course in Machine Learning & AI — built for an 8th grader,
with a **🧸 Little Kid Corner** in every chapter so a 4th grader can play along too.

No mystery. No hand-waving. Every idea starts with numbers small enough to work out with a
pencil, then becomes a picture you can poke with a slider, and only *then* becomes code.

---

## Start here

```bash
./run.sh app     # the interactive playground  ← start here
./run.sh lab     # JupyterLab notebooks (same chapters, with the code visible)
./run.sh test    # smoke tests
```

First run will take a few minutes while `uv` builds the environment. After that it's instant.

---

## How every chapter works

Every chapter — all 25 of them — follows the exact same six beats:

| Beat | What happens |
|------|--------------|
| 1. **Hook** 🎣 | A question or a game. Plain English. No math, no code. |
| 2. **By hand** ✏️ | A worksheet with 3–6 rows of tiny numbers. You solve it with a pencil. |
| 3. **See it** 👀 | A picture or animation of the exact thing you just did by hand. |
| 4. **Play** 🎛️ | Sliders and buttons. Move one knob, watch the picture change instantly. |
| 5. **For real** 💻 | 10–25 lines of actual code on actual data. |
| 6. **Challenge** 🏆 | Beat the machine, or break it on purpose. Plus the 🧸 Little Kid Corner. |

---

## The course

### Part 0 — What even *is* this?
| # | Chapter | The big idea |
|---|---------|--------------|
| 00 | The Guessing Machine | A computer can learn a rule from examples — and you can race it |

### Part 1 — Classical models, bare mechanics
| # | Chapter | The big idea |
|---|---------|--------------|
| 01 | Lines That Predict | `y = w·x + b` and the idea of "how wrong am I?" |
| 02 | Lines That Decide | A line can split the world in two |
| 03 | When a Ruler Isn't Enough | Some things a straight line simply cannot do |
| 04 | Maybe, Probably, Definitely | Squishing a number into a probability |
| 05 | Twenty Questions | Decision trees ask their way to an answer |
| 06 | A Crowd of Trees | Many weak guessers beat one strong one |
| 07 | The Widest Road | Don't just separate — separate with the biggest gap |
| 08 | The Model Zoo | Which model when, and how to not fool yourself |

### Part 2 — Escaping Flatland
| # | Chapter | The big idea |
|---|---------|--------------|
| 09 | Real Data, Real Mess | 🐧 penguins, 🍄 mushrooms, 🎮 Pokémon, 🚲 bikes |
| 10 | Where Models Go Wrong | Bias, leakage, and being confidently wrong |

### Part 3 — Neural networks, from one neuron up
| # | Chapter | The big idea |
|---|---------|--------------|
| 11 | One Neuron | It's Chapter 02 plus a squish. That's all. |
| 12 | How a Neuron Learns | Backprop, done by hand, then in 30 lines of NumPy |
| 13 | Two Layers, Three Neurons | Hidden neurons each draw a line — together they bend |
| 14 | Deeper and Wider | More layers, different squishes, and over-studying |
| 15 | Same Thing, in PyTorch | Nothing magic — we check its gradients against ours |

### Part 4 — Seeing
| # | Chapter | The big idea |
|---|---------|--------------|
| 16 | Pictures Are Just Numbers | Read a digit off a grid of numbers, then teach a net to |
| 17 | The Sliding Window | Convolutions by pencil, then a tiny CNN |

### Part 5 — Learning without answers
| # | Chapter | The big idea |
|---|---------|--------------|
| 18 | You Are Like Your Neighbors | The model that does no training at all |
| 19 | Sorting Without Labels | k-means — and squeezing your photo down to 5 colors |
| 20 | Squishing Dimensions | PCA is picking the best shadow to cast |

### Part 6 — Making things up
| # | Chapter | The big idea |
|---|---------|--------------|
| 21 | The Bigram Babbler | Count letter pairs, roll a die, invent words |
| 22 | Giving It a Memory | It discovers vowels on its own. Nobody told it. |
| 23 | Paying Attention | A tiny Transformer, with its attention shown live |
| 24 | So What Now? | The whole map, the honest limits, and what to build next |

---

## House rules (never broken)

- **Simple Python.** If it needs a clever trick to read, it gets rewritten.
- **Small data.** Everything ships in the repo or comes built into a library.
- **Laptop-sized.** Every chapter trains on a CPU in seconds to a couple of minutes.
  A test enforces this — if a chapter gets slow, the *dataset* shrinks, not the budget.
- **Notebook and app never disagree.** Both import the same `kidsml/` library.

## Repo layout

```
kidsml/       shared library — one source of truth for data, plots, and math
notebooks/    one notebook per chapter
app/          Streamlit playground, one page per chapter
data/         small bundled datasets and text corpora
worksheets/   printable pen-and-paper sheets (+ answer keys for grown-ups)
tests/        smoke tests and runtime budgets
```
