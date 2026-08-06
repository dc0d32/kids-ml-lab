# 🧪 Kids ML Lab

An interactive, hands-on crash course in Machine Learning & AI — built for an 8th grader,
with a **🧸 Little Kid Corner** in every chapter so a 4th grader can play along too.

26 chapters. Each one is a sequence of small screens: one idea, one picture, one thing to
try. Nothing is a wall of text and nothing is a video.

No mystery. No hand-waving. Every idea starts with numbers small enough to work out with a
pencil, then becomes a picture you can poke with a slider, and only *then* becomes code.

---

## Start here

You need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and nothing else —
it fetches the right Python and every package on first run.

**macOS / Linux**

```bash
./run.sh app     # the interactive playground  ← start here
./run.sh lab     # JupyterLab notebooks (same chapters, with the code visible)
./run.sh test    # the tests
```

**Windows (PowerShell)**

```powershell
.\run.ps1 app
.\run.ps1 lab
.\run.ps1 test
```

Or skip the launchers entirely, on any platform:

```bash
uv run streamlit run app/Home.py
uv run jupyter lab notebooks
```

First run takes a few minutes while `uv` builds the environment. After that it's instant.
Everything runs on the CPU — there is no GPU anywhere in this course, and nothing needs
an account, a key, or an internet connection (except a one-time 30 MB dataset in
Chapter 18).

---

## How every chapter works

Every chapter — all 25 of them — follows the exact same six beats:

| Beat | What happens |
|------|--------------|
| 1. **Hook** 🎣 | A question or a game. Plain English. No math, no code. |
| 2. **By hand** ✏️ | 3–6 rows of tiny numbers. You work them out, type your answer in, and find out *why* the question was asked. |
| 3. **See it** 👀 | A picture or animation of the exact thing you just did by hand. |
| 4. **Play** 🎛️ | Sliders and buttons. Move one knob, watch the picture change instantly. |
| 5. **For real** 💻 | 10–25 lines of actual code on actual data. |
| 6. **Challenge** 🏆 | Beat the machine, or break it on purpose. Plus the 🧸 Little Kid Corner. |

---

## The course

### Part 0 — What even is this?
| # | Chapter | The big idea |
|---|---------|--------------|
| 00 | The Guessing Machine | A computer can learn a rule from examples — and you can race it |

### Part 1 — Classical models
| # | Chapter | The big idea |
|---|---------|--------------|
| 01 | Lines That Predict | y = w·x + b, and the idea of 'how wrong am I?' |
| 02 | Lines That Decide | One line can split the whole world in two |
| 03 | When a Ruler Isn't Enough | Some things a straight line can never do |
| 04 | Maybe, Probably, Definitely | Squishing any number into a probability |
| 05 | Twenty Questions | Decision trees ask their way to an answer |
| 06 | A Crowd of Trees | Many weak guessers beat one strong one |
| 07 | The Widest Road | Don't just separate — separate with the biggest gap |
| 08 | You Are Like Your Neighbors | The model that does no training at all |
| 09 | The Model Zoo | Which model when, and how not to fool yourself |

### Part 2 — Escaping Flatland
| # | Chapter | The big idea |
|---|---------|--------------|
| 10 | Real Data, Real Mess | Penguins, mushrooms, monsters and bikes |
| 11 | Where Models Go Wrong | Bias, leakage, and being confidently wrong |

### Part 3 — Neural networks
| # | Chapter | The big idea |
|---|---------|--------------|
| 12 | Arrows and Grids | A matrix isn't a box of numbers. It's an instruction for moving space |
| 13 | One Neuron | It's Chapter 2 plus a squish. That's all |
| 14 | How a Neuron Learns | Backprop by hand, then in 30 lines of NumPy |
| 15 | Two Layers, Three Neurons | Hidden neurons each draw a line — together they bend |
| 16 | Deeper and Wider | More layers, different squishes, and over-studying |
| 17 | Same Thing, in PyTorch | Nothing magic — we check its gradients against ours |

### Part 4 — Seeing
| # | Chapter | The big idea |
|---|---------|--------------|
| 18 | Pictures Are Just Numbers | Read a digit off a grid of numbers, then teach a net to |
| 19 | The Sliding Window | Convolutions by pencil, then a tiny CNN |

### Part 5 — Without answers
| # | Chapter | The big idea |
|---|---------|--------------|
| 20 | Sorting Without Labels | k-means — and your photo squeezed to 5 colours |
| 21 | Squishing Dimensions | PCA is picking the best shadow to cast |

### Part 6 — Making things up
| # | Chapter | The big idea |
|---|---------|--------------|
| 22 | The Bigram Babbler | Count letter pairs, roll a die, invent words |
| 23 | Giving It a Memory | It discovers vowels on its own. Nobody told it |
| 24 | Paying Attention | A tiny Transformer, with its attention shown live |
| 25 | So What Now? | The whole map, the honest limits, and what to build next |

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
notebooks/    one notebook per chapter (built from notebooks/_src/)
app/          Streamlit playground, one page per chapter
data/         small bundled datasets and text corpora
worksheets/   interactive workbooks — questions with instant answer-checking
tools/        dataset preparation and notebook building
tests/        smoke tests and runtime budgets
docs/         BUILD_LOG.md (what was built and why) and TEACHING_NOTES.md (for the grown-up)
```

Teaching with this? Read [`docs/TEACHING_NOTES.md`](docs/TEACHING_NOTES.md) first.

Working on the code? Start with [`AGENTS.md`](AGENTS.md).
