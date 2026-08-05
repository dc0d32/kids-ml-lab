# AGENTS.md — how to work on Kids ML Lab

This file is the contract for anyone (human or agent) touching this repo. Read it before
changing anything. If you change how the project works, **update this file in the same
commit**.

---

## What this is

An interactive machine learning crash course, written for one specific audience: an 8th
grader who thinks AI is too complicated to understand, and their 4th-grade sibling.

The point is not to cover a syllabus. The point is that at some moment in every chapter
the reader goes *"oh — that's all it is?"*. If a change doesn't serve that, it isn't an
improvement.

---

## Non-negotiable house rules

1. **No printing, ever.** The kids work on laptops. There is no printer. Every worksheet,
   diagram and exercise is on screen. Do not add PDFs, print stylesheets, or anything
   phrased as "print this out".
2. **Very simple Python.** If a line needs a clever trick to read, rewrite it. No
   comprehension pyramids, no `functools`, no metaprogramming. The code is teaching
   material, not production code.
3. **Small datasets.** Everything comes from `kidsml.datasets` or is defined inline.
   Nothing downloads at runtime except Fashion-MNIST in Chapter 17, and that is cached.
4. **Laptop-sized, CPU-only.** No GPU anywhere. `tests/test_notebooks.py` enforces a
   wall-clock budget per chapter. **If a chapter blows its budget, shrink the dataset or
   the number of training steps — never the explanation.**
5. **Never name a thing before you explain it.** Introduce the idea in plain words, then
   say what grown-ups call it (`ui.jargon(...)` on a page, a `> 📖 **Grown-ups call
   this:**` blockquote in a notebook).
6. **Notebook and app must never disagree.** Both import from `kidsml/`. If logic is
   shared, it goes in a `kidsml/` module — never copy-pasted between the two.
7. **Comments explain *why*, not *what*.** Do not narrate obvious lines.

---

## Voice

- Short sentences. Concrete nouns. Second person — "your line", "you already know this".
- Never write "simply", "just", "obviously", or "trivial". They are the exact words that
  make a nervous reader give up.
- Humour is welcome. Condescension is not.
- Numbers in a by-hand question must be small and round enough to do with a pencil in
  under three minutes.
- Every chapter needs a **🧸 Little Kid Corner**: the same idea with zero algebra, usually
  a physical game or an analogy the 4th grader can act out.

---

## Repo layout

```
kidsml/        the shared library — the single source of truth
  ui.py          page furniture, the six beats, and CHAPTERS (the course map)
  plots.py       the house drawing style: colours, decision boundaries, loss curves
  datasets.py    every dataset — toy 2D shapes, bundled CSVs, text corpora
  nn_numpy.py    neural networks written out by hand, no frameworks
  text.py        char vocab, bigrams, context windows
  workbook.py    the interactive workbook machinery
  <topic>.py     chapter-specific helpers (zeeps.py, trees.py, ...)
app/
  Home.py        the course map landing page
  pages/NN_*.py  one Streamlit page per chapter
notebooks/
  _src/NN_*.py   chapter SOURCES in jupytext py:percent format  ← edit these
  NN_*.ipynb     GENERATED — do not hand-edit
data/            small bundled datasets and text corpora
worksheets/      NN_*.py — interactive workbooks, one per chapter
tools/           prepare_data.py, build_notebooks.py
tests/           notebook execution + budgets, and real Streamlit page runs
docs/BUILD_LOG.md   what was built, and why it was built that way
```

`CHAPTERS` in `kidsml/ui.py` is the course map. Chapter numbers, slugs, titles and
one-line ideas come from there and nowhere else — filenames must match it, and a test
checks that they do.

---

## Running it

The environment is `uv`-managed. On **NixOS**, pip wheels can't find `libstdc++`, so the
library path has to be set before any `uv run`:

```bash
cd /path/to/kids-ml-lab
export LD_LIBRARY_PATH=$(cat .nix-libs)     # .nix-libs is created by run.sh on first use
```

`run.sh` does this for you:

```bash
./run.sh app          # the Streamlit playground
./run.sh lab          # JupyterLab
./run.sh test         # all tests
./run.sh build        # regenerate every notebook from notebooks/_src/
./run.sh build 13 14  # regenerate only chapters 13 and 14
```

There is also a `flake.nix` devShell (`nix develop`) that sets the same paths
declaratively.

Datasets are committed, so `tools/prepare_data.py` only needs re-running if a source
changes.

---

## Adding or changing a chapter

A chapter is three files, where `NN` and `S` come from `CHAPTERS` in `kidsml/ui.py`:

| File | What it is |
|---|---|
| `app/pages/NN_S.py` | the Streamlit page — sliders and pictures, code hidden except in the "For Real" beat |
| `notebooks/_src/NN_S.py` | the notebook source, jupytext `py:percent` (`# %%` / `# %% [markdown]`) |
| `worksheets/NN_S.py` | the interactive workbook — a `WORKBOOK = Workbook(...)` |

Then:

```bash
./run.sh build NN
uv run pytest tests -q -k "NN_"
```

### The six beats — every chapter, same order

| Call | Header | What goes in it |
|---|---|---|
| `ui.beat("hook")` | 🎣 The Hook | A question or a game. Plain English. No math, no code. |
| `ui.beat("byhand")` | ✏️ Do It By Hand | A few rows of tiny numbers, worked out with a pencil, then the same thing in code so they see it match. |
| `ui.beat("seeit")` | 👀 See It | A picture of the exact thing they just did by hand. |
| `ui.beat("play")` | 🎛️ Play With It | Sliders. One knob → the picture changes within a second. This is the heart of the chapter. |
| `ui.beat("forreal")` | 💻 For Real | 10–25 lines of real code on real-ish data. |
| `ui.beat("challenge")` | 🏆 Challenge | Numbered quests — beat the machine, break it on purpose — ending with a 🧸 item. |

Start every page with `ui.page_setup(N)` and end it with `ui.worksheet_link(N)`.
End every notebook with a `---` and a one-line **Next up:** teaser.

Other helpers: `ui.aha`, `ui.careful`, `ui.little_kid_corner`, `ui.jargon`, `ui.figure`,
`ui.two_figures`, `ui.show`, `ui.shape_picker`, `ui.noise_slider`, `ui.sample_slider`,
`ui.seed_slider`.

### Workbooks

Questions are `Question(prompt=..., kind=..., answer=..., why=...)`. `kind` is one of
`number`, `choice`, `text`, `open`. **Every question needs a `why=`** — the teaching
point, not just the answer. That field is the most valuable thing in the file. Use `open`
for "why do you think…" questions; checking them just reveals the `why`.

### Working in parallel

If several agents are writing chapters at once:

- Do not edit another chapter's files, or any shared module
  (`ui.py`, `plots.py`, `datasets.py`, `nn_numpy.py`, `text.py`, `workbook.py`),
  or `README.md`, `pyproject.toml`, `run.sh`, `tests/`.
- Need a shared helper? Create a **new** `kidsml/<your_topic>.py`.
- Need a change to a shared file? Don't make it — report it, and let the coordinator apply it.
- Test with `-k "NN_"` so half-written chapters from other agents don't fail your run.

---

## Testing

```bash
./run.sh test
```

- `tests/test_notebooks.py` — executes every notebook and fails if it exceeds its
  wall-clock budget. Also fails if an `.ipynb` is out of sync with its `_src` file, which
  means someone forgot `./run.sh build`.
- `tests/test_pages.py` — really runs each Streamlit page through Streamlit's own test
  harness, so an exception in a page is a test failure. An HTTP 200 proves nothing.
- `tests/test_workbooks.py` — every workbook loads, and every non-open question's own
  stated answer passes its own checker.

Need a bigger budget for a chapter? Change `BUDGETS` in `tests/test_notebooks.py` **and
say why in the commit message**. Raising a budget is a decision, not a fix.

---

## Git

- Keep `docs/BUILD_LOG.md` current. Every meaningful change gets an entry, and every
  decision that a future reader would otherwise have to reverse-engineer gets a short
  "why" note. This is what makes the project portable between sessions and machines.
- Generated `.ipynb` files **are** committed, so the notebooks work on a fresh clone
  without a build step.
- **Never `git push` without asking the repo owner first.** Every single time.
