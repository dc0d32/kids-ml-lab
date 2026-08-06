# Copilot instructions for Kids ML Lab

**Read [`AGENTS.md`](../AGENTS.md) first — it is the full contract for this repo.**
This file is the short version.

## What this repo is

An interactive machine learning course for an 8th grader (with a 🧸 Little Kid Corner in
every chapter for their 4th-grade sibling). Every chapter exists as both a Jupyter
notebook and a Streamlit page, and both import the same `kidsml/` library.

## The rules that matter most

- **No printing.** The kids use laptops. Everything, including workbooks, is on screen.
- **Very simple Python.** This code is teaching material. Clever is a bug.
- **CPU-only, laptop-sized.** Every chapter trains in seconds to a couple of minutes.
  If a chapter is too slow, shrink the dataset — never the explanation.
- **Never use a term before explaining the idea behind it.** That includes across
  chapters: nothing may lean on a chapter that comes after it.
- **Notebook and page must never disagree** — shared logic lives in `kidsml/`.
- Avoid the words "simply", "just", "obviously", "trivial".
- **Plain words beat clever ones.** A metaphor may decorate an explanation; it must never
  *be* the explanation. If a sentence is the only place an idea is stated, state it as
  plain cause and effect.
- **Colours come from `kidsml/plots.py`.** The page is pure black — a hardcoded dark ink
  colour disappears.
- **Mermaid node labels are ASCII only.** `x₁` kills the whole diagram silently.
- **Animations go through `kidsml/anim.py`** (matplotlib frames → GIF). Animate a claim the
  reader has to take on trust; use a slider when they want to control the parameter.

## Where things live

- `kidsml/` — shared library. `ui.py` holds `CHAPTERS`, the course map that defines every
  chapter's number, slug and title. Filenames must match it.
- `notebooks/_src/NN_*.py` — **edit these**; the `.ipynb` files are generated.
- `app/pages/NN_*.py` — one Streamlit page per chapter.
- `worksheets/NN_*.py` — interactive workbooks (`Question` / `Workbook`).
- `docs/BUILD_LOG.md` — what was built and why. Keep it current.

## Commands

```bash
./run.sh app     # Streamlit playground
./run.sh lab     # JupyterLab
./run.sh build   # regenerate notebooks from notebooks/_src/
./run.sh test    # notebook execution + budgets + real Streamlit page runs
```

On NixOS, `run.sh` sets `LD_LIBRARY_PATH` automatically. Doing it by hand:
`export LD_LIBRARY_PATH=$(cat .nix-libs)`.

## Before finishing any change

1. `./run.sh build` if you touched a notebook source.
2. `./run.sh test` — must pass.
3. Add an entry to `docs/BUILD_LOG.md`.
4. **Never `git push` without asking the repo owner.**

## Explaining things (the most common review failure)

The kids often work alone. **One-line explanations are the bug we keep hitting.**

Every idea gets three moves: **setup** (the question they're already asking) → **the idea**
(plain words, real numbers) → **so what** (what it buys them). Three short paragraphs beats
one dense one, and beats one line by a mile.

Rhythm: ~60–120 words of prose, then *something happens* — a picture, a slider, a table, a
diagram. Never more than ~150 words unbroken. After every picture, say what to look at in it.

Use `ui.mermaid(...)` for structure diagrams (how a prediction flows, what order steps
happen in) and matplotlib for data. In notebooks use a fenced ```mermaid block — JupyterLab
renders it natively.

See the "How much to explain" section of AGENTS.md for the full standard.

## Voice

**Bill Nye energy**: fast, delighted, physical, never solemn. Short sentences, concrete
nouns, second person. Someone who thinks this is the coolest thing and can't wait to show
you.

**The running joke**: the reader gets secondhand embarrassment when a parent uses Gen-Z
slang, so the course does it on purpose — one or two per chapter, deliberately a beat
late or over-explained. The joke is the course being pleased with itself for knowing the
word. Keep it out of any explanation doing real work; captions, asides and metric labels
are the right home. If it isn't funny, leave it out.

See the "Voice" section of AGENTS.md for the full standard.
