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
- **Never use a term before explaining the idea behind it.**
- **Notebook and page must never disagree** — shared logic lives in `kidsml/`.
- Avoid the words "simply", "just", "obviously", "trivial".

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
