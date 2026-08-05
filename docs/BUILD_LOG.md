# Build Log

What was built, when, and — more importantly — **why it was built that way**. This file
exists so that a future session (or a different person, or a different machine) can pick
the project up without having to reverse-engineer the reasoning.

Newest entries at the bottom. Every meaningful change gets an entry.

---

## 2026-08-05 — Project kickoff

### The ask

Build a hands-on, interactive ML/AI crash course for two kids who find the subject
intimidating. Teach mechanics first and as bare as possible: simple equations they can run
with a pencil, diagrams, and interactive simulations where changing a parameter shows its
effect immediately. Progress from classical supervised models → neural networks →
clustering/kNN → a small vision model → a small generative language model.

### Decisions made up front

| Decision | Choice | Why |
|---|---|---|
| Audience level | One track aimed at the **8th grader**, plus a **🧸 Little Kid Corner** in every chapter | A 4th grader and an 8th grader are too far apart for one voice, but two full tracks would double the work and halve the polish |
| Delivery | **Both** a notebook and a Streamlit page per chapter | The app is for playing without reading code; the notebook is for looking under the hood. Both import `kidsml/` so they can't drift |
| Datasets | **Bundled in-repo** + built-in sklearn/torchvision sets | No Kaggle credentials, no network at runtime, no "it worked yesterday" |
| Neural nets | **NumPy from scratch first, PyTorch later** | The point of Part 3 is that nothing is hidden. PyTorch arrives only once the nets are too big to hand-write, and is introduced by *proving* its gradients match ours |
| Generative chapter | **Character-level**: bigram → MLP → tiny Transformer, on nursery rhymes / fables / names | Trains in 1–2 minutes on CPU, and its mistakes are funny rather than boring |
| Repo name | `kids-ml-lab`, private | Chosen by the owner |
| Course size | 25 chapters in 7 parts | Full menu, trimmable after Part 1 is reviewed |

### Hard constraints (the "house rules")

Very simple Python · small datasets · CPU-only · every chapter trains in seconds to a
couple of minutes. These are enforced by a test, not by good intentions: if a chapter goes
over its wall-clock budget, the **dataset** shrinks, never the explanation.

### What got built

**Scaffolding**
- `uv`-managed environment, Python 3.13, CPU-only torch via the PyTorch CPU index.
- `run.sh` with `app` / `lab` / `test` / `build` subcommands.
- `flake.nix` devShell.

**NixOS workaround.** This machine is NixOS with no `nix-ld`, so manylinux wheels fail
with `libstdc++.so.6: cannot open shared object file`. `run.sh` detects `/etc/NIXOS` and
sets `LD_LIBRARY_PATH` from `nix eval nixpkgs#stdenv.cc.cc.lib` and `nixpkgs#zlib`. The
result is cached in `.nix-libs` (gitignored) because `nix eval` is slow enough to be
annoying on every launch. `KIDSML_SKIP_NIX_LIBS=1` opts out, which the flake devShell sets
since it provides the paths itself.

**`kidsml/` — the shared library**
- `datasets.py` — 1D toys, six 2D toy shapes (blobs, moons, circles, xor, spiral,
  stripes), clustering toys, images, bundled tables, text corpora. Toy shapes are all
  normalised to a similar scale so one set of plot limits works everywhere.
- `plots.py` — the house style. One place decides that blue means class 0 and red means
  class 1. Includes `decision_boundary` (which shades by confidence when the model
  provides it), `regression_fit` (which can draw squared error as literal squares),
  `loss_surface`, and readable confusion/heatmap grids.
- `nn_numpy.py` — activations and their slopes, a `Neuron`, an `MLP` with hand-written
  backprop, the 1958 perceptron rule, and `numeric_gradient`.
- `text.py` — `CharVocab`, bigram counting, temperature sampling, context windows.
- `ui.py` — `CHAPTERS` (the course map, single source of truth), `page_setup`, the six
  beats, and the coloured boxes.
- `zeeps.py` — the Chapter 00 game.

**Verification worth recording:** hand-written backprop in `nn_numpy.py` agrees with
numerical gradients to ~3e-11, and the network solves XOR exactly. This is the thing
Chapter 15 later points at when it claims PyTorch isn't doing anything magic.

**Data.** `tools/prepare_data.py` downloads and reshapes everything into small,
kid-readable CSVs which are then committed (~900 KB total):

- `penguins.csv` — Palmer penguins (CC0)
- `mushrooms.csv` — UCI Mushroom (CC BY 4.0), with every single-letter code spelled out
  into real words so no one has to consult a legend
- `bikes.csv` — UCI Bike Sharing (CC BY 4.0), with the normalised columns converted back
  into real °C, % and km/h
- `names.txt` — 10 000 first names (public-domain SSA-derived)
- `rhymes.txt`, `fables.txt` — traditional public-domain rhymes and original short fables
- `creatures.csv` — 10 invented creatures, small enough to build a decision tree by hand
- `monsters.csv` — 800 invented trading-card monsters

**Decision: invented monsters instead of Pokémon.** The original plan used a Pokémon stats
dataset. Replaced with an original invented one to avoid trading on someone else's IP —
and it turned out *better* pedagogically, because we know the true generating rule
(`(attack + magic) > 150 AND speed < 90`, with 5% of labels deliberately flipped). Chapter
6 can therefore ask "did the model find the real rule?" and check the answer, and can
explain why scoring 100% would be a red flag rather than a triumph.

**Chapter pipeline.** Chapters are authored as jupytext `py:percent` sources in
`notebooks/_src/` and built into `.ipynb` by `tools/build_notebooks.py`. Rationale: plain
Python is far easier to write and review than notebook JSON, but kids should still open a
normal notebook. The generated `.ipynb` files are committed so a fresh clone works with no
build step, and a test fails if they drift out of sync with their sources.

**Testing.** `tests/test_pages.py` uses Streamlit's own `AppTest` harness rather than
hitting the server over HTTP — a page that throws still returns HTTP 200, so an HTTP check
would have been worthless. `tests/test_notebooks.py` actually executes each notebook and
asserts a per-chapter wall-clock budget.

**Chapter 00 · The Guessing Machine** built as the reference implementation: a secret-rule
game where the reader and a decision tree race to infer the rule from the same examples,
ending in the learning-curve plot that shows accuracy rising with *nothing changing but
the number of examples*.

---

## 2026-08-05 — Worksheets became on-screen workbooks

**The kids have no printer and work on laptops.** The original design had
`worksheets/NN_*.md` as printable pen-and-paper handouts with an answer key at the bottom.
That's now wrong by construction.

Replaced with `kidsml/workbook.py`: a `Workbook` is a list of `Question`s
(`number` / `choice` / `text` / `open`), each carrying a `why=` that explains the teaching
point. They render as real inputs with instant answer-checking — via Streamlit widgets on
a page, and via ipywidgets in a notebook, from the *same* definition.

Each chapter's questions live in `worksheets/NN_slug.py` rather than in one shared file,
so several chapters can be written in parallel without stepping on each other.

This turned out better than the printable version, not just different: the reader gets
feedback the moment they answer, and the `why` text does the teaching whether they got it
right or wrong. Scrap paper is still expected for the arithmetic — it's the *delivery* that
moved on screen, not the thinking.

`ui.worksheet_link(N)` is kept as an alias of `ui.workbook(N)` so pages written against
the earlier name keep working.

---

## 2026-08-05 — Agent instructions added

Added `AGENTS.md` (the full contract: house rules, voice, layout, how to add a chapter,
how to test, how to work in parallel) and `.github/copilot-instructions.md` (the short
version, pointing at it), plus this log. The goal is that a future session can resume
without re-deriving any of the reasoning above.
