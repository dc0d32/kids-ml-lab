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
Chapter 16 later points at when it claims PyTorch isn't doing anything magic.

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


---

## 2026-08-05 — Part 3 neural-network spine drafted

Built Chapters 12–15 as the first neural-network arc: one neuron, hand backprop, two-layer
feature invention, depth/width/overfitting, and the PyTorch translation. The shared helper
modules are `kidsml/nnplots.py` for neural-net diagrams/surfaces/snapshots and
`kidsml/torch_bits.py` for the small PyTorch API used in Chapter 16 and future chapters.

The main design choice was to keep every visual tied to the from-scratch `kidsml.nn_numpy`
objects rather than duplicate model logic in pages or notebooks. Chapter 16 copies NumPy
weights into PyTorch and asserts gradient agreement, so the framework appears as a faster
version of the same blame-passing idea instead of a new magic step.

---

## 2026-08-05 — Chapters 17 and 17 added

Built Part 4's first seeing chapters.

- Chapter 17 turns sklearn's 8×8 digits into visible number grids, trains a small seeded `MLPClassifier`, shows confusion-matrix mistakes, first-layer weights, and a canvas-to-8×8 digit demo.
- Chapter 18 adds `kidsml/vision.py` for plain valid convolutions, kernel presets, drawing preprocessing, and a small CPU PyTorch CNN/MLP comparison. Fashion-MNIST downloads to `data/torchvision/` when available and falls back to sklearn digits if needed.
- Both chapters use interactive Python workbooks rather than printable handouts.

---

## 2026-08-05 — Part 6 finale chapters

Built Chapters 22-24 as the generative-text finale: bigram counting, a fixed-window MLP with letter embeddings, a tiny causal Transformer with attention maps, and the course wrap-up. The shared language-model code lives in `kidsml/langmodels.py` so pages and notebooks use the same training, sampling, scoring, and attention helpers.

The models are intentionally small and CPU-only. Measured notebook runtimes in the chapter-only test run were 6.24s (Ch21), 6.81s (Ch22), 13.16s (Ch23), and 1.84s (Ch24), well below their budgets.

---

## 2026-08-05 — Chapters 01-04 teaching prose deepened

Reworked the Part 1 linear-model chapters so each teaching point has setup, concrete arithmetic, and a "why this matters" landing. The app pages, notebook sources, generated notebooks, and workbook `why=` explanations now explain squared error, gradients, perceptron geometry, XOR impossibility, feature lifting, sigmoid probabilities, and log-loss confidence in kid-readable steps.

Added Mermaid structure diagrams for the gradient-descent loop, perceptron score-to-class flow, 3D feature-lift pipeline, and logistic score-to-probability flow. Measured notebook runtimes after rebuilding: Ch01 3.60s, Ch02 3.22s, Ch03 5.01s, Ch04 2.98s.

---

## 2026-08-05 — All 25 chapters written, then deepened

Chapters 01-24 were written by parallel agents working from a shared brief, each owning
its own files and creating a new `kidsml/<topic>.py` rather than editing shared modules.
That kept eight agents out of each other's way.

**Then the prose failed review.** The owner's verdict:

> "the prose in general is weak i.e. explanations in most places are thin, one liners.
> If kids follow this course on their own, they will get confused. At the same time,
> they will give up if we throw a wall of text at them."

The chapters were structurally correct — six beats, working code, working sliders — and
still not good enough, because each teaching point was a single asserted line. The worst
example was Chapter 06, where the entire bridge from the jellybean hook to the actual idea
was *"Trees can do that too."* A reader who already knew the material would nod. A reader
who didn't would quietly fall behind.

This produced the **"How much to explain" standard** now in `AGENTS.md`, and it is the most
important thing in that file:

- Every idea gets **three moves**: setup (the question they're already asking) → the idea
  (plain words, real numbers) → so what (what it buys them).
- **Answer the question they're about to ask.** After each explanation, work out what a
  sharp 13-year-old would say next and answer it there.
- **Rhythm**: 60-120 words, then something happens. Never more than ~150 unbroken.
- **After every figure, say what to look at in it.** A figure with no pointer is decoration.

A second pass rewrote all 25 chapters against it. Chapter 06's page went from 133 lines to
250.

**Mermaid diagrams** were added at the same time, via `ui.mermaid` / `lesson.mermaid`
backed by `streamlit-mermaid`, which bundles its JavaScript locally and so needs no
network. Mermaid is for *structure* (how a prediction flows, what order steps happen in);
matplotlib stays for *data*. Notebooks use fenced mermaid blocks, which JupyterLab 4
renders natively.

---

## 2026-08-05 — Pages rebuilt as stepped lessons

The owner compared the app to **Brilliant** and asked for that polish. The pages were long
scrolling documents, and scrolling invites skimming.

`kidsml/lesson.py` turns a chapter into a sequence of screens — one idea each — with
Back/Next, a progress bar and a beat trail. Its most valuable piece is **`lesson.predict()`**:
it asks what the reader thinks will happen and returns `None` until they commit, so the
step can withhold the reveal. Being wrong and then surprised teaches far more than being
right by default. Every chapter has at least two, in front of the moments listed in
`docs/TEACHING_NOTES.md`.

Also added: `lesson.look_for()` (a pointer at what matters in each figure), a light theme,
and a 68-character reading width so no line of prose runs long enough to lose a reader.

**Testing had to change with it.** `tests/test_pages.py` now clicks through *every* step of
every chapter, so a broken step deep in a chapter can no longer hide behind a working first
screen. It also fails a chapter with fewer than 5 steps, checks the beats never run
backwards, and requires at least two predictions per chapter — the shape of a chapter is
now enforced rather than trusted.

One practical consequence worth knowing: **every Next click re-runs the page**, so anything
that trains a model must sit behind `@st.cache_resource`. Chapters 16, 17, 18, 23 and 24
would be unusable otherwise.

---

## 2026-08-05 — Chapter 11 added: linear algebra

Requested as "3blue1brown, but without the long videos", and explicitly as a playground:

> "I expect kids will spend hours playing with it and wrapping their head around the idea,
> and keep coming back to it."

Chapters 11-24 were renumbered to 12-25 by a one-shot migration to open the slot. It sits
immediately before the neural networks on purpose: **the fact that two linear steps collapse
into one linear step is the argument for why a neuron needs a squish.** A reader who has
watched that happen never has to wonder what activation functions are for.

`kidsml/linalg.py` supports it, and the two numbers that make the chapter work are measured
rather than asserted:

- applying two matrices in a row differs from applying their product by **exactly 0.0**
- inserting a `tanh` between them pushes that difference to **4.97**, and the grid lines
  visibly bend

16 steps, three prediction gates, and a rotatable 3D shadow game where the reader hunts for
the projection that keeps the most spread — and then finds out that is what PCA does in
Chapter 21.

The README's course tables are now **generated from `CHAPTERS`** rather than hand-written,
since a hand-maintained copy of the course map is exactly the sort of thing that drifts.

---

## 2026-08-05 — Teacher's guide, and a consistency sweep

`docs/TEACHING_NOTES.md` written for the parent: session shape, four alternative routes
through the course, the specific aha moment to wait for in each chapter, a ladder for when
a kid is stuck (do not rescue early), how to keep the 9-year-old involved, and honest
answers to the questions kids actually ask — including why models hallucinate, explained
mechanically rather than waved away.

Final sweep across the repo:

- migrated 77 uses of Streamlit's deprecated `use_container_width` to `width=`
- fixed a beat-order jump in Chapter 11 and two uses of a banned word
- regenerated the README course map from `CHAPTERS`
- rewrote the Home page, which still described the old scrolling format

**317 tests pass in 160s.** All 26 notebooks execute inside their budgets, the slowest
being the Transformer at 13s of 300s.

---

## 2026-08-05 — Voice, and a bug the tests could not see

**Voice.** The owner asked for Bill Nye energy — fast, delighted, never solemn — with a
twist: the 8th grader gets secondhand embarrassment when his dad uses Gen-Z slang, so the
course does it deliberately. The joke is not the slang; it's the course being visibly
pleased with itself for knowing the word, used a beat late or over-explained. One or two
per chapter, always in a caption, aside or metric label, never inside an explanation doing
real work. The standard is in `AGENTS.md`.

A follow-up pass thinned the repeats: "vibe check" had landed seven times across the
course, which is how a running joke stops being one.

**The bug worth remembering.** The owner reported that all six creatures in Chapter 00
showed as "not a zeep". He was right, and it was bad: chapter 00 drew six *random*
creatures, but the secret rule "big AND a square" is true of only 3 of the 18. A random
six came back all-negative. No positive example, nothing to generalise from, a rule that
could not be worked out — in the first five minutes of the course.

**Every test passed.** The page rendered, every step clicked through, nothing threw. The
tests verified that the code ran, and never asked whether the chapter *taught* anything.
That is the gap, and it is worth stating plainly: structural correctness proves very
little about a teaching artefact.

Fixes:

- `zeeps.teaching_examples()` deals on purpose. It starts with a **near-miss pair** — two
  creatures differing in exactly one way that still get different answers, which is the
  pair that kills every wrong hypothesis — then fills up keeping both answers present.
  `quiz_examples()` holds back creatures that were never shown.
- **`tests/test_teaching.py`** is a new category of test: does the teaching work, not does
  the code run. Every rule × seed must show at least two of each answer, never repeat a
  creature, never quiz on one already answered, and always include the pair that pins the
  rule down.
- It also now checks Chapter 23's payoff empirically: the learned letter embeddings really
  do put the vowels together, measured at roughly 3x closer to each other than to the
  consonants, on every seed tried. The chapter's most striking claim is verified rather
  than hoped for.

Anything a reader is asked to *deduce* should get a test in that file.

Also fixed: the empty radio label in `lesson.predict` that Streamlit warns about for
accessibility.

**495 tests pass in 188s.**

---

## 2026-08-05 — Chapter 02 had a hole between its first two screens

Owner's report: *"in 'lines that decide', between start here and work it out, we're
missing context."* Correct, and there were three separate gaps stacked on top of each
other:

1. **The data had no meaning.** Ten bare coordinates appeared in a table. No story, no
   picture, nothing to say what x1 and x2 *were*.
2. **Two features arrived unannounced.** Chapter 01 had a single input (weeks saved).
   Chapter 02 silently switched to two, which is precisely the change that makes a *line*
   necessary instead of a threshold — and it went unmentioned.
3. **The candidate line fell out of the sky.** `score = 1·x1 + 1·x2 − 8` was handed over
   with no account of where those three numbers came from.

Two screens now sit in the gap. **Ten dogs at the park** gives the data a story (how tall
in hand-spans, how heavy in bags of sugar; puppies versus grown dogs), plots it, and says
out loud that one number per thing puts you on a number line while two puts you on a map —
which is why we need a line. **Guess a line** then derives the score rule from an idea a
kid would actually have ("add them up, and if the total beats 8 call it a grown dog"),
draws it on the map, and admits we picked the numbers by eye — which is what sets up the
rest of the chapter.

### And the same bug as chapter 00, twice more

Looking at the by-hand table revealed it was slicing `X_tiny[:5]` — and the data is
ordered, so those five rows are **all puppies**. The worked example, the dog at (6, 5),
wasn't even in the table. An audit for the pattern found two more:

- **chapter 12** used the same `[:5]` slice, so the neuron's score never changed sign and
  the fence it was meant to demonstrate was invisible
- **chapter 09** opened the penguin data with `head(8)`, and the file is sorted by
  species, so the reader's first look at a three-species problem was eight Adelie

All three now take a deliberate mix, and `tests/test_teaching.py` guards it.

The general lesson, now stated twice in this log because it keeps costing us: **taking the
first N rows of ordered data is a bug in a teaching artefact even when it is not a bug in
the code.** Anything shown to a reader as evidence needs both answers in it.

**501 tests pass.**
