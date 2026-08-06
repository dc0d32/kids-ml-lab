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

---

## 2026-08-05 — Flow audit for chapters 00–05

Kids were losing the thread when prose screens introduced ideas before the matching picture
arrived. Chapters 01, 02, 03, 04 and 05 now keep the key visual on the same screen as the
idea: the first regression line, squared-error boxes, the perceptron before/after update,
XOR's contradiction table with its corner plot, the logistic shrug boundary, penguin
probability bars, and a one-hot before/after table.

Chapter 00 was checked and left alone: each screen already has either a table, interaction,
code, workbook, or challenge list that earns the click.

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

---

## 2026-08-05 — A prediction that asked about an invisible plot

Owner's report: in chapter 03, the XOR plot and table didn't show on the *Four dots are
enough to prove it* screen.

The cause was a misuse of the `lesson.predict` pattern. Everything after the gate is
withheld until the reader commits — which is right for the **reveal** and wrong for the
**setup**. The step asked *"can any straight line split these opposite-corner answers?"*
with the four points drawn only after the answer was locked in. The reader was being asked
to predict something about a picture they could not see.

The rule, now enforced: **the setup goes before the gate, the reveal goes after.** A
prediction is only interesting when the reader has enough in front of them to reason with.

Chapter 03's step now shows the four-row truth table and the plot side by side, invites
the reader to try a ruler in their head, and only then asks. The reveal — that whichever
way you tilt, one side always ends up holding one of each — stays behind the gate where it
belongs. The notebook gained the same table, since page and notebook must not disagree.

An audit found one more prediction using a demonstrative, in chapter 01. That one was
legitimate (it refers to a graph shown several steps earlier and correctly withholds the
squares) but opened with no framing at all, so it gained a sentence.

`tests/test_pages.py` now checks that a prediction whose question says "these" or "those"
has something drawn before its gate. Verified against the original broken code.

**527 tests pass.**

---

## 2026-08-05 — Continuity repair for chapters 10-15

Audited the neural-network spine against the "reader has what they need when they need it"
standard. The fixes were continuity, not new mechanics:

- Chapter 10 now introduces the confusion matrix before precision/recall formulas, defines
  precision and recall in sentences first, and points to Chapter 11 instead of skipping to
  Chapter 12.
- Chapter 11 now lands after Chapter 10 and explicitly carries the linear-collapse result
  into Chapter 12's squish and Chapter 14's hidden features.
- Chapter 12 now picks up Chapter 11's "straight layers collapse" argument, motivates the
  hand-picked neuron numbers, avoids using "loss" before Chapter 13, and keeps the workbook
  in the challenge beat.
- Chapter 13 now defines loss before gradient, gradient before chain rule, shows the
  sigmoid-slope arithmetic in the by-hand step, and keeps the workbook in the challenge
  beat.
- Chapter 14's XOR hidden-space table is now backed by worked output-score arithmetic; the
  notebook gained the same learned hidden-coordinate table and 3D hidden-space view as the
  page.
- Chapter 15's parameter-count table now shows the multiplication for every count, and the
  notebook mirrors the overfitting fixes: early stopping, weight decay, and more data.

Verified with `./run.sh build 10 11 12 13 14 15` and
`timeout 900 uv run pytest tests -q -k "10_ or 11_ or 12_ or 13_ or 14_ or 15_"`:
**60 passed, 467 deselected in 46.64s.**

---

## 2026-08-05 — Full continuity sweep

Owner, after finding two broken chapters in ten minutes: *"do a full sweep on continuity."*
Fair. Five agents read all 26 chapters end to end — page, notebook and workbook — against a
ten-point checklist built from the bugs already found.

**Clean chapters: none. Every single chapter had at least one break.**

The recurring shapes, all of which pass any test that only asks "does the code run":

- **Data used before it is introduced.** Tables of bare numbers with no story, no picture,
  and no statement of what is being decided. Chapter 09 modelled four datasets before
  introducing them.
- **Numbers with no provenance.** `b = -20`, `smoothing = 1`, a parameter table with no
  workings.
- **Terms used before they are explained.** kernel, residual, embedding, spread, loss,
  gradient, tensor — each named before the idea landed, somewhere.
- **Questions about invisible things.** Chapter 17 hid a grid behind its prediction gate,
  exactly as chapter 03 had.
- **Ordered data sampled with `.head()`.** Chapter 19 again. Fifth occurrence.
- **Page/notebook drift.** Chapter 00's notebook still had the broken random deal after the
  page was fixed — the point fix that missed half the chapter.
- **Stale artefacts from the renumber.** Widget keys naming the wrong chapter, a wrong
  "Next up" teaser, stale chapter *ranges* in chapter 25 (single references had been
  checked; ranges had not).
- **A correctness bug in shown code**: chapter 18's CNN snippet omitted `zero_grad()`.

### Plotly was rendering white in a dark app

Found while fixing chapter 03's missing 3D plot. **Every plotly figure in the course was on
a white background** — the exact flashbang the dark theme exists to prevent. It was missed
because the matplotlib figures beside them looked fine.

Fixed at the source rather than per call site: `use_house_style()` now registers a `kidsml`
plotly template and sets `plotly_dark+kidsml` as the **default**, so figures are dark even
when built inside a helper that never calls `style_plotly`. A first attempt checked call
sites for `style_plotly(...)` and missed three chapters for exactly that reason — the test
now checks the resolved template instead.

### Chapter 03's missing 3D plot

The step called *Invent a height* explained the lift, drew a flowchart, and never showed
it. The actual 3D view lived in the following step. It now shows the lifted cloud right
where the reader is told the ring rises, and the next step adds the cutting plane — see the
lift, then see the cut.

**529 tests pass.**

---

## 2026-08-05 — Bold was rendering as literal asterisks

Owner: *"streamlit renders `**<something>**` literally, not in bold/italic."*

`lesson.say` wraps its text in a `<div>` so the CSS can hold it to a readable column width.
That was the bug: **Streamlit parses markdown only when it owns the whole block.** The
moment the text sits inside our own HTML, everything within is treated as raw HTML and
`**like this**` arrives with the asterisks showing. 527 bold markers across the course,
none of them working.

Fixed by rendering the markdown ourselves before wrapping, using `markdown-it-py` — which
ships with Streamlit, so it costs no new dependency. `lesson.say`, `look_for`, `jargon`,
the prediction question and the chapter's one-line idea all go through it.

**A worse one hid underneath.** Chapter 04 had a `lesson.say` block indented eight spaces
to line up with the surrounding code. Four leading spaces mean *code block* in markdown, so
that entire passage — the explanation of the sigmoid, the one place `e` is introduced — was
rendering as a grey monospace box. `_dedent` now strips the indentation a triple-quoted
string picks up from living inside a function, including the awkward case where the first
line sits right after the quotes and has no indent while the rest do.

`tests/test_pages.py` now clicks through every step of every chapter and fails on a literal
`**` inside any styled box. It caught chapter 04 immediately, which is how the code-block
problem surfaced at all.

The workbook and the notebooks were checked and are unaffected — they hand their markdown
straight to Streamlit and Jupyter without wrapping it.

**555 tests pass.**

---

## 2026-08-05 — Flow audit for chapters 13–19

Merged split screens where a kid had to click before seeing the picture that completed the
thought. Chapter 14 now keeps the hidden-space table, hidden lines, and combined boundary
in one payoff sequence; Chapter 17 reveals the confusion matrix on the prediction screen;
Chapter 19 shows the k boundary and sweet-spot curve together.

Built chapters 13–19 and ran the targeted page/notebook/workbook tests: **77 passed**.

---

## 2026-08-05 — Flow audit for chapters 06–12

Moved visual payoffs onto the screens that introduce them: Chapter 06 now shows boosting
chasing noisy wiggles, Chapter 07 puts kernels beside the road-style switcher, Chapter 08
shows the baseline number, and Chapter 12 shows the squished output surface immediately.

Chapter 11 brings the grid mover up to the second screen so kids can start playing before
the vector details. Chapter 12 folds the trained divider into the training screen.

---

## 2026-08-05 — Flow: un-fragmenting the chapters

Owner, with his kids using the course:

> "Kids are pretty unhappy with the lack of flow at times, and get confused. More of that
> will mean they'll lose interest, and all this will have been for nothing."

Two bugs, and **the second one was self-inflicted**.

### Bug 1 — a concept introduced before there was anything to look at

A screen would explain an idea and put the picture of it on the *next* screen, so the reader
had to hold something abstract across a page turn. Chapter 05 explained one-hot encoding
with **no before/after table** — the single thing that makes one-hot obvious. Chapter 02
walked through a perceptron update across 25 lines of arithmetic with nothing to look at.
Chapter 03 argued the XOR contradiction with the four points invisible. Chapter 12 described
what the squish buys without showing the surface it changes.

Fixed by pulling each picture onto the screen that introduces it.

### Bug 2 — one thought split across two screens, and that one is on me

When the pages were converted to the stepped format I set a floor of 5 steps and a target of
8-14, and told the agents to hit it. **A count target gets met by splitting things that
should not be split.** Chapter 21 ended up with four separate screens of 4-7 lines. Chapter
19 had "Morph the boundary" and "There is a sweet spot" — one idea about `k`, cut in half.

23 screens merged away across 15 chapters. Chapter 21 went 11 → 8, chapter 25 went 10 → 7.
Chapter 11's grid mover — the thing the owner most expects his kids to sit and play with —
moved from deep in the chapter to step 2.

The rule is now stated properly: **a screen must stand on its own — something to read,
something to look at, and ideally something to move.** Seven steps is a fine chapter.

### The guard

`tests/test_pages.py` now fails any mid-chapter screen with nothing to look at and nothing
to move, exempting the opening hook and the challenge lists.

Writing it repeated a mistake from earlier the same day: the first version checked call
sites only, and reported a false failure on a chapter that draws its chart through a
module-level helper. It now resolves helpers first. **Twice in one day a grep-shaped test
has lied** — once about plotly styling, once here. Tests that pattern-match source text need
to follow the indirection the source actually uses.

**581 tests pass.** 271 screens across 26 chapters, none of them a dead click.


---

## 2026-08-05 — Kid-read continuity pass for chapters 00-04

Read the first five chapters as first-contact material rather than as a checklist. The fixes define the first uses of model, feature, weight, bias, loss, gradient, gradient descent, perceptron, probability, sigmoid, e, log loss, polynomial features, and the chapter 03/04 real-data axes before the reader has to use them.

The bike and penguin datasets now get a plain-language setup and sample rows before modeling in the page and notebook flows, and the workbooks now reinforce the same setup instead of assuming the app did the introduction.

---

## 2026-08-05 — Kid-read continuity pass for chapters 18-25

- Re-read chapters 18-25 as a standalone student and filled first-use gaps in pages, notebook sources, and workbooks: Fashion-MNIST details, image-kernel vocabulary, kNN scaling, photo pixels as 3D colour points, PCA/t-SNE cautions, Part 6 text-model continuity, attention query/key/value, and finale callbacks.
- Kept the fixes in the chapter files instead of shared helpers so parallel chapter work stays isolated. Generated notebooks must be rebuilt from `_src` after this entry.

---

## 2026-08-05 — Kid-read continuity pass for chapters 05-10

Read chapters 05-10 as first-contact student material and fixed the places where a reader would hit “wait, what?” instead of auditing against a checklist. The pass adds first-use explanations for Gini impurity/pure buckets, boosting/residuals, random forests, stumps, RBF/gamma/C/kernel language, folds/cross-validation, baselines/class imbalance, feature importance, confusion matrices, precision/recall, and leakage.

The real-data tables now get row/column/target setup before modeling: mushrooms in Chapter 05, monsters in Chapter 06 and 09, penguins across Chapters 07-09, bikes in Chapter 09, and failure scenarios in Chapter 10. Rebuilt notebooks 05-10 and ran the targeted chapter tests: **72 passed**.

---

## 2026-08-05 — Kid-read continuity pass for chapters 11-17

Read the neural-network spine as a first-contact student and fixed the “wait, what?” gaps in the pages, notebook sources, and workbooks: grid/matrix sliders and 3D axes, neuron diagram labels, by-hand gradient arithmetic, hidden-space h-coordinates, capacity/overfitting controls, PyTorch library/autograd vocabulary, and the digits dataset/canvas setup.

Rebuilt notebooks 11-17 and ran the targeted tests: `timeout 900 uv run pytest tests -q -k "11_ or 12_ or 13_ or 14_ or 15_ or 16_ or 17_"` → **84 passed**.

---

## 2026-08-05 — "Pretend that you're the kid taking the lesson"

The owner found that Chapter 04's entire introduction to the penguins — the **first real
dataset in the course** — was: *"Now compare a few probability promises before they get
turned into a hard Gentoo/not-Gentoo score."* No mention that penguins are real birds, that
there are three kinds, what a Gentoo is, what a flipper length is, or why we care.

Three audits had already run over that chapter and passed it.

Then, when the first response was to check dataset introductions specifically:

> "not just limited the defined dataset. Be more thorough. Pretend that you're the kid
> taking the lesson"

That is the correction that mattered, and it is now the method of record.

### Why the earlier audits kept missing this

They audited **against checklists**. A checklist finds the things on the checklist. The
reader does not experience a checklist — they experience a sequence, and they fall out of it
at the first sentence that assumes something they do not have.

The brief for this pass says instead: *you are 13, you did the previous chapters once a few
days ago, you remember the pictures but not the words, and there is nobody to ask. **You
will not admit you are lost. You will keep clicking Next, understand less and less, and then
quietly go and do something else.*** Then: log every "wait, what?", however small, and fix it.

It also said that reporting a chapter as clean is evidence of not reading properly, because
three previous audits had reported exactly that and been wrong. **No chapter came back
clean.**

### What was actually missing

Not just datasets. Terms reaching the reader with no explanation anywhere in the course
before them: *model*, *feature*, *weight*, *bias*, *loss*, *gradient*, *impurity*, *stump*,
*fold*, *R²*, *class imbalance*, *capacity*, *requires_grad*, *optimizer*, *channel*,
*pooling*, *centroid*, *inertia*, *corpus*, *token*, *head*. Confusion matrices shown
without saying what the rows and columns are. A slider with no stated purpose. Chapter 18's
image *kernel* colliding with chapter 07's SVM *kernel*, with nothing to tell the reader
they are different things. Chapter 20 treating a pixel as a point in 3D colour space, which
is a real leap, without saying so.

And in almost every case the page had been fixed at some point while **the notebook and the
workbook still had the original gap** — including chapter 04, where the page fix had been
made by hand an hour earlier.

### The guard

`tests/test_teaching.py` now checks, for 58 technical terms, that the first place a term
reaches the reader *anywhere in the course* has an explanation near it. First means first in
the course, not first in the chapter — the reader only gets one first time.

It is a rough check: it can miss a bad explanation, but it cannot miss a missing one. It
found exactly one survivor after the sweep (colour *channels* in chapter 17, used a chapter
before they are defined), which is a good enough signal-to-noise ratio to keep.

**639 tests pass.**

---

## 2026-08-05 — UI overhaul: readable, centred, responsive, animated

Owner: *"the lesson.say parts are word-wrapping on streamlit, to the point that single words
are getting their own lines. Do a UI overhaul in general. Alignment is all over the place,
Center is better. Add cool animations."* Then: *"also ensure reactive layout. Zooming in and
out, or resizing the window should work like any other modern web app."*

### The one-word-per-line bug

`lesson.say` wrapped its text in a `<div class='kml-say'>` with `max-width: 68ch` so prose
would not run in long lines. Streamlit sized that element to **fit-content**, and a block
with a max-width inside a fit-content parent collapses to its **min-content** width — which
is the width of the longest word. Hence one word per line.

This is the *second* problem caused by that wrapper; the first was markdown rendering as
literal asterisks. So the wrapper is gone: prose now goes straight to `st.markdown` and the
column width comes from the page container, where it belongs. The remaining coloured boxes
keep their HTML but are forced to stretch.

### Alignment

Chapters were `layout="wide"`, so some elements ran full-bleed while others shrank to their
content — which is what "alignment all over the place" was. Everything is now one centred
column, with the chapter heading, beat trail and step title centred, figures centred, and
the landing page sharing the same stylesheet via `lesson.apply_style()`.

### Responsive

No fixed pixel widths in the reading column. `max-width: min(62rem, 100%)`, padding in
`clamp()`, and headings and body type in `clamp()` so browser zoom scales properly. Below
46rem the knobs-beside-picture layout stacks instead of squeezing, and columns get
`min-width: 0` so they can actually shrink rather than forcing a horizontal scrollbar.
Figures now render with `width="stretch"` instead of at natural pixel size, so a wide plot
in a narrow column scales down.

### Animations

Each screen rises and fades in, staggered by element so the screen assembles rather than
snapping. Boxes pop, the active beat pill glows, buttons lift on hover, and the progress bar
eases. Streamlit re-runs the script on every click, so this replays on every Next — which is
the point.

All of it sits behind `@media (prefers-reduced-motion: reduce)`, because some kids get motion
sick and some school machines have it switched on.

### Testing

Layout is invisible to `AppTest`, which reports element values rather than rendered geometry
— that is exactly why this batch of bugs survived so long. `test_the_stylesheet_is_responsive`
asserts the properties that matter (shrinkable column, scaling type, a breakpoint, stacking
columns, no overflow, reduced-motion support) and fails if a hard pixel width comes back.

**640 tests pass.**

---

## 2026-08-05 — Bling: centring, hover, and reveal-on-scroll

Owner: *"text in buttons and list items is not vertically center aligned. Bling please.
Animations reveal as elements come in fov. Tasteful Hover animations on controls, buttons
etc."*

**Vertical centring.** A Streamlit button label is a `<p>` inside a `<div>` inside a taller
button box, so it sits high. The button is now a flex container that centres on both axes,
with the inner label forced to inline and a shared line-height. Radio and checkbox rows got
the same treatment so the dot and its text share a middle line. List items got sane spacing
and a green `::marker`.

**Reveal as you scroll.** Done with CSS scroll-driven animations —
`animation-timeline: view()` behind an `@supports` guard, so browsers that have it get
proper viewport-triggered reveals and everything else falls back to the existing entrance
animation. No JavaScript.

Two traps worth recording. A view-timeline animation with `both` fill sits at zero opacity
until its element enters, so anything in a clipped or scrolling ancestor that never resolves
a timeline would stay **invisible** — the reveal is therefore scoped to `.block-container`
rather than applied globally. And the range is `entry 0% entry 55%`, which means elements
already on screen at load are past the end of the range and simply appear, with no flash of
hidden content.

**Hover.** Buttons lift and gain a green edge, slider thumbs grow with a soft halo,
selectboxes and inputs glow on focus, metric cards lift, boxes nudge sideways, expander
summaries tint. Keyboard `:focus-visible` outlines were added at the same time, since making
everything mouse-reactive without that is a regression for anyone tabbing through.

Everything stays behind `prefers-reduced-motion`, which now also neutralises transforms and
timelines rather than only shortening durations.

**641 tests pass.**

---

## 2026-08-05 — The green came back

Owner: *"kids corner and 'Thats it' boxes etc used to be green. Bring that back"*

They were green when the app used Streamlit's own `st.success` / `st.info` on a **light**
theme. Switching to dark muted those alert colours into near-grey, and the green went with
them. Nothing in the code changed — the theme underneath it did.

The fix is not to re-tint Streamlit's alerts, because **Streamlit gives no reliable hook for
styling an alert by kind**. Its DOM exposes `stAlert`, `stAlertContainer` and
`stAlertContent`, with success/info/warning distinguished only inside baseweb's own
component. Worth noting: an earlier commit had a rule targeting
`[data-testid="stAlertContentSuccess"]`, which does not exist — dead CSS that had been
sitting there doing nothing.

So the boxes are ours now, like `kml-look` and `kml-jargon` already were:

- **green** for good news — the aha moment, the Little Kid Corner, and a correct
  prediction or workbook answer
- amber for *Careful*, violet for a surprising reveal, red for a wrong answer

The aha and Little Kid Corner boxes also get a one-shot glow on arrival, which is the
animation that was previously attached to the selector that never matched anything.

`kidsml/workbook.py` and `kidsml/ui.py` were moved onto the same boxes so feedback looks
identical whether it comes from a chapter, a workbook or the landing page.

**642 tests pass**, including one asserting the good-news boxes are on the course green and
have not drifted back onto a Streamlit alert.

---

## 2026-08-05 — The chapter list

Owner: *"the chapter list on the left has too much space between items now. clicking on one
item suddenly squishes it and removed the gap, and the gap comes back after the page loads.
No gap there please. also render the left side with chapter numbers"*

**The twitching.** The entrance animations were applied globally — to every
`[data-testid="stMarkdown"]` and every `stVerticalBlock` child on the page, **including the
sidebar**. So the chapter list re-animated on every rerun: it collapsed as the animation
restarted from `translateY`, then settled back. Exactly the squish-then-gap the owner
described.

All animation is now scoped to `.block-container`, and the sidebar is explicitly excluded
with `[data-testid="stSidebar"] * { animation: none !important; }`. The nav is furniture,
not content, and a list that re-lays itself out on every click is maddening. Spacing is now
set deliberately: flex rows, a 1px gap, and a hover that tints and nudges, with the current
chapter marked by a green inset bar.

**The numbers.** Streamlit derives a sidebar label from the filename and strips the leading
number, so `00_guessing_machine.py` was showing as "guessing machine". That is bad in a
course that says "remember Chapter 03" constantly — the reader had no way to tell which one
that was.

Fixed by declaring the navigation explicitly with `st.navigation`, built straight from
`CHAPTERS`, giving labels like `00 · The Guessing Machine`. That required splitting the old
`app/Home.py` in two, because with `st.navigation` the entry script runs on **every** page
view: `Home.py` is now the router, and the landing content moved to `app/welcome.py`.
Chapter URLs are unchanged. Adding a chapter to `CHAPTERS` still adds it to the list on its
own.

**644 tests pass.**

---

## 2026-08-05 — A mermaid diagram that would not parse

Owner: *"the one neuron page says there's syntax error in mermaid"*

Chapter 13 had this edge label:

```
O -->|2(out-y) = -1| L[loss]
```

Mermaid reads an opening bracket inside an **unquoted** label as the start of a node shape,
so the label ended at `2` and the rest was nonsense to the parser. Quoting it fixes it:
`O -->|"2(out-y) = -1"| L[loss]`.

**Why no test caught it.** Mermaid renders in the browser. A syntax error is a red box on
the page and produces nothing at all on the Python side — the same blind spot as the plotly
white backgrounds and the one-word-per-line wrapping. Everything about how this app *looks*
is invisible to the suite.

So the diagrams were checked properly, once, with the real parser: `npm install mermaid
jsdom`, then `mermaid.parse()` over every diagram extracted from the pages and notebook
sources, with a jsdom window because mermaid pulls in DOMPurify. **52 diagrams, 2 broken**
(the page and its notebook), both the same label.

That Node check is not wired into the suite — adding a JavaScript toolchain to a project
whose whole point is that it runs from one `uv` command is a bad trade. Instead
`tests/test_diagrams.py` lints for the mistakes that actually break diagrams: unquoted
brackets inside a label, and a missing diagram type on the first line. The command to run
the full parser check is recorded at the bottom of that file for the next time a diagram
misbehaves.

**749 tests pass.**

---

## 2026-08-05 — The stranded eyes emoji

Owner: *"why is there a line gap between the eyes emoji and 'Look for' text?"*

Self-inflicted, from the green-boxes commit. That change added:

```css
.kml-box > b { display: block; margin-bottom: 0.4rem; }
```

which is right for the boxes whose bold run is a **heading** — `<b>💡 Aha!</b>` followed by
prose. But the look-for box puts the emoji *outside* the bold run:

```html
<div class="kml-box kml-look">👀 <b>Look for:</b> ...text...</div>
```

Making that `<b>` a block pushed "Look for:" onto its own line and left the 👀 sitting alone
above it. Same for the jargon box, whose bold run is a term mid-sentence.

The rule is now scoped to the three heading boxes (aha, kid corner, careful), with the
mid-sentence ones explicitly kept inline.

Worth noting the shape of this mistake, since it is a common one: a rule written for one
member of a family, applied to the whole family. `.kml-box` covers eight boxes with two
quite different internal structures, and the selector did not distinguish them.

**750 tests pass.**

---

## 2026-08-05 — Fit and finish, with a browser this time

Owner: *"do a full sweep for such fit and finish issues"*

Every visual bug so far had survived because the suite cannot see rendering. So this pass
started by fixing that: Playwright driving a Nix-provided Chromium (the downloaded
Playwright browser will not run on NixOS), screenshotting real pages at desktop and phone
widths, and measuring the DOM. That turned a guessing game into a list.

What it found, in order of how bad it was:

**Mermaid diagrams were rendering on white.** Every diagram was a bright card in the middle
of a dark page. `streamlit-mermaid` exposes no theme setting, so the theme is now declared
in the diagram source itself with a `%%{init: ...}%%` directive carrying the course palette.

**Mermaid reserved 424px whatever it drew.** A 55px flowchart sat in a 424px box, leaving a
hole under it. The component ignores the height it is given, so the iframe is sized from CSS
via a keyed container (`st.container(key=...)` puts a `st-key-<key>` class in the DOM), with
the height estimated from the diagram — one row for a left-to-right flow, ~84px per box for
a top-down one. Deliberately over-estimated: slack is untidy, clipping is broken.

**Sixteen diagrams were unreadable.** A left-to-right flow with six long labels gets scaled
down until the text is a few pixels tall. Those are now top-down, so each box gets a full-
width row. A `graph LR` is fine for three or four short labels and nothing more.

**Figures were being scaled up.** `width="stretch"` blew a 704px chart up to 884px and
enlarged every label with it. Back to natural size, with the CSS cap handling narrow
windows — scaling *down* does no harm.

**Chart type was competing with the headings.** Streamlit renders a figure at about 175 dpi
and then scales it into the column, so a matplotlib point is worth roughly two screen
pixels — which made an 13pt chart title bigger than the step heading above it. Sizes are now
chosen for that effective scale, which looks absurdly small in the rcParams and correct on
screen.

**Streamlit's Deploy button** was sitting in the top right of every chapter. Gone, along
with the rest of its chrome.

Verified afterwards: no horizontal overflow at 1280px or 420px, no console errors, no
clipped diagrams (SVG height measured against iframe height on every page that has one).

The Playwright recipe is written up in `AGENTS.md` under "Seeing the app". It is not wired
into the suite — it needs Node and a browser, and this project runs from one `uv` command —
but it should be the first thing reached for whenever something looks wrong.

**750 tests pass.**

---

## 2026-08-05 — AMOLED black

Owner: *"instead of bluish gray, can we have amoled black theme across the board?"*

The dark theme had been a blue-grey (`#0E1117` page, `#171B26` panels) — Streamlit's own
dark palette. Now true black: on an OLED screen a black pixel is an off pixel, so the page
disappears and only the content is lit, and it is the least tiring thing to read at night,
which is when this gets used.

Done as one palette map applied across `kidsml/plots.py`, `kidsml/lesson.py` and
`.streamlit/config.toml`, so there is no chance of half the app moving and half staying:

- page `#000000`, panels `#0B0B0D`, raised surfaces `#131315`
- lines and dividers neutral grey rather than blue-tinted
- the tinted boxes keep their hue and lose almost all their lightness — good news green sits
  on `#04140D` now instead of `#10241C`

`.stApp` and the sidebar are forced to black explicitly, because Streamlit paints the
sidebar with its *secondary* background colour, which left a visible grey slab beside a
black page. The sidebar keeps a one-pixel divider so it still reads as a separate column.

Checked in the browser rather than assumed: `body` computes to `rgb(0, 0, 0)` on every page,
and the plots, the confusion grid, the digit-number grid and the mermaid diagrams were all
re-rendered and eyeballed against the new background.

A test asserts the plot background, the Streamlit config and the forced surfaces are black,
and fails if any of the five old blue-grey hexes reappears in the stylesheet.

**750 tests pass.**

---

## 2026-08-06 — Fit and finish: readability, a reordered curriculum, and plain English

A round of feedback from the repo owner, after reading with the 8th grader. Three themes:
the pages were hard to *see*, some of the writing was hard to *follow*, and several
controls were dead.

### The curriculum moved

**k-nearest-neighbours moved from chapter 19 to chapter 08.** It is a supervised
classifier, and "The Model Zoo" — the chapter about which model when — was racing every
guesser in the course *except* the simplest one. Chapters 08–18 all shifted up by one.

k-means and PCA stayed in Part 5. They are genuinely unsupervised, and moving clustering
in front of the Zoo would have weakened the Zoo's "pick a supervised model" race rather
than strengthening it. Part 5 now opens with chapter 20 instead of 19, and the framing
sentences moved with it.

`CHAPTERS` in `kidsml/ui.py` is still the only place the order is written down, so the
sidebar, the course map and the filename tests all followed automatically. The wall-clock
budgets in `tests/test_notebooks.py` were remapped by hand.

### Every mermaid diagram in the neural-network chapters was invisible

The single worst bug found in this round. The mermaid build bundled with
`streamlit-mermaid` has an **ASCII-only lexer for unquoted node labels**. A label like
`X1[x₁]` throws in the browser, and because the failure is a JavaScript exception inside
the component's iframe, the page renders a **blank gap** where the picture should be — the
"giant blank space" the owner reported on the One Neuron hook, and the missing picture on
"XOR is back".

Three fixes, because this class of bug must not come back:

1. The subscripts are gone from the diagram sources. `x1` reads better for a 13-year-old
   than `x₁` anyway, and "add them up, plus b" beats a sigma.
2. `lesson.mermaid` now quotes any label containing a non-ASCII character on its way
   through, so a future author cannot silently break a page.
3. `tests/test_diagrams.py` fails the build on a non-ASCII unquoted label.

While in there: `_estimated_height` used to assume a left-to-right diagram was one row of
boxes tall, which clipped the bottom off every branching one. It now ranks the graph
properly (longest path, with loops broken) and sizes from the widest rank.

### Contrast

The AMOLED theme landed before some of the older figures were checked against it.

- `kidsml/plots.py` gained a named palette — `SHAPE`, `GHOST`, `EDGE`, `GRIDLINE`, `AMBER`,
  `VIOLET`, `PINK`, `TEAL` — and the grid, axes and class-region colours were all lifted.
  The rule is now written down in AGENTS.md: **never hardcode a hex**.
- Chapter 12 (Arrows and Grids) was still drawing on a light-theme palette: the little
  house was `#0F172A`, a near-black outline on a near-black page, so the shape the whole
  chapter asks you to watch was not there. It is now `SHAPE`, the brightest thing in the
  figure, with the "before" grid and house in `GHOST`.
- The neuron and network diagrams in `kidsml/nnplots.py` used pale pastel fills with the
  default near-white text on top, so each box hid its own label. Dark panels, bright rims.
- Markers no longer get white outlines: the house style cuts them out of the panel colour.

### The Next button

Streamlit puts white text on the green primary button — about 1.8:1, which is unreadable.
It is now very dark green ink on the same green slab, over 10:1, and a disabled Next no
longer looks live.

### Dead controls, and things that would not re-roll

| Where | Was | Now |
|---|---|---|
| 14 · The weights walk | A straight line at every setting: the two weights grew in proportion on a separable problem, so there was nothing to see | Draws the loss valley behind the path (bias held fixed, so the 2D map is honest), with start/end markers. A small rate crawls, a good one lands on the star, a large one overshoots |
| 15 · Scrub the training | Almost all the movement happened between the first two snapshots | Snapshots resampled by arc length in weight space, plus a learning-rate control |
| 16 · Weight decay | A linear slider to 0.08, dead above 0.02 | A `select_slider` over the values that actually do something |
| 17 · PyTorch knobs | No way to train for longer or shorter | A training-steps control; the loss curve and boundary move with it |
| 18 · Draw a digit | The canvas component's own toolbar draws in near-black — the clear button was invisible | `display_toolbar=False` plus our own bright "Clear the pad and start over" |
| 22–25 · anything that samples | Cached against a stale roll, so changing the prompt changed nothing | `lesson.regenerate(...)` — a button returning a counter to feed in as the seed |

`lesson.regenerate` is the general fix and is now the documented house rule: **anything
that rolls dice needs a way to roll again.**

### Teaching

- **Chapter 12 no longer mentions neurons**, which chapter 13 introduces. The chapter's
  punchline — stacked straight moves collapse into one straight move, so layers buy nothing
  without a squish — stands perfectly well in the chapter's own vocabulary, with the word
  itself held back as a teaser.
- **Chapter 15 animates the fold.** The reader used to be told that a hidden layer moves
  the points into a space where one line works, and had to take it on faith from two static
  scatter plots. Now the four XOR corners visibly slide into their hidden coordinates and
  the separating line drops in. (`kidsml/foldspace.py`, a cached ~30-frame GIF built with
  Pillow — no new dependency, well inside the notebook budget.)
- **Chapter 08's k curve now has a hump.** It was scored on one split of easy data and came
  out flat, which quietly contradicted the chapter's main claim. It is now 5-fold
  cross-validated, averaged over several datasets, and swept far enough for the collapse:
  it climbs from k = 1, peaks around k = 5, and falls off a cliff. The failure ends are
  labelled on the figure.
- **Chapter 19 shows the data before asking for a prediction.** Betting on a model when you
  have never seen the pictures is a coin flip, and a coin flip teaches nothing.
- **Chapter 25 has 30 projects in six labelled groups**, each with the chapters it uses and
  where the data comes from — all of it collectable by hand or already bundled.

### Language

A pass over every chapter. The rule added to AGENTS.md: **a metaphor may decorate an
explanation; it must never be the explanation.** Casualties included "Neural networks are
not a new planet", "a line wearing a costume", "More wiggle is a trade", "two magnets
clicking shut", "a flat pca shadow still tangles the lifted circles", and "half a fact
wearing a full-size hat".

The dad-slang running joke stays, but it is now confined to captions, asides, metric labels
and challenge titles — never inside a sentence that is doing the teaching. Every chapter is
down to one or two.

Three stacked "Grown-ups call this" boxes in chapter 14 became one. A wall of three
identical grey boxes reads as a wall.

### Tooling

`tools/shots.mjs` drives a real browser over a chapter's steps and writes screenshots.
Every visual bug in this round was invisible to `AppTest`, which reports element values and
not rendered geometry — it cannot see a diagram that failed to parse or a dark shape on a
dark page. `node_modules/` is gitignored; `npm install playwright` is a developer step, not
a project dependency.

### A warning for future sessions

Part of this round's work was destroyed mid-flight by a `git reset --hard` run inside a
worktree that had hours of uncommitted changes in it, and had to be redone from scratch.
**Commit early and often when several things are editing the same tree**, and never reach
for `git reset --hard`, `git checkout .` or `git stash` as a way of tidying up.
