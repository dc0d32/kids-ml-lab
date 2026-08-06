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
   Nothing downloads at runtime except Fashion-MNIST in Chapter 18, and that is cached.
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

Aim for **Bill Nye**: fast, delighted, physical, never solemn. Short sentences. Concrete
nouns. Second person — "your line", "you already know this". The energy of someone who
genuinely thinks this is the coolest thing and cannot wait to show you.

- Never write "simply", "just", "obviously", or "trivial". They are the exact words that
  make a nervous reader give up.
- Humour is welcome. Condescension is not.
- Numbers in a by-hand question must be small and round enough to do with a pencil in
  under three minutes.
- Every chapter needs a **🧸 Little Kid Corner**: the same idea with zero algebra, usually
  a physical game or an analogy the 4th grader can act out.

### The running joke: dad slang, used slightly wrong

The 8th grader this was written for gets secondhand embarrassment when his dad uses
Gen-Z/Gen-Alpha slang. So the course does it on purpose. **The joke is not the slang. The
joke is the course being visibly pleased with itself for knowing the slang.**

How to land it:

- **Rarely.** One or two per chapter, maximum. Peppered, not poured. The gag dies instantly
  if it's on every screen.
- **Slightly off**, the way a parent uses it — a beat late, too formal, or over-explained.
  *"A tree this deep has, and I believe I am using this correctly, no aura."*
- **Never inside an explanation that's doing real work.** Put it in a caption, an aside, a
  challenge title, a metric label — the places where a reader losing a second to an eye-roll
  costs nothing. Clarity always outranks the bit.
- **Keep it kind and keep it clean.** It's never at the reader's expense.
- Usable vocabulary: cooked, mid, goated, no cap, bet, sus, lowkey, based, cringe, ate,
  let him cook, touch grass, main character, NPC, aura, delulu, vibe check, side quest,
  glazing. Avoid anything crude or body-related.

Good: `st.metric("Model's aura", "0")` under a chart of a model that just failed.
Good: *"Nine multiplies and one add. That's it. That's the whole convolution. It is, as
they say, mid — and mid is exactly why it's fast enough to run on every pixel."*
Bad: three slang terms in one paragraph, or slang in the middle of the backprop derivation.

If you can't make it funny, leave it out. An unfunny bit is worse than no bit.


---

## How much to explain (read this twice)

**These kids will often work through a chapter alone, with nobody to ask.** That sets the
bar: if a reader could get stuck and have no way out, the chapter is not finished.

There are two ways to fail, and they are opposites:

- **Too thin.** A teaching point delivered as a single asserted line. "Trees can do that
  too." The reader nods, doesn't actually understand, and quietly falls behind. *This is
  the failure mode we keep hitting — check for it specifically.*
- **Too thick.** Six paragraphs before anything happens on screen. The reader's eyes
  glaze and they close the tab.

### The rule: every idea gets three moves

Never state an idea and move on. Give it:

1. **Setup** — the problem, or the question the reader is already asking.
   *"Averaging helps when guesses are wrong in different directions. But why would two
   trees ever disagree? They'd see the same data and make the same tree."*
2. **The idea** — plainly, with a concrete example and real numbers.
   *"So we don't give them the same data. Each tree gets a random sample of the rows and
   is only allowed to look at some of the columns. Tree 1 might never even see the `speed`
   column."*
3. **So what** — what it buys you, or what it lets you do next.
   *"Now their mistakes are different mistakes, and different mistakes cancel out when you
   vote. That's the entire trick."*

Three short paragraphs beats one dense one, and it beats one line by a mile.

### Rhythm

- **Roughly 60–120 words of prose, then something happens** — a picture, a slider, a
  table, a diagram, a line of output. Never more than ~150 words unbroken.
- Break long explanations with a diagram (`ui.mermaid`) or a small figure. A picture
  between two paragraphs is worth more than a better paragraph.
- Immediately after a picture, say **what to look at in it**. A figure with no pointer is
  decoration. "Notice that the boundary between the two clumps is perfectly straight —
  that's the part that's about to become a problem."

### Answer the question they're about to ask

After each explanation, ask yourself what a sharp 13-year-old would say next — *"but why
does squaring it help?"*, *"wait, where did that 0.5 come from?"*, *"what if they're all
wrong in the same direction?"* — and answer it right there. That question is the single
best guide to what the next paragraph should contain.

### Don't skip the arithmetic

When a number appears, show where it came from. `z = 2(1) + (-1)(3) + 0.5 = -0.5` is more
use than "compute the weighted sum". Kids trust numbers they've watched being built.

---

## Diagrams

Use **Mermaid** for structure — how a prediction flows, what order steps happen in, how a
thing is wired together. Use **matplotlib** for data — boundaries, curves, points, images.

On a page:

```python
lesson.mermaid("""
graph LR
    X1[x1] --> S[add them up, plus b]
    X2[x2] --> S
    S --> A[squish]
    A --> Y[output]
""")
```

**Node labels must be plain ASCII.** The mermaid build bundled with `streamlit-mermaid`
has an ASCII-only lexer for unquoted labels: one `x₁` or `Σ` and the whole diagram throws
in the browser, which the reader sees as a blank gap in the middle of the page and which
no other test can see. `tests/test_diagrams.py` fails the build if you do it. Write `x1`,
not `x₁`, and say "add them up, plus b" rather than reaching for a sigma.

In a notebook, a fenced ```mermaid block inside a markdown cell — JupyterLab renders it
natively, and so does GitHub. Keep the page version and the notebook version in step.

Keep diagrams to six or seven boxes. A diagram that needs studying has stopped helping.
Good uses: the neuron, the layers of a network, the boosting loop, the convolution slide,
the train/test split, the attention flow, the course map.

---

## Animation

Some ideas are about *change over time*, and a before/after pair asks the reader to do the
animating in their head. A grid bending under a matrix, a window sliding over a picture,
cluster centres drifting into place — in those the motion **is** the teaching point.

`kidsml/anim.py` is the one recipe: matplotlib frames stitched into a looping GIF with
Pillow, entirely in memory. Both are already dependencies. Do not add manim — it needs
cairo and ffmpeg, which breaks "runs with one `uv` command".

```python
from kidsml import anim

fig, ax = lesson.figure()
dots = ax.scatter(...)

def draw(i, progress):          # progress runs 0 -> 1, eased, across the moving section
    dots.set_offsets(start + progress * (end - start))

data = anim.gif_bytes(fig, draw, frames=30)
```

Then `st.image(data)` on a page (behind `@st.cache_data`), and
`IPython.display.Image(data=data)` in the notebook.

Rules that were learned the hard way:

- **Animate a claim, not a control.** If the reader benefits from *turning* the parameter,
  a slider beats a clip. Animate the things the chapter currently asks them to take on
  trust. Keep the slider as well — the clip explains, the control lets them poke.
- **25–45 frames, 5–7 seconds.** Every clip in the course together builds in under 8
  seconds, which is what keeps the notebook budgets safe.
- **Say what to watch.** A clip gets a `lesson.look_for(...)` like any other picture.
- **Leave something still.** A looping GIF cannot be paused, so keep a static figure or a
  before/after pair nearby for a reader who wants to stare.
- **Give the title room.** `fig.tight_layout()` measures an empty title and then clips the
  text the moment `draw` sets it. Use `fig.subplots_adjust(top=0.90)` instead.
- **Check the frames, not the page.** A browser screenshot catches one frame of a loop.
  Decode the GIF instead — and call `.convert("RGB")` on each frame, because they are mode
  `"P"` and reading them raw gives palette indices that look like garbage.

---

## Colour

The app is pure black (AMOLED). Every colour comes from `kidsml/plots.py` and nowhere
else, because a colour picked against a white page usually vanishes on this one:

| Name | For |
|---|---|
| `COOL` / `WARM` | class 0 / class 1, everywhere in the course |
| `ACCENT` | the model's own line or prediction |
| `AMBER`, `VIOLET`, `PINK`, `TEAL` | extra series when three colours are not enough |
| `SHAPE` | a drawn outline that must be the brightest thing in the figure |
| `GHOST` | the faint "before" layer: the original grid, the untrained boundary |
| `MUTED` | unlabelled points |
| `EDGE`, `GRIDLINE` | the axes box and the graph paper |
| `PANEL`, `BACKGROUND`, `INK` | figure furniture |

Never hardcode a hex. In particular never use a dark navy, slate or near-black as ink, and
never outline a marker in white — the house style cuts markers out of the panel colour
instead of putting a halo round them.


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
  Home.py        the router: declares the chapter list via st.navigation
  welcome.py     the course map landing page
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

The sidebar chapter list is declared explicitly in `app/Home.py` with `st.navigation`,
because Streamlit strips the leading number off a filename and the course refers to itself
by chapter number constantly. Adding a chapter to `CHAPTERS` is enough; the list builds
itself.

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
./run.sh build 13 14  # regenerate only chapters 14 and 14
```

There is also a `flake.nix` devShell (`nix develop`) that sets the same paths
declaratively.

**Windows** uses `run.ps1`, which takes the same subcommands. Both launchers set
`PYTHONUTF8=1`, because the chapters are full of emoji and the Windows console default
encoding is not UTF-8. If you read a repo file in Python, **pass `encoding="utf-8"`
explicitly** — the platform default is not the same everywhere.

Datasets are committed, so `tools/prepare_data.py` only needs re-running if a source
changes.

---

## Adding or changing a chapter

A chapter is three files, where `NN` and `S` come from `CHAPTERS` in `kidsml/ui.py`:

| File | What it is |
|---|---|
| `app/pages/NN_S.py` | the Streamlit chapter — **stepped**, one idea per screen |
| `notebooks/_src/NN_S.py` | the notebook source, jupytext `py:percent` (`# %%` / `# %% [markdown]`) |
| `worksheets/NN_S.py` | the interactive workbook — a `WORKBOOK = Workbook(...)` |

Then:

```bash
./run.sh build NN
uv run pytest tests -q -k "NN_"
```

### Pages are stepped — one idea per screen

A chapter page is **not** a long scrolling document. Scrolling invites skimming, and
skimming is how a reader reaches the end of a chapter having understood none of it.
`kidsml/lesson.py` turns a chapter into a sequence of screens with Back/Next, a progress
bar and a beat trail.

```python
from kidsml import lesson

lesson.begin(3)

@lesson.step("Four points, four answers", beat="hook")
def _():
    lesson.say("Here is the smallest hard problem in machine learning.")
    ...

lesson.finish()
```

Rules for steps:

- **A step is 1–3 sentences and one thing to look at.** If a step needs scrolling, it
  should have been two steps.
- **8–14 steps per chapter.** A test fails below 5. Above ~20 it drags.
- **Predict, then reveal.** Use `lesson.predict(question, choices, correct=, why=)` before
  showing a result. It returns `None` until they commit, so the step can withhold the
  reveal. A wrong prediction followed by a surprise teaches far more than a correct one
  that was never in doubt. This is the single most valuable pattern in the framework —
  every chapter should use it at least twice.
- **Nothing passive.** Every step wants a slider to move, a question to answer, or a
  picture with `lesson.look_for("...")` telling them what to notice in it.
- Steps must still run in the six beats' order: hook → byhand → seeit → play → forreal →
  challenge.

Helpers: `lesson.say`, `predict`, `look_for`, `aha`, `careful`, `kid_corner`, `jargon`,
`figure`, `show`, `mermaid`, `workbook`, `controls`, `regenerate`.

**Anything that rolls dice needs `lesson.regenerate(...)`.** Streamlit only re-runs when
something changes, so a step that samples once looks frozen — the reader edits the prompt
and the same words come back. `regenerate` draws a button and returns a number that goes
up on every press; feed it in as the seed *and* into the `@st.cache_data` key. Say in one
line that the model rolls dice, so a different answer is the model working, not a bug.

**One jargon box per step.** Three `lesson.jargon(...)` calls in a row is a grey wall.
Name all the words the step just taught in a single box.

Notebooks stay linear — scrolling is the right shape for a document you edit and re-run.
`kidsml/ui.py` keeps the older non-stepped helpers for the Home page.


### The six beats — every chapter, same order

| Call | Header | What goes in it |
|---|---|---|
| `ui.beat("hook")` | 🎣 Start here | A question or a game. Plain English. No math, no code. |
| `ui.beat("byhand")` | ✏️ Work it out | A few rows of tiny numbers, worked out with a pencil, then the same thing in code so they see it match. |
| `ui.beat("seeit")` | 👀 Take a look | A picture of the exact thing they just did by hand. |
| `ui.beat("play")` | 🎛️ Your turn | Sliders. One knob → the picture changes within a second. This is the heart of the chapter. |
| `ui.beat("forreal")` | 💻 In real code | 10–25 lines of real code on real-ish data. |
| `ui.beat("challenge")` | 🏆 Go further | Numbered quests — beat the machine, break it on purpose — ending with a 🧸 item. |

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

---

## Seeing the app

Most of this project's bugs have been visual, and **none of them were visible to the test
suite**. `AppTest` reports element values, not rendered geometry: it cannot see a white
plotly box on a dark page, text wrapping one word per line, a mermaid syntax error, a
clipped diagram, or a chart whose title is twice the size of the prose.

So look at it. On NixOS the downloaded Playwright browser will not run, but a Nix one will:

```bash
npm install playwright
nix build --no-link --print-out-paths nixpkgs#chromium     # gives CHROMIUM
./run.sh app &                                             # or a fixed --server.port
```

```js
import { chromium } from "playwright";
const browser = await chromium.launch({ executablePath: CHROMIUM + "/bin/chromium",
                                        args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
await page.goto("http://localhost:8501/04_maybe_probably", { waitUntil: "networkidle" });
await page.waitForTimeout(3500);          // Streamlit renders after load
await page.screenshot({ path: "shot.png", fullPage: true });
```

Worth checking every time: a 420px-wide viewport, `document.scrollWidth` against
`clientWidth` for accidental horizontal scroll, and `naturalWidth` against the displayed
width of any image to see whether a figure is being scaled up.
