# Teaching Notes

For the grown-up. The kids never need to read this.

> **Chapter numbers in this file are checked against `CHAPTERS` in `kidsml/ui.py`.**
> If you reorder the course, fix them here too.

---

## The one thing to get right

You are not covering a syllabus. You are waiting for a specific moment, once per chapter,
where the kid's face changes and they say some version of **"oh — that's *all* it is?"**

Everything else is scaffolding for that moment. If you get it, the chapter worked, even if
you skipped half the material. If you don't get it, moving on faster won't help.

The second thing: **they are allowed to not finish a chapter.** Stopping at the good bit
beats grinding to the end.

---

## How a session actually goes

A chapter is roughly **35–50 minutes** at a comfortable pace. That is one sitting. Two
chapters in a row is usually one too many — the second one gets nodded at rather than
understood.

A session that works:

1. **Read the Hook out loud, together.** Don't let them read it silently. The hook is
   written to be a conversation opener, and it dies on the page.
2. **Do the "By Hand" part with a pencil, on scrap paper, before touching the workbook
   boxes.** This is the part they will want to skip. Don't let them. The whole design of
   this course assumes they have personally done the arithmetic at least once.
3. **Let them loose on the sliders.** Say nothing for a few minutes. Seriously — sit on
   your hands. The play beat is where the understanding actually forms, and narrating over
   it interrupts exactly the process you want.
4. **Ask them to explain the picture back to you.** Not "do you understand?" — they will
   always say yes. Ask "so what happens if I drag this one all the way over?" and let them
   predict *before* they drag it. A wrong prediction followed by a surprise is worth more
   than ten correct ones.
5. **Challenges are optional and should feel like dares**, not homework. "Bet you can't
   break it" works. "Please complete exercises 1 through 4" does not.

### The app or the notebook?

Start every chapter in the **app** (`./run.sh app`). It's the playground; there's no code
in the way.

Open the **notebook** (`./run.sh lab`) when they ask "but how does it actually *do* that?"
— which they will, at different chapters for different kids. The notebook has the same
chapter with the code showing and editable. Some kids live in the notebook from Chapter 02
onwards; some never open it. Both are fine.

---

## Pacing

There is no correct route. Pick the one that matches the kid in front of you.

**The full course**, in order, about 15–20 sittings. Best if they're enjoying it.

**The impatient route** — for a kid who only wants to know how ChatGPT works. Go
00 → 01 → 02 → 03 → 11 (linear algebra) → 12 → 13 → 14 → 23 → 24 → 25. Skips the
classical models entirely. You can always come back for them, and after seeing a
Transformer they often *want* to.

**The "I like puzzles" route** — 00 → 05 → 06 → 07 → 08 → 09 → 10. Trees and honest
evaluation, no calculus-shaped ideas anywhere. Genuinely satisfying on its own.

**Rainy afternoon, one chapter, no commitment** — 00, 05, 11, 20 and 22 all stand alone
and each has a strong payoff inside 40 minutes. Chapter 11 in particular is the one they
are most likely to wander back to on their own.

### If you only have time for six chapters

00 (what learning from examples means) · 02 (a line that decides) · 03 (why one line isn't
enough) · 11 (linear algebra, where space starts moving) · 14 (hidden layers — the payoff)
· 24 (the Transformer). That's the spine.

---

## The moment to wait for, chapter by chapter

Your job is mostly to not talk over these.

| Ch | Wait for |
|---|---|
| 00 | The learning curve rising while **not one line of code changed**. Only the number of examples changed. |
| 01 | The squared error drawn as an actual square. "Oh, that's why it's called *squared*." |
| 02 | They move the line by hand, one update, and it lands correctly. They just trained a model with a pencil. |
| 03 | Four points. Four answers. And no line on Earth works. Then the circles dataset lifting into 3D and a flat plane slicing it. |
| 04 | A point sitting on the line, and the model finally saying "honestly, I'm not sure". |
| 05 | Their hand-picked first question turns out to be the same one sklearn picks. |
| 06 | The boosting staircase — a smooth curve being built out of little steps, one leftover at a time. |
| 07 | Deleting a point and the road **not moving**. Then deleting a support vector and it jumping. |
| 08 | The same two models swapping places on the leaderboard when you change one random seed. |
| 09 | Their own feature picks losing to the model's feature importances. Or beating them. |
| 10 | A model that is 99% accurate and completely useless. |
| 11 | Two matrices applied in a row being **exactly** the same as one matrix — difference 0.00. Then a squish going in the middle and the grid bending. |
| 12 | The neuron diagram, and the realisation that the inside of the brackets is Chapter 02 and the squish is Chapter 04. |
| 13 | The slow obvious gradient and the clever fast one agreeing to eleven decimal places. |
| 14 | The three hidden lines sliding into place during training, and XOR becoming separable in the new space. **This is the big one.** |
| 15 | Test loss turning around and climbing while train loss keeps falling. |
| 16 | PyTorch's gradients matching the ones they computed by hand. |
| 17 | Reading a digit off a grid of numbers with their own eyes. |
| 18 | Detecting an edge with a pencil, then finding the same filter inside a trained network. |
| 19 | k=1 carving an island around every noisy point — Chapter 05's overfitting, wearing a different hat. |
| 20 | Their own photo collapsing to five colours. |
| 21 | An 8 still looking like an 8 with most of its information thrown away. |
| 22 | The tally chart inventing pronounceable names that don't exist. |
| 23 | **The vowels clustering on their own.** Nobody told it what a vowel is. |
| 24 | The attention map lighting up, and the honest scale conversation afterwards. |
| 25 | The whole course on one page, and realising they built all of it. |

---

## When they get stuck

**Do not rescue early.** Being stuck for two minutes is where learning happens. Being stuck
for ten is where quitting happens. Aim for the gap.

The ladder, in order:

1. *"What do you think will happen if you drag that?"* — hand it back.
2. *"Which bit is the confusing bit?"* — often they know, and saying it out loud fixes it.
3. Point at the **workbook hint** ("Stuck? Nudge me"). Every question that could plausibly
   stump someone has one.
4. Do one line of the arithmetic **with** them, then hand the pencil back mid-problem.
5. Only then, explain.

If a whole chapter isn't landing, the fix is almost never to explain harder. It's to go
back to the previous chapter's *play* section for five minutes. Nearly every stuck moment
in this course is a gap one chapter earlier.

### Things that look like confusion but aren't

- **Randomly dragging sliders to see what breaks.** That's the intended use. Leave them.
- **Getting a workbook answer wrong.** The `why` text does the teaching either way; a wrong
  answer that gets read is worth more than a right one that gets skipped.
- **Wanting to change the code before understanding it.** Fine. Let them break it. `git
  checkout` exists.

---

## Keeping the 4th grader in it

Every chapter has a **🧸 Little Kid Corner** — same idea, no algebra, usually something to
do away from the laptop.

What works:

- Give the younger one **the sliders** while the older one reads. One drives, one navigates.
- The Little Kid Corner games are genuinely better done on a carpet than on a screen —
  Chapter 02's is a pencil between two piles of toys, Chapter 20's is sorting laundry.
- Let them **be the model**: you give examples, they guess the rule. They'll beat the
  computer at Chapter 00 and it will make their week.
- Don't ask them to follow the maths. If they want to, let them; if not, they're getting
  the idea anyway, which is the whole point of the corner.

Chapters that work particularly well for a 9-year-old: **00, 05, 11, 17, 20, 22**.

---

## Questions they will ask

Short, honest answers. You don't have to be an expert to give them.

**"Is it thinking?"**
No. It's finding patterns in numbers and picking the most likely next thing. That turns out
to be enough to do some genuinely amazing stuff, and it's also why it can be confidently,
completely wrong. Both facts come from the same place.

**"Is it alive / does it know it exists?"**
No. Chapter 24 builds a small one from scratch, and there's nowhere in there for that to
live. It's matrix multiplications and a squish, repeated. Being unimpressed by the mechanism
while being impressed by the results is the correct position.

**"Will it take everyone's jobs?"**
It changes what jobs look like — a real thing worth taking seriously, and not the same as
"replaces people". Point out that they now understand more about how it works than most
adults do, which is the actually useful position to be in.

**"Why does it lie?"**
It was never trained to be right. It was trained to produce likely-looking text. Looking
right and being right are different targets, and it only ever practised one of them.
Chapter 25 shows this using their own model.

**"Could I build ChatGPT?"**
You already built a tiny one in Chapter 24. The difference is quantity — a few hundred
thousand pounds of computing time, and roughly a billion times more of everything. Not a
missing secret. That's the honest and much more interesting answer.

**"Why is my model worse than yours?"**
Usually a different random seed. Chapter 08 is entirely about how much of a score is luck.

---

## Practical stuff

```bash
./run.sh app      # the playground — start here
./run.sh lab      # the notebooks
./run.sh test     # check everything still works
```

- **Everything runs on the laptop.** No GPU, no cloud, no accounts, nothing to sign up for.
- **Nothing goes on the internet** except a one-time ~30 MB dataset download in Chapter 18,
  which is then cached.
- **Every chapter finishes in seconds.** If something takes minutes, something is wrong —
  run `./run.sh test`.
- **They can't break it permanently.** Everything is in git. `git checkout .` undoes any
  amount of enthusiastic editing.
- **Let them edit the code.** Especially `kidsml/zeeps.py` (Chapter 00 invites them to add
  their own secret rule) and any chapter's parameters. Breaking things on purpose is a
  listed challenge in most chapters.

---

## If you want to go further with them

The honest next steps, in order of payoff:

1. **Their own data.** Anything they collect themselves for two weeks — screen time, sleep,
   how far a paper aeroplane goes with different folds. The models in this course are more
   than good enough, and it stops being an exercise.
2. **Chapter 25's project list.** Ten of them, each about a weekend.
3. **Maths, in this order**: algebra → vectors and matrices (Chapter 11 is the on-ramp) →
   derivatives. That sequence unlocks the next tier and nothing skips well.
4. **A GPU**, only when a project actually needs one. Not before.
