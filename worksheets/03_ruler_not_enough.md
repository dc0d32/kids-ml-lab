# Chapter 03 · When a Ruler Isn't Enough — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — XOR truth table

Opposite corners match.

| x1 | x2 | answer |
|---:|---:|---|
| 0 | 0 | blue |
| 0 | 1 | red |
| 1 | 0 | red |
| 1 | 1 | blue |

Draw these four points. Try to draw one line with red on one side and blue on the other.

Could you do it? ______

---

## Part 2 — The contradiction

A line score is:

**w1 × x1 + w2 × x2 + b**

Blue means negative. Red means positive.

Fill the missing left sides.

| point | answer | line needs |
|---|---|---|
| (0, 0) | blue | b < 0 |
| (1, 1) | blue | __________ < 0 |
| (1, 0) | red | __________ > 0 |
| (0, 1) | red | __________ > 0 |

Now add the two red rows:

```
______________________________________________________________
```

Add the two blue rows:

```
______________________________________________________________
```

Why is that impossible?

```
______________________________________________________________
```

---

## Part 3 — Lift the data

For a circle problem, invent a new feature:

**x3 = x1² + x2²**

Compute x3.

| x1 | x2 | x3 |
|---:|---:|---:|
| 0 | 0 | |
| 1 | 0 | |
| 0 | 1 | |
| 2 | 0 | |
| 0 | 2 | |

What does x3 measure in plain words?

```
______________________________________________________________
```

---

## Part 4 — Invent a feature

The stripes shape changes class again and again as x1 moves left to right.

What kind of feature might help? Circle one.

- a feature that grows forever
- a feature that repeats
- a feature that ignores x1

Why?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Put a donut-shaped ring and a button on a table. Try to separate the ring from the
button with one straight piece of string.

Now lift the ring pieces higher than the button. Could a flat book separate high from
low?

What changed?

```
______________________________________________________________
```

---
---

## Answers (for grown-ups)

**Part 1:** No. Many kids will draw a near miss; that is useful. The point is to feel the
failure before seeing the proof.

**Part 2:** The missing expressions are w1 + w2 + b, w1 + b, and w2 + b. Adding red rows
gives w1 + w2 + 2b > 0. Adding blue rows gives w1 + w2 + 2b < 0. The same number cannot
be both positive and negative. This is the clean 8th-grade proof that XOR is not linearly
separable.

**Part 3:** x3 values are 0, 1, 1, 4, 4. It measures distance from the middle, squared.
Teaching point: the model can stay straight if the features are cleverer.

**Part 4:** A feature that repeats. Stripes are periodic, so a periodic feature such as
sine/cosine of x1 can line them up in a space where a straight model has a chance.

**Little Kid Corner:** Lifting adds a new direction. That is the physical version of
adding a feature.
