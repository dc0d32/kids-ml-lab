# Chapter 04 · Maybe, Probably, Definitely — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — The S-curve table

Use or look up:

**sigmoid(z) = 1 / (1 + e^-z)**

| z | probability |
|---:|---:|
| -4 | |
| -2 | |
| -1 | |
| 0 | |
| 1 | |
| 2 | |
| 4 | |

Plot the seven points on graph paper. What letter shape does the curve look like?

```
______________________________________________________________
```

---

## Part 2 — The exact middle

At z = 0:

**sigmoid(0) = 1 / (1 + e⁰)**

And **e⁰ = 1**.

So sigmoid(0) = ______

What should a model feel right on the decision line: certain or unsure?

```
______________________________________________________________
```

---

## Part 3 — Confidence is a promise

Fill in whether the penalty should be small, medium, or huge.

| model says red | truth | penalty |
|---:|---|---|
| 90% | red | |
| 60% | red | |
| 10% | red | |
| 99% | blue | |
| 50% | blue | |

Why should 99% and wrong hurt so much?

```
______________________________________________________________
```

---

## Part 4 — Still a line

Logistic regression adds probability around the line.

Does it make the boundary bend? ______

What is new compared with the perceptron?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Put tape down the middle of a room.

- Far on the left: say "I am sure it is blue."
- Far on the right: say "I am sure it is red."
- On the tape: say "I do not know."

Where do you feel 50/50?

```
______________________________________________________________
```

---
---

## Answers (for grown-ups)

**Part 1:** Approximate values: 0.018, 0.119, 0.269, 0.5, 0.731, 0.881, 0.982. The curve
looks like an S. The table matters because students draw the squish before naming it.

**Part 2:** 1/2. A point on the line should be maximally unsure: 50% red, 50% blue.

**Part 3:** 90% red and true = small. 60% red and true = medium-small. 10% red and true =
huge. 99% red when truth is blue = huge. 50% red when truth is blue = medium. Teaching
point: log loss punishes confident wrong answers because confidence is a promise.

**Part 4:** No, the boundary is still straight. The new idea is confidence: a fade from
blue to red instead of a hard jump.

**Little Kid Corner:** The tape line is the 50/50 place. Farther away from it, confidence
grows.
