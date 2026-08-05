# Chapter 02 · Lines That Decide — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — Which side of the line?

Use this rule:

**score = x1 + x2 - 8**

Positive score means **red**. Negative score means **blue**.

| # | x1 | x2 | score | guess |
|---|---:|---:|---:|---|
| 1 | 1 | 1 | | |
| 2 | 2 | 1 | | |
| 3 | 1 | 2 | | |
| 4 | 6 | 5 | | |
| 5 | 7 | 6 | | |

---

## Part 2 — One perceptron update

Start with a bad line:

**w = (1, 1), b = -20**

Point **(6, 5)** is really red.

**a)** What score does the bad line give it? ______

**b)** Does the line guess red or blue? ______

**c)** Because it was wrong, add the point to the weights and add 1 to the bias.

New w1 = ______

New w2 = ______

New b = ______

---

## Part 3 — Bias trouble

Draw any line that passes through (0, 0). Now try to separate these points:

Blue: (1, 4), (2, 5)

Red: (1, 1), (2, 2)

Can your line do it while still passing through (0, 0)? ______

What does this tell you about **b**?

```
______________________________________________________________
```

---

## Part 4 — The big idea

A perceptron only stops changing when it can get every training point right.

What might happen if the red and blue piles overlap?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Make two piles of toys: red team and blue team. Put a pencil between them.

1. Move the pencil so all red toys are on one side.
2. Move one toy into the other pile.
3. Can the pencil still make everyone happy?

What happened?

```
______________________________________________________________
```

---
---

## Answers (for grown-ups)

**Part 1:** Scores are -6, -5, -5, 3, 5. Guesses are blue, blue, blue, red, red. Teaching
point: the model first makes a raw score, then turns the sign into a class.

**Part 2:** Score = 6 + 5 - 20 = -9, so it guesses blue. The truth is red. New weights are
(7, 6), and new bias is -19. Teaching point: the update is small enough to do by hand,
which makes "training" feel less magical.

**Part 3:** No, not with a single straight line through the origin. b lets the line slide;
without it, every line is pinned to (0, 0).

**Part 4:** It may keep changing forever because perfection is impossible. This is an
honest limitation of the perceptron, not a coding bug.

**Little Kid Corner:** The moved toy creates overlap. A straight divider only works when
the piles can be cleanly separated.
