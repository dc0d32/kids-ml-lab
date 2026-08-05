# Chapter 06 · A Crowd of Trees — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — Tally the forest vote

Five tiny trees vote on four mystery points.

| point | tree 1 | tree 2 | tree 3 | tree 4 | tree 5 | crowd vote |
|---|---|---|---|---|---|---|
| A | red | red | blue | red | red | |
| B | blue | blue | blue | blue | red | |
| C | red | blue | red | red | red | |
| D | blue | blue | red | blue | blue | |

---

## Part 2 — Why make trees disagree?

A random forest gives each tree a slightly different job: different rows, and different
columns.

Why might that help the crowd?

```
______________________________________________________________
```

---

## Part 3 — Boosting leftovers

Four data points have real answers:

| point | real answer | first guess | leftover = real - guess |
|---|---:|---:|---:|
| A | 2 | 5 | |
| B | 4 | 5 | |
| C | 8 | 5 | |
| D | 10 | 5 | |

Now the next tiny tree fixes **half** of each leftover.

| point | old guess | half leftover | new guess | new leftover |
|---|---:|---:|---:|---:|
| A | 5 | | | |
| B | 5 | | | |
| C | 5 | | | |
| D | 5 | | | |

---

## Part 4 — Forest or boosting?

Write **forest** or **boosting**.

1. Trees can be trained independently: __________________
2. Trees must be trained in order: __________________
3. Usually hard to mess up: __________________
4. Often a bit stronger on table data: __________________
5. Easier to overfit: __________________

---

## 🧸 Little Kid Corner

Guess jellybeans in a jar with a group.

Your guess: ________

Five other guesses: ________ ________ ________ ________ ________

Average guess: ________

Which was closer, one guess or the average? __________________

---
---

## Answers (for grown-ups)

**Part 1:** A = red, B = blue, C = red, D = blue. The teaching point is majority vote:
an ensemble can turn several noisy answers into one steadier answer.

**Part 2:** Different trees make different mistakes. Voting helps when the mistakes do not
all point the same way. Random rows and columns create that useful disagreement.

**Part 3:** Leftovers are -3, -1, 3, 5. Half leftovers are -1.5, -0.5, 1.5, 2.5. New
guesses are 3.5, 4.5, 6.5, 7.5. New leftovers are -1.5, -0.5, 1.5, 2.5.

**Part 4:** forest, boosting, forest, boosting, boosting. Forests vote independently.
Boosting is a sequence of fixes, so it can chase noise if the fixes are too strong or too
many.
