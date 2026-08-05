# Chapter 08 · The Model Zoo — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — Pick the animal

Match each model personality.

| model | personality |
|---|---|
| logistic regression | |
| decision tree | |
| random forest | |
| RBF SVM | |
| kNN | |

Personalities:

A. asks nearby points

B. one straight line

C. smooth islands

D. boxes and stairs

E. many boxy votes

---

## Part 2 — Five folds

Ten rows are split into five folds of two rows.

| round | test rows | score |
|---|---|---:|
| 1 | 1, 2 | 0.80 |
| 2 | 3, 4 | 0.70 |
| 3 | 5, 6 | 0.90 |
| 4 | 7, 8 | 0.80 |
| 5 | 9, 10 | 0.60 |

Average score:

`(____ + ____ + ____ + ____ + ____) / 5 = ________`

Highest score: ________

Lowest score: ________

Spread from highest to lowest: ________

---

## Part 3 — The fake 100%

A deep tree gets:

- 100% on training data
- 78% on test data

Which number should you tell people? Why?

```
______________________________________________________________
```

---

## Part 4 — Baseline trap

A dataset has 90 cats and 10 dogs. A lazy model always says **cat**.

Accuracy = ____ / 100 = ________%

Is the model useful?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Race three toys down the same ramp three times.

| toy | run 1 | run 2 | run 3 | average |
|---|---:|---:|---:|---:|
| toy A | | | | |
| toy B | | | | |
| toy C | | | | |

Did the winner change on any single run? ________

---
---

## Answers (for grown-ups)

**Part 1:** logistic regression = B, decision tree = D, random forest = E, RBF SVM = C,
kNN = A. The teaching point is that models have different biases, so there is no permanent
champion.

**Part 2:** Average is `(0.80 + 0.70 + 0.90 + 0.80 + 0.60) / 5 = 0.76`. Highest is 0.90,
lowest is 0.60, range is 0.30. A score without a spread hides how jumpy the result was.

**Part 3:** Tell people the test score, or better, cross-validation mean and spread. The
training score answers "did it memorise these rows?" The test score asks a more honest
question: "does it work on rows it did not study?"

**Part 4:** The lazy model scores 90%. It is not useful for finding dogs. This foreshadows
why accuracy can mislead on lopsided data.
