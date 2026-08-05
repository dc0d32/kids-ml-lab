# Chapter 01 · Lines That Predict — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — A piggy-bank line

Use this line:

**dollars = 3 × weeks + 5**

| # | weeks saved | real dollars | prediction | real - prediction | mistake² |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 8 | | | |
| 2 | 2 | 11 | | | |
| 3 | 3 | 15 | | | |
| 4 | 4 | 17 | | | |

Add the last column: **total squared mistake = ______**

---

## Part 2 — Why square it?

A model misses by +2 on one point and -2 on another.

**a)** If you added the raw mistakes, what total would you get? ______

**b)** If you square each mistake first, what total do you get? ______

**c)** Which total feels more honest? Why?

```
______________________________________________________________
```

---

## Part 3 — Find a better line

Try your own line:

**dollars = ____ × weeks + ____**

Fill the table again for the same four points.

| # | prediction | real - prediction | mistake² |
|---|---:|---:|---:|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

Your total squared mistake: ______

Did you beat 3 × weeks + 5? ______

---

## Part 4 — The big idea

After training, the whole model is two numbers: **w** and **b**.

What does **w** control on the graph?

```
______________________________________________________________
```

What does **b** control?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Put four toy cars in a row, not perfectly straight. Lay a string near them.

1. Move the string until it is close to all the cars.
2. Point to where the next car might go.
3. Move one car far away. What happens to the string?

What did you notice?

```
______________________________________________________________
```

---
---

## Answers (for grown-ups)

**Part 1:** Predictions are 8, 11, 14, 17. Mistakes are 0, 0, 1, 0. Squared mistakes are
0, 0, 1, 0. Total = 1. The exact app data has decimals, but this printed version uses
round numbers so the pencil work stays friendly.

**Part 2:** Raw total is 0. Squared total is 8. Teaching point: raw errors can cancel out,
which hides bad predictions. Squared errors preserve "wrongness" and make large misses
matter more.

**Part 3:** Answers vary. The useful conversation is not whether the child found the
perfect line, but whether they can compare two candidate lines by one score.

**Part 4:** w tilts the line. b slides it up or down. The model is not a mysterious box
here; it is two learned numbers.

**Little Kid Corner:** The far-away car is an outlier. It can pull the best line toward
it, which is why Chapter 01 asks what happens when one point gets dragged away.
