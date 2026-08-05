# Chapter 07 · The Widest Road — Worksheet

**Print this. You need a pencil and a ruler.**

---

## Part 1 — Six points

Blue points: `(1,1)`, `(1,3)`, `(2,2)`

Red points: `(5,1)`, `(5,3)`, `(4,2)`

Draw them on graph paper.

---

## Part 2 — Two roads

Try two centre lines:

- Road A: `x = 2.5`
- Road B: `x = 3.0`

For each road, measure the distance to the nearest blue point and nearest red point.

| road | nearest blue gap | nearest red gap | smallest safety gap |
|---|---:|---:|---:|
| x = 2.5 | | | |
| x = 3.0 | | | |

Which road would you trust for a new point? Why?

```
______________________________________________________________
```

---

## Part 3 — Support points

Circle the points closest to your winning road.

If you delete a far-away point, should the road move much?

```
______________________________________________________________
```

If you delete a closest point, should the road move much?

```
______________________________________________________________
```

---

## Part 4 — C and gamma in kid words

Match the knob to the meaning.

| knob | meaning |
|---|---|
| C | |
| gamma | |

Meanings:

A. How far each point's influence reaches.

B. How much the model cares about getting every training dot right instead of keeping the
road wide.

---

## 🧸 Little Kid Corner

Put blue and red stickers on paper. Draw a road between them.

Now make the road wider until it touches a sticker. Which stickers stopped the road?

```
______________________________________________________________
```

---
---

## Answers (for grown-ups)

**Part 2:** For `x = 2.5`, the nearest blue gap is 0.5 and the nearest red gap is 1.5, so
the smallest safety gap is 0.5. For `x = 3.0`, both nearest gaps are 1.0, so the smallest
safety gap is 1.0. The widest-road choice is `x = 3.0`.

**Part 3:** The closest points are `(2,2)` and `(4,2)`. These are the support-vector idea:
points on the edge of the margin determine the road. Far-away points often do not matter.

**Part 4:** C = B. gamma = A. Low C allows a wider road with a few mistakes. High gamma can
make tiny islands around points.
