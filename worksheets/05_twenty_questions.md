# Chapter 05 · Twenty Questions — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — Creature questions

A tree is trying to predict **can fly?**

| creature | wings? | bigger than cat? | feathers? | water? | can fly? |
|---|---|---|---|---|---|
| sparrow | yes | no | yes | no | yes |
| eagle | yes | yes | yes | no | yes |
| penguin | yes | yes | yes | yes | no |
| ostrich | yes | yes | yes | no | no |
| bat | yes | no | no | no | yes |
| bumblebee | yes | no | no | no | yes |
| cat | no | no | no | no | no |
| elephant | no | yes | no | no | no |
| dolphin | no | yes | no | yes | no |
| goldfish | no | no | no | yes | no |

Try the question **has wings?**

Yes bucket: ____ fly, ____ do not fly.

No bucket: ____ fly, ____ do not fly.

---

## Part 2 — Bucket mix

For a bucket with two answers:

`mix = 1 - p_yes² - p_no²`

1. A bucket has 4 yes and 0 no. Mix = __________________
2. A bucket has 3 yes and 3 no. Mix = __________________
3. A bucket has 2 yes and 2 no. Mix = __________________

Now compute the yes bucket for **has wings?**:

`1 - (____/____)² - (____/____)² = ________`

---

## Part 3 — Pick the first question

Fill what you can. Circle the least mixed first question.

| first question | yes bucket | no bucket | good first question? |
|---|---|---|---|
| has wings? | 4 fly, 2 no | 0 fly, 4 no | |
| bigger than cat? | ____ | ____ | |
| has feathers? | ____ | ____ | |
| lives in water? | ____ | ____ | |

Why did you circle that one?

```
______________________________________________________________
```

---

## Part 4 — Over-studying

A depth-1 tree asks one question.
A depth-20 tree can ask question after question after question.

Which one is more likely to memorise last year's test answers?

```
______________________________________________________________
```

What picture clue would show memorising?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Play animal Guess Who.

1. Pick 8 animals.
2. Choose a secret answer, like **can fly**.
3. Ask yes/no questions to split the animals.
4. Which first question helped most?

Best first question: ___________________________

---
---

## Answers (for grown-ups)

**Part 1:** The yes bucket for wings has 4 flyers and 2 non-flyers. The no bucket has 0
flyers and 4 non-flyers.

**Part 2:** 4/0 has mix 0. A 3/3 bucket has mix 0.5. A 2/2 bucket also has mix 0.5.
For wings, `1 - (4/6)^2 - (2/6)^2 = 4/9 ≈ 0.44` in the yes bucket, and 0 in the no
bucket. The weighted score is about 0.27.

**Part 3:** `has_wings` is the best first question in this dataset. The teaching point is
that a tree is not magic; it tries candidate questions and chooses the split that makes
cleaner buckets.

**Part 4:** The depth-20 tree is more likely to memorise. The picture clue is a boundary
with many tiny boxes around individual training points, plus train accuracy much higher
than test accuracy.
