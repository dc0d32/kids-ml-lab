# Chapter 00 · The Guessing Machine — Worksheet

**Print this. You need a pencil.**

---

## Part 1 — Find the rule

I have a secret rule that decides whether a creature is a **zeep**.
Here are six creatures I have already sorted:

| # | shape | colour | size | zeep? |
|---|----------|--------|-------|-----------|
| 1 | square | red | big | ✅ zeep |
| 2 | circle | red | big | ❌ no |
| 3 | square | blue | big | ✅ zeep |
| 4 | square | red | small | ❌ no |
| 5 | triangle | green | big | ❌ no |
| 6 | square | green | small | ❌ no |

**Write the rule in your own words:**

```
______________________________________________________________
```

---

## Part 2 — Use your rule

Now use your rule on three creatures you have never seen:

| # | shape | colour | size | your answer |
|---|----------|--------|-------|-------------|
| 7 | square | green | big | |
| 8 | circle | blue | small | |
| 9 | triangle | red | big | |

*(Answers on the last page. No peeking — that's cheating, and cheating is exactly the
thing Chapter 10 is about.)*

---

## Part 3 — Think about it

**a)** Which of the six examples was the most useful to you? Circle its number.
Why that one?

```
______________________________________________________________
```

**b)** Suppose I had only shown you examples **1, 3 and 5**. Could you still have found
the rule? What would you have guessed instead?

```
______________________________________________________________
```

**c)** Suppose all six examples had been ✅ zeep. What rule would you guess then?

```
______________________________________________________________
```

**d)** Here is a harder secret rule: *a creature is a zeep if **exactly one** of these is
true — it is red, or it is big.* Fill in the table:

| shape | colour | size | zeep? |
|----------|--------|-------|-------|
| circle | red | big | |
| circle | red | small | |
| square | blue | big | |
| square | blue | small | |

How many examples do you think someone would need before they could work *that* one out?
More or fewer than the first rule? Why?

```
______________________________________________________________
```

---

## Part 4 — The big question

The computer running this chapter used the **exact same program** whether it saw 2
examples or 17. Not one line of code changed.

So what actually made it better?

```
______________________________________________________________
```

---

## 🧸 Little Kid Corner

Play this with someone in your family.

1. Think of a secret rule about things in your room. Something easy, like
   *anything blue*, or *anything you can eat*, or *anything smaller than your hand*.
2. Point at five things one at a time and say **yes** or **no** for each.
   Don't explain anything.
3. Now point at a sixth thing and ask them to guess.
4. Keep going until they get three in a row right.

**How many examples did they need?** ______

Now swap. **How many did you need?** ______

---
---

## Answers (for grown-ups)

**Part 1:** The rule is *it is big **AND** it is a square*.
Both parts have to be true. Row 3 is the one that rules out "it is red", and row 4 is
the one that rules out "it is a square".

**Part 2:** 7 = ✅ zeep. 8 = ❌ no. 9 = ❌ no.

**Part 3a:** Any answer with a reason is a good answer. Row 3 and row 4 are the two that
carry the most information — each one eliminates a whole family of wrong rules. This is
the seed of a real idea in ML: not all data points are equally valuable.

**Part 3b:** No. Examples 1, 3 and 5 are consistent with *lots* of rules — "it is big",
"it is a square", "it is not green", "it is not a triangle". With too few examples,
many different rules survive, and the learner has to pick one at random. That is exactly
what the left-hand end of the graph in the chapter shows.

**Part 3c:** "Everything is a zeep." That's the only thing the evidence supports. This is
a real failure mode: a model trained on data where every answer is the same will happily
learn to always give that answer, and will look 100% accurate on its own examples.

**Part 3d:** big/red = ❌ no (both true), small/red = ✅ zeep, big/blue = ✅ zeep,
small/blue = ❌ no (neither true).

This rule is genuinely harder because **no single column tells you anything on its own**.
Knowing the colour alone is useless. Knowing the size alone is useless. You have to look
at two columns *together*. In the chapter's graph this is the curve that stays low the
longest — and it is the same reason Chapter 03 needs to exist.

**Part 4:** More examples. That's it. This is the single most important idea in the
course and it is worth spending real time on.
