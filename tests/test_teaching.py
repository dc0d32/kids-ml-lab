"""Checks that the *teaching* works, not only that the code runs.

The page tests prove a chapter doesn't crash. They cannot tell you that a chapter is
teaching something impossible — which is exactly what happened here: chapter 00 dealt six
random creatures, all of which happened to be "not a zeep", leaving a reader with no
positive example and a rule that could not be worked out. It ran perfectly. It was also
the first five minutes of the course.

Anything a reader is asked to deduce gets a test here.
"""

from __future__ import annotations

import pytest

from kidsml.zeeps import (
    RULES,
    all_zeeps,
    label_with,
    quiz_examples,
    teaching_examples,
)

SEEDS = range(8)


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_examples_show_both_answers(rule: str, seed: int):
    """A pile of examples that are all 'no' teaches nothing at all."""
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    shown = teaching_examples(rule, n=6, seed=seed)

    assert labels[shown].sum() >= 2, (
        f"rule {rule!r}, seed {seed}: fewer than two zeeps among the examples — "
        "there is nothing positive to generalise from"
    )
    assert (~labels[shown]).sum() >= 2, (
        f"rule {rule!r}, seed {seed}: fewer than two non-zeeps among the examples"
    )


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_examples_are_distinct(rule: str, seed: int):
    shown = teaching_examples(rule, n=6, seed=seed)
    assert len(set(shown.tolist())) == len(shown), "the same creature shown twice"


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("seed", SEEDS)
def test_the_quiz_asks_about_unseen_creatures(rule: str, seed: int):
    """Quizzing on a creature they were already given the answer for proves nothing."""
    shown = teaching_examples(rule, n=6, seed=seed)
    quiz = quiz_examples(rule, shown, n=3, seed=seed)

    assert not set(quiz.tolist()) & set(shown.tolist()), "the quiz reuses a shown example"
    assert len(set(quiz.tolist())) == len(quiz), "the same creature quizzed twice"


@pytest.mark.parametrize("rule", list(RULES))
def test_a_near_miss_pair_is_present_when_one_exists(rule: str):
    """The pair that differs in one way and still disagrees is what kills wrong guesses."""
    creatures = all_zeeps()
    labels = label_with(creatures, rule)
    rows = [tuple(r) for r in creatures.to_numpy()]

    exists = any(
        sum(1 for a, b in zip(rows[i], rows[j]) if a != b) == 1
        for i in range(len(rows))
        for j in range(len(rows))
        if labels[i] and not labels[j]
    )
    if not exists:
        pytest.skip(f"rule {rule!r} has no near-miss pair to include")

    shown = teaching_examples(rule, n=6, seed=0)
    found = any(
        sum(1 for a, b in zip(rows[i], rows[j]) if a != b) == 1
        for i in shown
        for j in shown
        if labels[i] and not labels[j]
    )
    assert found, f"rule {rule!r}: the examples never pin the rule down"
