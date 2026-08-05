"""Chapter 14 workbook · Deeper and Wider."""

import pandas as pd

from kidsml.workbook import Question, Workbook

LOSS_TABLE = pd.DataFrame(
    [[0, 0.30, 0.31], [100, 0.20, 0.22], [200, 0.13, 0.16], [400, 0.09, 0.14], [800, 0.05, 0.20], [1200, 0.03, 0.28]],
    columns=['checkpoint', 'train loss', 'test loss'],
)

WORKBOOK = Workbook(
    chapter=14,
    title='Workbook · Bigger is not always kinder',
    intro='A big network can over-study its practice dots. Use the table to decide when to stop.',
    questions=[
        Question(
            prompt='Which checkpoint has the lowest test loss?',
            kind='number', answer=400, tolerance=0.001, table=LOSS_TABLE,
            why='We ship the model that works best on hidden test examples, not the one that memorised the practice set.',
        ),
        Question(
            prompt='At checkpoint 1200, train loss is tiny. Why is that not the best model?',
            kind='open',
            why='The test loss climbed. The network learned details that helped the practice dots but hurt fresh dots.',
        ),
        Question(
            prompt='For [2, 5, 5, 1], how many weight numbers are in the first layer?',
            kind='number', answer=10, tolerance=0.001,
            why='Each of 2 inputs connects to each of 5 hidden neurons: 2*5 = 10.',
        ),
        Question(
            prompt='How many weight numbers are in all three layers? 2*5 + 5*5 + 5*1 = ?',
            kind='number', answer=40, tolerance=0.001,
            why='Weights are the arrows. Counting arrows makes parameter count concrete.',
        ),
        Question(
            prompt='How many bias numbers does [2, 5, 5, 1] have?',
            kind='number', answer=11, tolerance=0.001,
            why='Every non-input neuron gets one bias: 5 + 5 + 1 = 11.',
        ),
        Question(
            prompt='Total parameters: weights plus biases?',
            kind='number', answer=51, tolerance=0.001,
            why='The network must learn 51 numbers. That is why bigger nets can memorise more.',
        ),
    ],
    kid_corner='Studying helps until you memorise the exact practice quiz. Then a new quiz with the same idea can feel harder.',
)
