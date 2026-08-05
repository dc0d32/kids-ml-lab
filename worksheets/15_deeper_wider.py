"""Chapter 15 workbook · Deeper and Wider."""

import pandas as pd

from kidsml.workbook import Question, Workbook

LOSS_TABLE = pd.DataFrame(
    [[0, 0.30, 0.31], [100, 0.20, 0.22], [200, 0.13, 0.16], [400, 0.09, 0.14], [800, 0.05, 0.20], [1200, 0.03, 0.28]],
    columns=['checkpoint', 'train loss', 'test loss'],
)

WORKBOOK = Workbook(
    chapter=15,
    title='Workbook · Bigger is not always kinder',
    intro='A big network has more capacity: more room to fit shapes. It can overfit by over-studying practice dots, so use the train/test table to decide when early stopping should save the calmer model.',
    questions=[
        Question(
            prompt='Which checkpoint has the lowest test loss?',
            kind='number', answer=400, tolerance=0.001, table=LOSS_TABLE,
            why='Checkpoint 400 has test loss 0.14, the lowest value in the table. We ship the model that works best on hidden test examples, not the one that memorised the practice dots.',
        ),
        Question(
            prompt='At checkpoint 1200, train loss is tiny. Why is that not the best model?',
            kind='open',
            why='The test loss climbed to 0.28. The network learned tiny details that helped the practice dots but hurt fresh dots, which is overfitting: practice memorising instead of reusable learning.',
        ),
        Question(
            prompt='For [2, 5, 5, 1], how many weight numbers are in the first layer?',
            kind='number', answer=10, tolerance=0.001,
            why='Each of 2 inputs connects to each of 5 hidden neurons: 2*5 = 10. Counting arrows makes the architecture concrete.',
        ),
        Question(
            prompt='How many weight numbers are in all three layers? 2*5 + 5*5 + 5*1 = ?',
            kind='number', answer=40, tolerance=0.001,
            why='Weights are the arrows: 10 in the first layer, 25 in the second, and 5 in the output layer. 10 + 25 + 5 = 40.',
        ),
        Question(
            prompt='How many bias numbers does [2, 5, 5, 1] have?',
            kind='number', answer=11, tolerance=0.001,
            why='Every non-input neuron gets one bias: 5 + 5 + 1 = 11. Biases let each neuron slide its line or ramp.',
        ),
        Question(
            prompt='Total parameters: weights plus biases?',
            kind='number', answer=51, tolerance=0.001,
            why='The network must learn 40 + 11 = 51 numbers. More numbers can fit more real shape, but they can also memorise more noise.',
        ),
    ],
    kid_corner='Studying helps until you memorise the exact practice quiz. Then a new quiz with the same idea can feel harder.',
)
