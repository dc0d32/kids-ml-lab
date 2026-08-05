"""Chapter 14 workbook · Two Layers, Three Neurons."""

import pandas as pd

from kidsml.workbook import Question, Workbook

XOR_TABLE = pd.DataFrame(
    [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0]],
    columns=['x1', 'x2', 'OR-ish h1', 'AND-ish h2', 'XOR answer'],
)

WORKBOOK = Workbook(
    chapter=14,
    title='Workbook · Solve XOR by inventing two features',
    intro='Hidden neuron 1 says OR-ish and writes h1. Hidden neuron 2 says AND-ish and writes h2. Those h numbers are new coordinates for the same XOR dots; fill the new two-column world.',
    questions=[
        Question(
            prompt='For point (0, 0), what does OR-ish h1 output?',
            kind='number', answer=0, tolerance=0.001, table=XOR_TABLE,
            why='Neither input is on, so OR-ish stays off. This parks (0, 0) at h-space point (0, 0).',
        ),
        Question(
            prompt='For point (1, 1), what does AND-ish h2 output?',
            kind='number', answer=1, tolerance=0.001,
            why='Both inputs are on, so AND-ish turns on. This pulls (1, 1) away from the two red XOR points.',
        ),
        Question(
            prompt='In the new (h1, h2) space, which point is the red XOR class?',
            kind='choice', choices=['(0, 0)', '(1, 0)', '(1, 1)'], answer='(1, 0)',
            why='The red rows are exactly the ones where OR is on but AND is off. The hidden layer moved both red corners onto the same h-space spot, side by side.',
        ),
        Question(
            prompt='Try output score = 1*h1 - 2*h2 - 0.5. What score does (h1, h2) = (1, 0) get?',
            kind='number', answer=0.5, tolerance=0.001,
            why='1*1 - 2*0 - 0.5 = 0.5, so the output neuron says red. In h-space, one straight line can now catch the red point.',
        ),
        Question(
            prompt='With the same output score, what score does (h1, h2) = (1, 1) get?',
            kind='number', answer=-1.5, tolerance=0.001,
            why='1*1 - 2*1 - 0.5 = -1.5. Both original inputs on should be blue for XOR, and the negative score gives blue.',
        ),
        Question(
            prompt='What did the hidden layer really invent?',
            kind='text', answer=['features', 'new features', 'better features'],
            why='Each hidden neuron draws a line and reports a new number. Those reports are new features, which is the Chapter 3 escape route built automatically.',
        ),
    ],
    kid_corner='Two friends stand by two tape lines. Each friend shouts yes or no. A third friend listens to those shouts and makes the final call.',
)
