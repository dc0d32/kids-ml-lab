"""Chapter 13 workbook · One Neuron."""

import pandas as pd

from kidsml.workbook import Question, Workbook

POINTS = pd.DataFrame(
    [[1, 0, 2.5, 0.92], [0, 1, -0.5, 0.38], [1, 2, 0.5, 0.62]],
    columns=['x1', 'x2', 'z = 2*x1 - x2 + 0.5', 'sigmoid(z) approx'],
)

WORKBOOK = Workbook(
    chapter=13,
    title='Workbook · One circle with a squish',
    intro='Use w1 = 2, w2 = -1, and b = 0.5. The weights multiply x1 and x2, b is the extra push, and the squish clips the raw score to 0..1. Build the raw score first. Then send it through the activation squish.',
    questions=[
        Question(
            prompt='For point (1, 0), what is the raw score z?',
            kind='number', answer=2.5, tolerance=0.001, table=POINTS,
            hint='2*1 - 1*0 + 0.5',
            why='The raw score is the Chapter 2 line inside the neuron: 2*1 - 0 + 0.5 = 2.5. Build this number first; the squish comes after.',
        ),
        Question(
            prompt='For point (0, 1), sigmoid(z) is about 0.38. Which class does the neuron pick with a 0.5 cutoff?',
            kind='choice', choices=['blue / 0', 'red / 1'], answer='blue / 0',
            why='A sigmoid output below 0.5 lands on the blue side. The squish turns the line score into confidence, while the boundary stays nailed at z = 0.',
        ),
        Question(
            prompt='For point (1, 2), what is z?',
            kind='number', answer=0.5, tolerance=0.001,
            why='2*1 - 2 + 0.5 = 0.5. Positive puts the point on the red side; the small size says it is standing near the fence.',
        ),
        Question(
            prompt='Now double w1, w2, and b. What is the new z for point (1, 2)?',
            kind='number', answer=1.0, tolerance=0.001,
            why='Every raw score doubles: 2*(0.5) = 1.0. Confidence changes because the point now looks farther from the fence.',
        ),
        Question(
            prompt='After doubling all three numbers, what happens to the decision boundary?',
            kind='choice', choices=['it moves', 'it stays put'], answer='it stays put',
            why='The boundary is where z = 0. Multiplying the whole score by 2 keeps the zero places fixed, so confidence changes while the line stays put.',
        ),
        Question(
            prompt='What weights make a neuron maximally unsure everywhere?',
            kind='text', answer=['w1=0 w2=0 b=0', '0 0 0', 'all zeros', 'w=0 b=0'],
            why='If all weights and the bias are zero, every point has z = 0 and sigmoid(0) = 0.5. The neuron stands on the fence everywhere.',
        ),
    ],
    kid_corner='Put tape down as a line on the floor. Standing far from the tape feels confident. Standing on the tape feels like a shrug.',
)
