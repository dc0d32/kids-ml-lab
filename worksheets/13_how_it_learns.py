"""Chapter 13 workbook · How a Neuron Learns."""

from kidsml.workbook import Question, Workbook

WORKBOOK = Workbook(
    chapter=13,
    title='Workbook · One downhill nudge',
    intro='One data point: x1 = 1, x2 = 2, answer y = 1. Start with w1 = 0, w2 = 0, b = 0. Loss is squared error, and the gradient tells each number which way to move. Use lr = 0.5.',
    questions=[
        Question(
            prompt='What is z = w1*x1 + w2*x2 + b at the start?',
            kind='number', answer=0, tolerance=0.001,
            why='All three learned numbers start at zero, so z = 0*1 + 0*2 + 0 = 0. This is the raw line score before the squish touches it.',
        ),
        Question(
            prompt='sigmoid(0) equals what?',
            kind='number', answer=0.5, tolerance=0.001,
            why='Zero sits in the middle of the S-curve. The neuron is perfectly unsure, which is a bad answer here because the target is 1.',
        ),
        Question(
            prompt='For squared error, dL/dout = 2*(out - y). What is it here?',
            kind='number', answer=-1, tolerance=0.001,
            hint='2*(0.5 - 1)',
            why='2*(0.5 - 1) = -1. The negative sign says raising the output would lower the loss.',
        ),
        Question(
            prompt='The sigmoid slope at zero is 0.25. What is dL/dz = dL/dout * slope?',
            kind='number', answer=-0.25, tolerance=0.001,
            why='This is blame squeezing through the squish: output blame times squish slope, so -1 * 0.25 = -0.25.',
        ),
        Question(
            prompt='What is dw1 = dL/dz * x1?',
            kind='number', answer=-0.25, tolerance=0.001,
            why='Weight 1 gets -0.25 * 1 = -0.25. The input value tells how strongly this weight could affect z.',
        ),
        Question(
            prompt='What is dw2 = dL/dz * x2?',
            kind='number', answer=-0.5, tolerance=0.001,
            why='Weight 2 gets -0.25 * 2 = -0.5. Because x2 is twice as large as x1, this weight matters twice as much for this point.',
        ),
        Question(
            prompt='After one step, new w2 = old w2 - lr*dw2. What is new w2?',
            kind='number', answer=0.25, tolerance=0.001,
            why='0 - 0.5*(-0.5) = 0.25. Subtracting a negative gradient moves w2 upward, which pushes the too-low output upward.',
        ),
        Question(
            prompt='If lr = 0, what changes after a step?',
            kind='choice', choices=['the weights change', 'nothing changes'], answer='nothing changes',
            why='Learning rate is the step size. Step size zero means the gradient points downhill, but the model plants its feet and stands still.',
        ),
    ],
    kid_corner='Imagine a friend says your throw was too short. That clue tells your arm to throw a little harder next time.',
)
