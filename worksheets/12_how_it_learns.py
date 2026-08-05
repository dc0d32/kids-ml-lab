"""Chapter 12 workbook · How a Neuron Learns."""

from kidsml.workbook import Question, Workbook

WORKBOOK = Workbook(
    chapter=12,
    title='Workbook · One downhill nudge',
    intro='One data point: x1 = 1, x2 = 2, answer y = 1. Start with w1 = 0, w2 = 0, b = 0. Use lr = 0.5.',
    questions=[
        Question(
            prompt='What is z = w1*x1 + w2*x2 + b at the start?',
            kind='number', answer=0, tolerance=0.001,
            why='All three learned numbers start at zero, so the raw score is zero.',
        ),
        Question(
            prompt='sigmoid(0) equals what?',
            kind='number', answer=0.5, tolerance=0.001,
            why='Zero is the middle of the S-curve. The neuron is perfectly unsure.',
        ),
        Question(
            prompt='For squared error, dL/dout = 2*(out - y). What is it here?',
            kind='number', answer=-1, tolerance=0.001,
            hint='2*(0.5 - 1)',
            why='The negative sign says the output is too small. We need to push it upward.',
        ),
        Question(
            prompt='The sigmoid slope at zero is 0.25. What is dL/dz = dL/dout * slope?',
            kind='number', answer=-0.25, tolerance=0.001,
            why='This is blame passing through the squish: output blame times squish slope.',
        ),
        Question(
            prompt='What is dw1 = dL/dz * x1?',
            kind='number', answer=-0.25, tolerance=0.001,
            why='Weight 1 gets the same blame because x1 is 1.',
        ),
        Question(
            prompt='What is dw2 = dL/dz * x2?',
            kind='number', answer=-0.5, tolerance=0.001,
            why='Weight 2 gets twice as much blame because x2 is 2.',
        ),
        Question(
            prompt='After one step, new w2 = old w2 - lr*dw2. What is new w2?',
            kind='number', answer=0.25, tolerance=0.001,
            why='0 - 0.5*(-0.5) = 0.25. Subtracting a negative number moves the weight up.',
        ),
        Question(
            prompt='If lr = 0, what changes after a step?',
            kind='choice', choices=['the weights change', 'nothing changes'], answer='nothing changes',
            why='Learning rate is the step size. Step size zero means standing still while holding a map.',
        ),
    ],
    kid_corner='Imagine a friend says your throw was too short. That blame tells your arm to throw a little harder next time.',
)
