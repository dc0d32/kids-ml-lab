"""Chapter 15 workbook · Same Thing, in PyTorch."""

from kidsml.workbook import Question, Workbook

WORKBOOK = Workbook(
    chapter=15,
    title='Workbook · Translate NumPy to PyTorch',
    intro='PyTorch uses different names, but the pieces are the ones you already built.',
    questions=[
        Question(
            prompt='Our raw layer a @ W + b matches which PyTorch layer?',
            kind='choice', choices=['nn.Linear', 'nn.Tanh', 'nn.MSELoss'], answer='nn.Linear',
            why='A Linear layer stores weights and biases, then multiplies and adds.',
        ),
        Question(
            prompt='Our tanh squish matches which PyTorch piece?',
            kind='choice', choices=['nn.Linear', 'nn.Tanh', 'optimizer.step'], answer='nn.Tanh',
            why='The activation function is still the squish between layers.',
        ),
        Question(
            prompt='What command asks PyTorch to pass blame backward through the graph?',
            kind='text', answer=['backward', '.backward()', 'loss.backward()', 'backward()'],
            why='backward() fills in the gradients for every parameter that helped make the loss.',
        ),
        Question(
            prompt='Why do we call optimizer.zero_grad() before each new step?',
            kind='open',
            why='PyTorch adds gradients into the same buckets unless we clear them. Forgetting to clear them mixes old blame with new blame.',
        ),
        Question(
            prompt='If our gradient and PyTorch param.grad match, what does that prove?',
            kind='choice', choices=['PyTorch is using the same blame-passing idea', 'PyTorch guessed randomly'], answer='PyTorch is using the same blame-passing idea',
            why='Matching gradients mean the update direction is the same. The framework is faster, not a mystery machine.',
        ),
    ],
    kid_corner='A tensor is like a trail of footprints. PyTorch can walk backward along the footprints to see which step caused the mess.',
)
