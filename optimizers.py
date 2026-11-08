"""Parameter updates from stored gradients."""

class SGD:
    """Stochastic gradient descent: parameter -= learning_rate * gradient."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            for parameter, gradient in layer.parameters():
                parameter -= self.learning_rate * gradient
