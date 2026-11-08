"""Neural network layers with explicit forward and backward passes."""

import numpy as np

class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        """Return (parameter, gradient) pairs; empty for layers without parameters."""
        return ()

class Dense(Layer):
    """Fully connected layer: outputs = inputs @ weights + biases."""

    def __init__(self, in_features, out_features, rng=None):
        rng = np.random.default_rng(rng)
        # He initialization for ReLU networks.
        self.weights = rng.standard_normal((in_features, out_features)) * np.sqrt(
            2.0 / in_features
        )
        self.biases = np.zeros((1, out_features))

    def forward(self, inputs):
        # Keep the input for the weight gradient.
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output):
        # The loss gradient is already averaged; do not average again here.
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_biases = grad_output.sum(axis=0, keepdims=True)
        return grad_output @ self.weights.T

    def parameters(self):
        return (
            (self.weights, self.grad_weights),
            (self.biases, self.grad_biases),
        )

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0.0, inputs)

    def backward(self, grad_output):
        # Use a zero derivative at x = 0.
        return grad_output * (self.inputs > 0)

class Sigmoid(Layer):
    def forward(self, inputs):
        # Use separate forms for positive and negative inputs to avoid overflow.
        self.outputs = np.empty_like(inputs, dtype=float)
        positive = inputs >= 0
        self.outputs[positive] = 1.0 / (1.0 + np.exp(-inputs[positive]))
        exp_inputs = np.exp(inputs[~positive])
        self.outputs[~positive] = exp_inputs / (1.0 + exp_inputs)
        return self.outputs

    def backward(self, grad_output):
        return grad_output * self.outputs * (1.0 - self.outputs)

class Tanh(Layer):
    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs

    def backward(self, grad_output):
        return grad_output * (1.0 - self.outputs**2)
