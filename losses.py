"""Loss functions and their gradients with respect to model outputs."""

import numpy as np

class MeanSquaredError:
    """Mean squared error for matching (N, n_outputs) arrays."""

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have matching shapes: (n_samples, n_outputs).")
        self.difference = predictions - targets
        return np.mean(self.difference**2)

    def backward(self):
        # The mean is taken over all output elements.
        return 2.0 * self.difference / self.difference.size

class BinaryCrossEntropy:
    """Binary cross-entropy for probabilities and matching 0/1 targets."""

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have matching shapes: (n_samples, n_outputs).")
        # Clip probabilities to avoid log(0) and division by zero.
        self.clip_mask = (predictions >= 1e-7) & (predictions <= 1.0 - 1e-7)
        self.predictions = np.clip(predictions, 1e-7, 1.0 - 1e-7)
        self.targets = targets
        return -np.mean(
            targets * np.log(self.predictions)
            + (1.0 - targets) * np.log1p(-self.predictions)
        )

    def backward(self):
        probabilities = self.predictions
        # The clipped loss is constant outside the clipping interval.
        return self.clip_mask * (probabilities - self.targets) / (
            probabilities * (1.0 - probabilities) * probabilities.size
        )

class CrossEntropy:
    """Softmax cross-entropy for logits (N, C) and integer labels (N,).

    Pass raw logits; do not add a separate Softmax layer.
    """

    def forward(self, logits, targets):
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("Class labels must have shape (n_samples,).")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Class labels must be integers.")
        if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
            raise ValueError("Class labels must be between 0 and n_classes - 1.")

        # Subtract the row maximum to avoid overflow without changing Softmax.
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted_logits - np.log(
            np.exp(shifted_logits).sum(axis=1, keepdims=True)
        )
        self.probabilities = np.exp(log_probabilities)
        self.targets = targets
        return -np.mean(log_probabilities[np.arange(len(targets)), targets])

    def backward(self):
        grad_logits = self.probabilities.copy()
        # Softmax cross-entropy gradient: probabilities - one_hot(targets).
        grad_logits[np.arange(len(self.targets)), self.targets] -= 1.0
        return grad_logits / len(self.targets)
