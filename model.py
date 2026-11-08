"""A sequential network trained with mini-batch gradient descent."""

import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def predict(self, X):
        """Return final layer outputs without converting them to class labels."""
        outputs = np.asarray(X, dtype=float)
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def fit(
        self, X, y, *, loss, optimizer, epochs=100, batch_size=None,
        shuffle=True, verbose=False, rng=None,
    ):
        """Train the network and return the mean batch loss for each epoch.

        X has shape (n_samples, n_features); y depends on the chosen loss.
        batch_size=None uses the full training set for each update.
        rng accepts a NumPy Generator or an integer seed.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a nonempty array with shape (n_samples, n_features).")
        if y.ndim == 0 or len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        batch_size = len(X) if batch_size is None else batch_size
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")

        rng = np.random.default_rng(rng)
        history = []
        for epoch in range(epochs):
            indices = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            epoch_loss = 0.0

            for start in range(0, len(X), batch_size):
                batch = indices[start:start + batch_size]
                predictions = self.predict(X[batch])
                batch_loss = loss.forward(predictions, y[batch])

                # Apply the chain rule from the output layer back to the input.
                grad_output = loss.backward()
                for layer in reversed(self.layers):
                    grad_output = layer.backward(grad_output)
                # Compute all gradients before updating any weights.
                optimizer.step(self.layers)
                # Weight by sample count so a smaller final batch contributes correctly.
                epoch_loss += batch_loss * len(batch)

            epoch_loss /= len(X)
            history.append(float(epoch_loss))
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        return history
