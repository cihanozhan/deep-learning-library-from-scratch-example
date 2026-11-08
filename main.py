"""Classify two concentric circles with a small dense network."""

import numpy as np

from datasets import make_circles
from layers import Dense, ReLU, Sigmoid
from losses import BinaryCrossEntropy
from model import Sequential
from optimizers import SGD

def main():
    X, y = make_circles(n_samples=1000, noise=0.05, rng=42)
    # Binary targets have the same (N, 1) shape as the predictions.
    y = y.reshape(-1, 1)
    # Reserve 200 examples for evaluation.
    split = 800
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rng = np.random.default_rng(42)
    model = Sequential([
        Dense(2, 16, rng=rng),
        ReLU(),
        Dense(16, 16, rng=rng),
        ReLU(),
        Dense(16, 1, rng=rng),
        Sigmoid(),
    ])
    loss = BinaryCrossEntropy()
    initial_loss = loss.forward(model.predict(X_train), y_train)

    history = model.fit(
        X_train,
        y_train,
        loss=loss,
        optimizer=SGD(learning_rate=0.1),
        epochs=200,
        batch_size=32,
        rng=42,
    )

    train_probabilities = model.predict(X_train)
    test_probabilities = model.predict(X_test)
    final_loss = loss.forward(train_probabilities, y_train)
    train_accuracy = np.mean((train_probabilities >= 0.5) == y_train)
    test_accuracy = np.mean((test_probabilities >= 0.5) == y_test)

    print(f"Epochs: {len(history)}")
    print(f"Training loss: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Test accuracy:     {test_accuracy:.2%}")

if __name__ == "__main__":
    main()
