"""Train a dense network on MNIST. Run with python train_mnist.py."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from layers import Dense, ReLU
from losses import CrossEntropy
from mnist import DEFAULT_DATA_DIR, load_mnist
from model import Sequential
from optimizers import SGD

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a NumPy neural network on MNIST.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Use a random training subset instead of all 60,000 images.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive.")
    if args.train_samples is not None and args.train_samples < 1:
        parser.error("train-samples must be positive.")

    (images_train, y_train), (images_test, y_test) = load_mnist(args.data_dir)
    rng = np.random.default_rng(args.seed)
    if args.train_samples is not None:
        if args.train_samples > len(y_train):
            parser.error(f"train-samples cannot exceed {len(y_train)}.")
        # Sample only the training split, without replacement.
        indices = rng.choice(len(y_train), size=args.train_samples, replace=False)
        images_train, y_train = images_train[indices], y_train[indices]

    # Flatten 28x28 images and scale pixels from [0, 255] to [0, 1].
    X_train = images_train.reshape(len(images_train), -1) / 255.0
    X_test = images_test.reshape(len(images_test), -1) / 255.0

    model = Sequential([
        Dense(784, 128, rng=rng),
        ReLU(),
        Dense(128, 64, rng=rng),
        ReLU(),
        Dense(64, 10, rng=rng),
    ])
    # CrossEntropy expects raw logits and computes Softmax internally.
    loss = CrossEntropy()
    print(f"NumPy: {np.__version__}")
    print(f"Training samples: {len(y_train)} | Test samples: {len(y_test)}")
    print("Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (logits)")
    start = perf_counter()
    history = model.fit(
        X_train, y_train,
        loss=loss,
        optimizer=SGD(learning_rate=args.learning_rate),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        rng=rng,
    )

    # Evaluate the official test split only after training.
    logits = model.predict(X_test)
    test_loss = float(loss.forward(logits, y_test))
    predictions = logits.argmax(axis=1)
    test_accuracy = float(np.mean(predictions == y_test))
    elapsed = perf_counter() - start

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Training and evaluation: {elapsed:.2f} seconds")
    return {
        "numpy": np.__version__,
        "architecture": [784, 128, 64, 10],
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "training_loss": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "duration_seconds": elapsed,
    }

if __name__ == "__main__":
    main()
