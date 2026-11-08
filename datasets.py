"""Synthetic datasets for classification examples."""

import numpy as np

def make_circles(n_samples=1000, noise=0.05, factor=0.5, rng=None):
    """Return shuffled points X (N, 2) and binary labels y (N,).

    factor is the inner-to-outer radius ratio. noise is the standard
    deviation of Gaussian noise added to the coordinates.
    """
    rng = np.random.default_rng(rng)
    # Each class lies on a circle with a different radius.
    y = np.arange(n_samples) % 2
    angles = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    radii = np.where(y == 0, 1.0, factor)
    X = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    X += rng.normal(0.0, noise, X.shape)
    # Shuffle points and labels together.
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]
