"""Download and load MNIST as NumPy arrays."""

from pathlib import Path
from urllib.request import urlopen

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

def load_mnist(data_dir=DEFAULT_DATA_DIR):
    """Return (images, labels) pairs for the training and test splits.

    Images are uint8 arrays of shape (N, 28, 28); labels have shape (N,).
    The archive is downloaded once and cached in data_dir.
    """
    path = Path(data_dir) / "mnist.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MNIST to {path}")
        # Publish the cache file only after the download is complete.
        with urlopen(MNIST_URL, timeout=60) as response:
            content = response.read()
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(path)

    with np.load(path, allow_pickle=False) as data:
        train = data["x_train"], data["y_train"]
        test = data["x_test"], data["y_test"]
    return train, test
