"""Offline checks for MNIST loading and training."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mnist import load_mnist
from train_mnist import main

class MNISTTests(unittest.TestCase):
    def make_archive(self):
        # Build a small MNIST-shaped archive without a network request.
        rng = np.random.default_rng(4)
        data = {
            "x_train": rng.integers(0, 256, (20, 28, 28), dtype=np.uint8),
            "y_train": np.tile(np.arange(10, dtype=np.uint8), 2),
            "x_test": rng.integers(0, 256, (10, 28, 28), dtype=np.uint8),
            "y_test": np.arange(10, dtype=np.uint8),
        }
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        return data, buffer.getvalue()

    def test_load_cached_archive(self):
        expected, content = self.make_archive()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "mnist.npz").write_bytes(content)
            with patch("mnist.urlopen", side_effect=AssertionError("Unexpected download")):
                train, test = load_mnist(directory)
        for actual, key in zip((*train, *test), expected):
            np.testing.assert_array_equal(actual, expected[key])
            self.assertEqual(actual.dtype, np.uint8)

    def test_download_is_cached(self):
        expected, content = self.make_archive()
        with tempfile.TemporaryDirectory() as directory:
            with patch("mnist.urlopen", return_value=io.BytesIO(content)) as download:
                with contextlib.redirect_stdout(io.StringIO()):
                    train, test = load_mnist(directory)
                    cached_train, cached_test = load_mnist(directory)
                self.assertEqual(download.call_count, 1)
            self.assertTrue((Path(directory) / "mnist.npz").exists())
            self.assertFalse((Path(directory) / "mnist.tmp").exists())
        np.testing.assert_array_equal(train[0], expected["x_train"])
        np.testing.assert_array_equal(test[1], expected["y_test"])
        np.testing.assert_array_equal(cached_train[0], train[0])
        np.testing.assert_array_equal(cached_test[0], test[0])

    def test_failed_download_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("mnist.urlopen", side_effect=OSError("Download interrupted")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(OSError):
                        load_mnist(directory)
            self.assertFalse((Path(directory) / "mnist.npz").exists())

    def test_training_application_offline(self):
        _, content = self.make_archive()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "mnist.npz").write_bytes(content)
            with patch("mnist.urlopen", side_effect=AssertionError("Unexpected download")):
                with contextlib.redirect_stdout(io.StringIO()):
                    results = main([
                        "--data-dir", directory, "--epochs", "1",
                        "--train-samples", "12", "--batch-size", "5",
                    ])
        self.assertEqual(results["train_samples"], 12)
        self.assertEqual(results["test_samples"], 10)
        self.assertEqual(len(results["training_loss"]), 1)
        self.assertTrue(np.isfinite(results["test_loss"]))
        self.assertTrue(np.isfinite(results["training_loss"]).all())
        self.assertGreaterEqual(results["test_accuracy"], 0.0)
        self.assertLessEqual(results["test_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
