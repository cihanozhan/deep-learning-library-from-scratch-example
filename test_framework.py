"""Gradient and training checks. Run with python -m unittest -v."""

import unittest

import numpy as np

from layers import Dense, ReLU, Sigmoid, Tanh
from losses import BinaryCrossEntropy, CrossEntropy, MeanSquaredError
from model import Sequential
from optimizers import SGD
from datasets import make_circles

def numerical_gradient(function, values, epsilon=1e-6):
    """Central differences: df/dx ~ (f(x + h) - f(x - h)) / (2h)."""
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + epsilon
        positive = function()
        values[index] = original - epsilon
        negative = function()
        values[index] = original
        gradient[index] = (positive - negative) / (2.0 * epsilon)
    return gradient

class FrameworkTests(unittest.TestCase):
    def assert_gradient(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_dense_gradients(self):
        rng = np.random.default_rng(7)
        inputs = rng.normal(size=(4, 3))
        grad_output = rng.normal(size=(4, 2))
        layer = Dense(3, 2, rng=rng)
        layer.forward(inputs)
        grad_inputs = layer.backward(grad_output)

        def objective():
            return np.sum(layer.forward(inputs) * grad_output)

        self.assert_gradient(grad_inputs, numerical_gradient(objective, inputs))
        self.assert_gradient(layer.grad_weights, numerical_gradient(objective, layer.weights))
        self.assert_gradient(layer.grad_biases, numerical_gradient(objective, layer.biases))

    def test_activation_gradients(self):
        inputs = np.array([[-1.2, -0.4], [0.6, 1.3]])
        grad_output = np.array([[0.5, -0.3], [0.7, 0.2]])
        for layer in (ReLU(), Sigmoid(), Tanh()):
            with self.subTest(layer=type(layer).__name__):
                layer.forward(inputs)
                gradient = layer.backward(grad_output)
                expected = numerical_gradient(
                    lambda: np.sum(layer.forward(inputs) * grad_output), inputs
                )
                self.assert_gradient(gradient, expected)

    def test_loss_gradients(self):
        cases = (
            (MeanSquaredError(), np.array([[0.2, 0.8], [0.4, 0.6]]),
             np.array([[0.0, 1.0], [1.0, 0.0]])),
            (BinaryCrossEntropy(), np.array([[0.2, 0.8], [0.4, 0.6]]),
             np.array([[0.0, 1.0], [1.0, 0.0]])),
            (CrossEntropy(), np.array([[0.2, -0.5, 0.8], [1.1, 0.3, -0.4]]),
             np.array([2, 0])),
        )
        for loss, predictions, targets in cases:
            with self.subTest(loss=type(loss).__name__):
                loss.forward(predictions, targets)
                gradient = loss.backward()
                expected = numerical_gradient(
                    lambda: loss.forward(predictions, targets), predictions
                )
                self.assert_gradient(gradient, expected)

    def test_network_gradients(self):
        rng = np.random.default_rng(12)
        model = Sequential([Dense(2, 3, rng=rng), Tanh(), Dense(3, 1, rng=rng), Sigmoid()])
        X = np.array([[0.2, -0.6], [0.7, 0.3], [-0.4, 0.9]])
        y = np.array([[0.0], [1.0], [1.0]])
        loss = BinaryCrossEntropy()
        loss.forward(model.predict(X), y)
        gradient = loss.backward()
        for layer in reversed(model.layers):
            gradient = layer.backward(gradient)

        for layer in model.layers:
            for parameter, analytic_gradient in layer.parameters():
                expected = numerical_gradient(
                    lambda: loss.forward(model.predict(X), y), parameter
                )
                self.assert_gradient(analytic_gradient, expected)

    def test_numerical_stability(self):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            sigmoid = Sigmoid()
            probabilities = sigmoid.forward(np.array([[-1000.0], [1000.0]]))
            np.testing.assert_allclose(probabilities, [[0.0], [1.0]])
            self.assertTrue(np.isfinite(sigmoid.backward(np.ones((2, 1)))).all())

            binary_loss = BinaryCrossEntropy()
            self.assertTrue(np.isfinite(binary_loss.forward(probabilities, 1.0 - probabilities)))
            self.assertTrue(np.isfinite(binary_loss.backward()).all())

            loss = CrossEntropy()
            logits = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]])
            # Confidently wrong predictions must still give finite loss and gradients.
            self.assertAlmostEqual(loss.forward(logits, np.array([1, 0])), 2000.0)
            self.assertTrue(np.isfinite(loss.backward()).all())
            np.testing.assert_allclose(loss.backward().sum(axis=1), 0.0, atol=1e-15)

    def test_regression_learning(self):
        X = np.linspace(-1.0, 1.0, 33).reshape(-1, 1)
        y = 2.0 * X - 1.0
        model = Sequential([Dense(1, 1, rng=5)])
        loss = MeanSquaredError()
        history = model.fit(
            X, y, loss=loss, optimizer=SGD(0.1), epochs=150, shuffle=False
        )
        self.assertEqual(len(history), 150)
        self.assertLess(loss.forward(model.predict(X), y), 1e-7)
        np.testing.assert_allclose(model.layers[0].weights, [[2.0]], atol=1e-3)
        np.testing.assert_allclose(model.layers[0].biases, [[-1.0]], atol=1e-3)

    def test_multiclass_learning(self):
        rng = np.random.default_rng(5)
        centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
        y = np.repeat(np.arange(3), 12)
        X = centers[y] + rng.normal(0.0, 0.15, (len(y), 2))
        model = Sequential([Dense(2, 3, rng=5)])
        loss = CrossEntropy()
        model.fit(
            X, y, loss=loss, optimizer=SGD(0.1),
            epochs=60, batch_size=7, rng=5,
        )
        self.assertLess(loss.forward(model.predict(X), y), 0.02)
        np.testing.assert_array_equal(model.predict(X).argmax(axis=1), y)

    def test_partial_batch_history(self):
        X = np.arange(5.0).reshape(-1, 1)
        y = np.zeros((5, 1))
        model = Sequential([Dense(1, 1, rng=3)])
        loss = MeanSquaredError()
        expected_loss = loss.forward(model.predict(X), y)
        # Freeze the parameters to isolate batch-loss weighting.
        history = model.fit(
            X, y, loss=loss, optimizer=SGD(0.0),
            epochs=1, batch_size=2, shuffle=False,
        )
        self.assertAlmostEqual(history[0], expected_loss)

    def test_loss_shapes_prevent_broadcasting(self):
        for loss in (MeanSquaredError(), BinaryCrossEntropy()):
            with self.subTest(loss=type(loss).__name__):
                with self.assertRaises(ValueError):
                    loss.forward(np.ones((4, 1)), np.ones(4))

    def test_binary_cross_entropy_clipping_gradient(self):
        predictions = np.array([[0.0], [5e-8], [1.0 - 5e-8], [1.0]])
        targets = np.array([[1.0], [1.0], [0.0], [0.0]])
        loss = BinaryCrossEntropy()
        loss.forward(predictions, targets)
        gradient = loss.backward()
        expected = numerical_gradient(
            lambda: loss.forward(predictions, targets), predictions, epsilon=1e-9
        )
        self.assert_gradient(gradient, expected)

    def test_sgd_step_matches_numerical_gradient(self):
        X = np.array([[0.2, -0.6], [0.7, 0.3], [-0.4, 0.9]])
        cases = (
            (MeanSquaredError(), np.array([[0.1, 0.8], [0.4, 0.2], [0.6, 0.3]]), Tanh),
            (CrossEntropy(), np.array([0, 1, 0]), ReLU),
        )
        for loss, y, activation in cases:
            with self.subTest(loss=type(loss).__name__):
                rng = np.random.default_rng(23)
                model = Sequential([
                    Dense(2, 4, rng=rng), activation(),
                    Dense(4, 3, rng=rng), activation(), Dense(3, 2, rng=rng),
                ])
                expected_parameters = []
                for layer in model.layers:
                    if isinstance(layer, Dense):
                        for parameter in (layer.weights, layer.biases):
                            gradient = numerical_gradient(
                                lambda: loss.forward(model.predict(X), y), parameter
                            )
                            expected_parameters.append((parameter, parameter.copy() - 0.07 * gradient))
                model.fit(X, y, loss=loss, optimizer=SGD(0.07), epochs=1, shuffle=False)
                for parameter, expected in expected_parameters:
                    self.assert_gradient(parameter, expected)

    def test_xor_learning(self):
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = np.array([[0.0], [1.0], [1.0], [0.0]])
        rng = np.random.default_rng(7)
        model = Sequential([Dense(2, 8, rng=rng), Tanh(), Dense(8, 1, rng=rng), Sigmoid()])
        loss = BinaryCrossEntropy()
        model.fit(X, y, loss=loss, optimizer=SGD(0.3), epochs=1000, shuffle=False)
        probabilities = model.predict(X)
        self.assertLess(loss.forward(probabilities, y), 0.02)
        np.testing.assert_array_equal(probabilities >= 0.5, y)

    def test_nonlinear_multioutput_regression(self):
        X = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
        y = np.column_stack((X[:, 0] ** 2, np.sin(2.0 * X[:, 0])))
        rng = np.random.default_rng(19)
        model = Sequential([Dense(1, 12, rng=rng), Tanh(), Dense(12, 2, rng=rng)])
        loss = MeanSquaredError()
        model.fit(X, y, loss=loss, optimizer=SGD(0.1), epochs=2000, shuffle=False)
        X_test = rng.uniform(-1.0, 1.0, (41, 1))
        y_test = np.column_stack((X_test[:, 0] ** 2, np.sin(2.0 * X_test[:, 0])))
        self.assertLess(loss.forward(model.predict(X_test), y_test), 0.01)

    def test_deep_multiclass_learning(self):
        rng = np.random.default_rng(42)
        y = np.repeat(np.arange(3), 80)
        radii = np.array([0.4, 0.9, 1.4])[y]
        angles = rng.uniform(0.0, 2.0 * np.pi, len(y))
        X = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
        X += rng.normal(0.0, 0.02, X.shape)
        # Use 60 examples per class for training and 20 for evaluation.
        train = np.arange(len(y)) % 80 < 60
        model = Sequential([
            Dense(2, 16, rng=rng), ReLU(),
            Dense(16, 16, rng=rng), ReLU(), Dense(16, 3, rng=rng),
        ])
        loss = CrossEntropy()
        model.fit(
            X[train], y[train], loss=loss, optimizer=SGD(0.1),
            epochs=350, batch_size=32, rng=42,
        )
        logits = model.predict(X[~train])
        self.assertLess(loss.forward(logits, y[~train]), 0.1)
        self.assertGreaterEqual(np.mean(logits.argmax(axis=1) == y[~train]), 0.95)

    def test_training_reproducibility(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.2]])
        y = np.array([[1.0], [1.0], [0.0], [0.0], [1.0]])
        models, histories = [], []
        for _ in range(2):
            rng = np.random.default_rng(17)
            model = Sequential([Dense(2, 3, rng=rng), Tanh(), Dense(3, 1, rng=rng), Sigmoid()])
            histories.append(model.fit(
                X, y, loss=BinaryCrossEntropy(), optimizer=SGD(0.1),
                epochs=5, batch_size=2, rng=17,
            ))
            models.append(model)
        np.testing.assert_array_equal(histories[0], histories[1])
        np.testing.assert_array_equal(models[0].predict(X), models[1].predict(X))

    def test_circles_geometry(self):
        X, y = make_circles(n_samples=11, noise=0.0, factor=0.3, rng=15)
        self.assertEqual(X.shape, (11, 2))
        self.assertEqual(y.shape, (11,))
        np.testing.assert_array_equal(np.bincount(y), [6, 5])
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), np.where(y == 0, 1.0, 0.3))

    def test_minibatch_updates_include_last_sample(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = 2.0 * X
        layer = Dense(1, 1, rng=1)
        layer.weights[:] = 0.0
        layer.biases[:] = 0.0
        model = Sequential([layer])
        history = model.fit(
            X, y, loss=MeanSquaredError(), optimizer=SGD(0.1),
            epochs=1, batch_size=2, shuffle=False,
        )
        # First batch: w=1, b=0.6. Final sample: w=2.44, b=1.08.
        np.testing.assert_allclose(layer.weights, [[2.44]])
        np.testing.assert_allclose(layer.biases, [[1.08]])
        self.assertAlmostEqual(history[0], (2.0 * 10.0 + 5.76) / 3.0)

if __name__ == "__main__":
    unittest.main()
