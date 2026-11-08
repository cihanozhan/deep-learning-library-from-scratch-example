# Deep Learning Library from Scratch

A small neural network library built with NumPy. Forward passes, gradients,
and parameter updates are written out so the training process is easy to follow.

It includes dense layers, ReLU, sigmoid, tanh, three loss functions, and SGD.
The examples cover a synthetic classification problem and handwritten digit
recognition with MNIST.

## Run

Use Python 3.10 or later.

```bash
git clone https://github.com/cihanozhan/deep-learning-library-from-scratch-example.git
cd deep-learning-library-from-scratch-example
python -m pip install -r requirements.txt
python main.py
```

`main.py` trains a small network to separate two concentric circles. It prints
the training loss and accuracy on the training and held-out test samples.

## MNIST

```bash
python train_mnist.py
```

The MNIST archive is included at `data/mnist.npz`. If it is missing,
`mnist.py` downloads it from the same source used by Keras. Keras itself
is not required.

The model has two hidden layers:

```text
784 inputs -> 128 ReLU -> 64 ReLU -> 10 logits
```

Images are flattened from 28x28 pixels and scaled to [0, 1]. Training uses
the official 60,000-image training split. The 10,000-image test split is
evaluated after training.

The defaults are 5 epochs, batch size 128, learning rate 0.1, and seed 42.
A checked run with NumPy 2.5.4 reached 96.70% test accuracy.

For a shorter run, use a subset of the training data:

```bash
python train_mnist.py --train-samples 10000 --epochs 3
```

The test split remains unchanged. Run `python train_mnist.py --help` for
the available options.

## Code

- `layers.py`: dense layers and activation functions.
- `losses.py`: mean squared error, binary cross-entropy, and Softmax cross-entropy.
- `optimizers.py`: SGD updates using the gradients stored by each layer.
- `model.py`: forward prediction and mini-batch training.
- `datasets.py`: synthetic circle data.
- `mnist.py`: MNIST loading and caching.

Each row is one sample: inputs have shape `(n_samples, n_features)`.
Binary targets use `(N, 1)`; multiclass targets use integer labels with shape
`(N,)`. CrossEntropy takes raw logits, so it does not need a separate Softmax
output layer.

## Tests

```bash
python -B -m unittest -v
```

The tests check gradients against central differences, SGD updates, numerical
stability, and learning on small datasets. MNIST loader tests use synthetic
archives and do not access the network.

## MNIST attribution

MNIST is by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges.
The bundled archive is unchanged from the
[Keras MNIST source](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py).
MNIST is distributed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), as documented
in that loader. This dataset license applies to `data/mnist.npz`.
