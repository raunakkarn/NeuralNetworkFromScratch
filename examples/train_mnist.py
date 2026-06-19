import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import Adam
from nn.metrics import accuracy

print("Loading MNIST...")

X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

X = X.astype(np.float32) / 255.0
y = y.astype(int)

# Use a subset first so training is fast
X = X[:10000]
y = y[:10000]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 64),
    ReLU(),
    Dense(64, 10),
    Softmax()
])

loss_fn = CategoricalCrossEntropy()
optimizer = Adam(lr=0.001)

epochs = 20
batch_size = 64

for epoch in range(epochs):

    indices = np.random.permutation(
        len(X_train)
    )

    X_train = X_train[indices]
    y_train = y_train[indices]

    for start in range(
        0,
        len(X_train),
        batch_size
    ):

        end = start + batch_size

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        probs = model.forward(
            X_batch
        )

        loss = loss_fn.forward(
            probs,
            y_batch
        )

        grad = loss_fn.backward(
            probs,
            y_batch
        )

        model.backward(grad)

        optimizer.step(
            model.layers
        )

    train_probs = model.forward(
        X_train
    )

    train_acc = accuracy(
        y_train,
        train_probs
    )

    train_loss = loss_fn.forward(
        train_probs,
        y_train
    )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss {train_loss:.4f} "
        f"Acc {train_acc:.4f}"
    )

test_probs = model.forward(
    X_test
)

test_acc = accuracy(
    y_test,
    test_probs
)

print(
    f"MNIST Test Accuracy: {test_acc:.4f}"
)

