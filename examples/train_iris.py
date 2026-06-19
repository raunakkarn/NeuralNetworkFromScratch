import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
import numpy as np


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CategoricalCrossEntropy
from nn.model import Sequential
from nn.optimizers import Adam
from nn.metrics import accuracy

iris = load_iris()

X = iris.data
y = iris.target

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential([
    Dense(4,16),
    ReLU(),
    Dense(16,16),
    ReLU(),
    Dense(16,3),
    Softmax()
])

loss_fn = CategoricalCrossEntropy()
optimizer = Adam(lr=0.01)

epochs = 500

batch_size = 32

for epoch in range(epochs):

    for start in range(0, len(X_train), batch_size):

        end = start + batch_size

        X_batch = X_train[start:end]
        y_batch = y_train[start:end]

        probs = model.forward(X_batch)

        loss = loss_fn.forward(
            probs,
            y_batch
        )

        grad = loss_fn.backward(
            probs,
            y_batch
        )

        model.backward(grad)

        optimizer.step(model.layers)

    if epoch % 50 == 0:

        train_probs = model.forward(X_train)

        acc = accuracy(
            y_train,
            train_probs
        )

        train_loss = loss_fn.forward(
            train_probs,
            y_train
        )

        print(
            f"Epoch {epoch} "
            f"Loss {train_loss:.4f} "
            f"Acc {acc:.4f}"
        )

test_probs = model.forward(X_test)

print(
    "Test Accuracy:",
    accuracy(y_test, test_probs)
)
