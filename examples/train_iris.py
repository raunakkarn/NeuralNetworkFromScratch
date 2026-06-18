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

for epoch in range(epochs):

    probs = model.forward(X_train)

    loss = loss_fn.forward(
        probs,
        y_train
    )

    grad = loss_fn.backward(
        probs,
        y_train
    )

    model.backward(grad)

    optimizer.step(model.layers)

    if epoch % 50 == 0:

        acc = accuracy(
            y_train,
            probs
        )

        print(
            f"Epoch {epoch} "
            f"Loss {loss:.4f} "
            f"Acc {acc:.4f}"
        )

test_probs = model.forward(X_test)

print(
    "Test Accuracy:",
    accuracy(y_test, test_probs)
)
