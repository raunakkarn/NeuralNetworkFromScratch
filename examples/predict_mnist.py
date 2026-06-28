import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.model import Sequential
from nn.checkpoint import load_model

print("Loading MNIST...")

X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

X = X.astype(np.float32) / 255.0
y = y.astype(int)

# Same subset used during training
X = X[:10000]
y = y[:10000]

_, X_test, _, y_test = train_test_split(
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

load_model(
    model,
    "models/best_mnist_model.npz"
)

idx = random.randint(0, len(X_test) - 1)

image = X_test[idx]
label = y_test[idx]

probs = model.forward(image.reshape(1, 784))

prediction = np.argmax(probs)
confidence = probs[0][prediction]

print(f"True Label      : {label}")
print(f"Prediction      : {prediction}")
print(f"Confidence      : {confidence:.4f}")

plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"True: {label}  Pred: {prediction}"
)

plt.axis("off")

plt.show()
