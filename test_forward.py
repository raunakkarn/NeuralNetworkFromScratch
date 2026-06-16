import numpy as np

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.model import Sequential

model = Sequential([
    Dense(784, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
])

X = np.random.randn(5, 784)

output = model.forward(X)

print(output.shape)
print(output)
