
from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.model import Sequential

model = Sequential([
    Dense(784,128),
    ReLU(),
    Dense(128,64),
    ReLU(),
    Dense(64,10),
    Softmax()
])

print("Model initialized.")
