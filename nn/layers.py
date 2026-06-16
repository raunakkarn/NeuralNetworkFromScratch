
import numpy as np

class Dense:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2/in_features)
        self.W = np.random.randn(in_features, out_features)*limit
        self.b = np.zeros((1,out_features))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad / self.x.shape[0]
        self.db = grad.mean(axis=0, keepdims=True)
        return grad @ self.W.T
