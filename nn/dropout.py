import numpy as np

class Dropout:

    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, training=True):

        if not training:
            return x

        self.mask = (
            np.random.rand(*x.shape) > self.p
        ) / (1 - self.p)

        return x * self.mask

    def backward(self, grad):
        return grad * self.mask
