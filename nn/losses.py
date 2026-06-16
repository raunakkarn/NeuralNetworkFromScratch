
import numpy as np

class CategoricalCrossEntropy:
    def forward(self, probs, y):
        n=len(y)
        return -np.mean(np.log(probs[np.arange(n),y]+1e-12))

    def backward(self, probs, y):
        n=len(y)
        g=probs.copy()
        g[np.arange(n),y]-=1
        return g/n
