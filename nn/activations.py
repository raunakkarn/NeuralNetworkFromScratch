
import numpy as np

class ReLU:
    def forward(self,x):
        self.mask = x > 0
        return np.maximum(0,x)

    def backward(self,grad):
        return grad*self.mask

class Sigmoid:
    def forward(self,x):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self,grad):
        return grad*self.y*(1-self.y)

class Tanh:
    def forward(self,x):
        self.y=np.tanh(x)
        return self.y

    def backward(self,grad):
        return grad*(1-self.y**2)

class Softmax:
    def forward(self,x):
        e=np.exp(x-np.max(x,axis=1,keepdims=True))
        self.p=e/e.sum(axis=1,keepdims=True)
        return self.p
