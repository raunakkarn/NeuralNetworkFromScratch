
class Sequential:
    def __init__(self,layers):
        self.layers=layers

    def forward(self,x):
        for l in self.layers:
            x=l.forward(x)
        return x

    def backward(self,grad):
        for l in reversed(self.layers):
            if hasattr(l,'backward'):
                grad=l.backward(grad)
