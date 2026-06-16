
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr=lr

    def step(self,layers):
        for l in layers:
            if hasattr(l,'W'):
                l.W -= self.lr*l.dW
                l.b -= self.lr*l.db

class Adam:
    def __init__(self, lr=0.001,b1=0.9,b2=0.999,eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.t=0; self.state={}

    def step(self,layers):
        self.t+=1
        for i,l in enumerate(layers):
            if not hasattr(l,'W'): continue
            if i not in self.state:
                self.state[i]=[0*l.W,0*l.W,0*l.b,0*l.b]
            mw,vw,mb,vb=self.state[i]
            mw=self.b1*mw+(1-self.b1)*l.dW
            vw=self.b2*vw+(1-self.b2)*(l.dW**2)
            mb=self.b1*mb+(1-self.b1)*l.db
            vb=self.b2*vb+(1-self.b2)*(l.db**2)
            mwh=mw/(1-self.b1**self.t)
            vwh=vw/(1-self.b2**self.t)
            mbh=mb/(1-self.b1**self.t)
            vbh=vb/(1-self.b2**self.t)
            l.W-=self.lr*mwh/(np.sqrt(vwh)+self.eps)
            l.b-=self.lr*mbh/(np.sqrt(vbh)+self.eps)
            self.state[i]=[mw,vw,mb,vb]
