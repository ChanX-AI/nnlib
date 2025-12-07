import numpy as np
from ..layer import Layer


class Dense(Layer):
    def __init__(self, inp_dim, out_dim):
        self.W = np.random.randn(out_dim, inp_dim)
        self.b = np.zeros(out_dim)
        self.dW = None
        self.db = None
        
    def forward(self, X):
        self.X = X
        Z = X @ self.W.T + self.b 
        return Z

    def backward(self, grad):
        self.dW = grad.T @ self.X
        self.db = grad.sum(axis=0)
        # grad.T @ np.ones((grad.shape[0]))
        # grad.sum(axis=0)
        return grad @ self.W

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]