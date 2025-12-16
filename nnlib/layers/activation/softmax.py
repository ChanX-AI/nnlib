import numpy as np
from ..layer import Layer

class Softmax(Layer):
    def forward(self, Z):
        self.A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return self.A

    def backward(self, grad):
        return grad * self.A * (1 - self.A)
