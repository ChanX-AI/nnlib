import numpy as np
from ..layer import Layer

class Tanh(Layer):
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, grad):
        return grad * (1 - self.A ** 2)