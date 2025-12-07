import numpy as np
from ..layer import Layer

class Sigmoid(Layer):
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, grad):
        return grad * self.A * (1 - self.A)