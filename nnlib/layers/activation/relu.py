import numpy as np
from ..layer import Layer

class ReLU(Layer):
    def forward(self, Z):
        self.Z = Z
        return np.maximum(Z, 0)

    def backward(self, grad):
        return grad * (self.Z > 0).astype(float)