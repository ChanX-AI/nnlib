import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params