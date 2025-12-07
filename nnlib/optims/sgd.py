import numpy as np

class SGD:
    def __init__(self,lr):
        self.lr = lr

    def update(self, params):
        for param, grad in params:
            param -= self.lr * grad