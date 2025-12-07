import numpy as np

class MSE:
    def forward(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        return np.mean(np.mean((Y_true - Y_pred) ** 2, axis=0))

    def backward(self):
        m, n = self.Y_true.shape
        return (-2 / (m*n)) * (self.Y_true - self.Y_pred)