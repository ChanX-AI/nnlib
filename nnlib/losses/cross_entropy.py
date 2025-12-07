import numpy as np

class CrossEntropy:
    def forward(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        return -np.mean(np.mean(Y_true * np.log(Y_pred), axis=0))

    def backward(self):
        m, n = self.Y_true.shape
        return (-1 / (m*n)) * (self.Y_true / self.Y_pred)