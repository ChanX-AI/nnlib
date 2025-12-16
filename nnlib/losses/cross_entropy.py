import numpy as np

class CrossEntropy:
    def forward(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        return -np.mean(np.mean(Y_true * np.log(Y_pred), axis=0))

    def backward(self):
        m, n = self.Y_true.shape
        return (-1 / (m*n)) * (self.Y_true / self.Y_pred)

# cross_entropy balanced

class BalancedCrossEntropy:
    def forward(self, logits, labels):
        self.labels = labels
        self.logits = logits - np.max(logits, axis=1, keepdims=True)
        self.probs = np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1, keepdims=True)
        m = labels.shape[0]
        rows = np.arange(m)
        episilon = 1e-15
        loss = -np.sum(np.log(self.probs[rows, labels] + episilon)) / m
        return loss

    def backward(self):
        m = self.labels.shape[0]
        grad = self.probs.copy()
        grad[np.arange(m), self.labels] -= 1
        return grad / m

