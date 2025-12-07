import numpy as np

class BatchLoader:
    def __init__(self, X, Y, batch_size=1, shuffle=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_data(self):
        size = self.X.shape[0]
        batches = []
        if self.shuffle:
            permutes = np.random.permutation(size)
            X = self.X[permutes]
            Y = self.Y[permutes]
        for i in range(0, size, batch_size):
            X_batch = self.X[i : i + batch_size]
            Y_batch = self.Y[i : i + batch_size]
            batches.extend([(X_batch, Y_batch)])
            
        return batches