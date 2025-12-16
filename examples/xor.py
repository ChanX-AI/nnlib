import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from nnlib.layers import Dense, ReLU, Softmax
from nnlib.losses import CrossEntropy, BalancedCrossEntropy
from nnlib.model import Sequential
from nnlib.optims import SGD
from nnlib.utils import softmax

np.random.seed(42)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

model = Sequential([
    Dense(2, 32),
    ReLU(),
    Dense(32, 16),
    ReLU(),
    Dense(16, 2),
])

lr = 0.1
epochs = 100
optim = SGD(lr=lr)
loss_fn = BalancedCrossEntropy()

for _ in range(epochs):
    Y_pred = model.forward(X)
    loss = loss_fn.forward(Y_pred, y)
    grad = loss_fn.backward()
    model.backward(grad)
    optim.update(model.parameters())

logits = model.forward(X)
predictions = softmax(logits)

print("Final predictions:", predictions)
