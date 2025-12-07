import numpy as np
from nnlib.layers import Dense, ReLU, Softmax
from nnlib.losses import CrossEntropy
from nnlib.model import Sequential
from nnlib.optims import SGD

np.random.seed(42)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])  # XOR problem

model = Sequential([
    Dense(2, 32),
    ReLU(),
    Dense(32, 16),
    ReLU(),
    Dense(16, 2),
    Softmax()
])

lr = 0.001
epochs = 1000
optim = SGD(lr=lr)
loss_fn = CrossEntropy()

for _ in range(epochs):
    Y_pred = model.forward(X)
    loss = loss_fn.forward(Y, Y_pred)
    grad = loss_fn.backward()
    model.backward(grad)
    optim.update(model.parameters())

print("Final predictions:", np.round(model.forward(X)))