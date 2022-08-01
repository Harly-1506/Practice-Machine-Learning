import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Linear(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, X):
        Y = X @ self.W + self.b
        return Y

    def __call__(self, X):
        return self.forward(X)


class Sigmoid(object):
    def __init__(self):
        pass

    def forward(self, Z):
        X = 1.0/(1.0+np.exp(-Z))
        return X

    def __call__(self, Z):
        return self.forward(Z)


class LogitRegression(object):
    def __init__(self, W, b, threshold=[0.5]):
        self.linear = Linear(W, b)
        self.sigmoid = Sigmoid()
        self.threshold = threshold

    def predict(self, X):
        Z = self.linear(X)
        Z = self.sigmoid(Z)

        preds = np.where(Z > self.threshold, 1, 0)

        return preds

    def __call__(self, X):
        return self.predict(X)


Nin, Nout = 4, 3
N = 5
W = np.random.normal(0, 1, (Nin, Nout))
b = np.random.normal(0, 1, (Nout))
X = np.random.normal(0, 1, (N, Nin))
y = LogitRegression(W, b)(X)
print(y)
