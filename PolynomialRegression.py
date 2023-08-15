import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression3:
    def __init__(self, M=3):
        self.M = M

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert X.ndim==t.ndim
        # degree M must be less than N number of data ponits(M<N)
        assert self.M<X.shape[0]
        X = np.vander(X, self.M + 1, increasing=True)
        self.w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def predict(self, X):
        X = np.array(X)
        X = np.vander(X, self.M + 1, increasing=True)
        return np.dot(X, self.w)
