import numpy as np


class MyLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        
        # error management
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("Alpha must be a positive float")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Theta must be a list or numpy array")

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)

    def gradient(self, x, y):
        if not (
            isinstance(x, np.ndarray)
            and isinstance(y, np.ndarray)
            and isinstance(self.theta, np.ndarray)
        ):
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None

        m = x.shape[0]
        x = np.insert(x, 0, 1, axis=1)

        y_hat = x.dot(self.theta)
        error = y_hat - y

        return (1 / m) * x.T.dot(error)

    def fit_(self, x, y):
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
            or not isinstance(self.alpha, float)
            or not isinstance(self.max_iter, int)
        ):
            return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None
        if x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2:
            return None
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        if self.max_iter < 0:
            return None

        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient(x, y)

        return self.theta

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
            return None
        if x.size == 0 or self.theta.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.theta.shape[0] != x.shape[1] + 1:
            return None

        # Add an intercept term of 1s as the first column of x
        intercept = np.ones((x.shape[0], 1))
        x = np.hstack((intercept, x))

        return x.dot(self.theta)

    def loss_elem_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        m = y.shape[0]
        return (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

    @staticmethod
    def mse_(y, y_hat):
        squared_diff = (y_hat - y) ** 2
        if squared_diff is None:
            return None
        return np.mean(squared_diff)
