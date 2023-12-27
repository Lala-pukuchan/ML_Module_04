from ex06.mylinearregression import MyLinearRegression
import numpy as np

class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "thetas": self.thetas,
            "lambda_": self.lambda_,
        }

    def set_params_(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def loss_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        m = y.shape[0]
        loss = np.sum((y_hat - y) ** 2)
        penalty = self.lambda_ * np.sum(self.thetas[1:] ** 2)
        return (1 / (2 * m)) * (loss + penalty)
    

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

        # change theta0 to 0
        theta = np.insert(self.theta[1:], 0, 0)

        return (1 / m) * (x.T.dot(error) + self.lambda_ * theta)
    

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