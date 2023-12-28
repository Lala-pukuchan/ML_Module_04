from ex06.mylinearregression import MyLinearRegression
import numpy as np


class MyRidge(MyLinearRegression):
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.01, max_iter=1000, lambda_=1):
        super().__init__(theta, alpha=alpha, max_iter=max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "theta": self.theta,
            "lambda_": self.lambda_,
        }

    def set_params_(self, alpha=None, max_iter=None, theta=None, lambda_=None):
        if alpha is not None:
            self.alpha = alpha
        if max_iter is not None:
            self.max_iter = max_iter
        if theta is not None:
            self.theta = theta
        if lambda_ is not None:
            self.lambda_ = lambda_

    def loss_elem_(self, y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2

    # def loss_(self, y, y_hat):
    #     if y.shape != y_hat.shape:
    #         return None
    #     m = y.shape[0]
    #     loss = np.sum((y_hat - y) ** 2)
    #     penalty = self.lambda_ * np.sum(self.theta[1:] ** 2)
    #     return (1 / (2 * m)) * (loss + penalty)
    def loss_(self, y, y_hat):
        if y.size == 0 or y_hat.size == 0 or self.theta.size == 0:
            return None

        if y.shape != y_hat.shape:
            return None

        loss_elements = self.loss_elem_(y, y_hat)
        loss = np.sum(loss_elements)

        regularization_term = self.lambda_ * np.dot(self.theta[1:].T, self.theta[1:])

        return float(1 / (2 * len(y)) * (loss + regularization_term))

    # def gradient(self, y, x):
    #     if (
    #         not isinstance(x, np.ndarray)
    #         or x.size == 0
    #         or not isinstance(y, np.ndarray)
    #         or y.size == 0
    #         or not isinstance(self.theta, np.ndarray)
    #         or self.theta.size == 0
    #         or x.shape[0] != y.shape[0]
    #         or (x.shape[1] + 1) != self.theta.shape[0]
    #         or y.shape[1] != 1
    #         or self.theta.shape[1] != 1
    #         or not (isinstance(self.lambda_, int) or isinstance(self.lambda_, float))
    #     ):
    #         return None

    #     # calculate hypothesis
    #     x_dash = np.insert(x, 0, 1, axis=1)
    #     h = np.dot(x_dash, self.theta)

    #     # calculate gradient descent
    #     m = y.shape[0]
    #     theta_dash = np.insert(self.theta[1:], 0, 0).reshape(-1, 1)

    #     return (1 / m) * (np.dot(x_dash.T, (h - y)) + self.lambda_ * theta_dash)

    # def fit_(self, x, y):
    #     if (
    #         not isinstance(x, np.ndarray)
    #         or not isinstance(y, np.ndarray)
    #         or not isinstance(self.theta, np.ndarray)
    #         or not isinstance(self.alpha, float)
    #         or not isinstance(self.max_iter, int)
    #     ):
    #         return None
    #     if x.size == 0 or y.size == 0 or self.theta.size == 0:
    #         return None
    #     if x.ndim != 2 or y.ndim != 2 or self.theta.ndim != 2:
    #         return None
    #     if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
    #         return None
    #     if self.max_iter < 0:
    #         return None

    #     for _ in range(self.max_iter):
    #         self.theta = self.theta - self.alpha * self.gradient(y, x)

    #     return self.theta

    def gradient_(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        gradient = np.zeros((n + 1, 1))
        X = np.hstack((np.ones((m, 1)), x))

        for i in range(m):
            xi = X[i].reshape(n + 1, 1).astype(float)
            yi = y[i]
            hypothesis = np.dot(self.theta.T, xi)[0, 0]
            gradient += (hypothesis - yi) * xi

        gradient /= m
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]

        return gradient

    def fit_(self, x, y):
        m = len(y)
        n = x.shape[1]

        # if (x.shape[0] != m) or (self.the.shape[0] != (n + 1)):
        #     return None
        for i in range(self.max_iter):
            # Use the gradient_ method from MyRidge class
            gradient_update = self.gradient_(x, y)
            if gradient_update is None:
                return None
            self.theta = self.theta.astype(np.float64)

            # Regularization term
            regularization_term = (self.lambda_ / m) * self.theta

            # Update theta with regularization
            self.theta -= self.alpha * (gradient_update + regularization_term)

        return self.theta


if __name__ == "__main__":
    X = np.array(
        [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
    )
    Y = np.array([[23.0], [48.0], [218.0]])
    mylr = MyRidge(np.array([[1.0], [1.0], [1.0], [1.0], [1]]), lambda_=0.0)
    # print("mylr.theta:", mylr.theta)

    # Example 0:
    y_hat = mylr.predict_(X)  # Output: array([[8.], [48.], [323.]])
    print(y_hat)

    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))  # Output: array([[225.], [0.], [11025.]])

    # Example 2:
    print(mylr.loss_(Y, y_hat))  # Output: 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    # Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
    print(mylr.theta)

    # Example 4:
    # Output: array([[23.417..], [47.489..], [218.065...]])
    y_hat = mylr.predict_(X)
    print(y_hat)

    # Example 5:
    # Output: array([[0.174..], [0.260..], [0.004..]])
    print(mylr.loss_elem_(Y, y_hat))

    # Example 6:
    print(mylr.loss_(Y, y_hat))  # Output: 0.0732..
    mylr.set_params_(lambda_=2.0)
    print(mylr.get_params_())
