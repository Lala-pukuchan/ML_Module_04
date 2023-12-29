import numpy as np


class MyLogisticRegression:
    """
    Description:
        My personnal logistic regression to classify things.
    """

    supported_penalities = ["l2"]

    def __init__(self, theta, alpha=0.01, max_iter=1000, penalty="l2", lambda_=1.0):
        """
        Description:
            generator of the class, initialize self.
        """
        # error management
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("Alpha must be a positive float")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(theta, (list, np.ndarray)):
            raise TypeError("Theta must be a list or numpy array")

        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalities else 0

    def set_params_(self, alpha=None, max_iter=None, theta=None, lambda_=None):
        if alpha is not None:
            self.alpha = alpha
        if max_iter is not None:
            self.max_iter = max_iter
        if theta is not None:
            self.theta = theta
        if lambda_ is not None:
            self.lambda_ = lambda_

    def sigmoid_(self, x):
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
            The sigmoid value as a numpy.ndarray of shape (m, 1).
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (not isinstance(x, np.ndarray)) or x.size == 0:
            return None
        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        """
        Description:
            Prediction of output using the hypothesis function (sigmoid).
        Args:
            x: a numpy.ndarray with m rows and n features.
        Returns:
            The prediction as a numpy.ndarray with m rows.
            None if x is an empty numpy.ndarray.
            None if x does not match the dimension of the
            training set.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (
            not isinstance(x, np.ndarray)
            or x.size == 0
            or x.shape[1] + 1 != self.theta.shape[0]
        ):
            return None

        # append 1 to x's first column
        x_dash = np.insert(x, 0, 1, axis=1)

        # compute y_hat
        y_hat = self.sigmoid_(np.dot(x_dash, self.theta))
        return y_hat

    def loss_elem_(self, y, yhat):
        """
        Description:
            Calculates the loss of each sample.
        Args:
            y: has to be an numpy.ndarray, a vector of dimension m.
            y_hat: has to be an numpy.ndarray, a vector of dimension m.
        Returns:
            yhat - y as a numpy.ndarray of dimension (1, m).
            None if y or y_hat are empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        if (
            not isinstance(y, np.ndarray)
            or not isinstance(yhat, np.ndarray)
            or y.size == 0
            or yhat.size == 0
            or y.shape != yhat.shape
        ):
            return None

        # prevent log(0), clip is to limit the value between min and max
        eps = 1e-15
        yhat = np.clip(yhat, eps, 1 - eps)

        # calculate loss with using cross-entropy
        return y * np.log(yhat) + (1 - y) * np.log(1 - yhat)

    def l2(self):
        """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
        Args:
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
        Returns:
            The L2 regularization as a float.
            None if theta in an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        # error handling for theta
        if (
            not isinstance(self.theta, np.ndarray)
            or self.theta.size == 0
            or self.theta.shape[1] != 1
        ):
            return None

        # change theta0 to 0
        theta_dash = np.insert(self.theta[1:], 0, 0).reshape(1, -1)

        # return dot product
        return np.dot(theta_dash, theta_dash.T)[0][0]

    def loss_(self, y, y_hat):
        """
        Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for l
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta is empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        y = y.reshape(-1, 1)
        # error handling for y
        if (
            not isinstance(y, np.ndarray)
            or y.size == 0
            or y.shape[1] != 1
            or y.shape[0] != y.shape[0]
        ):
            return None

        # error handling for y_hat
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 or y_hat.shape[1] != 1:
            return None

        # error handling for theta
        if (
            not isinstance(self.theta, np.ndarray)
            or self.theta.size == 0
            or self.theta.shape[1] != 1
        ):
            return None

        # error handling for lambda_
        if not isinstance(self.lambda_, float):
            return None

        m = y.shape[0]
        eps = 1e-15
        loss = -(1 / m) * (
            np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        )
        penalty = (1 / (2 * m)) * (self.lambda_ * self.l2())
        return loss + penalty

    def vec_reg_logistic_grad(self, y, x):
        """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of shape m * n.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
        Raises:
            This function should not raise any Exception.
        """
        if (
            not isinstance(x, np.ndarray)
            or x.size == 0
            or not isinstance(y, np.ndarray)
            or y.size == 0
            or not isinstance(self.theta, np.ndarray)
            or self.theta.size == 0
            or x.shape[0] != y.shape[0]
            or (x.shape[1] + 1) != self.theta.shape[0]
            or y.shape[1] != 1
            or self.theta.shape[1] != 1
            or not (isinstance(self.lambda_, int) or isinstance(self.lambda_, float))
        ):
            return None

        # calculate hypothesis
        x_dash = np.insert(x, 0, 1, axis=1)
        h = 1 / (1 + np.exp(-np.dot(x_dash, self.theta)))

        # calculate gradient descent
        m = y.shape[0]
        theta_dash = np.insert(self.theta[1:], 0, 0).reshape(-1, 1)

        return (1 / m) * (np.dot(x_dash.T, (h - y)) + self.lambda_ * theta_dash)

    def fit_(self, x, y):
        """
        Description:
            Find the right theta to make loss minimum.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * n.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
            New theta as a numpy.ndarray, a vector of dimension n * 1
        Raises:
            This function should not raise any Exception.
        """
        # input validation
        # if x, y and theta are not numpy.ndarray, return None
        if (
            not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or not isinstance(self.theta, np.ndarray)
        ):
            return None

        # if x, y and theta are empty numpy.ndarray, return None
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            return None

        # if x, y and theta do not have compatible shapes, return None
        if not (x.shape[0] == y.shape[0] and x.shape[1] + 1 == self.theta.shape[0]):
            return None

        # gradient descent
        for _ in range(self.max_iter):
            self.theta = self.theta - self.alpha * self.vec_reg_logistic_grad(y, x)

        return self.theta
