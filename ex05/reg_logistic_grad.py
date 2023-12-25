import numpy as np


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
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
        or not isinstance(theta, np.ndarray)
        or theta.size == 0
        or x.shape[0] != y.shape[0]
        or (x.shape[1] + 1) != theta.shape[0]
        or y.shape[1] != 1
        or theta.shape[1] != 1
        or not (isinstance(lambda_, int) or isinstance(lambda_, float))
    ):
        return None

    # calculate hypothesis
    x_dash = np.insert(x, 0, 1, axis=1)
    h = 1 / (1 + np.exp(-np.dot(x_dash, theta)))

    # calculate gradient descent
    m = y.shape[0]
    n = theta.shape[0]
    grad = np.zeros((n, 1))

    for j in range(n):
        sum_grad = 0.0
        for i in range(m):
            sum_grad += (h[i] - y[i]) * x_dash[i, j]
        if j == 0:
            grad[j] = (1 / m) * sum_grad
        else:
            grad[j] = (1 / m) * sum_grad + (lambda_ / m) * theta[j]

    return grad


def vec_reg_logistic_grad(y, x, theta, lambda_):
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
        or not isinstance(theta, np.ndarray)
        or theta.size == 0
        or x.shape[0] != y.shape[0]
        or (x.shape[1] + 1) != theta.shape[0]
        or y.shape[1] != 1
        or theta.shape[1] != 1
        or not (isinstance(lambda_, int) or isinstance(lambda_, float))
    ):
        return None

    # calculate hypothesis
    x_dash = np.insert(x, 0, 1, axis=1)
    h = 1 / (1 + np.exp(-np.dot(x_dash, theta)))

    # calculate gradient descent
    m = y.shape[0]
    theta_dash = np.insert(theta[1:], 0, 0).reshape(-1, 1)

    return (1 / m) * (np.dot(x_dash.T, (h - y)) + lambda_ * theta_dash)
