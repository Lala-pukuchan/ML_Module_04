import numpy as np
from ex01.l2_reg import l2


def reg_log_loss_(y, y_hat, theta, lambda_):
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
    if not isinstance(theta, np.ndarray) or theta.size == 0 or theta.shape[1] != 1:
        return None

    # error handling for lambda_
    if not isinstance(lambda_, float):
        return None

    m = y.shape[0]
    eps = 1e-15
    loss = -(1 / m) * (
        np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    )
    loss += (1 / (2 * m)) * (lambda_ * l2(theta))
    return loss
