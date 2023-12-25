import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    # error handling for theta
    if not isinstance(theta, np.ndarray) or theta.size == 0 or theta.shape[1] != 1:
        return None

    # return L2 regulatization with a for-loop
    l2 = 0.0
    for i in range(1, theta.shape[0]):
        l2 += theta[i][0] ** 2
    return l2


def l2(theta):
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
    if not isinstance(theta, np.ndarray) or theta.size == 0 or theta.shape[1] != 1:
        return None

    # change theta0 to 0
    theta_dash = np.insert(theta[1:], 0, 0).reshape(1, -1)

    # return dot product
    return np.dot(theta_dash, theta_dash.T)[0][0]
