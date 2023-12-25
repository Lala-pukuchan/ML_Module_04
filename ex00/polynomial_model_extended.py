import numpy as np


def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """

    # error handling for x
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None

    # error handling for power
    if not isinstance(power, int):
        return None

    # app polynomial feature
    pol = x
    for i in range(2, power + 1):
        pol = np.concatenate((pol, x ** i), axis=1)
    return pol
