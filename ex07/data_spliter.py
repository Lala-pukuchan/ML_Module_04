import numpy as np


def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
      x: numpy.array, a matrix of dimension m * n.
      y: numpy.array, a vector of dimension m * 1.
      proportion: float, the proportion of the dataset that will be assigned to the training set.
    Return:
      (x_train, x_test, y_train, y_test) as a tuple of numpy.array
      None if x or y is an empty numpy.array.
      None if x and y do not share compatible dimensions.
      None if x, y or proportion is not of expected type.
    Raises:
      This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(proportion, float)
    ):
        return None
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return None

    # Combine x and y and shuffle
    combined = np.hstack((x, y.reshape(-1, 1)))
    np.random.shuffle(combined)

    # Split the combined array back into x and y
    split_idx = int(combined.shape[0] * proportion)
    x_train = combined[:split_idx, :-1]
    x_test = combined[split_idx:, :-1]
    y_train = combined[:split_idx, -1:]
    y_test = combined[split_idx:, -1:]

    return (x_train, x_test, y_train, y_test)
