import numpy as np


def get_tp_fp_tn_fn(y, y_hat, label=1):
    tp = 0  # True Positives
    fp = 0  # False Positives
    tn = 0  # True Negatives
    fn = 0  # False Negatives

    for actual, predicted in zip(y, y_hat):
        if actual == label and predicted == label:
            tp += 1
        elif actual != label and predicted == label:
            fp += 1
        elif actual != label and predicted != label:
            tn += 1
        elif actual == label and predicted != label:
            fn += 1

    return tp, fp, tn, fn


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y: a numpy.ndarray for the correct labels
    y_hat: a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None

    correct_predictions = 0
    for actual, predicted in zip(y, y_hat):
        if actual == predicted:
            correct_predictions += 1

    return correct_predictions / len(y)


def precision_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp, fp, tn, fn = get_tp_fp_tn_fn(y, y_hat, label=pos_label)
    if tp + fp == 0:
        return 0  # Avoid division by zero
    return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    tp, fp, tn, fn = get_tp_fp_tn_fn(y, y_hat, label=pos_label)
    if tp + fn == 0:
        return 0  # Avoid division by zero
    return tp / (tp + fn)


def f1_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    if precision + recall == 0:
        return 0  # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)
