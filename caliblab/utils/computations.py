import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def make_one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converts a 1D array of integer labels into a 2D one-hot encoded array.
    """
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true] = 1.0
    return one_hot
