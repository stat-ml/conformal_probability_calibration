import numpy as np

def make_one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converts a 1D array of integer labels into a 2D one-hot encoded array.
    """
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true] = 1.0
    return one_hot


def get_cumulative_mass_scores(
    probs: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """
    Computes the cumulative mass score for each sample.

    The score is the sum of sorted probabilities down to the rank of the true class.
    """
    # Get sorted probabilities and indices
    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = -np.sort(-probs, axis=1)

    # Find the rank of the true class for each sample
    true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

    # Calculate cumulative probability sums
    cum_probs = np.cumsum(sorted_probs, axis=1)

    # Get the cumulative mass at the rank of the true class
    scores = cum_probs[np.arange(len(y_true)), true_class_ranks]
    return scores


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
