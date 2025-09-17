import numpy as np


def make_one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converts a 1D array of integer labels into a 2D one-hot encoded array.
    """
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true] = 1.0
    return one_hot


def cumulative_mass_and_coverage(
    probs: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute descending-sorted cumulative probabilities and coverage indicators.

    Given class probabilities `probs` of shape (n_samples, n_classes) and true
    labels `y_true` of shape (n_samples,), this returns three arrays:
    - cum_probs: cumulative sums of sorted probabilities (descending), shape (n_samples, n_classes)
    - coverage_matrix: boolean matrix where coverage_matrix[i, k] is True if the
      top-(k+1) set contains the true class for sample i, shape (n_samples, n_classes)
    - sorted_indices: indices that sort probabilities descending per row, shape (n_samples, n_classes)

    This utility centralizes common logic used by visualizers and metrics.
    """
    if probs.ndim != 2:
        raise ValueError("`probs` must be a 2D array of shape (n_samples, n_classes)")
    if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
        raise ValueError("`y_true` must be shape (n_samples,) and align with `probs` rows")

    n_samples, n_classes = probs.shape

    # Sort probabilities descending per sample
    # argsort ascending then reverse to avoid negative trick and keep stability
    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)

    # Cumulative sums along sorted probabilities
    cum_probs = np.cumsum(sorted_probs, axis=1)

    # True class rank within the sorted indices
    true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

    # Coverage matrix: for each k, whether true rank <= k
    ranks = np.arange(n_classes)
    coverage_matrix = true_class_ranks[:, np.newaxis] <= ranks[np.newaxis, :]

    return cum_probs, coverage_matrix, sorted_indices


def get_cumulative_mass_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes the cumulative mass score for each sample.

    The score is the sum of sorted probabilities down to the rank of the true class.
    """
    cum_probs, _, sorted_indices = cumulative_mass_and_coverage(probs, y_true)
    true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]
    scores = cum_probs[np.arange(len(y_true)), true_class_ranks]
    return scores


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
