from typing import Tuple

import numpy as np

from ..utils.bins import get_bin_lowers_uppers


def calculate_confidence_bins(
    probs: np.ndarray, y_true: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates accuracy, confidence, and counts for each confidence bin.

    Args:
        probs (np.ndarray): Array of predicted probabilities for each class.
        y_true (np.ndarray): Array of true labels.
        bins (np.ndarray): The bins to use for calibration analysis.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - bin_confidences: Average confidence for each bin.
            - bin_accuracies: Accuracy for each bin.
            - bin_counts: Number of samples in each bin.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_lowers, bin_uppers = bins[:-1], bins[1:]
    n_bins = len(bin_lowers)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(accuracies[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])

    return bin_confidences, bin_accuracies, bin_counts


def calculate_cumulative_mass_bins(
    probs: np.ndarray, y_true: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the empirical coverage for binned cumulative probability masses.

    This function is central to plotting a cumulative mass calibration curve.
    The curve shows, for a given cumulative probability mass `p`, what percentage
    of the time the true class is found within the smallest set of predicted
    classes whose probabilities sum to `p`.

    For each sample, we generate `C` (number of classes) data points. Each point
    `(p, c)` corresponds to a prediction set, where `p` is the cumulative
    probability mass of the set and `c` is a boolean (0 or 1) indicating if
    the true class is in that set.

    These `N x C` points are then binned by their `p` value to compute the
    empirical coverage (mean of `c`) for each bin.

    Args:
        probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes).
        y_true (np.ndarray): True labels, shape (n_samples,).
        bins (np.ndarray): The number of bins to use for the analysis.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - bin_mean_scores: The mean cumulative mass score for non-empty bins.
            - bin_mean_coverages: The empirical coverage for non-empty bins.
            - bin_counts: The number of data points in each bin.
    """
    if probs.ndim != 2:
        raise ValueError("`probs` must be a 2D array.")
    if y_true.ndim != 1:
        raise ValueError("`y_true` must be a 1D array.")
    if probs.shape[0] != len(y_true):
        raise ValueError("`probs` and `y_true` must have the same number of samples.")

    n_samples, n_classes = probs.shape

    # Step 1: For each sample, sort the predicted probabilities in descending order
    # using a stable sort so ties are handled consistently with other metrics.
    # Example: probs=[0.1, 0.7, 0.2] -> sorted_indices=[1, 2, 0], sorted_probs=[0.7, 0.2, 0.1]
    sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)

    # Step 2: Calculate the cumulative sum of the sorted probabilities. This gives
    # the "cumulative mass" for prediction sets of increasing size (1, 2, ..., C).
    # Example: sorted_probs=[0.7, 0.2, 0.1] -> cum_probs=[0.7, 0.9, 1.0]
    cum_probs = np.cumsum(sorted_probs, axis=1)

    # Step 3: Find the rank (0-indexed position) of the true class within the
    # sorted list of predictions for each sample.
    # Example: y_true=0, sorted_indices=[1, 2, 0] -> true_class_rank=2
    # The `np.where` trick efficiently finds the column index of the true label.
    true_class_ranks = np.where(sorted_indices == y_true[:, np.newaxis])[1]

    # Step 4: Create a "coverage" matrix. An element (i, j) is True if the
    # prediction set of size (j+1) for sample i contains the true class. This
    # is equivalent to checking if the true class's rank is j or less.
    # Example: true_class_rank=2 -> coverage=[F, F, T] (since 2<=0 is F, 2<=1 is F, 2<=2 is T)
    ranks = np.arange(n_classes)
    coverage_matrix = true_class_ranks[:, np.newaxis] <= ranks[np.newaxis, :]

    # Step 5: Flatten both the cumulative probabilities and the coverage matrix.
    # We now have two long vectors. `all_cum_scores[k]` is the cumulative mass
    # and `all_coverages[k]` is the corresponding coverage (0 or 1). These
    # represent all (N x C) data points for our plot.
    all_cum_scores = cum_probs.flatten()
    all_coverages = coverage_matrix.flatten().astype(float)

    # Step 6: Bin the data points based on their cumulative mass score.
    bin_lowers, bin_uppers = bins[:-1], bins[1:]
    n_bins = len(bin_lowers)

    bin_mean_scores = np.zeros(n_bins)
    bin_mean_coverages = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find all scores that fall into the current bin.
        # Use [lower, upper) for all bins except the last, which is [lower, upper].
        if i == n_bins - 1:
            in_bin_mask = (all_cum_scores >= lower) & (all_cum_scores <= upper)
        else:
            in_bin_mask = (all_cum_scores >= lower) & (all_cum_scores < upper)
        bin_counts[i] = np.sum(in_bin_mask)

        if bin_counts[i] > 0:
            # For this bin, calculate the average cumulative mass (x-axis)
            # and the average coverage (y-axis, the empirical coverage).
            bin_mean_scores[i] = np.mean(all_cum_scores[in_bin_mask])
            bin_mean_coverages[i] = np.mean(all_coverages[in_bin_mask])

    return bin_mean_scores, bin_mean_coverages, bin_counts
