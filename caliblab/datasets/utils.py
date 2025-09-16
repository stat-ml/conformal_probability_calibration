from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
from typing import Tuple


import numpy as np

def collapse_rare_logits(X: np.ndarray, y: np.ndarray, min_count: int = 3):
    """
    Collapse all classes with freq < min_count in y into a single UNK class (last column).
    X is logits of shape (n_samples, n_classes).
    Returns: (X_new, y_new, mapping, kept, rare, unk_idx)
    """
    n, C = X.shape
    msk = (y >= 0) & (y < C)
    X, y = X[msk], y[msk]

    counts = np.bincount(y, minlength=C)
    kept = np.flatnonzero(counts >= min_count)
    rare = np.setdiff1d(np.arange(C), kept)
    unk = len(kept)

    # all rare → single UNK via log-sum-exp over all columns
    if kept.size == 0:
        s = X.max(1, keepdims=True)
        Xn = s + np.log(np.exp(X - s).sum(1, keepdims=True))
        yn = np.zeros_like(y)
        mapping = {int(c): 0 for c in range(C)}
        return Xn, yn, mapping, kept, rare, 0

    # label remap via array (faster than dict-loop)
    remap = np.full(C, unk, dtype=int)
    remap[kept] = np.arange(len(kept))
    yn = remap[y]

    kept_logits = X[:, kept]
    if rare.size:
        s = X[:, rare].max(1, keepdims=True)
        unk_col = s + np.log(np.exp(X[:, rare] - s).sum(1, keepdims=True))
    else:
        unk_col = np.full((X.shape[0], 1), -np.inf, dtype=X.dtype)

    Xn = np.concatenate([kept_logits, unk_col], axis=1)
    mapping = {int(c): int(remap[c]) for c in range(C)}
    return Xn, yn



def split_data(
    outputs: np.ndarray,
    labels: np.ndarray,
    cal_ratio: float,
    seed: int,
    subset_items: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < cal_ratio < 1.0:
        raise ValueError("cal_ratio must be between 0 and 1.")

    if subset_items > 0:
        if subset_items < len(outputs):
            np.random.seed(seed)
            indices = np.random.choice(len(outputs), subset_items, replace=False)
            outputs = outputs[indices]
            labels = labels[indices]

    # outputs, labels = collapse_rare_logits(outputs, labels)

    if outputs.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    test_ratio = 1.0 - cal_ratio

    cal_outputs, test_outputs, cal_labels, test_labels = train_test_split(
        outputs, labels, test_size=test_ratio, random_state=seed
    )

    return cal_outputs, test_outputs, cal_labels, test_labels
