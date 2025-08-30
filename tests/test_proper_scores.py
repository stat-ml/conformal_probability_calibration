import pytest
import numpy as np

from caliblab.metrics.proper_scores import BrierScore, NegativeLogLikelihood
from caliblab.utils.computations import make_one_hot


class TestBrierScore:
    """Concise tests for Brier score."""

    def test_brier_with_labels(self):
        """Brier score when y_true are integer class labels."""
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ])
        y_true = np.array([0, 1, 2, 0])

        # one-hot targets
        one_hot = make_one_hot(y_true, probs.shape[1])

        expected = np.mean(np.sum((probs - one_hot) ** 2, axis=1), axis=0)

        brier = BrierScore()
        assert brier(probs=probs, y_true=y_true) == pytest.approx(expected)

    def test_brier_with_soft_true_proba(self):
        """Brier score when true labels are soft distributions (non-constant)."""
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ])
        true_proba = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ])

        expected = np.mean(np.sum((probs - true_proba) ** 2, axis=1), axis=0)

        brier = BrierScore()
        assert brier(probs=probs, true_proba=true_proba) == pytest.approx(expected)

class TestNegativeLogLikelihood:
    """Concise tests for NLL metric."""

    def test_exact_calculation(self):
        """Test NLL with manual calculation for 3-class problem and labels."""
        probs = np.array([
            [0.7, 0.2, 0.1],  # true=0, prob=0.7
            [0.1, 0.8, 0.1],  # true=1, prob=0.8
            [0.2, 0.3, 0.5],  # true=2, prob=0.5
            [0.6, 0.3, 0.1],  # true=0, prob=0.6
        ])
        y_true = np.array([0, 1, 2, 0])
        
        # NLL = [-log(0.7) - log(0.8) - log(0.5) - log(0.6)] / 4
        expected = (-np.log(0.7) - np.log(0.8) - np.log(0.5) - np.log(0.6)) / 4
        
        nll = NegativeLogLikelihood()
        assert nll(probs=probs, y_true=y_true) == pytest.approx(expected)

    def test_true_proba_soft_labels(self):
        """Test NLL when true labels are provided as soft distributions."""
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ])
        true_proba = np.array([
            [0.6, 0.3, 0.1],  # soft distribution
            [0.2, 0.5, 0.3],  # soft distribution
        ])

        # NLL = mean over samples of -sum(p_true * log(p_pred))
        expected_0 = -(0.8 * np.log(0.7) + 0.1 * np.log(0.2) + 0.1 * np.log(0.1))
        expected_1 = -(0.2 * np.log(0.1) + 0.5 * np.log(0.8) + 0.3 * np.log(0.1))
        expected = (expected_0 + expected_1) / 2

        nll = NegativeLogLikelihood()
        assert nll(probs=probs, true_proba=true_proba) == pytest.approx(expected)
