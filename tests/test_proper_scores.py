import pytest
import numpy as np
from caliblab.metrics.proper_scores import NegativeLogLikelihood


class TestNegativeLogLikelihood:
    """Concise tests for NLL metric."""

    def test_exact_calculation(self):
        """Test NLL with manual calculation for 3-class problem."""
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

