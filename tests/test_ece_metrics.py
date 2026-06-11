import pytest
import numpy as np
from caliblab.metrics.calibration_errors import (
    AccuracyPreservingRatio,
    ExpectedCalibrationError,
    MaximumCalibrationError,
    ClasswiseExpectedCalibrationError,
    OrderPreservingRatio,
)


class TestCalibrationMetricsExact:
    """Precise tests with exact expected values for 3-class, 10-sample calibration."""

    @staticmethod
    def get_test_data():
        """Shared test data for all calibration metrics."""
        probs = np.array(
            [
                [0.2, 0.3, 0.5],  # sample 0: max=0.5, pred=2, true=1 → WRONG
                [0.1, 0.8, 0.1],  # sample 1: max=0.8, pred=1, true=1 → CORRECT
                [0.9, 0.05, 0.05],  # sample 2: max=0.9, pred=0, true=0 → CORRECT
                [0.4, 0.4, 0.2],  # sample 3: max=0.4, pred=0, true=2 → WRONG
                [0.6, 0.2, 0.2],  # sample 4: max=0.6, pred=0, true=0 → CORRECT
                [0.3, 0.6, 0.1],  # sample 5: max=0.6, pred=1, true=1 → CORRECT
                [0.1, 0.1, 0.8],  # sample 6: max=0.8, pred=2, true=2 → CORRECT
                [0.7, 0.2, 0.1],  # sample 7: max=0.7, pred=0, true=0 → CORRECT
                [0.3, 0.3, 0.4],  # sample 8: max=0.4, pred=2, true=2 → CORRECT
                [0.2, 0.2, 0.6],  # sample 9: max=0.6, pred=2, true=2 → CORRECT
            ]
        )
        y_true = np.array([1, 1, 0, 2, 0, 1, 2, 0, 2, 2])
        return probs, y_true

    def test_ece_exact_calculation(self):
        """Test ECE with step-by-step manual calculation."""
        probs, y_true = self.get_test_data()

        # Manual ECE calculation for 3 bins: [0, 1/3), [1/3, 2/3), [2/3, 1]
        # Bin 0 [0, 1/3): no samples (empty bin)
        #
        # Bin 1 [1/3, 2/3): samples 0,3,4,5,8,9 with max_probs [0.5,0.4,0.6,0.6,0.4,0.6]
        #   avg_confidence = (0.5+0.4+0.6+0.6+0.4+0.6)/6 ≈ 0.517
        #   correct = [0,0,1,1,1,1] → avg_accuracy = 4/6 ≈ 0.667
        #   bin_error = |0.517 - 0.667| = 0.15
        #   weighted_error = (6/10) × 0.15 = 0.09
        #
        # Bin 2 [2/3, 1]: samples 1,2,6,7 with max_probs [0.8,0.9,0.8,0.7]
        #   avg_confidence = (0.8+0.9+0.8+0.7)/4 = 0.8
        #   correct = [1,1,1,1] → avg_accuracy = 4/4 = 1.0
        #   bin_error = |0.8 - 1.0| = 0.2
        #   weighted_error = (4/10) × 0.2 = 0.08
        #
        # ECE = 0.09 + 0.08 = 0.17

        ece = ExpectedCalibrationError(n_bins=3)
        result = ece(probs=probs, y_true=y_true)
        assert result == pytest.approx(0.17, abs=1e-6)

    def test_mce_exact_calculation(self):
        """Test MCE with detailed manual calculation."""
        probs, y_true = self.get_test_data()

        # Manual MCE calculation for 3 bins: [0, 1/3), [1/3, 2/3), [2/3, 1]
        # Bin 0 [0, 1/3): no samples (empty bin)
        #
        # Bin 1 [1/3, 2/3): samples 0,3,4,5,8,9 with max_probs [0.5,0.4,0.6,0.6,0.4,0.6]
        #   avg_confidence = (0.5+0.4+0.6+0.6+0.4+0.6)/6 ≈ 0.517
        #   correct = [0,0,1,1,1,1] → avg_accuracy = 4/6 ≈ 0.667
        #   calibration_error = |0.517 - 0.667| = 0.15
        #
        # Bin 2 [2/3, 1]: samples 1,2,6,7 with max_probs [0.8,0.9,0.8,0.7]
        #   avg_confidence = (0.8+0.9+0.8+0.7)/4 = 0.8
        #   correct = [1,1,1,1] → avg_accuracy = 4/4 = 1.0
        #   calibration_error = |0.8 - 1.0| = 0.2
        #
        # MCE = max(0.15, 0.2) = 0.2

        mce = MaximumCalibrationError(n_bins=3)
        result = mce(probs=probs, y_true=y_true)
        assert result == pytest.approx(0.2, abs=1e-6)

    def test_classwise_ece_exact_calculation(self):
        """Test cw-ECE with detailed per-class calculation."""
        probs, y_true = self.get_test_data()

        # Manual cw-ECE calculation (ECE computed separately for each class):
        # For each class, we use that class's probability column and binary correctness
        #
        # CLASS 0: probs=[0.2,0.1,0.9,0.4,0.6,0.3,0.1,0.7,0.3,0.2], correct=[0,0,1,0,1,0,0,1,0,0]
        #   Bin 0 [0,0.33): 6 samples → avg_conf=0.2, avg_acc=0.0, error=0.2, weight=0.6
        #   Bin 1 [0.33,0.67): 2 samples → avg_conf=0.5, avg_acc=0.5, error=0.0, weight=0.2
        #   Bin 2 [0.67,1]: 2 samples → avg_conf=0.8, avg_acc=1.0, error=0.2, weight=0.2
        #   Class 0 ECE = 0.6×0.2 + 0.2×0.0 + 0.2×0.2 = 0.16
        #
        # CLASS 1: probs=[0.3,0.8,0.05,0.4,0.2,0.6,0.1,0.2,0.3,0.2], correct=[1,1,0,0,0,1,0,0,0,0]
        #   Bin 0 [0,0.33): 7 samples → avg_conf≈0.2, avg_acc≈0.29, error=0.05, weight=0.7
        #   Bin 1 [0.33,0.67): 2 samples → avg_conf=0.5, avg_acc=0.5, error=0.0, weight=0.2
        #   Bin 2 [0.67,1]: 1 sample → avg_conf=0.8, avg_acc=1.0, error=0.2, weight=0.1
        #   Class 1 ECE = 0.7×0.05 + 0.2×0.0 + 0.1×0.2 = 0.055
        #
        # CLASS 2: probs=[0.5,0.1,0.05,0.2,0.2,0.1,0.8,0.1,0.4,0.6], correct=[0,0,0,1,0,0,1,0,1,1]
        #   Bin 0 [0,0.33): 6 samples → avg_conf≈0.175, avg_acc≈0.33, error≈0.042, weight=0.6
        #   Bin 1 [0.33,0.67): 3 samples → avg_conf≈0.5, avg_acc≈0.33, error≈0.167, weight=0.3
        #   Bin 2 [0.67,1]: 1 sample → avg_conf=0.8, avg_acc=1.0, error=0.2, weight=0.1
        #   Class 2 ECE = 0.6×0.042 + 0.3×0.167 + 0.1×0.2 = 0.095
        #
        # cw-ECE = (0.16 + 0.055 + 0.095) / 3 = 0.103333

        cw_ece = ClasswiseExpectedCalibrationError(n_bins=3)
        result = cw_ece(probs=probs, y_true=y_true)

        # cw-ECE = (0.16 + 0.055 + 0.095) / 3 = 0.103333
        assert result == pytest.approx(0.103333, abs=1e-6)


class TestOrderPreservationMetrics:
    def test_accuracy_preserving_ratio(self):
        uncalibrated_probs = np.array(
            [
                [0.7, 0.2, 0.1],  # top-1 stays class 0
                [0.45, 0.4, 0.15],  # top-1 swaps 0 -> 1
                [0.2, 0.7, 0.1],  # top-1 stays class 1
                [0.2, 0.3, 0.5],  # top-1 stays class 2
            ]
        )
        calibrated_probs = np.array(
            [
                [0.6, 0.25, 0.15],
                [0.35, 0.5, 0.15],
                [0.15, 0.75, 0.1],
                [0.21, 0.29, 0.5],
            ]
        )

        metric = AccuracyPreservingRatio()
        result = metric(
            probs=calibrated_probs,
            y_true=np.array([0, 1, 1, 2]),
            uncalibrated_probs=uncalibrated_probs,
        )
        assert result == pytest.approx(0.75, abs=1e-9)

    def test_order_preserving_ratio(self):
        uncalibrated_probs = np.array(
            [
                [0.7, 0.2, 0.1],  # order preserved
                [0.45, 0.4, 0.15],  # order changed
                [0.2, 0.7, 0.1],  # order preserved
                [0.3, 0.4, 0.3],  # order changed
            ]
        )
        calibrated_probs = np.array(
            [
                [0.6, 0.25, 0.15],  # same: 0 > 1 > 2
                [0.35, 0.5, 0.15],  # now: 1 > 0 > 2
                [0.15, 0.75, 0.1],  # same: 1 > 0 > 2
                [0.25, 0.35, 0.4],  # now: 2 > 1 > 0
            ]
        )

        metric = OrderPreservingRatio()
        result = metric(
            probs=calibrated_probs,
            y_true=np.array([0, 1, 1, 2]),
            uncalibrated_probs=uncalibrated_probs,
        )
        assert result == pytest.approx(0.5, abs=1e-9)
