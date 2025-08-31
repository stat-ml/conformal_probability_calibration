import numpy as np
import pytest
import torch

from caliblab.calibrators import IsotonicRegression, TemperatureScaling


def test_temperature_scaling_on_overconfident_model():
    """
    Test that TemperatureScaling finds T > 1 for an overconfident model.
    """
    # High confidence logits, but only 50% correct
    logits = np.array([[10, 0], [10, 0], [10, 0], [10, 0]])
    y_true = np.array([0, 0, 1, 1])  # Model is wrong 50% of the time

    calibrator = TemperatureScaling(lr=0.1, max_iter=100)
    calibrator.fit(logits=logits, y_true=y_true)

    # For an overconfident model, temperature should be > 1 to soften probabilities
    assert calibrator.temperature.item() > 1.0


def test_temperature_scaling_on_underconfident_model():
    """
    Test that TemperatureScaling finds T < 1 for an underconfident model.
    """
    # Low confidence logits, but 100% correct
    logits = np.array([[0.1, 0], [0.1, 0], [0.1, 0], [0.1, 0]])
    y_true = np.array([0, 0, 0, 0])  # Model is always correct

    calibrator = TemperatureScaling(lr=0.01, max_iter=50)
    calibrator.fit(logits=logits, y_true=y_true)

    # For an underconfident model, temperature should be < 1 to sharpen probabilities
    assert calibrator.temperature.item() < 1.0


def test_temperature_scaling_predict_proba():
    """
    Test that TemperatureScaling predict_proba returns valid probabilities.
    """
    logits = np.random.randn(10, 3)
    y_true = np.random.randint(0, 3, size=10)

    calibrator = TemperatureScaling()
    calibrator.fit(logits=logits, y_true=y_true)
    probs = calibrator.predict_proba(logits=logits)

    assert probs.shape == (10, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_isotonic_regression_predict_proba():
    """
    Test that IsotonicRegression predict_proba returns valid probabilities.
    """
    # Create probabilities that are systematically underestimated for class 1
    probs = np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
        ]
    )
    y_true = np.array([0, 1, 1, 1])  # Model is wrong once on class 0, twice on class 1

    calibrator = IsotonicRegression()
    calibrator.fit(probs=probs, y_true=y_true)
    calibrated_probs = calibrator.predict_proba(probs=probs)

    assert calibrated_probs.shape == probs.shape
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0)
    assert np.all(calibrated_probs >= 0)
    assert np.all(calibrated_probs <= 1)

    # After calibration, the probability for class 1 should increase for the second sample
    # where the model was correct but underconfident.
    # Note: This is a weak test, as the exact value is hard to predict,
    # but we can check the direction of change.
    original_prob_class_1 = probs[1, 1]
    calibrated_prob_class_1 = calibrated_probs[1, 1]
    assert calibrated_prob_class_1 > original_prob_class_1
