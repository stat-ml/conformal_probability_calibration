from caliblab.calibrators.temperature_scaling import TemperatureScaling
from caliblab.metrics.calibration_errors import ExpectedCalibrationError, MaximumCalibrationError, ClasswiseExpectedCalibrationError
from caliblab.metrics.proper_scores import NegativeLogLikelihood
from caliblab.eval.runner import evaluate
import numpy as np

logits = np.random.randn(128, 5)
y_true = np.random.randint(0, 5, size=128)


cal = TemperatureScaling()
cal.fit(logits=logits, y_true=y_true)


probs_cal = cal.predict_proba(logits=logits)


metrics = [
    ExpectedCalibrationError(n_bins=15),
    MaximumCalibrationError(n_bins=15),
    ClasswiseExpectedCalibrationError(n_bins=15),
    NegativeLogLikelihood(),
]


report = evaluate(
    probs_cal=probs_cal,
    y_true=y_true,
    metrics=metrics,
)

print("Calibration Metrics Results:")
for metric_name, value in report.metrics.items():
    print(f"  {metric_name}: {value:.6f}")
print(f"Samples: {report.n_samples}, Classes: {report.n_classes}")
