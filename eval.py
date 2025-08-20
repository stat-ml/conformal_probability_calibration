from caliblab.calibrators.temperature_scaling import TemperatureScaling
from caliblab.metrics.ece import ExpectedCalibrationError
from caliblab.eval.runner import evaluate
import numpy as np

logits = np.random.randn(128, 5)
y_true = np.random.randint(0, 5, size=128)


cal = TemperatureScaling()
cal.fit(logits=logits, y_true=y_true)


probs_cal = cal.predict_proba(logits=logits)


metrics = [ExpectedCalibrationError(n_bins=15)]


report = evaluate(
    probs_cal=probs_cal,
    y_true=y_true,
    metrics=metrics,
)
