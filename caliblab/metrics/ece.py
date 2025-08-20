from .base import LabelBasedMetricBase


class ExpectedCalibrationError(LabelBasedMetricBase):
    def __init__(self, n_bins: int = 15, strategy: str = "uniform") -> None:
        super().__init__()
        self.n_bins = n_bins
        self.strategy = strategy

    def compute(self, *, probs, y_true, true_proba):
        raise NotImplementedError("ECE.compute is not implemented.")
