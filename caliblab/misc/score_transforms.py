import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from .functions import nameStrings, optimize


class NormalizingFlowTransform(BaseEstimator, RegressorMixin):
    def __init__(self, flow_name="ER", max_iter=3000, n_val=0):
        self.flow_name = flow_name
        self.max_iter = max_iter
        self.n_val = n_val

    def fit(self, scores: np.ndarray, X: np.ndarray, y=None):
        a = torch.tensor(scores, dtype=torch.float64)
        X_t = torch.tensor(X, dtype=torch.float64)

        if self.flow_name not in nameStrings:
            raise ValueError(f"flow_name must be one of {nameStrings}")
        k = nameStrings.index(self.flow_name)

        # Train the normalizing flow model
        model, _ = optimize(k, (a, X_t), self.max_iter, self.n_val)
        self.model_ = model

        # Store flow outputs on calibration set for later CDF estimation
        with torch.no_grad():
            self.b_cal_ = self.model_(a, X_t)
        return self

    def forward(self, scores: np.ndarray, X: np.ndarray):
        scores = torch.tensor(scores, dtype=torch.float64)
        X = torch.tensor(X, dtype=torch.float64)
        return self.model_.forward(scores, X).detach().numpy()

    def inverse(self, scores: np.ndarray, X: np.ndarray):
        # returns MATRIX of size [X.shape[0], 1]
        scores = torch.tensor(scores, dtype=torch.float64)
        X = torch.tensor(X, dtype=torch.float64)
        return self.model_.inverse(scores, X).detach().numpy()[:, None]


class IdentityTransform:
    def fit(self, scores: np.ndarray, logits: np.ndarray, y=None):
        pass

    def forward(self, scores: np.ndarray, logits: np.ndarray):
        return scores

    def inverse(self, scores: np.ndarray, logits: np.ndarray):
        return scores
