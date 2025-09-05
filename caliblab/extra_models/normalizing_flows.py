import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from torch.distributions import Normal, Uniform  # or Uniform, depending on your flow

from .functions import nameStrings, optimize


class ConformalFlowCalibrator(BaseEstimator, RegressorMixin):
    """
    A scikit-learn-compatible conformal calibrator using Normalizing Flows.

    After fitting, you can:
    - query an alpha to get the quantile threshold on nonconformity scores for new inputs,
    - or invert for a score to get its corresponding alpha level.
    """

    def __init__(self, flow_name="Uniform", max_iter=3000, n_val=0):
        self.flow_name = flow_name
        self.max_iter = max_iter
        self.n_val = n_val

        if self.flow_name == "Uniform":
            self.base = Uniform(0.0, 1.0)
        elif self.flow_name == "ML":
            self.base = Normal(0.0, 1.0)
        else:
            raise ValueError(f"flow_name must be one of ML or Uniform")

    def fit(self, scores: np.ndarray, X: np.ndarray, y=None):
        """
        Fit the flow-based conformal calibrator.

        Parameters
        ----------
        scores : array-like, shape (n_samples,)
            Nonconformity scores on calibration data.
        X : array-like, shape (n_samples, n_features)
            Features associated with calibration scores.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        self
        """
        a = torch.tensor(scores, dtype=torch.float32)
        X_t = torch.tensor(X, dtype=torch.float32)

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

    def predict_cdf_for_scores(self, scores_new: np.ndarray, X_new: np.ndarray):
        if not hasattr(self, "model_"):
            raise NotFittedError("Call fit() before predict_alpha().")

        # Ensure shapes align
        scores_new = np.atleast_2d(scores_new)  # Shape: (n_samples, n_score_dims)
        X_new = np.atleast_2d(X_new)  # Shape: (n_samples, n_features)
        
        if scores_new.shape[0] != X_new.shape[0]:
            raise ValueError(f"scores_new and X_new must have the same number of samples. "
                           f"Got {scores_new.shape[0]} and {X_new.shape[0]}")

        # Prepare torch inputs
        scores_t = torch.tensor(scores_new, dtype=torch.float32)
        X_t = torch.tensor(X_new, dtype=torch.float32)

        # Result will have shape (n_samples, n_score_dims)
        cdfs_result = []
        
        for i in range(X_t.shape[0]):  # Iterate over samples
            x_val = X_t[i]  # Shape: (n_features,)
            sample_cdfs = []
            
            for j in range(scores_t.shape[1]):  # Iterate over score dimensions
                s_val = scores_t[i, j]  # Scalar score
                
                # 1) map (score, x) to latent z
                z = self.model_(s_val.unsqueeze(0), x_val.unsqueeze(0)).squeeze()

                # 2) exact CDF under base
                F_z = self.base.cdf(z)  # a float in [0,1]

                # 3) append CDF value
                sample_cdfs.append(F_z.item())
            
            cdfs_result.append(sample_cdfs)

        return np.array(cdfs_result)  # Shape: (n_samples, n_score_dims)
