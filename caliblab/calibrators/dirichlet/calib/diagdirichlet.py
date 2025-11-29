import numpy as np
from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator
from ..utils import clip_for_log
from sklearn.metrics import log_loss

import numpy as np
import jax.numpy as jax_np

from .multinomial import MultinomialRegression
from .fulldirichlet import FullDirichletCalibrator

def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)


def clip(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)

    
def clip_jax(X):
    eps = jax_np.finfo(X.dtype).eps
    return jax_np.clip(X, eps, 1-eps)

class DiagonalDirichletCalibrator(FullDirichletCalibrator):
    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.weights_ = None
        self.ref_row = True
        self.optimizer = 'auto'

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        self.calibrator_ = MultinomialRegression(
            method='Diag', reg_lambda=self.reg_lambda, reg_mu=self.reg_mu,
            reg_norm=self.reg_norm, ref_row=self.ref_row,
            optimizer=self.optimizer)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        self.final_loss = log_loss(y_val, self.calibrator_.predict_proba(_X_val))

        return self