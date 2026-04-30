import logging
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss

from .calib.fulldirichlet import FullDirichletCalibrator
from .calib.multinomial import MultinomialRegression
from .utils import clip_for_log


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, matrix_type='full', l2=0.0, comp_l2=False,
                 initializer='identity', maxiter=None):
        if matrix_type not in ['full', 'full_gen', 'diagonal', 'fixed_diagonal']:
            raise(ValueError)

        self.matrix_type = matrix_type
        self.l2 = l2
        if isinstance(l2, list):
            self.l2_grid = l2
        else:
            self.l2_grid = [l2]
        if isinstance(comp_l2, list):
            self.comp_l2 = comp_l2
        else:
            self.comp_l2 = [comp_l2]
        self.initializer = initializer
        self.maxiter = maxiter

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        if self.matrix_type == 'full':
            self.calibrator_ = FullDirichletCalibrator(reg_lambda_list=self.l2_grid,
                                                       reg_mu_list=self.comp_l2,
                                                       initializer=self.initializer)
        elif self.matrix_type in ['diagonal', 'fixed_diagonal']:
            method = 'Diag' if self.matrix_type == 'diagonal' else 'FixDiag'
            self.calibrator_ = _RegularizedDirichletCalibrator(
                method=method,
                reg_lambda_list=self.l2_grid,
                initializer=self.initializer,
                maxiter=self.maxiter,
            )
        else:
            raise(ValueError)

        _X = np.copy(X)
        if len(X.shape) == 1:
            _X = np.vstack(((1-_X), _X)).T

        _X_val = X_val
        if X_val is not None:
            _X_val = np.copy(X_val)
            if len(X_val.shape) == 1:
                _X_val = np.vstack(((1-_X_val), _X_val)).T

        self.calibrator_ = self.calibrator_.fit(_X, y, X_val=_X_val,
                                                y_val=y_val, **kwargs)

        if hasattr(self.calibrator_, 'l2'):
            self.l2 = self.calibrator_.l2
        if hasattr(self.calibrator_, 'weights_'):
            self.weights_ = self.calibrator_.weights_
        if hasattr(self.calibrator_, 'coef_'):
            self.coef_ = self.calibrator_.coef_
        if hasattr(self.calibrator_, 'intercept_'):
            self.intercept_ = self.calibrator_.intercept_
        return self

    @property
    def cannonical_weights(self):
        b = self.weights_[:, -1]
        W = self.weights_[:,:-1]
        col_min = np.min(W,axis=0)
        A = W - col_min
        softmax = lambda z:np.divide(np.exp(z), np.sum(np.exp(z)))
        c = softmax(np.matmul(W, np.log(np.ones(len(b))/len(b))) + b)
        return np.hstack((A, c.reshape(-1,1)))

    def predict_proba(self, S):

        _S = np.copy(S)
        if len(S.shape) == 1:
            _S = np.vstack(((1-_S), _S)).T
            return self.calibrator_.predict_proba(_S)[:,1]

        return self.calibrator_.predict_proba(_S)

    def predict(self, S):

        _S = np.copy(S)
        if len(S.shape) == 1:
            _S = np.vstack(((1-_S), _S)).T
            return self.calibrator_.predict(_S)[:,1]

        return self.calibrator_.predict(_S)


class _RegularizedDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, method, reg_lambda_list, initializer='identity', maxiter=None):
        self.method = method
        self.reg_lambda_list = reg_lambda_list
        self.initializer = initializer
        self.maxiter = maxiter
        self.calibrator_ = None
        self.reg_lambda = None

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):
        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.log(clip_for_log(np.copy(X)))
        _X_val = np.log(clip_for_log(np.copy(X_val)))

        for i, reg_lambda in enumerate(self.reg_lambda_list):
            tmp_cal = MultinomialRegression(
                method=self.method,
                reg_lambda=reg_lambda,
                initializer=self.initializer,
                maxiter=self.maxiter,
            )
            tmp_cal.fit(_X, y, *args, **kwargs)
            tmp_loss = log_loss(
                y_val,
                tmp_cal.predict_proba(_X_val),
                labels=np.arange(X.shape[1]),
            )

            if i == 0 or tmp_loss < final_loss:
                final_cal = tmp_cal
                final_loss = tmp_loss
                final_reg_lambda = reg_lambda

        self.calibrator_ = final_cal
        self.reg_lambda = final_reg_lambda
        self.weights_ = self.calibrator_.weights_

        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict_proba(S))

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict(S))
