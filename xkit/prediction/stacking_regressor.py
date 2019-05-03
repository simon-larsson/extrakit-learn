'''
-------------------------------------------------------
    Stacking Regressor - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class StackingRegressor(BaseEstimator, RegressorMixin):
    ''' Stacking Regressor

    Ensemble regressor that uses one meta regressor and several sub-regressors.
    The sub-regressors give their output to to the main regressor which will
    use them as input features.

    Parameters
    ----------
    regs : Regressors who's output will assist the meta_reg, list regressor

    meta_reg : Ensemble regressor that makes the final output, regressor

    keep_features : If original input features should be used by meta_reg, bool

    refit : If sub-regressors should be refit, bool
    '''

    def __init__(self, regs, meta_reg, keep_features=True, refit=True):
        self.regs = regs
        self.meta_reg = meta_reg
        self.keep_features = keep_features
        self.refit = refit

    def fit(self, X, y):
        ''' Fitting of the regressor

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        '''

        X, y = check_X_y(X, y, accept_sparse=True)

        # Refit of regressor ensemble
        if self.refit:
            for reg in self.regs:
                reg.fit(X, y)

        # Build new tier-2 features
        X_meta = build_meta_X(self.regs, X, self.keep_features)

        # Fit meta regressor, stacking the ensemble
        self.meta_reg.fit(X_meta, y)

        self.n_features_ = X.shape[1]
        self.n_meta_features_ = X_meta.shape[1]
        self.n_regs = len(self.regs)

        return self

    def predict(self, X):
        ''' Regression

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The prediction input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of classifications, bools.
        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_features_')

        # Build new tier-2 features
        X_meta = build_meta_X(self.regs, X, self.keep_features)

        return self.meta_reg.predict(X_meta)

def build_meta_X(regs, X=None, keep_features=True):
    ''' Build features that includes outputs of the sub-regressors

    Parameters
    ----------
    regs : Regressors who's output will assist the meta_reg, list regressor

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The prediction input samples.

    Returns
    -------
    X_meta : {array-like, sparse matrix}, shape (n_samples, n_features + n_regs)
                 The prediction input samples for the meta clf.
    '''

    if keep_features:
        X_meta = X
    else:
        X_meta = None

    for reg in regs:

        if X_meta is None:
            X_meta = reg.predict(X)
        else:
            y_ = reg.predict(X)
            X_meta = np.hstack([X_meta, y_])

    return X_meta
