'''
-------------------------------------------------------
    Fold Estimator - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from copy import copy
import numpy as np

class FoldEstimator(BaseEstimator):
    ''' Fold Estimator

    Meta estimator that performs cross validation over k folds. Can optionally
    be used as a ensemble of k estimators.

    Parameters
    ----------
    est : Base estimator

    fold : Fold cross validation object

    metric : Evaluations metric, func(y, y_)

    regressor : Flag when using a regressor instead of a classifier, bool

    proba : Flag to use predict_proba instead of predict during fit, bool

    ensemble : Flag for post fit behaviour
                True: Continue as a ensemble trained on separate folds
                False: Retrain one estimator on full data

    verbose : Printing of intermediate results, bool or int
    '''

    def __init__(self, est, fold, metric, regressor=False, proba=False, ensemble=False, verbose=0):

        if proba and regressor:
            raise ValueError('Cannot be both a regressor and use `predict_proba`')

        if proba and not hasattr(est, 'predict_proba'):
            raise ValueError('Cannot have `proba=True` for a classifier without `predict_proba`')

        if not regressor and ensemble and not hasattr(est, 'predict_proba'):
            raise ValueError('Can only ensemble probabilistic classifiers')

        self.est = est
        self.fold = fold
        self.metric = metric
        self.regressor = regressor
        self.proba = proba
        self.ensemble = ensemble
        self.verbose = verbose

    def fit(self, X, y):
        ''' Fitting of the estimator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        '''
 
        X, y = check_X_y(X, y, accept_sparse=True)

        if self.ensemble:
            self.ests_ = []

        self.oof_scores_ = []

        if not self.regressor:
            self.n_classes_ = np.unique(y).shape[0]

        if self.proba:
            self.oof_y_ = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        else:
            self.oof_y_ = np.zeros_like(y)

        current_fold = 1
        # Iterate over folds
        for fold_idx, oof_idx in self.fold.split(X, y):

            # Prepare fold
            X_fold, y_fold = X[fold_idx], y[fold_idx]
            X_oof, y_oof = X[oof_idx], y[oof_idx]

            if self.ensemble:
                est = copy(self.est)
            else:
                est = self.est

            est.fit(X_fold, y_fold)

            if self.proba:
                y_oof_ = est.predict_proba(X_oof)
                self.oof_y_[oof_idx] = y_oof_
                y_oof_ = y_oof_[:,0]
            else:
                y_oof_ = est.predict(X_oof)
                self.oof_y_[oof_idx] = y_oof_

            oof_score = self.metric(y_oof, y_oof_)
            self.oof_scores_.append(oof_score)

            if self.ensemble:
                self.ests_.append(est)

            if self.verbose:
                print('Finished fold {} with score: {}'.format(current_fold, oof_score))

            current_fold += 1

        if not self.ensemble:
            self.est.fit(X, y)

        if len(self.oof_y_.shape) > 1:
            self.oof_score_ = self.metric(y, self.oof_y_[:,0])
        else:
            self.oof_score_ = self.metric(y, self.oof_y_)

        if self.verbose:
            print('Finished with a total score of: {}'.format(self.oof_score_))

        self.n_features_ = X.shape[1]
        self.n_folds_ = self.fold.n_splits

        return self

    def predict_proba(self, X):
        ''' Probability prediction

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The prediction input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of probabilities, floats.
        '''

        if not hasattr(self.est, 'predict_proba'):
            raise ValueError('Base estimator does not support `predict_proba`')

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_features_')

        if self.ensemble:
            y_ = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

            for est in self.ests_:
                y_ += est.predict_proba(X) / self.n_folds_
        else:
            y_ = self.est.predict_proba(X)

        return y_

    def predict(self, X):
        ''' Prediction

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The prediction input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of predictions.
        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_features_')

        if self.regressor and self.ensemble:

            y_ = np.zeros((X.shape[0],), dtype=np.float64)

            for est in self.ests_:
                y_ += est.predict(X) / self.n_folds_

        elif self.ensemble:
            y_ = np.argmax(self.predict_proba(X), axis=1)
        else:
            y_ = self.predict(X)

        return y_
