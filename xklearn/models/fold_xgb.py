'''
-------------------------------------------------------
    Fold XGBoost - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from copy import copy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class FoldXGBoost(BaseEstimator):
    ''' Fold XGBoost

    Meta estimator that performs cross validation over k folds on a XGBoost
    estimator. Can optionally be used as an ensemble of k estimators.

    Parameters
    ----------
    xgb : Base estimator

    fold : Fold cross validation object

    metric : Evaluations metric, func(y, y_)

    fit_params : Parameters that should be fed to estimator during fit. 
                 Dictionary (string -> object)

    refit_full : Flag for post fit behaviour
                 True: Retrain one estimator on full data
                 False: Continue as an ensemble trained on separate folds

    refit_params : Parameters that should be fed to estimator during refit. 
                   Dictionary (string -> object)

    verbose : Printing of fold scores, bool or int
    '''

    def __init__(self, xgb, fold, metric, fit_params={}, refit_full=False, refit_params={}, verbose=1):

        is_proba_metric = metric.__name__ in ['roc_auc_score']
        is_regressor = issubclass(type(xgb), RegressorMixin)

        if is_proba_metric and is_regressor:
            raise ValueError('Cannot be both a regressor and use a metric that '
                             'requires `predict_proba`')

        if is_proba_metric and not hasattr(xgb, 'predict_proba'):
            raise ValueError('Metric `{}` requires a classifier that implements '
                             '`predict_proba`'.format(metric.__name__))

        self.xgb = xgb
        self.fit_params = fit_params
        self.fold = fold
        self.metric = metric
        self.is_regressor_ = is_regressor
        self.is_proba_metric_ = is_proba_metric
        self.refit_full = refit_full
        self.refit_params = refit_params
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

        if not self.refit_full:
            self.xgbs_ = []

        self.oof_scores_ = []

        if not self.is_regressor_:
            self.n_classes_ = np.unique(y).shape[0]

        if self.is_proba_metric_:
            self.oof_y_ = np.zeros((X.shape[0], self.n_classes_),
                                   dtype=np.float64)
        else:
            self.oof_y_ = np.zeros_like(y)

        current_fold = 1
        for fold_idx, oof_idx in self.fold.split(X, y):

            X_fold, y_fold = X[fold_idx], y[fold_idx]
            X_oof, y_oof = X[oof_idx], y[oof_idx]

            if self.refit_full:
                xgb = self.xgb 
            else:
                xgb = copy(self.xgb)

            xgb.fit(X_fold, y_fold,
                    sample_weight=self.fit_params.get('sample_weight'),
                    eval_set=[(X_oof, y_oof)],
                    eval_metric=self.fit_params.get('eval_metric'),
                    early_stopping_rounds=self.fit_params.get('early_stopping_rounds'),
                    verbose=self.fit_params.get('verbose', self.verbose),
                    xgb_model=self.fit_params.get('xgb_model'),
                    sample_weight_eval_set=self.fit_params.get('sample_weight_eval_set'),
                    callbacks=self.fit_params.get('callbacks')
                    )

            if self.is_proba_metric_:
                y_oof_ = xgb.predict_proba(X_oof)
                self.oof_y_[oof_idx] = y_oof_
                y_oof_ = y_oof_[:, 0]
            else:
                y_oof_ = xgb.predict(X_oof)
                self.oof_y_[oof_idx] = y_oof_

            oof_score = self.metric(y_oof, y_oof_)
            self.oof_scores_.append(oof_score)

            if not self.refit_full:
                self.xgbs_.append(xgb)

            if self.verbose:
                print('Finished fold {} with score: {}'.format(current_fold,
                                                               oof_score))

            current_fold += 1

        if self.refit_full:
            xgb.fit(X_fold, y_fold,
                    sample_weight=self.refit_params.get('sample_weight'),
                    eval_set=self.refit_params.get('eval_set'),
                    eval_metric=self.refit_params.get('eval_metric'),
                    early_stopping_rounds=self.refit_params.get('early_stopping_rounds'),
                    verbose=self.refit_params.get('verbose', self.verbose),
                    xgb_model=self.refit_params.get('xgb_model'),
                    sample_weight_eval_set=self.refit_params.get('sample_weight_eval_set'),
                    callbacks=self.refit_params.get('callbacks')
                    )

        if len(self.oof_y_.shape) > 1:
            self.oof_score_ = self.metric(y, self.oof_y_[:, 0])
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

        if not hasattr(self.xgb, 'predict_proba'):
            raise ValueError('Base estimator does not support `predict_proba`')

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_features_')

        if self.refit_full:
            y_ = self.xgb.predict_proba(X)
        else:
            y_ = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

            for xgb in self.xgbs_:
                y_ += xgb.predict_proba(X) / self.n_folds_

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

        if not (self.is_regressor_ or self.refit_full):

            y_ = np.zeros((X.shape[0],), dtype=np.float64)

            for xgb in self.xgbs_:
                y_ += xgb.predict(X) / self.n_folds_

        elif self.refit_full:
            y_ = self.xgb.predict(X)     
        else:
            y_ = np.argmax(self.predict_proba(X), axis=1)

        return y_
