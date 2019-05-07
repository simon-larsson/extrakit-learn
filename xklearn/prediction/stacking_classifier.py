'''
-------------------------------------------------------
    Stacking Classifier - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class StackingClassifier(BaseEstimator, ClassifierMixin):
    ''' Stacking Classifier

    Ensemble classifier that uses one meta classifiers and several sub-classifiers.
    The sub-classifiers give their output to to the main classifier which will use
    them as input features.

    Parameters
    ----------
    clfs : Classifiers who's output will assist the meta_clf, list classifier

    meta_clf : Ensemble classifier that makes the final output, classifier

    keep_features : If original input features should be used by meta_clf, bool

    refit : If sub-classifiers should be refit, bool
    '''

    def __init__(self, clfs, meta_clf, keep_features=True, refit=True):
        self.clfs = clfs
        self.meta_clf = meta_clf
        self.keep_features = keep_features
        self.refit = refit

    def fit(self, X, y):
        ''' Fitting of the classifier

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

        # Refit of classifier ensemble
        if self.refit:
            for clf in self.clfs:
                clf.fit(X, y)

        # Build new tier-2 features
        X_meta = build_meta_X(self.clfs, X, self.keep_features)

        # Fit meta classifer, stacking the ensemble
        self.meta_clf.fit(X_meta, y)

        # set attributes
        self.n_features_ = X.shape[1]
        self.n_meta_features_ = X_meta.shape[1]
        self.n_clfs_ = len(self.clfs)

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

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_features_')

        # Build new tier-2 features
        X_meta = build_meta_X(self.clfs, X, self.keep_features)

        return self.meta_clf.predict_proba(X_meta)

    def predict(self, X):
        ''' Classification

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
        X_meta = build_meta_X(self.clfs, X, self.keep_features)

        return self.meta_clf.predict(X_meta)

def build_meta_X(clfs, X=None, keep_features=True):
    ''' Build features that includes outputs of the sub-classifiers

    Parameters
    ----------
    clfs : Classifiers that who's output will assist the meta_clf, list classifier

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The prediction input samples.

    Returns
    -------
    X_meta : {array-like, sparse matrix}, shape (n_samples, n_features + n_clfs*classes)
                 The prediction input samples for the meta clf.
    '''

    if keep_features:
        X_meta = X
    else:
        X_meta = None

    for clf in clfs:

        if X_meta is None:
            X_meta = clf.predict_proba(X)
        else:
            y_ = clf.predict_proba(X)
            X_meta = np.hstack([X_meta, y_])

    return X_meta
