'''
-------------------------------------------------------
    Target Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''
from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted
import numpy as np

class TargetEncoder(BaseEstimator, TransformerMixin):
    ''' Target Encoder

    Transforms categorical values to their respective target mean.

    Parameters
    ----------
    smoothing : Smooth means by weighting target mean, float
    '''

    def __init__(self, smoothing=0):
        self.smoothing = smoothing

    def fit(self, X, y):
        ''' Fitting of the transformer

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The class values. An array of int.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        '''

        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        target_mean = np.mean(y)

        self.classes_, counts = np.unique(X, return_counts=True)
        self.class_means_ = np.zeros_like(self.classes_, dtype='float64')

        for i, c in enumerate(self.classes_):
            self.class_means_[i] = np.mean(y[np.where(X == c)])

        if self.smoothing != 0:

            #                 class_counts x class_means + smoothing x global mean
            #  smooth mean =  ----------------------------------------------------
            #                           (class_counts + smoothing)

            self.class_means_ = (counts * self.class_means_ + self.smoothing * target_mean)\
                                / (counts + self.smoothing)

        self.lut_ = np.hstack([self.classes_.reshape(-1, 1), self.class_means_.reshape(-1, 1)])

        if self.class_means_.shape[0] != np.unique(self.class_means_).shape[0]:
            warn('Duplicate target encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        # `fit` should always return `self`
        return self

    def transform(self, X):
        ''' Applying transformation on the data

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The class values. An array of int.

        Returns
        -------
        X : array-like, shape (n_samples,)
            The encoded values. An array of float.
        '''

        check_is_fitted(self, 'class_means_')
        X = column_or_1d(X, warn=True)

        classes, indices = np.unique(X, return_inverse=True)

        # Check if unique values match
        diff = list(np.setdiff1d(classes, self.classes_))

        if diff:
            raise ValueError('X contains previously unseen classes: %s' % str(diff))

        # Perform replacement with lookup
        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_), indices))

        return X

    def fit_transform(self, X, y):
        ''' Fitting of the transformer

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The class values. An array of int.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        X : array-like, shape (n_samples,)
            The encoded values. An array of float.
        '''

        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        target_mean = np.mean(y)

        self.classes_, indices, counts = np.unique(X, return_inverse=True, return_counts=True)
        self.class_means_ = np.zeros_like(self.classes_, dtype='float64')

        for i, c in enumerate(self.classes_):
            self.class_means_[i] = np.mean(y[np.where(X == c)])

        if self.smoothing != 0:

            #                 class_counts x class_means + smoothing x global mean
            #  smooth mean =  ----------------------------------------------------
            #                           (class_counts + smoothing)

            self.class_means_ = (counts * self.class_means_ + self.smoothing * target_mean) \
                                / (counts + self.smoothing)

        self.lut = np.hstack([self.classes_.reshape(-1, 1), self.class_means_.reshape(-1, 1)])

        if self.class_means_.shape[0] != np.unique(self.class_means_).shape[0]:
            warn('Duplicate target encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_), indices))

        return X
