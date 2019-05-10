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

    unseen : Replacement strategy for unseen values, str
             One of ['global', 'nan', 'error']

    missing : Replacement strategy for missing values, str
              One of ['global', 'nan', 'error']
    '''

    def __init__(self, smoothing=0, unseen='global', missing='global'):
        replace_strats = ['global', 'nan', 'error']

        if unseen not in replace_strats:
            raise ValueError('Value of `unseen` {} is not a valid replacement '
                             'strategy, {}'.format(unseen, replace_strats))

        if missing not in replace_strats:
            raise ValueError('Value of `missing` {} is not a valid replacement '
                             'strategy, {}'.format(missing, replace_strats))

        self.unseen = unseen
        self.missing = missing
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

        if self.missing == 'error' and np.isnan(X).any():
            error_index = list(np.where(np.isnan(X))[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        target_mean = np.mean(y)

        if self.unseen == 'global':
            self.default_unseen_ = target_mean
        elif self.unseen == 'nan':
            self.default_unseen_ = np.nan

        if self.missing == 'global':
            self.default_missing_ = target_mean
        elif self.missing == 'nan':
            self.default_missing_ = np.nan

        self.classes_, counts = np.unique(X, return_counts=True)
        self.class_means_ = np.zeros_like(self.classes_, dtype='float64')

        for i, c in enumerate(self.classes_):
            class_mask = np.where(X == c)

            if class_mask[0].shape[0] > 0:
                self.class_means_[i] = np.mean(y[class_mask])
            else:
                self.class_means_[i] = 1.0

        if self.smoothing != 0:

            #                 class_counts x class_means + smoothing x global mean
            #  smooth mean =  ----------------------------------------------------
            #                           (class_counts + smoothing)

            self.class_means_ = (counts * self.class_means_ + self.smoothing * target_mean)\
                                / (counts + self.smoothing)

        self.lut_ = np.hstack([self.classes_.reshape(-1, 1),
                               self.class_means_.reshape(-1, 1)])

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

        missing_mask = np.isnan(X)
        unseen_index = list(np.setdiff1d(classes, self.classes_))

        if missing_mask.any() and self.missing == 'error':
            error_index = list(np.where(missing_mask)[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        if self.unseen == 'error' and unseen_index:
            raise ValueError('X contains previously a unseen classes at index '
                             '{}'.format(unseen_index))

        # Perform replacement with lookup
        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_),
                            indices))

        X[missing_mask] = self.default_missing_

        return X

    def fit_transform(self, X, y):
        ''' Combined fit and transform

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

        if self.missing == 'error' and np.isnan(X).any():
            error_index = list(np.where(np.isnan(X))[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        target_mean = np.mean(y)

        if self.unseen == 'global':
            self.default_unseen_ = target_mean
        elif self.unseen == 'nan':
            self.default_unseen_ = np.nan

        if self.missing == 'global':
            self.default_missing_ = target_mean
        elif self.missing == 'nan':
            self.default_missing_ = np.nan

        self.classes_, indices, counts = np.unique(X, return_inverse=True, return_counts=True)
        self.class_means_ = np.zeros_like(self.classes_, dtype='float64')

        missing_mask = np.isnan(X)

        if missing_mask.any() and self.missing == 'error':
            error_index = list(np.where(missing_mask)[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        for i, c in enumerate(self.classes_):
            class_mask = np.where(X == c)

            if class_mask[0].shape[0] > 0:
                self.class_means_[i] = np.mean(y[class_mask])
            else:
                self.class_means_[i] = 1.0

        if self.smoothing != 0:

            #                 class_counts x class_means + smoothing x global mean
            #  smooth mean =  ----------------------------------------------------
            #                           (class_counts + smoothing)

            self.class_means_ = (counts * self.class_means_ + self.smoothing * target_mean) \
                                / (counts + self.smoothing)

        self.lut_ = np.hstack([self.classes_.reshape(-1, 1),
                               self.class_means_.reshape(-1, 1)])

        if self.class_means_.shape[0] != np.unique(self.class_means_).shape[0]:
            warn('Duplicate target encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_),
                            indices))

        X[missing_mask] = self.default_missing_

        return X
