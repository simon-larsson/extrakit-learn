'''
-------------------------------------------------------
    Target Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from warnings import warn
from ..preprocessing.util import check_error_strat
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
            The target values.

        Returns
        -------
        self : object
            Returns self.
        '''

        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        missing_mask = np.isnan(X)
        encode_mask = np.invert(missing_mask)

        check_error_strat(missing_mask, self.missing, 'missing')

        target_mean = np.mean(y)

        self.default_unseen_ = strat_to_default(self.unseen, 
                                                target_mean)

        self.default_missing_ = strat_to_default(self.missing, 
                                                target_mean)

        self.classes_, counts = np.unique(X[encode_mask], return_counts=True)
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
            
        if self.unseen is not 'error':
            self.classes_ = np.append(self.classes_, [np.max(self.classes_) + 1])
            self.class_means_ = np.append(self.class_means_, [self.default_unseen_])

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

        missing_mask = np.isnan(X)
        encode_mask = np.invert(missing_mask)
        unseen_mask = np.bitwise_xor(np.isin(X, self.classes_, invert=True), 
                                     missing_mask)

        check_error_strat(missing_mask, self.missing, 'missing')
        check_error_strat(unseen_mask, self.unseen, 'unseen')

        # Make all unseen to the same class outside of classes
        X[unseen_mask] = np.max(self.classes_)

        _, indices = np.unique(X[encode_mask], return_inverse=True)

        # Perform replacement with lookup
        X[encode_mask] = np.take(self.lut_[:, 1], \
                                 np.take(np.searchsorted(self.lut_[:, 0], 
                                                         self.classes_),
                                         indices))

        if np.any(missing_mask):
            X[missing_mask] = self.default_missing_

        return X

    def fit_transform(self, X, y):
        ''' Combined fit and transform

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The class values. An array of int.

        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        X : array-like, shape (n_samples,)
            The encoded values. An array of float.
        '''

        X = column_or_1d(X, warn=True)
        y = column_or_1d(y, warn=True)

        missing_mask = np.isnan(X)
        encode_mask = np.invert(missing_mask)

        check_error_strat(missing_mask, self.missing, 'missing')

        target_mean = np.mean(y)

        self.default_unseen_ = strat_to_default(self.unseen, 
                                                target_mean)

        self.default_missing_ = strat_to_default(self.missing, 
                                                target_mean)

        self.classes_, indices, counts = np.unique(X[encode_mask], 
                                                   return_inverse=True, 
                                                   return_counts=True)

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

            self.class_means_ = (counts * self.class_means_ + self.smoothing * target_mean) \
                                / (counts + self.smoothing)

        self.classes_ = np.append(self.classes_, [np.max(self.classes_) + 1])
        self.class_means_ = np.append(self.class_means_, [self.default_unseen_])
        self.lut_ = np.hstack([self.classes_.reshape(-1, 1),
                               self.class_means_.reshape(-1, 1)])

        if self.class_means_.shape[0] != np.unique(self.class_means_).shape[0]:
            warn('Duplicate target encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        # Perform replacement with lookup
        X[encode_mask] = np.take(self.lut_[:, 1], \
                                 np.take(np.searchsorted(self.lut_[:, 0], 
                                                         self.classes_),
                                         indices))

        if np.any(missing_mask):
            X[missing_mask] = self.default_missing_

        return X


def strat_to_default(strat, global_mean=None):

    if strat == 'global':
        return global_mean
    elif strat == 'nan':
        return np.nan
    else:
        return None
