'''
-------------------------------------------------------
    Count Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from warnings import warn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted
import numpy as np

class CountEncoder(BaseEstimator, TransformerMixin):
    ''' Count Encoder

    Transforms categorical values to their respective value count.

    Parameters
    ----------
    unseen : Replacement strategy for unseen values, str
             One of ['one', 'nan', 'error']

    missing : Replacement strategy for missing values, str
              One of ['one', 'nan', 'error']
    '''

    def __init__(self, unseen='one', missing='one'):
        replace_strats = ['one', 'nan', 'error']

        if unseen not in replace_strats:
            raise ValueError('Value of `unseen` {} is not a valid replacement '
                             'strategy, {}'.format(unseen, replace_strats))

        if missing not in replace_strats:
            raise ValueError('Value of `missing` {} is not a valid replacement '
                             'strategy, {}'.format(missing, replace_strats))

        if unseen == 'one':
            self.default_unseen = 1
        elif unseen == 'nan':
            self.default_unseen = np.nan

        if missing == 'one':
            self.default_missing = 1
        elif missing == 'nan':
            self.default_missing = np.nan

        self.requires_float = missing == 'nan' or unseen == 'nan'

        self.unseen = unseen
        self.missing = missing

    def fit(self, X, y=None):
        ''' Fitting of the transformer

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The class values. An array of int.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''

        X = column_or_1d(X, warn=True)

        if self.requires_float:
            X = X.astype('float64')

        if self.missing == 'error' and np.isnan(X).any():
            error_index = list(np.where(np.isnan(X))[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        self.classes_, self.counts_ = np.unique(X, return_counts=True)

        if self.classes_.shape[0] != np.unique(self.counts_).shape[0]:
            warn('Duplicate count encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        self.classes_ = np.append(self.classes_, [-1])
        self.counts_ = np.append(self.counts_, [self.default_unseen])
        self.lut_ = np.hstack([self.classes_.reshape(-1, 1),
                               self.counts_.reshape(-1, 1)])

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
            The count values. An array of int/float.
        '''

        check_is_fitted(self, 'classes_')
        X = column_or_1d(X, warn=True)

        if self.requires_float:
            X = X.astype('float64')

        missing_mask = np.isnan(X)
        unseen_mask = np.isin(X, self.classes_, invert=True)

        if unseen_mask.any() and self.unseen == 'error':
            error_index = list(np.where(unseen_mask)[0])
            raise ValueError('Unseen value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        if missing_mask.any() and self.missing == 'error':
            error_index = list(np.where(missing_mask)[0])
            raise ValueError('Missing value found at index {}. Aborting '
                             'according to set strategy'.format(error_index))

        _, indices = np.unique(X, return_inverse=True)

        # Perform replacement with lookup
        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_),
                            indices))

        X[missing_mask] = self.default_missing

        return X
