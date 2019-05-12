'''
-------------------------------------------------------
    Count Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from warnings import warn
from ..preprocessing.util import check_error_strat, is_float_array
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

        self.default_unseen_ = strat_to_default(unseen)
        self.default_missing_ = strat_to_default(missing)

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

        check_error_strat(np.isnan(X), self.missing, 'missing')

        self.classes_, self.counts_ = np.unique(X[np.isfinite(X)], 
                                                return_counts=True)

        if self.classes_.shape[0] != np.unique(self.counts_).shape[0]:
            warn('Duplicate count encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        self.classes_ = np.append(self.classes_, [np.max(self.classes_) + 1])
        self.counts_ = np.append(self.counts_, [self.default_unseen_])
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

        # Cast as int if possible
        if is_float_array(X) and np.all(np.isfinite(X)):
            X = X.astype('int64')

        return X

def strat_to_default(strat):

    if strat == 'one':
        return 1
    elif strat == 'nan':
        return np.nan
    else:
        return None
