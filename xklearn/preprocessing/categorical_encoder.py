'''
-------------------------------------------------------
    Categorical Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    ''' Categorical Encoder

    Extends scikit's labels encoder by allowing nan values.

    Parameters
    ----------
    unseen : Replacement strategy for unseen values, str
             One of ['encode', 'nan', 'error']

    missing : Replacement strategy for missing values, str
              One of ['encode', 'nan', 'error']
    '''

    def __init__(self, unseen='nan', missing='nan'):

        replace_strats = ['encode', 'nan', 'error']

        if unseen not in replace_strats:
            raise ValueError('Value of `unseen` {} is not a valid replacement '
                             'strategy, {}'.format(unseen, replace_strats))

        if missing not in replace_strats:
            raise ValueError('Value of `missing` {} is not a valid replacement '
                             'strategy, {}'.format(missing, replace_strats))

        if unseen == 'encode':
            self.default_unseen_ = -1
        elif unseen == 'nan':
            self.default_unseen_ = np.nan

        if missing == 'encode':
            self.default_missing_ = -1
        elif missing == 'nan':
            self.default_missing_ = np.nan

        self.le_ = LabelEncoder()

    def fit(self, X, y=None):
        ''' Fitting of the transformer

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''

        X = column_or_1d(X.copy(), warn=True)

        if is_object_array(X):
            self.le_.fit(X[[x is not np.nan for x in X]])
        elif is_float_array(X):
            self.le_.fit(X[np.isfinite(X)])
        else:
            self.le_.fit(X)

        # `fit` should always return `self`
        return self

    def transform(self, X):
        ''' Applying transformation on the data

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        Returns
        -------
        X : array-like, shape (n_samples,)
            The count values. An array of int.
        '''

        X = column_or_1d(X.copy(), warn=True)

        unseen_mask = np.isin(X, self.le_.classes_, invert=True)

        if is_object_array(X):
            missing_mask = [x is np.nan for x in X]
            unseen_mask = np.bitwise_xor(unseen_mask, missing_mask)

            X = encode_with_masks(X,
                                  self.le_,
                                  self.default_unseen_,
                                  unseen_mask,
                                  self.default_missing_,
                                  missing_mask)

            X = correct_dtype(X,
                              self.default_unseen_,
                              unseen_mask,
                              self.default_missing_,
                              missing_mask)

        elif is_float_array(X):
            missing_mask = np.isnan(X)
            unseen_mask = np.bitwise_xor(unseen_mask, missing_mask)

            X = encode_with_masks(X,
                                  self.le_,
                                  self.default_unseen_,
                                  unseen_mask,
                                  self.default_missing_,
                                  missing_mask)
        else:
            X = self.le_.transform(X)

        return X

def is_object_array(X):
    return X.dtype.type is np.object_

def is_float_array(X):
    return X.dtype.type in [np.float16, np.float32, np.float64]

def correct_dtype(X, default_unseen, unseen_mask, default_missing, missing_mask):

    if (default_unseen is np.nan and np.any(unseen_mask)) or \
       (default_missing is np.nan and np.any(missing_mask)):
        return X.astype('float')
    else:
        return X.astype('int')

def encode_with_masks(X, le, default_unseen, unseen_mask, default_missing, missing_mask):
    encode_mask = np.invert(unseen_mask | missing_mask)
    X[encode_mask] = le.transform(X[encode_mask])
    X[unseen_mask] = default_unseen
    X[missing_mask] = default_missing
    return X
