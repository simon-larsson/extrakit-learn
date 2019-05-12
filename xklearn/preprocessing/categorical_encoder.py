'''
-------------------------------------------------------
    Categorical Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from ..preprocessing.util import is_float_array, \
is_object_array, check_error_strat, correct_dtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted
from sklearn.preprocessing import LabelEncoder
import numpy as np

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    ''' Categorical Encoder

    Extends scikit's labels encoder by allowing to encode missing
    and previously unseen values.

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
        else:
            self.default_unseen_ = None

        if missing == 'encode':
            self.default_missing_ = -1
        elif missing == 'nan':
            self.default_missing_ = np.nan
        else:
            self.default_missing_ = None

        self.unseen = unseen
        self.missing = missing

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

            missing_mask = [x is np.nan for x in X]
            check_error_strat(missing_mask, self.missing, 'missing')
            self.le_.fit(X[np.invert(missing_mask)])

        elif is_float_array(X):

            missing_mask = np.isnan(X)
            check_error_strat(missing_mask, self.missing, 'missing')
            self.le_.fit(X[np.invert(missing_mask)])

        else:
            self.le_.fit(X)

        self.classes_ = self.le_.classes_

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
            The count values. An array of int/float.
        '''


        X = column_or_1d(X.copy(), warn=True)
        check_is_fitted(self, 'n_classes')

        if is_object_array(X):
            missing_mask = [x is np.nan for x in X]
            unseen_mask = np.bitwise_xor(np.isin(X, self.le_.classes_, invert=True), 
                                        missing_mask)
            
            check_error_strat(missing_mask, self.missing, 'missing')
            check_error_strat(unseen_mask, self.unseen, 'unseen')

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
            unseen_mask = np.bitwise_xor(np.isin(X, self.le_.classes_, invert=True), 
                                        missing_mask)

            check_error_strat(missing_mask, self.missing, 'missing')
            check_error_strat(unseen_mask, self.unseen, 'unseen')

            X = encode_with_masks(X,
                                  self.le_,
                                  self.default_unseen_,
                                  unseen_mask,
                                  self.default_missing_,
                                  missing_mask)

            X = correct_dtype(X,
                              self.default_unseen_,
                              [],
                              self.default_missing_,
                              missing_mask)            

        else:
            X = self.le_.transform(X)

        return X

    def fit_transform(self, X, y=None):
        ''' Combined fit and transform

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X : array-like, shape (n_samples,)
            The count values. An array of int/float.
        '''

        X = column_or_1d(X.copy(), warn=True)

        if is_object_array(X):

            missing_mask = [x is np.nan for x in X]
            check_error_strat(missing_mask, self.missing, 'missing')
            encode_mask = np.invert(missing_mask)
            X[encode_mask] = self.le_.fit_transform(X[encode_mask])
            X[missing_mask] = self.default_missing_

            X = correct_dtype(X,
                              None,
                              [],
                              self.default_missing_,
                              missing_mask) 

        elif is_float_array(X):

            missing_mask = np.isnan(X)
            check_error_strat(missing_mask, self.missing, 'missing')
            encode_mask = np.invert(missing_mask)
            X[encode_mask] = self.le_.fit_transform(X[encode_mask])
            X[missing_mask] = self.default_missing_
            
            X = correct_dtype(X,
                              None,
                              [],
                              self.default_missing_,
                              missing_mask) 

        else:
            X = self.le_.transform(X)

        self.classes_ = self.le_.classes_

        return X

def encode_with_masks(X, le, default_unseen, unseen_mask, default_missing, missing_mask):
    encode_mask = np.invert(unseen_mask | missing_mask)

    X[encode_mask] = le.transform(X[encode_mask])
    X[unseen_mask] = default_unseen
    X[missing_mask] = default_missing
    return X
