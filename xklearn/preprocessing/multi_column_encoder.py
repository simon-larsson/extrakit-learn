'''
-------------------------------------------------------
    Multi-Column Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from copy import copy
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MultiColumnEncoder(BaseEstimator, TransformerMixin):
    ''' Multi-Column Encoder

    Use column encoders, such as sklearn's LabelEncoder, on multiple columns.

    Parameters
    ----------
    enc : Base encoder that should be used on columns 

    columns : Indices or mask to select columns for encoding, list-like
              `columns=None` encodes all columns.
    '''

    def __init__(self, enc, columns=None):
        self.enc = enc
        self.columns = columns

    def fit(self, X, y=None):
        ''' Fitting of the transformer

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        '''

        X = check_array(X, accept_sparse=True)

        self.columns = to_column_indices(X, self.columns)

        self.encs_ = []

        if self.columns is None:

            for col in X.T:
                enc = copy(self.enc)
                enc.fit(col)
                self.encs_.append(enc)

        elif len(self.columns) > 0:

            for col in X[:, self.columns].T:            
                enc = copy(self.enc)
                enc.fit(col)
                self.encs_.append(enc)

        self.n_features_ = X.shape[1]
        self.n_encoders_ = len(self.encs_)

        # `fit` should always return `self`
        return self

    def transform(self, X):
        ''' Applying transformation on the data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The class values.

        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The encoded values.
        '''

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'n_encoders_')

        if self.columns is None:

            for i, col in enumerate(X.T):

                enc = self.encs_[i]
                X[:, i] = enc.transform(col)
        else:

            for i, col_idx in enumerate(self.columns):

                enc = self.encs_[i]
                X[:, col_idx] = enc.transform(X[:, col_idx])

        return X

    def fit_transform(self, X, y=None):
        ''' Combined fit and transform

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The class values.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The encoded values.
        '''

        X = check_array(X, accept_sparse=True)

        self.columns = to_column_indices(X, self.columns)

        self.encs_ = []

        if self.columns is None:

            for i, col in enumerate(X.T):

                enc = copy(self.enc)
                X[:, i] = enc.fit_transform(col)
                self.encs_.append(enc)
        else:

            for col_idx in self.columns:

                enc = copy(self.enc)
                X[:, col_idx] = enc.fit_transform(X[:, col_idx])
                self.encs_.append(enc)

        self.n_features_ = X.shape[1]
        self.n_encoders_ = len(self.encs_)

        return X

def to_column_indices(X, columns):

    if columns is None:
        return None
    else:
        columns = np.array(columns).reshape(-1)

        if X.shape[1] == columns.shape[0]:
            return np.where(columns)[0]
        else:
            return columns
