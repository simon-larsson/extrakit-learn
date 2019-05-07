'''
-------------------------------------------------------
    Label Encoder - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import column_or_1d, check_is_fitted
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
import numpy as np

class LabelEncoder(BaseEstimator, TransformerMixin):
    ''' Label Encoder

    Extends scikit's labels encoder by allowing nan values.

    Parameters
    ----------
    None
    '''
    def __init__(self):
        self.le = SkLabelEncoder()

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
        
        if X.dtype == np.object:
            X = X.astype('str')
            self.le.fit(X[np.where(X!= 'nan')])
        else:
            self.le.fit(X[np.isfinite(X)])

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

        if X.dtype == np.object:
            X_transformed = np.zeros(X.shape, dtype='float64')
            # Hacky to find `NaN`, convert to string and match with 'nan'
            X = X.astype('str')
            nan_mask = np.where(X == 'nan')
            not_nan_mask = np.where(X != 'nan')
            X_transformed[nan_mask] = np.nan
            X_transformed[not_nan_mask] = self.le.transform(X[not_nan_mask])

            return X_transformed
        else:
            X[np.isfinite(X)] = self.le.transform(X[np.isfinite(X)])
            return X
