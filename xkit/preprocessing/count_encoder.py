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
    one_to_nan : Flag for replacing one counts with np.nan, bool
    '''
    def __init__(self, one_to_nan=False):
        self.one_to_nan = one_to_nan

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

        self.classes_, self.counts_ = np.unique(X, return_counts=True)

        if self.classes_.shape[0] != np.unique(self.counts_).shape[0]:
            warn('Duplicate count encoding for muliple classes. This will '
                 'make two or more categories indistinguishable.')

        self.classes_ = np.append(self.classes_, [-1])
        self.counts_ = np.append(self.counts_, [1])
        self.lut_ = np.hstack([self.classes_.reshape(-1, 1), self.counts_.reshape(-1, 1)])

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

        _, indices = np.unique(X, return_inverse=True)

        unseen_mask = list(np.where(np.isin(X, self.classes_, invert=True))[0])

        if unseen_mask:
            warn('Unseen or nan value at index {} will be encoded to default value'\
                .format(unseen_mask))

        # Perform replacement with lookup
        X = np.take(self.lut_[:, 1], \
                    np.take(np.searchsorted(self.lut_[:, 0], self.classes_), indices))

        if self.one_to_nan:
            X = X.astype('float64')
            X[unseen_mask] = np.nan
            X[np.where(X == 1.0)] = np.nan
        else:
            X[unseen_mask] = 1

        return X
