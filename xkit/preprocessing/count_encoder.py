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
    None
    '''

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

        self.lut = np.concatenate([self.classes_.reshape(-1, 1),
                                   self.counts_.reshape(-1, 1)],
                                  axis=1)

        if self.classes_.shape[0] != np.unique(self.counts_).shape[0]:
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
            The count values. An array of int.
        '''

        # Check is fit had been called
        check_is_fitted(self, 'classes_')

        # Input validation
        X = column_or_1d(X, warn=True)

        unseen_mask = list(np.where(np.isin(X, self.classes_, invert=True))[0])

        if unseen_mask:
            warn('Unseen value at index {} will be encoded to 1'.format(unseen_mask[0]))

        _, indices = np.unique(X, return_inverse=True)

        X = np.take(self.lut[:, 1], \
                    np.take(np.searchsorted(self.lut[:, 0], self.classes_), indices))

        X[unseen_mask] = 1

        return X
