'''
-------------------------------------------------------
    Preprocessing Utils - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

import numpy as np

def is_object_array(X):
    ''' Check if an array is an object array

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        truth :  bool
    '''
    return X.dtype.type is np.object_

def is_float_array(X):
    ''' Check if an array is a float array

        One of [float16, float32, float64]

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        truth :  bool
    '''
    return X.dtype.type in [np.float16, np.float32, np.float64]

def check_error_strat(mask, strat, name):
    ''' Check error strategies for encoders that can allow missing and
        unseen values.

        Parameters
        ----------
        masks : array-like, indices or bool mask relevant to strategy

        strat : strategy for handling occurances in mask, string

        name : name of error check instance, string
    '''

    if strat == 'error' and np.any(mask):

        indices = list(np.where(mask)[0])

        raise ValueError('Error value found at index {}. Aborting '
                         'according to {} strategy.'.format(indices, name))
