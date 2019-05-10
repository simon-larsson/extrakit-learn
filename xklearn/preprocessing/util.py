'''
-------------------------------------------------------
    Preprocessing Utils - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

import numpy as np

def is_object_array(X):
    return X.dtype.type is np.object_

def is_float_array(X):
    return X.dtype.type in [np.float16, np.float32, np.float64]

def check_error_strat(mask, strat, name):
    
    if strat == 'error' and np.any(mask):
        
        indices = list(np.where(mask)[0])
        
        raise ValueError('Error value found at index {}. Aborting '
                         'according to {} strategy'.format(indices, name))

def correct_dtype(X, default_unseen, unseen_mask, default_missing, missing_mask):

    if (default_unseen is np.nan and np.any(unseen_mask)) or \
       (default_missing is np.nan and np.any(missing_mask)):
        return X.astype('float')
    else:
        return X.astype('int')

