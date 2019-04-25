'''
-------------------------------------------------------
    Prediction - extrakit-learn

    Author: Simon Larsson <larssonsimon0@gmail.com>

    License: MIT
-------------------------------------------------------
'''

from .stacking_classifier import StackingClassifier
from .stacking_regressor import StackingRegressor

__all__ = ['StackingClassifier',
           'StackingRegressor']
