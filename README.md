# extrakit-learn

[![PyPI version](https://badge.fury.io/py/xklearn.svg)](https://pypi.python.org/pypi/xklearn/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/simon-larsson/extrakit-learn/blob/master/LICENSE)

Machine learnings components built to extend scikit-learn. All components use scikit's [object API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) to work interchangably with scikit components. It is mostly a collection of tools that have been useful for [Kaggle](https://www.kaggle.com) competitions. extrakit-learn is in no way affiliated with scikit-learn in anyway, just inspired by it.

## Installation

    pip install xklearn

## Components
- **CategoryEncoder** - Like scikit's LabelEncoder but supports NaNs and missing values.
- **CountEncoder** - Categorical feature engineering based on value counts.
- **TargetEncoder** - Categorical feature engineering based on target means.
- **MultiColumnEncoder** - Apply a column encoder to multiple columns
- **FoldEstimator** - K-fold cross validation meta estimator.
- **FoldLGBM** - K-fold cross validation meta LGBM.
- **StackingClassifier** - Stack an ensemble of classifiers with a meta classifier.
- **StackingRegressor** - Stack an ensemble of regressors with a meta regressor.

### Hierachy
    xklearn
    |
    ├── preprocessing
    │   ├── CountEncoder      
    │   └── TargetEncoder
    |
    └── models
        ├── FoldEstimator
        ├── FoldLGBM
        ├── StackingClassifier
        └── StackingRegressor

##### Example

    from xklearn.models import FoldEstimator

### CategoryEncoder
Wraps scikit's LabelEncoder, allowing missing and unseen values to be handled.

#### Arguments
`unseen` - Strategy for handling unseen values. See replacement strategies below for options.

`missing` - Strategy for handling missing values. See replacement strategies below for options.

##### Replacement strategies

`'encode'` - Replace value with -1.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example:
```python
from xklearn.preprocessing import CategoryEncoder

ce = CategoryEncoder(unseen='nan', missing='nan')
X[:, 0] = ce.fit_transform(X[:, 0])
```

### CountEncoder
Replaces categorical values with their respective value count during training. Classes with a count of one and previously unseen classes during prediction are encoded as either one or NaN.

#### Arguments
`unseen` - Strategy for handling unseen values. See replacement strategies below for options.

`missing` - Strategy for handling missing values. See replacement strategies below for options.

##### Replacement strategies

`'one'` - Replace value with 1.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example:
```python
from xklearn.preprocessing import CountEncoder

ce = CountEncoder(unseen='one')
X[:, 0] = ce.fit_transform(X[:, 0])
```

### TargetEncoder
Performs target mean encoding of categorical features with optional smoothing.

#### Arguments
`smoothing` - Smoothing weight.

`unseen` - Strategy for handling unseen values. See replacement strategies below for options.

`missing` - Strategy for handling missing values. See replacement strategies below for options.

##### Replacement strategies

`'one'` - Replace value with 1.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example:

```python
from xklearn.preprocessing import TargetEncoder

te = TargetEncoder(smoothing=10)
X[:, 0] = te.fit_transform(X[:, 0], y)
```

### MultiColumnEncoder
Applies a column encoder over multiple columns.

#### Arguments
`enc` - Base encoder that will be applied to selected columns

`columns` - Column selection, either bool-mask, indices or None (default=None).

#### Example:
```python
from xklearn.preprocessing import CountEncoder
from xklearn.preprocessing import MultiColumnEncoder

columns = [1, 3, 4]
enc = CountEncoder()

mce = MultiColumnEncoder(enc, columns)
X = mce.fit_transform(X)
```

### FoldEstimator
Meta estimator that performs cross validation over k folds. Can optionally be used as a stacked ensemble of k estimators.

#### Arguments
`est` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`ensemble` - Flag indicting post fit behaviour. True will make it a stacked ensemble, False will do a full refit on the full data.

`verbose` - Flag for printing intermediate scores during fit.

#### Example:
```python
from xklearn.models import FoldEstimator

base = RandomForestRegressor(n_estimators=10)
fold = KFold(n_splits=5)

est = FoldEstimator(base, fold=fold, metric=mean_squared_error, verbose=1)

est.fit(X_train, y_train)
est.predict(X_test)
```

### FoldLGBM
Meta estimator that performs cross validation over k folds on a LightGBM estimator. Can optionally be used as a ensemble of k estimators.

#### Arguments
`lgbm` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`fit_params` - Dictionary of parameter that should be fed to the fit method.

`ensemble` - Flag indicting post fit behaviour. True will make it a stacked ensemble, False will do a full refit on the full data.

`refit_params` - Dictionary of parameter that should be fed to the refit if `ensemble=False`.

`verbose` - Flag for printing intermediate scores during fit.

#### Example:
```python
from xklearn.models import FoldLGBM

base = LGBMClassifier(n_estimators=1000)
fold = KFold(n_splits=5)
fit_params = {'eval_metric': 'auc',
              'early_stopping_rounds': 50,
              'verbose': 0}
              
fold_lgbm = FoldLGBM(base, 
                     fold=fold, 
                     metric=roc_auc_score,
                     fit_params=fit_params,
                     verbose=1)
               
fold_lgbm.fit(X_train, y_train)
fold_lgbm.predict(X_test)
```

### StackingClassifier
Ensemble classifier that stacks an ensemble of classifiers by using their outputs as input features.

#### Arguments
`clfs` - List of ensemble of classifiers.

`meta_clf` - Meta classifier that stacks the predictions of the ensemble.

`keep_features` - Flag to train the meta classifier on the original features too.

`refit` - Flag to retrain the ensemble of classifiers.

#### Example:
```python
from xklearn.models import StackingClassifier

meta_clf = RidgeClassifier()
ensemble = [RandomForestClassifier(), KNeighborsClassifier(), SVC()]

stack_clf = StackingClassifier(clfs=ensemble, meta_clf=meta_clf, refit=True)

stack_clf.fit(X_train, y_train)
y_ = stack_clf.predict(X_test)
```

### StackingRegressor
Ensemble regressor that stacks an ensemble of regressors by using their outputs as input features.

#### Arguments
`regs` - List of ensemble of regressors.

`meta_reg` - Meta regressor that stacks the predictions of the ensemble.

`keep_features` - Flag to train the meta regressor on the original features too.

`refit` - Flag to retrain the ensemble of regressors.

#### Example:
```python
from xklearn.models import StackingRegressor

meta_reg = RidgeRegressor()
ensemble = [RandomForestRegressor(), KNeighborsRegressor(), SVR()]

stack_reg = StackingRegressor(regs=ensemble, meta_reg=meta_reg, refit=True)

stack_reg.fit(X_train, y_train)
y_ = stack_reg.predict(X_test)
```
