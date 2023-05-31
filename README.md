# extrakit-learn

[![PyPI version](https://badge.fury.io/py/xklearn.svg)](https://pypi.python.org/pypi/xklearn/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/simon-larsson/extrakit-learn/blob/master/LICENSE)

Machine learnings components built to extend scikit-learn. All components use scikit's [object API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) to work interchangably with scikit components. It is mostly a collection of tools that have been useful for [Kaggle](https://www.kaggle.com) competitions.

## Installation

    pip install xklearn

## Components
- [CategoryEncoder](https://github.com/simon-larsson/extrakit-learn#categoryencoder) - Like scikit's LabelEncoder but supports NaNs and unseen values.
- [CountEncoder](https://github.com/simon-larsson/extrakit-learn#countencoder) - Categorical feature engineering on a column based on value counts.
- [TargetEncoder](https://github.com/simon-larsson/extrakit-learn#targetencoder) - Categorical feature engineering on a column based on target means.
- [MultiColumnEncoder](https://github.com/simon-larsson/extrakit-learn#multicolumnencoder) - Apply a column encoder to multiple columns.
- [FoldEstimator](https://github.com/simon-larsson/extrakit-learn#foldestimator) - K-fold on scikit estimator wrapped into an estimator.
- [FoldLightGBM](https://github.com/simon-larsson/extrakit-learn#foldlightgbm) - K-fold on LGBM wrapped into an estimator.
- [FoldXGBoost](https://github.com/simon-larsson/extrakit-learn#foldxgboost) - K-fold on XGBoost wrapped into an estimator.
- [StackClassifier](https://github.com/simon-larsson/extrakit-learn#stackclassifier) - Stack an ensemble of classifiers with a meta classifier.
- [StackRegressor](https://github.com/simon-larsson/extrakit-learn#stackregressor) - Stack an ensemble of regressors with a meta regressor.
- [compress_dataframe](https://github.com/simon-larsson/extrakit-learn#compress_dataframe) - Reduce memory of a Pandas dataframe.

### Hierachy
    xklearn
    │
    ├── preprocessing
    │   ├── CategoryEncoder
    │   ├── CountEncoder
    │   ├── TargetEncoder      
    │   └── MultiColumnEncoder
    │
    ├── models
    │   ├── FoldEstimator
    │   ├── FoldLightGBM
    |   ├── FoldXGBoost
    |   ├── StackClassifier
    |   └── StackRegressor
    |
    └── utils

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

#### Example
```python
from xklearn.preprocessing import CategoryEncoder
...

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

#### Example
```python
from xklearn.preprocessing import CountEncoder
...

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

`'global'` - Replace value with global target mean.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example

```python
from xklearn.preprocessing import TargetEncoder
...

te = TargetEncoder(smoothing=10)
X[:, 0] = te.fit_transform(X[:, 0], y)
```

### MultiColumnEncoder
Applies a column encoder over multiple columns.

#### Arguments
`enc` - Base encoder that will be applied to selected columns

`columns` - Column selection, either bool-mask, indices or None (default=None).

#### Example
```python
from xklearn.preprocessing import CountEncoder
from xklearn.preprocessing import MultiColumnEncoder
...

columns = [1, 3, 4]
enc = CountEncoder()

mce = MultiColumnEncoder(enc, columns)
X = mce.fit_transform(X)
```

### FoldEstimator
K-fold wrapped into an estimator that performs cross validation over a selected folding method automatically when fit. Can optionally be used as a stacked ensemble of k estimators after fit.

#### Arguments
`est` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`refit_full` - Flag indicting post fit behaviour. True will do a full refit on the full data, False will make it a stacked ensemble trained on the different folds.

`verbose` - Flag for printing fold scores during fit.

#### Example
```python
from xklearn.models import FoldEstimator
...

base = RandomForestRegressor(n_estimators=10)
fold = KFold(n_splits=5)

est = FoldEstimator(base, fold=fold, metric=mean_squared_error, verbose=1)

est.fit(X_train, y_train)
est.predict(X_test)
```
Output:
```
Finished fold 1 with score: 200.8023
Finished fold 2 with score: 261.2365
Finished fold 3 with score: 169.2404
Finished fold 4 with score: 186.7915
Finished fold 5 with score: 205.0894
Finished with a total score of: 204.6813
```

### FoldLightGBM
K-fold wrapped into an estimator that performs cross validation on a LGBM over a selected folding method automatically when fit. Can optionally be used as a stacked ensemble of k estimators after fit.

#### Arguments
`lgbm` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`fit_params` - Dictionary of parameter that should be fed to the fit method.

`refit_full` - Flag indicting post fit behaviour. True will do a full refit on the full data, False will make it a stacked ensemble trained on the different folds.

`refit_params` - Dictionary of parameter that should be fed to the refit if refit_full=False.

`verbose` - Flag for printing fold scores during fit.

#### Example
```python
from xklearn.models import FoldLightGBM
...

base = LGBMClassifier(n_estimators=1000)
fold = KFold(n_splits=5)
fit_params = {'eval_metric': 'auc',
              'early_stopping_rounds': 50,
              'verbose': 0}
              
fold_lgbm = FoldLightGBM(base, 
                         fold=fold, 
                         metric=roc_auc_score,
                         fit_params=fit_params,
                         verbose=1)
               
fold_lgbm.fit(X_train, y_train)
fold_lgbm.predict(X_test)
```
Output:
```
Finished fold 1 with score: 0.9114
Finished fold 2 with score: 0.9265
Finished fold 3 with score: 0.9419
Finished fold 4 with score: 0.9189
Finished fold 5 with score: 0.9152
Finished with a total score of: 0.9225
```

### FoldXGBoost
K-fold wrapped into an estimator that performs cross validation on a XGBoost over a selected folding method automatically when fit. Can optionally be used as a stacked ensemble of k estimators after fit.

#### Arguments
`xgb` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`fit_params` - Dictionary of parameter that should be fed to the fit method.

`refit_full` - Flag indicting post fit behaviour. True will do a full refit on the full data, False will make it a stacked ensemble trained on the different folds.

`refit_params` - Dictionary of parameter that should be fed to the refit if refit_full=False.

`verbose` - Flag for printing fold scores during fit.

#### Example
```python
from xklearn.models import FoldXGBoost
...

base = XGBRegressor(objective="reg:linear", random_state=42)
fold = KFold(n_splits=5)
fit_params = {'eval_metric': 'mse',
              'early_stopping_rounds': 5,
              'verbose': 0}
              
fold_xgb = FoldXGBoost(base, 
                       fold=fold, 
                       metric=mean_squared_error,
                       fit_params=fit_params,
                       verbose=1)
               
fold_xgb.fit(X_train, y_train)
fold_xgb.predict(X_test)
```
Output:
```
Finished fold 1 with score: 3212.8362
Finished fold 2 with score: 2179.7843
Finished fold 3 with score: 2707.8460
Finished fold 4 with score: 2988.6643
Finished fold 5 with score: 3281.4299
Finished with a total score of: 3274.9001
```

### StackClassifier
Ensemble classifier that stacks an ensemble of classifiers by using their outputs as input features.

#### Arguments
`clfs` - List of ensemble of classifiers.

`meta_clf` - Meta classifier that stacks the predictions of the ensemble.

`keep_features` - Flag to train the meta classifier on the original features too.

`refit` - Flag to retrain the ensemble of classifiers during fit.

#### Example
```python
from xklearn.models import StackClassifier
...

meta_clf = RidgeClassifier()
ensemble = [RandomForestClassifier(), KNeighborsClassifier(), SVC()]

stack_clf = StackClassifier(clfs=ensemble, meta_clf=meta_clf, refit=True)

stack_clf.fit(X_train, y_train)
y_ = stack_clf.predict(X_test)
```

### StackRegressor
Ensemble regressor that stacks an ensemble of regressors by using their outputs as input features.

#### Arguments
`regs` - List of ensemble of regressors.

`meta_reg` - Meta regressor that stacks the predictions of the ensemble.

`drop_first` : Drop first class probability to avoid multi-collinearity.

`keep_features` - Flag to train the meta regressor on the original features too.

`refit` - Flag to retrain the ensemble of regressors during fit.

#### Example
```python
from xklearn.models import StackRegressor
...

meta_reg = RidgeRegressor()
ensemble = [RandomForestRegressor(), KNeighborsRegressor(), SVR()]

stack_reg = StackRegressor(regs=ensemble, meta_reg=meta_reg, refit=True)

stack_reg.fit(X_train, y_train)
y_ = stack_reg.predict(X_test)
```

### compress_dataframe
Reduce memory usage of a Pandas dataframe by finding columns that use larger variable types than unnecessary.

#### Arguments
`df` - Dataframe for memory reduction.

`verbose` - Flag for printing result of memory reduction.

#### Example
```python
from xklearn.utils import compress_dataframe
...

train = compress_dataframe(train, verbose=1)
```
Output:
```
Dataframe memory decreased to 169.60 MB (64.6% reduction)
```
