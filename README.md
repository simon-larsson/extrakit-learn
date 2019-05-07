# extrakit-learn

Machine learnings components built to extend scikit-learn. All components use scikit's [object API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) to work interchangably with scikit components.

## Components
- **TargetEncoder** - Categorical feature engineering based on target means.
- **CountEncoder** - Categorical feature engineering based on value counts.
- **FoldEstimator** - K-fold cross validation meta estimator.
- **FoldLGBM** - K-fold cross validation meta LGBM.
- **StackingClassifier** - Stack an ensemble of classifiers with a meta classifier.
- **StackingRegressor** - Stack an ensemble of regressors with a meta regressor.

### TargetEncoder
Performs target mean encoding of categorical features with optional smoothing.

#### Arguments
`smoothing` - Smoothing weight.

`unseen` - Strategy for handling unseen values. Se replacement strategies below for options.

`missing` - Strategy for handling missing values. Se replacement strategies below for options.

##### Replacement strategies

`'one'` - Replace value with 1.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example:

```python
te = TargetEncoder(smoothing=10)
X[0] = te.fit_transform(X[0], y)
```

### CountEncoder
Replaces categorical values with their respective value count during training. Classes with a count of one and previously unseen classes during prediction are encoded as either one or nan.

#### Arguments
`unseen` - Strategy for handling unseen values. Se replacement strategies below for options.

`missing` - Strategy for handling missing values. Se replacement strategies below for options.

##### Replacement strategies

`'one'` - Replace value with 1.

`'nan'` - Replace value with np.nan.

`'error'` - Raise ValueError.

#### Example:
```python
ce = TargetEncoder(one_to_nan=True)
X[0] = ce.fit_transform(X[0], y)
```

### FoldEstimator
Meta estimator that performs cross validation over k folds. Can optionally be used as a stacked ensemble of k estimators.

#### Arguments
`est` - Base estimator.

`fold` - Folding cross validation object, i.e KFold and StratifedKfold.

`metric` - Evaluation metric.

`ensemble` - Flag indicting that the estimator should be a stacked ensemble after fit.

`verbose` - Flag for printing intermediate scores during fit.

#### Example:
```python
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

`ensemble` - Flag indicting that the estimator should be a stacked ensemble after fit.

`verbose` - Flag for printing intermediate scores during fit.

#### Example:
```python
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
meta_reg = RidgeRegressor()
ensemble = [RandomForestRegressor(), KNeighborsRegressor(), SVR()]

stack_reg = StackingRegressor(regs=ensemble, meta_reg=meta_reg, refit=True)

stack_reg.fit(X_train, y_train)
y_ = stack_reg.predict(X_test)
```
