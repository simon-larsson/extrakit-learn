# extrakit-learn

Machine learnings components built to extend scikit-learn. All components use scikit's [object API](https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects) to work interchangably with scikit components.

### TargetEncoder
Performs target mean encoding of categorical features. Performs optional smoothing of means with:

$$means_{smooth} = \frac {counts_{class} \cdot means_{class} + smoothing \cdot mean_{global}}{smoothing \cdot counts_{class}}$$

#### Arguments
```smoothing``` - Smoothing weight.
#### Example:

```python
te = TargetEncoder(smoothing=10)
X[0] = te.fit_transform(X[0], y)
```

### CountEncoder
Replaces categorical values with their respective value count during training. Classes with a count of one and previously unseen classes during prediction are encoded as either one or nan.

#### Arguments
```one_to_nan``` - Flag for using nans instead of ones for unseen/single classes.

#### Example:
```python
ce = TargetEncoder(one_to_nan=True)
X[0] = ce.fit_transform(X[0], y)
```

### StackingClassifier
Ensemble classifier that stacks an ensemble of classifiers by using their outputs as input features.

#### Arguments
```clfs``` - List of ensemble of classifiers
```meta_clf``` - Meta classifier that stacks the predictions 
```keep_features``` - Flag to train the meta classifier on the original features too
```refit``` - Flag to retrain the ensemble of classifiers

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
```regs``` - List of ensemble of regressors
```meta_reg``` - Meta regressor that stacks the predictions 
```keep_features``` - Flag to train the meta regressor on the original features too
```refit``` - Flag to retrain the ensemble of regressors

#### Example:
```python
meta_reg = RidgeRegressor()
ensemble = [RandomForestRegressor(), KNeighborsRegressor(), SVR()]
stack_reg = StackingRegressor(regs=ensemble, meta_reg=meta_reg, refit=True)
stack_reg.fit(X_train, y_train)
y_ = stack_reg.predict(X_test)
```