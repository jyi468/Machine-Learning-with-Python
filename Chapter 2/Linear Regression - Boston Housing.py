import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# Boston Housing dataset for regression.
# Will use to predict median value of homes in several Boston neighborhoods in the 1970's
# (506, 13)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston = load_boston()
# Load derived dataset
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)

# We see training set is very accurate but test set is much lower.
# This is likely due to overfitting due to the number of features in the training set
# We need a model to control complexity
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Ridge training score lower but test set score is higher
print("\nTraining set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("\nTraining set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("\nTraining set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))