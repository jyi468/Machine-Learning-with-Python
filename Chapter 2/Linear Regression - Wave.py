import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# mglearn.plots.plot_linear_regression_wave()
# plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
# coef_ has coefficients/weights (w) and intercept (b) is stored in intercept_ attribute
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Test Linear Regression score
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Training set score is 0.67 and test set score is 0.66
