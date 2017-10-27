# Individual items are called SAMPLES in machine learning and their properties are called FEATURES
# Convention in scikit-learn: shape of data array is num of samples (rows) times the num of features (columns)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import load_iris

iris = load_iris()

# iris.keys()
# dict_keys(['DESCR', 'data', 'target_names', 'feature_names', 'target'])

# Value of key DESCR is a short description of the data set
# print(iris.keys())
# print(iris['DESCR'][:150] + "\n...")

# Species of Flowers we want to predict
# print(iris['target_names'])

# Description of each feature (sepal length, sepal width, petal length, petal width)
# print(iris['feature_names'])

# Data for features above. Each row corresponds to a flower, each column represents 4 measurements for each flower
# print(iris['data'][:5])

# Measurements for 150 different flowers (150, 4)
# print(iris['data'].shape)

# Target is species of each flowers measured. Matches 'target_names'.
# 0 is Setosa, 1 is Versicolor, 2 is Virginica;
# print(iris['target'])

# train_test_split will shuffle dataset, extracts 75% of rows as training set, 25% are declared test set
# scikit-learn: data usually denoted with capital X, labels denoted by y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
# print(y_train)

# Visualize data. Scatter plot is a common way. You can only display 2 or 3 features usually, so you can do a pair plot
# Which has pairs of all the features
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i + 1])
        if j > i:
            ax[i, j].set_visible(False)
plt.show()

# Plots show relatively well separated classes. We will use k-nearest neighbor classifier to separate out