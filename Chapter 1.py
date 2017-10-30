# Individual items are called SAMPLES in machine learning and their properties are called FEATURES
# Convention in scikit-learn: shape of data array is num of samples (rows) times the num of features (columns)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
# fig, ax = plt.subplots(3, 3, figsize=(15, 15))
# plt.suptitle("iris_pairplot")
# for i in range(3):
#     for j in range(3):
#         ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
#         ax[i, j].set_xticks(())
#         ax[i, j].set_yticks(())
#         if i == 2:
#             ax[i, j].set_xlabel(iris['feature_names'][j])
#         if j == 0:
#             ax[i, j].set_ylabel(iris['feature_names'][i + 1])
#         if j > i:
#             ax[i, j].set_visible(False)
# plt.show()

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris.feature_names)
# create a scatter matrix from the dataframe, color by y_train. This will modify the plt object we imported
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=60, alpha=.8)
# plt.show()

# Plots show relatively well separated classes. We will use k-nearest neighbor classifier to separate out
# All machine learning models in scikit-learn are in their own class, the Estimator class.

# Use KNeighborsClassifier class and use an instance
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# Build the model on the training set using fit
knn.fit(X_train, y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1,
#                      p=2, weights='uniform')

# Create test data with one flower
X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)

# Predict with knn. Our prediction shows us 0, which means it's Setosa
prediction = knn.predict(X_new)
print(iris.target_names[prediction])

# Evaluate the model. Use predictions for iris using test data and compare it against its label (the known species)
# Compute accuracy, fraction of flowers for which the right species was predicted
y_pred = knn.predict(X_test)
print(X_test)
# Array of boolean values. When mean is calculated here, it will treat true as 1 and false as 0
# print(y_pred == y_test)
# print(np.mean(y_pred == y_test))

# Can also use knn.score to compute the test set accuracy
print(knn.score(X_test, y_test))

print(iris)
