import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# Forge: 2 class Classification dataset
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
# print("X.shape: {}".format(X.shape))
print(X)
print(y)

# Implement knn
# Split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Import Knn classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# fit using training set
clf.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3,
                     p=2, weights='uniform')
clf.predict(X_test)
# 0.857... Accuracy
# print(X_test)
# print(clf.score(X_test, y_test))

# # mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.title("forge_three_neighbors")
# plt.show()

# Decision boundaries for 1, 3 and 9 neighbors
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
    ax.set_title("%d neighbor(s)" % n_neighbors)

plt.show()
