import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train, y_train)
# # 100 percent accurate training set because all leaves are pure
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# # Test set accuracy slightly worse than linear models, which were around 95% accurate
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# We pre-prune by setting max_depth to 4 to avoid overfitting on training set and better generalization
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
# Get better test set but worse training set (okay)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

print("Feature importances:\n{}".format(tree.feature_importances_))

# Plot feature importances
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


plot_feature_importances_cancer(tree)
plt.show()
