import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# 2 class Classification dataset
# generate dataset
# X, y = mglearn.datasets.make_forge()
# # plot dataset
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X.shape: {}".format(X.shape))
# plt.show()

# Regression dataset. Use low dimensional datasets for simplicity of learning
# X, y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()

# Wisconsin Breast Cancer dataset which records clinical measurements of breast cancer tumors
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
# 569 data points, 30 features each
print(cancer.data.shape)
