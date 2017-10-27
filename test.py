from sklearn.datasets import load_iris
iris = load_iris()

# iris.keys()
# dict_keys(['DESCR', 'data', 'target_names', 'feature_names', 'target'])

# Value of key DESCR is a short description of the data set
print(iris['DESCR'][:150] + "\n...")
# print(iris.keys())
