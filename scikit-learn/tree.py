
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.externals.six import StringIO 
import pydot 
from sklearn import tree

"""
train_features = np.loadtxt('prototask_train_features.csv', delimiter=',')
train_targets = np.loadtxt('prototask_train_targets.csv')

test_features = np.loadtxt('prototask_test_features.csv', delimiter=',')
test_targets = np.loadtxt('prototask_test_targets.csv')
"""

train_features = np.loadtxt('big_data_train_features.csv', delimiter=',')
train_targets = np.loadtxt('big_data_train_targets.csv')

test_features = np.loadtxt('big_data_test_features.csv', delimiter=',')
test_targets = np.loadtxt('big_data_test_targets.csv')

""" 
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
"""

X_train, y_train = train_features, train_targets
X_test, y_test = test_features, test_targets

# Fit regression model
clf_1 = DecisionTreeRegressor(max_depth=2)
clf_1.fit(X_train, y_train)


dot_data = StringIO() 
tree.export_graphviz(clf_1, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("big_tree_d2.pdf") 

# Predict
print("train MSE tree: %.4f" % mean_squared_error(y_train, clf_1.predict(X_train)))
print("test MSE tree: %.4f" % mean_squared_error(y_test, clf_1.predict(X_test)))

params = {'n_estimators': 10, 'max_depth': 2, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

print("train MSE boosting: %.4f" % mean_squared_error(y_train, clf.predict(X_train)))
print("test MSE boosting: %.4f" % mean_squared_error(y_test, clf.predict(X_test)))

