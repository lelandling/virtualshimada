import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

x, y = datasets.load_iris(return_X_y=True)
print(x.shape, y.shape)

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, x, y, cv=5)
print(scores)

from sklearn.model_selection import ShuffleSplit
n_samples = x.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
print(cross_val_score(clf, x, y, cv=cv))
