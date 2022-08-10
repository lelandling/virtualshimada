#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import copy as cp
import seaborn as sns
from typing import Tuple
import csv

import keygen
	# read key and sanitize

def main() :
	key = keygen.makekey()

	# print(key)
	#Creating Dataset and including the first row by setting no header as input
	rawData = pd.read_csv('test5-1.csv', header = None)

	#Renaming the columns
	rawData.columns = rawData.head(1).values.flatten()
	rawData = rawData.iloc[1: , :]
	rawData = rawData.dropna(axis='columns')

	print(rawData.head())
	# print(rawData.columns)
	types = []
	# calculate 
	for patient in (rawData['pid']):
		# print(patient[0:6])
		ptype = key[patient[0:6]]
		# pclass = ptype.iloc[0]
		types.append(ptype)

	# print(types)

	#add outcome row and set to dataset var
	ds = rawData.assign(outcomes = types)
	factor = pd.factorize(ds['outcomes'])
	# print(factor)
	ds.outcomes = factor[0]
	definitions = factor[1]
	print(definitions)

	#Splitting the data into independent and dependent variables
	X = ds.iloc[:,1:145].values
	y = ds.iloc[:,145].values
	# print('The independent features set: ')
	# print(X[:7,:])
	# print('The dependent variable: ')
	# print(y[:146])

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1, random_state = 0)

	scaler = StandardScaler()
	# X_train = scaler.fit_transform(X)
	# X_test = scaler.transform(X_test)

	cv = LeaveOneOut()
	y_true, y_pred = list(), list()
	yset = []
	i = 1
	for train_ix, test_ix in cv.split(X):
		# split data
		# print("TRAIN:", train_ix, "TEST:", test_ix)

		X_train, X_test = X[train_ix, :], X[test_ix, :]
		y_train, y_test = y[train_ix], y[test_ix]
		# fit model
		model = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
		model.fit(X_train, y_train)
		# evaluate model
		yhat = model.predict(X_test)
		# store
		y_true.append(y_test[0])
		y_pred.append(yhat[0])
	print("set %d: " % i)
	print(y_true)
	print(y_pred)
		
	acc = accuracy_score(y_true, y_pred)
	print('Accuracy: %.3f' % acc)

	reversefactor = dict(zip(range(3),definitions))
	y_test = np.vectorize(reversefactor.get)(y)
	y_pred = np.vectorize(reversefactor.get)(y_pred)
	print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))

	fpred = pd.factorize(y_pred)
	# print("prediction key: ")
	# print(fpred[1])
	prediction = list(zip(list(rawData["pid"]),y_true, fpred[0]))

	# for pred in prediction:
	# 	print(pred)
	# print(fpred[1])
	return prediction, fpred[1]


if __name__ == "__main__" :
	[pred, key] = main()
	key = list(zip(range(3), key))
	print(pred[1][1])
	with open("output1.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow([y for x in [["key: "], key] for y in x])
		writer.writerow(["slide-region, actual species, predicted species"])
		writer.writerows(pred)

	with open("misclassified.csv", "w", newline="") as f:
		writer = csv.writer(f)
		for i in range(len(pred)):
			if pred[i][1] != pred[i][2]:
				writer.writerow(pred[i])