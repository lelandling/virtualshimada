#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
from scipy.io import loadmat
import h5py
import tables
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


#Creating Dataset and including the first row by setting no header as input
rawData = pd.read_csv('test3.csv', header = None)
#Renaming the columns
# print(rawData.head(1).values.flatten())
# feature_names = rawData.head(1).values.flatten()
rawData.columns = rawData.head(1).values.flatten()
rawData = rawData.iloc[1: , :]

rawData = rawData.dropna(axis='columns')
print('Shape of the dataset: ' + str(rawData.shape))
print(rawData.columns)

#add outcome row and set to dataset var
ds = rawData.assign(outcomes = ['p','p','p','p','p','u','p','p','p','p','p','p','d','p','p','p','p','u','d'])


factor = pd.factorize(ds['outcomes'])
ds.outcomes = factor[0]
definitions = factor[1]
print(ds.outcomes.head())
print(definitions)

#Splitting the data into independent and dependent variables
x = ds.iloc[:,1:145].values
y = ds.iloc[:,145].values
# print('The independent features set: ')
# print(x[:7,:])
# print('The dependent variable: ')
# print(y[:146])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(x)
# X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(x, y)

# Predicting the Test set results
y_pred = classifier.predict(x)
#Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
