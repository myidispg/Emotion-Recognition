#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:46:57 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('extracted_landmarks_data_new.csv')
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, 172]

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# One Hot encoding the categorical data
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder()
#y_train = y_train.reshape(-1, 1)
#y_train = onehotencoder.fit_transform(y_train).toarray()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)

from sklearn.externals import joblib
scaler_filename = 'min-max-scaler.sav'
joblib.dump(sc, scaler_filename)

# Test train Split
validation_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=0)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

# Train Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_valid)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred) # Around 85%
print("Accuracy- {}".format(accuracy(cm)))

# Save the classifier to disk
joblib.dump(classifier, 'logistic-regression-emotion-recognition.pkl')
import pickle
# Save the classifier
with open('logistic-regression-emotion-recognition.pkl', 'wb')  as file:
    pickle.dump(classifier, file)

# Try Regression Forest
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(max_depth=80, max_features=3, 
#                                    min_samples_leaf=3, min_samples_split=8,
#                                    n_estimators=1000, criterion='entropy',
#                                    random_state=0)
#classifier.fit(X_train, y_train)
#
## Use confusion matrix to analyse
#y_pred = classifier.predict(X_valid)
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_valid, y_pred) # Around 85%
#print("Accuracy- {}".format(accuracy(cm)))
#
## Apply gridsearch to find the best parameters
#from sklearn.model_selection import GridSearchCV
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [80, 90, 100, 110],
#    'max_features': [2, 3],
#    'min_samples_leaf': [3, 4, 5],
#    'min_samples_split': [8, 10, 12],
#    'n_estimators': [100, 200, 300, 1000]
#}
#
#grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, 
#                          cv = 10, n_jobs = -1, verbose = 2)
#
#grid_search.fit(X_train, y_train)
#
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
#
## Save the classifier to disk
#import pickle
## Save the classifier
#with open('random-forest-emotion-recognition.pkl', 'wb')  as file:
#    pickle.dump(classifier, file)
#    
