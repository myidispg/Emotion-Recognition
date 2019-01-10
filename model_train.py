#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:46:57 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('extracted_landmarks_data_new.csv')
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, 172]

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# One Hot encoding the categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y_train = y_train.reshape(-1, 1)
y_train = onehotencoder.fit_transform(y_train).toarray()

 Test train Split. REQUIRED ONLY FOR ANN training. For Logistic Regression, validation is used for accuracy.
validation_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=0)
#
## Try an Artificial Neural Network
#from keras.models import Sequential
#from keras.layers import Dense
#
#model = Sequential()
#model.add(Dense(128, input_dim=172, init='uniform', activation='relu'))
#model.add(Dense(64, init='uniform', activation='relu'))
#model.add(Dense(7, init='uniform', activation='sigmoid'))
#
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#print(model.summary())
#
#history = model.fit(X_train, y_train, epochs=100, verbose=2, validation_data=(X_valid, y_valid))
#
#import matplotlib.pyplot as plt
#
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.legend(['training', 'validation'])
#plt.title('Loss')
#plt.xlabel('Epochs')
#
#model.save('model.h5')


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

