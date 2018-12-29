#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:46:57 2018

@author: myidispg
"""

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('extracted_landmarks_data.csv')
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, 172]

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# One Hot encoding the categorical data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
y_train = y_train.reshape(-1, 1)
y_train = onehotencoder.fit_transform(y_train).toarray()

# Test train Split
validation_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

# Define the ANN.
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adagrad
from keras import regularizers

from keras.layers.advanced_activations import LeakyReLU

def model():
    model = Sequential()
    model.add(Dense(128, input_dim = 172, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(7,  activation='sigmoid'))
#    optimizer = Adam(lr=0.001, clipnorm=0.9)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

#def model():
#    model = Sequential()
#    model.add(Dense(128, input_dim = 172))
#    model.add(LeakyReLU(alpha=0.1))
##    model.add(Dropout(0.5))
#    model.add(Dense(64, ))
#    model.add(LeakyReLU(alpha=0.1))
##    model.add(Dropout(0.5))
#    model.add(Dense(32,))
#    model.add(LeakyReLU(alpha=0.1))
##    model.add(Dropout(0.5))
#    model.add(Dense(7,  activation='sigmoid'))
#    optimizer = Adam(lr=0.001, clipnorm=0.9)
#    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    
#    return model

model = model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 30, batch_size= 32, validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')