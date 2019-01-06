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
X_valid = sc.transform(X_valid)

# Test train Split
validation_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=0)

# Define the ANN.
from keras.models import Sequential
from keras.layers import Dense, Dropout

def model():
    model = Sequential()
    model.add(Dense(128, input_dim = 172, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(7,  activation='sigmoid'))
#    optimizer = Adam(lr=0.001, clipnorm=0.9)
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = model()
print(model.summary())

history = model.fit(X_train, y_train, epochs = 125, batch_size= 32, validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

model.save('ann-emotion-recognition.h5')

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

# ANN did not work, try Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Use confusion matrix to analyse
y_pred = classifier.predict(X_valid)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred) # 73%
print("Accuracy- {}".format(accuracy(cm)))

import pickle
filename = 'logistic-emotion-recognition.sav'
pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)

# Try Regression Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=24, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Use confusion matrix to analyse
y_pred = classifier.predict(X_valid)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred) # Around 85%
print("Accuracy- {}".format(accuracy(cm)))

# Try Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
# Use confusion matrix to analyse
y_pred = classifier.predict(X_valid)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred) # Around 86%
print("Accuracy- {}".format(accuracy(cm)))

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Use confusion matrix to analyse
y_pred = classifier.predict(X_valid)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred) # 59%
print("Accuracy- {}".format(accuracy(cm)))

# ---- Test on a real world image
import cv2
# Load the required stuff for landmark detection.
from imutils import face_utils
import dlib
from landmarks_calculation import left, right

image_path = '/home/myidispg/My Files/Machine-Learning-Projects/Emotion-Dataset/cohn-kanade-images/S042/001/S042_001_00000019.png'
#image_path = '../Google Photos/2018/00100sPORTRAIT_00100_BURST20180619133945605_COVER.jpg'
#image_path = 'angry-image.jpeg'

p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# t display the image with rectangle
win = dlib.image_window()

#Find face box in the image
image = cv2.imread(image_path, 0)
image = cv2.resize(image, (640, 490))

rects = face_detector(image, 1)

print("rects- {}".format(rects))
print("left- {}, right- {}, top- {}, bottom- {}".format(rects[0].left(), rects[0].right(), rects[0].top(), rects[0].bottom()))

# Visualize the drawn rectangle
win.clear_overlay()
win.set_image(image)
win.add_overlay(rects)
dlib.hit_enter_to_continue()


for (i, rect) in enumerate(rects):
        # Find landmark
        shape = landmark_predictor(image, rect)
        # Convert to numpy array
        shape = face_utils.shape_to_np(shape)
        
# Required landmark calculations for a single image        
x_coords = []
y_coords = []
for i in range(len(left)):
    x_coords.append(shape[left[i]][0] - shape[right[i]][0])
    y_coords.append(shape[left[i]][1] - shape[right[i]][1])

x_coords = np.asarray(x_coords)
y_coords = np.asarray(y_coords)
distance_between = np.sqrt(np.square(x_coords) + np.square(y_coords))

predict = classifier.predict(distance_between.reshape(1, -1))
    
