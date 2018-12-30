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

history = model.fit(X_train, y_train, epochs = 50, batch_size= 32, validation_data=(X_valid, y_valid))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

model.save('ann-emotion-recognition.h5')

# ANN did not work, try Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
#X_train = X_train.reshape(172, 8196)
#y_train = y_train.reshape(7, 8196)
y_train = np.ravel(y_train)
history = classifier.fit(X_train, y_train)

# ---- Test on a real world image
import os
import cv2

casc_directory = 'face-cascades/'

def find_face(image_path):
    # Create the haar cascade
    faceCascade1 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt.xml'))
    faceCascade2 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt2.xml'))
    faceCascade3 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt_tree.xml'))
    faceCascade4 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_default.xml'))
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image using 4 different classifiers
    faces1 = faceCascade1.detectMultiScale(
                gray, 
                scaleFactor=1.2,
                minNeighbors=5
                )
    faces2 = faceCascade2.detectMultiScale(
                gray, 
                scaleFactor=1.2,
                minNeighbors=5
                )
    faces3 = faceCascade3.detectMultiScale(
                gray, 
                scaleFactor=1.2 ,
                minNeighbors=5
                )
    faces4 = faceCascade4.detectMultiScale(
                gray, 
                scaleFactor=1.2,
                minNeighbors=5
                )
    
    # Go over the detected faces of all the classifiers and select one.
    if len(faces1) == 1:
        faces = faces1
    elif len(faces2) == 1:
        faces = faces2
    elif len(faces3) == 1:
        faces = faces3
    elif len(faces4) == 1:
        faces = faces4
    else:
        print('No faces found')
        
    # Return new image
    for (x, y, w, h) in faces:
        new_image = gray[y:y+h+20, x:x+w+20]
    
    return new_image

face_image = find_face(image_path)

cv2.imshow('face', face_image)
cv2.waitKey(0)

# Load the required stuff for landmark detection.
from imutils import face_utils
import dlib
from landmarks_calculation import left, right
import cv2

image_path = '../Google Photos/2018/00000PORTRAIT_00000_BURST20180407170851895.jpg'

p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

#Find face box in the image
image = cv2.imread(image_path, 0)
rect = face_detector(image)
for (i, rect) in enumerate(rect):
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
average_coords = ((x_coords + y_coords)/2).reshape(1, -1)

# Feature scaling
test_x = sc.transform(average_coords)


model = keras.models.load_model('ann-emotion-recognition.h5')
predict = classifier.predict(test_x)
    
