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

# Test train Split
validation_size = 0.2

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size, random_state=0)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

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

# ---- Test on a real world image-------------
import cv2
# Load the required stuff for landmark detection.
from imutils import face_utils
import dlib
from landmarks_calculation import left, right


# Using Haar Cascade for face detection if dlib fails
import os

casc_directory = 'face-cascades/'
def find_face(gray):
    # Create the haar cascade
    faceCascade1 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt.xml'))
    faceCascade2 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt2.xml'))
    faceCascade3 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt_tree.xml'))
    faceCascade4 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_default.xml'))
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
        return (False, 0)
        
    return (True, faces)


#image_path = '/home/myidispg/My Files/Machine-Learning-Projects/Emotion-Dataset/cohn-kanade-images/S042/001/S042_001_00000019.png'
image_path = '../Google Photos/2018/IMG-20180407-WA0005.jpg'
#image_path = 'angry-image.jpeg'

p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# display the image with rectangle
#win = dlib.image_window()

#Find face box in the image
image = cv2.imread(image_path, 0)
image = cv2.resize(image, (640, 490))

rects = face_detector(image)

if len(rects) == 0:
    print('here\n\n\n')
    rects = find_face(image)
    if not rects[0]:
        import sys 
        sys.exit()
    else:
        rects = dlib.rectangle(left = rects[0][0], top=rects[0][1], right=rects[0][2], bottom=rects[0][3])
        print("rects- {}".format(rects))
        print("left- {}, right- {}, top- {}, bottom- {}".format(rects.left(), rects.right(), rects.top(), rects.bottom()))
        shape = landmark_predictor(image, rects)
        # Convert to numpy array
        shape = face_utils.shape_to_np(shape)
else:
    print('2nd here- \n\n\n')
    print("rects- {}".format(rects))
    print("left- {}, right- {}, top- {}, bottom- {}".format(rects[0].left(), rects[0].right(), rects[0].top(), rects[0].bottom()))
    rects = dlib.rectangle(left = rects[0].left(), top=rects[0].top(), right=rects[0].right(), bottom= rects[0].bottom())
    # Find landmark
    shape = landmark_predictor(image, rects)
    # Convert to numpy array
    shape = face_utils.shape_to_np(shape)
#    for (i, rect) in enumerate(rects):
#        # Find landmark
#        shape = landmark_predictor(image, rects)
#        # Convert to numpy array
#        shape = face_utils.shape_to_np(shape)
#    
    


# Visualize the drawn rectangle
#win.clear_overlay()
#win.set_image(image)
#win.add_overlay(rects)
#dlib.hit_enter_to_continue()


#for (i, rect) in enumerate(rects):
#        # Find landmark
#        shape = landmark_predictor(image, rects)
#        # Convert to numpy array
#        shape = face_utils.shape_to_np(shape)
        
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
    
