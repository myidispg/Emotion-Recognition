#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:09:03 2019

@author: myidispg
"""

import cv2
import os
import numpy as np
from imutils import face_utils
import dlib
from landmarks_calculation import left, right

from emotion_key import emotions_list

# For command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '-image', required=True, help='The path of the image with a face')
args = vars(parser.parse_args())

print(args)

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

# Open the image from path specified by cmd line

image = cv2.imread(args['i'], 0)
#image = cv2.imread('/home/myidispg/My Files/Machine-Learning-Projects/Emotion-Dataset/cohn-kanade-images/S061/001/S061_001_00000012.png', 0)


p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# Get a cropped image with dimensions similar to cohn kanade images.
from crop_face_area import CropImage
crop_image = CropImage(image)
image = crop_image.get_cropped_image()

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

rects = face_detector(image)

if len(rects) == 0:
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
    print("rects- {}".format(rects))
    print("left- {}, right- {}, top- {}, bottom- {}".format(rects[0].left(), rects[0].right(), rects[0].top(), rects[0].bottom()))
    rects = dlib.rectangle(left = rects[0].left(), top=rects[0].top(), right=rects[0].right(), bottom= rects[0].bottom())
    # Find landmark
    shape = landmark_predictor(image, rects)
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

import pickle
from sklearn.externals import joblib

# Load the saved classifier
#with open('logistic-regression-emotion-recognition.pkl', 'rb') as file:
#    classifier = pickle.load(file)

classifier = joblib.load('logistic-regression-emotion-recognition.pkl')

# Load the MinMaxScaler
scaler = joblib.load('min-max-scaler.sav') 

distance_between = distance_between.reshape(1, -1)
distance_between = scaler.transform(distance_between)

predict = classifier.predict(distance_between)

print(predict[0])

print("The person's emotion seems to be- {}".format(emotions_list[predict[0]]))


