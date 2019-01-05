#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:19:56 2018

@author: myidispg
"""

import keras
import os
import random
import numpy as np
import pandas as pd

import cv2
from imgaug import augmenters as iaa

face_data_dir = '../faces-data-new/'

categories = os.listdir(face_data_dir)

dataset = {}

# Create a dictionary with the dataset
for i in range(len(categories)):
    dataset[categories[i]] = os.listdir(os.path.join(face_data_dir, categories[i]))
    
# 2 empty lists to hold image paths and labels.
data_x = []
data_y = []
    
for category in dataset:
#    data_y.append(category)
    for image in dataset[category]:
        data_x.append(face_data_dir + category + '/' +image)
        data_y.append(category)

# Shuffle the dataset. Otherwise all samples of a category will be fed into the model in sequence.
shuffled_data = list(zip(data_x, data_y))
random.shuffle(shuffled_data)
data_x[:], data_y[:] = zip(*shuffled_data)

# Convert the data to numpy array.
dataX = np.asarray(data_x)
dataY = np.asarray(data_y, dtype='int64')

# Perform some augmentation to the images. This will help in adding generality to the model.

#image = cv2.imread(os.path.join(face_data_dir, '1', dataset['1'][1]))
#
#def zoom(image):
#    zoom = iaa.Affine(scale=(1, 1.3))
#    return zoom.augment_image(image)
#
#def pan(image):
#    pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y':(-0.1, 0.1)})
#    return pan.augment_image(image)
#
#def img_random_brightness(image):
#    brightness = iaa.Multiply((0.2, 1.2))
#    return brightness.augment_image(image)
#
#def img_random_flip(image):
#    image = cv2.flip(image, 1)
#    return image
#
#def random_augment(image_path):
#    image = cv2.imread(image_path)
#    if np.random.rand()< 0.5:
#        image = pan(image)
#    if np.random.rand()< 0.5:
#        image = zoom(image)
#    if np.random.rand()< 0.5:
#        image = img_random_brightness(image)
#    if np.random.rand()< 0.5:
#        image = img_random_flip(image)
#        
#    return image

import gc

del categories, category, data_x, data_y, i, image, shuffled_data
gc.collect()

# A function to take an image and return only the face from it.
casc_directory = 'face-cascades/'
def find_face(image):
    # Create the haar cascade
    faceCascade1 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt.xml'))
    faceCascade2 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt2.xml'))
    faceCascade3 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_alt_tree.xml'))
    faceCascade4 = cv2.CascadeClassifier(os.path.join(casc_directory, 'haarcascade_frontalface_default.xml'))
    # Read the image
#    image = cv2.imread(image_path)
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
        return (False, 0)
        
    return (True, faces)

#y = find_face(image)

# Load the required stuff for landmark detection.
from imutils import face_utils
import dlib
from landmarks_calculation import left, right


p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# -----------Calculations to be made for facial muscle approximation.------------

# 2 lists to hold face-masks values and their corresponding categories.
face_masks_x = []
face_masks_y = []
#categories = []
distance_between = []

# Loop over all images, detect landmarks, calculate face-masks and append to the lists.
for i in range(len(dataX)):
    print('Working on image {}\n'.format(i))
    image = cv2.imread(dataX[i])
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find faces.
#    rects = face_detector(image, 0)
    rects = find_face(image)
    if rects[0]:
        rects = rects[1]
        rects = dlib.rectangle(left = rects[0][0], top=rects[0][1], right=rects[0][2], bottom=rects[0][3])
        # Find landmark
        shape = landmark_predictor(image, rects)
        # Convert to numpy array
        shape = face_utils.shape_to_np(shape)
    else: # If haar cascade fails to find face
        rects = rects[1]
        rects = face_detector(image, 0)
        for (i, rect) in enumerate(rects):
            # Find landmark
            shape = landmark_predictor(image, rect)
            # Convert to numpy array
            shape = face_utils.shape_to_np(shape)
    # List to hold values of single image
    single_x = []
    single_y = []
    for j in range(len(left)):
        single_x.append(shape[left[j]][0] - shape[right[j]][0])
        single_y.append(shape[left[j]][1] - shape[right[j]][1])
    # Append to the list to hold values from images.
    face_masks_x.append(single_x)
    face_masks_y.append(single_y)
    single_x = np.asarray(single_x)
    single_y = np.asarray(single_y)
#    distance = np.sqrt(np.square(single_x) + np.square(single_y))
#    distance_between.append((single_x + single_y)/2)
    distance_between.append(np.sqrt(np.square(single_x) + np.square(single_y)))
#    categories.append(y_train[i])
    

distance_between = np.asarray(distance_between)
categories = np.asarray(dataY, dtype='int32')


# Convert the extracted data to Pandas DataFrame and save to CSV. 
features_df = pd.DataFrame(distance_between)
categories_df = pd.DataFrame(categories)
features_df = pd.concat([features_df, categories_df], axis=1)
features_df.to_csv('extracted_landmarks_data_new.csv', index=False)
