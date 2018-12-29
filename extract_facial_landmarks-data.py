#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:19:56 2018

@author: myidispg
"""

import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import pandas as pd

import cv2
from imgaug import augmenters as iaa

face_data_dir = '../faces-data/'

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

# Split the data into train and validation set.
validation_size = 0.0

X_train, X_valid, y_train, y_valid = train_test_split(data_x, data_y, test_size = validation_size, random_state=12)

# Perform some augmentation to the images. This will help in adding generality to the model.

image = cv2.imread(os.path.join(face_data_dir, '001', dataset['001'][1]))

def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    return zoom.augment_image(image)

def pan(image):
    pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y':(-0.1, 0.1)})
    return pan.augment_image(image)

def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(image)

def img_random_flip(image):
    image = cv2.flip(image, 1)
    return image

def random_augment(image_path):
    image = cv2.imread(image_path)
    if np.random.rand()< 0.5:
        image = pan(image)
    if np.random.rand()< 0.5:
        image = zoom(image)
    if np.random.rand()< 0.5:
        image = img_random_brightness(image)
    if np.random.rand()< 0.5:
        image = img_random_flip(image)
        
    return image

import gc

del categories, category, dataX, dataY, data_x, data_y, i, image, shuffled_data, validation_size
gc.collect()

# Load the required stuff for landmark detection.
from imutils import face_utils
import dlib
from landmarks_calculation import left, right

p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# Calculations to be made for facial muscle approximation.

# 2 lists to hold face-masks values and their corresponding categories.
face_masks_x = []
face_masks_y = []
categories = []
average_coords = []

# Loop over all images, detect landmarks, calculate face-masks and append to the lists.
for i in range(len(X_train)):
    image = cv2.imread(X_train[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find faces.
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
    average_coords.append((single_x + single_y)/2)
    categories.append(y_train[i])
    

    
        
        
    
    
    


