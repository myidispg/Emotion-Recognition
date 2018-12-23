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
import os
import random
import numpy as np

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
        data_x.append(image)
        data_y.append(category)

# Shuffle the dataset. Otherwise all samples of a category will be fed into the model in sequence.
shuffled_data = list(zip(data_x, data_y))
random.shuffle(shuffled_data)
data_x[:], data_y[:] = zip(*shuffled_data)

# Convert the data to numpy array.
dataX = np.asarray(data_x)
dataY = np.asarray(data_y, dtype='int64')


# Perform some augmentation to the images. This will help in adding generality to the model.
