#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 20:20:12 2018

This file is to be executed to extract only faces from the dataset and save them in a different directory.
The images for a particular class will be stored in a single folder.

@author: myidispg
"""

import os
import cv2
import gc

data_dir_base = '../Emotion-Dataset/cohn-kanade-images/'

# Get a list of all the subject directories in the dataset folder.
folders_list = os.listdir(data_dir_base)

# All the subfolders in the directories for each subject
sub_folders_dict = {}
for folder in folders_list:
    sub_folders_dict[folder] = os.listdir(data_dir_base + folder)
# Since some folders have a .DS_Store, removing it
for folder in sub_folders_dict:
    valid_folders = []
    for sub_folder in sub_folders_dict[folder]:
        if sub_folder != '.DS_Store':
            valid_folders.append(sub_folder)
    sub_folders_dict[folder] = valid_folders
    
# A function to take an image and return only the face from it.
face_directory = 'faces-data/'
casc_direcory = 'face-cascades/'
def find_face(image_path, face_dir):
    # Create the haar cascade
    faceCascade1 = cv2.CascadeClassifier(os.path.join(casc_direcory, 'haarcascade_frontalface_alt.xml'))
    faceCascade2 = cv2.CascadeClassifier(os.path.join(casc_direcory, 'haarcascade_frontalface_alt2.xml'))
    faceCascade3 = cv2.CascadeClassifier(os.path.join(casc_direcory, 'haarcascade_frontalface_alt_tree.xml'))
    faceCascade4 = cv2.CascadeClassifier(os.path.join(casc_direcory, 'haarcascade_frontalface_default.xml'))
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
    
# Go over all images, extract faces and save them
for folder in sub_folders_dict:
    for sub_folder in sub_folders_dict[folder]:
        images = os.listdir(os.path.join(data_dir_base, folder, sub_folder))
        for image in images:
            print(image)
            face = find_face(os.path.join(data_dir_base, folder, sub_folder, image), face_directory)
        

    
    
    