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
    
def find_face(image_path):
    