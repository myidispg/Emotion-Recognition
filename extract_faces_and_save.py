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
import matplotlib.pyplot as plt

data_dir_base = '../Emotion-Dataset/cohn-kanade-images/'
emotion_dir = '../Emotion-Dataset/Emotion/'

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
    
# Go over all images, extract faces and save them
    
face_directory = '../faces-data-new/' # Root directory where the faces will be saved.

# adictionary which will serve as a counter for image names while saving them.
count_dictionary = {
        '1': 0000,
        '2': 0000,
        '3': 0000,
        '4': 0000,
        '5': 0000,
        '6': 0000,
        '7': 0000
        }

# The prefix is supposed to make the file name of format 00000.png
def get_file_prefix(count):
    if count/10 < 1:
        return '0000'
    elif count/100 < 1:
        return '000'
    elif count/1000 < 1:
        return '00'
    elif count/10000 < 1:
        return '0'

# A function to get the emotion category from the Emotions folder
def find_emotion_category(path):
    # This check is made because sometimes, emotion label directory is non existent.
    if os.path.isdir(path):
        file_in_dir = os.listdir(path)
        # This check is made to find if the emotion label is present.
        if len(file_in_dir) == 0:
            return None
        else:
            with open(os.path.join(path, file_in_dir[0]), 'r') as file:
                data = file.read()
                data = data.split('.')[0].strip()
                return data
    return None
        
test = find_emotion_category(os.path.join(emotion_dir, folder, sub_folder))
        
    
# loop over all the images in dataset and save to another folder with its sub folders acting as categories.
for folder in sub_folders_dict:
    for sub_folder in sub_folders_dict[folder]:
        images = os.listdir(os.path.join(data_dir_base, folder, sub_folder))
        for image in images:
            print(image)
            emotion_category = find_emotion_category(os.path.join(emotion_dir, folder, sub_folder))
            if emotion_category is None:
                pass
            else:
#            emotion_category = image.split('_')[1]
                face = find_face(os.path.join(data_dir_base, folder, sub_folder, image))
                save_path = os.path.join(os.getcwd(), face_directory, emotion_category)
                file_prefix = get_file_prefix(count_dictionary[emotion_category]) 
                final_path = os.path.join(save_path, file_prefix + str(count_dictionary[emotion_category]) + '.png')
                if os.path.isdir(save_path):
                    print(final_path)
                    cv2.imwrite(final_path, face)
                    count_dictionary[emotion_category] += 1
                else:
                    print('Creating directory')
                    os.makedirs(os.path.join(os.getcwd(), face_directory, emotion_category), exist_ok=True)
                    print(final_path)
                    cv2.imwrite(final_path, face)
                    count_dictionary[emotion_category] += 1
  
total = 0

for category in count_dictionary:
    total += count_dictionary[category]
    
print('Total images processed {}'.format(total))
# Freeing up some memmory
del casc_directory, count_dictionary, category, emotion_category, face, file_prefix, final_path, folder, folders_list, image, images, save_path, sub_folder, total, valid_folders
gc.collect()

# Now to get a count of all the images in all the created directories. 
processed_folders = os.listdir(face_directory)
count_images = {
        '1': 0000,
        '2': 0000,
        '3': 0000,
        '4': 0000,
        '5': 0000,
        '6': 0000,
        '7': 0000
        }

for folder in processed_folders:
    count_images[folder] = len(os.listdir(os.path.join(face_directory, folder)))

# Plotting the number of image samples in each folder
plot_x = []
plot_y = []
for key in count_images:
    plot_x.append(key)
    plot_y.append(count_images[key])

plt.bar(plot_x, plot_y)
#plt.yticks(plot_y)
#plt.xticks(plot_x)
plt.xlabel('Folder')
plt.ylabel('Count')
plt.title('No. of images in each folder')
plt.show()    

# Load all images, convert to grayscale and resize to 192x192. Also normalize values to be between 0 and 1

folders = os.listdir(face_directory)

all_images = {}

for folder in folders:
    all_images[folder] = os.listdir(os.path.join(face_directory, folder))
    
for folder in all_images:
    for image_name in all_images[folder]:
        image = cv2.imread(os.path.join(face_directory, folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64,64))
#        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print('Saving image- {}'.format(os.path.join(face_directory, folder, image_name)))
        cv2.imwrite(os.path.join(face_directory, folder, image_name), image)
