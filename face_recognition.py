#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 18:05:18 2018

@author: myidispg
"""

import os
import cv2

#imagePath = '../Google Photos/2018/IMG_20180702_101025.jpg'
imagePath = '../Emotion-Dataset/cohn-kanade-images/S005/001/S005_001_00000003.png'

cascDir = 'face-cascades/'

# Create the haar cascade
faceCascade1 = cv2.CascadeClassifier(os.path.join(cascDir, 'haarcascade_frontalface_alt.xml'))
faceCascade2 = cv2.CascadeClassifier(os.path.join(cascDir, 'haarcascade_frontalface_alt2.xml'))
faceCascade3 = cv2.CascadeClassifier(os.path.join(cascDir, 'haarcascade_frontalface_alt_tree.xml'))
faceCascade4 = cv2.CascadeClassifier(os.path.join(cascDir, 'haarcascade_frontalface_default.xml'))

# Read the image
image = cv2.imread(imagePath)
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
            scaleFactor=1.2,
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
    exit(0)
   
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w+20, y+h+20), (0, 255, 0), 2)
    
image = cv2.resize(image, (800, 600))
    
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
