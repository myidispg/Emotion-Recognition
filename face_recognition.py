#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 18:05:18 2018

@author: myidispg
"""

import os
import cv2

imagePath = '../Google Photos/2018/IMG_20181107_185651.jpg'

cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,
        minNeighbors=5
        )

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
image = cv2.resize(image, (800, 600))
gray = cv2.resize(gray, (800, 600))
    
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
