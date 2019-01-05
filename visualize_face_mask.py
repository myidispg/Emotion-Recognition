#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:33:30 2019

@author: myidispg
"""

import cv2

image = cv2.imread('/home/myidispg/My Files/Machine-Learning-Projects/Emotion-Dataset/cohn-kanade-images/S081/005/S081_005_00000019.png', 0)
image = cv2.imread('../Google Photos/2018/IMG-20180328-WA0000.jpg', 0)

from imutils import face_utils
import dlib
from landmarks_calculation import left, right

p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

rects = face_detector(image)
for (i, rect) in enumerate(rects):
        # Find landmark
        shape = landmark_predictor(image, rect)
        # Convert to numpy array
        shape = face_utils.shape_to_np(shape)
        
for i in range(len(left)):
    
    cv2.line(image, tuple(shape[left[i]]), tuple(shape[right[i]]), (0, 255, 0), 2)
    
cv2.imshow('window', image)
cv2.waitKey(0)