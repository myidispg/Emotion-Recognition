#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 14:02:45 2019

@author: myidispg
"""

import os
import cv2


class CropImage:
    
    def __init__(self, image):
        self.image = image
        self.casc_directory = 'face-cascades/'
    
    def find_face(self, image):
        # Create the haar cascade
        faceCascade1 = cv2.CascadeClassifier(os.path.join(self.casc_directory, 'haarcascade_frontalface_alt.xml'))
        faceCascade2 = cv2.CascadeClassifier(os.path.join(self.casc_directory, 'haarcascade_frontalface_alt2.xml'))
        faceCascade3 = cv2.CascadeClassifier(os.path.join(self.casc_directory, 'haarcascade_frontalface_alt_tree.xml'))
        faceCascade4 = cv2.CascadeClassifier(os.path.join(self.casc_directory, 'haarcascade_frontalface_default.xml'))
        
        # Detect faces in the image using 4 different classifiers
        faces1 = faceCascade1.detectMultiScale(
                    image, 
                    scaleFactor=1.2,
                    minNeighbors=5
                    )
        faces2 = faceCascade2.detectMultiScale(
                    image, 
                    scaleFactor=1.2,
                    minNeighbors=5
                    )
        faces3 = faceCascade3.detectMultiScale(
                    image, 
                    scaleFactor=1.2 ,
                    minNeighbors=5
                    )
        faces4 = faceCascade4.detectMultiScale(
                    image, 
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
            print('No faces found by Haar Cascade, will try dlibs now.')
            return (False, 0)
                
        return (True, faces)
    
    def get_cropped_image(self):
        face = self.find_face(self.image)#[1][0]
        print(face[0])
        if face[0] is False:
            from imutils import face_utils
            import dlib
            face_detector = dlib.get_frontal_face_detector()
            rects = face_detector(self.image)
            
            left, right, top, bottom = rects[0][0], rects[0][2], rects[0][1], rects[0][3]
        else:
            face = face[1][0]
            left, top, right, bottom = face[0], face[1], face[0] + face[2], face[1] + face[3]
    
        image_size = self.image.shape
        hori_adjustment_scale = image_size[1]/10
        vert_adjustment_scale = image_size[0]/10
#        print('hori_adjustment_scale- {}'.format(hori_adjustment_scale))
#        print('vert_adjustment_scale- {}'.format(vert_adjustment_scale))
#        
#        print('top- {}'.format(top))
#        print('bottom- {}'.format(bottom))
#        print('left- {}'.format(left))
#        print('right- {}'.format(right))
#        
        if (left-hori_adjustment_scale) >= 0:
            left -= hori_adjustment_scale
            left = int(left)
        else:
            left = 0
            
        if (right + hori_adjustment_scale) <= image_size[1]:
            right += hori_adjustment_scale
            right = int(right)
        else:
            right = image_size[1]
            
        if (top - vert_adjustment_scale) >= 0:
            print('ye')
            top -= vert_adjustment_scale
            top = int(top)
        else:
            top = 0
            
        if (bottom + vert_adjustment_scale) <= image_size[0]:
            bottom += vert_adjustment_scale
            bottom = int(bottom)
        else:
            bottom = image_size[0]
            
#        print('top- {}'.format(top))
#        print('bottom- {}'.format(bottom))
#        print('left- {}'.format(left))
#        print('right- {}'.format(right))
            
        self.image = self.image[top:bottom, left:right]
        
        self.image = cv2.resize(self.image, (640, 490))
        
        return self.image
