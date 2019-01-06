# Emotion-Recognition #

This is my attempt at Emotion Recognition through images.

My first approach was going to be simply training a Convolutional Neural Network over the Cohn-Kanade dataset. But, I knew they were techniques to simplify this. Hence, I did some research and applied a different technique detailed below.

First I downloaded the dataset and sorted it. All the images of a particular emotion class are saved in a single directory outside the directory in which this repository is saved. For example, the numerical code for 'surprise' emotion is '7', hence all images labeled as surprised are saved in a single directory with directory name- faces-data/7/00000.png. This helped in easier understanding for later use.

## Extract Facial Features ##
I used Haar Cascade and dlibs to detect faces and extract the 68 landmarks.
The landmarks look like this- 
![alt text](https://github.com/myidispg/Emotion-Recognition/blob/master/facial_landmarks_68markup-768x619.jpg)

The reason behind landmark detection is that these landmarks can be used to approximate the facial muscles and facial muscles can be used to classify emotions more easily. Hence, using these landmarks, I extracted a mask approximating the facial muscles. The mask looks like this-
![alt text](https://github.com/myidispg/Emotion-Recognition/blob/master/face_mask.jpg)

Now, all I calculated was length of the lines shown in the above face mask and use them to train a model.

## Model Training ##
I tried many models and found that Logistic Regression gave the best results. Hence, that is the one I used. 

I trained my model over 5680 images and used a validation set of over 1420 images.

I achieved an accuracy of 71.26% over the validation set.

## Description of files ##
**emotion-recognition.py** - The file to run the trained model over a real world image.
**crop_face_area.py**- This file detects a face in the image, crops the image to a size near but larger than that of the face and resized it to 640x490 pixels and grayscale format. This is similar to cohn-kanade dataset images.
**extract_faces_save.py**- File to extract faces from raw dataset and sort them in above mentioned format.
**extract_facial_landmarks_data.py**- Calculate the face landmarks, and get distance between landmarks as per the face mask and save in a csv file.
**landmarks_calculation.py**- Two lists to hold which landmark index to use for feature extraction.
**model_train.py**- The file to train the Logistic Regression model over the extracted feature set.
