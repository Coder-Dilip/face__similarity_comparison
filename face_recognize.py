# done in google colab, head over to this link colab.research.google.com and copy this code and press control+s and run 

import os
import cv2
from os import listdir
import face_recognition
from google.colab.patches import cv2_imshow


# first image 
frame1=cv2.imread("./dilip.jpg")
Torgb1=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)

# encoding means extracting the features from the image eg. nose, eyes, ears, etc
encoding1=face_recognition.face_encodings(Torgb1)[0]

# second image to compare with
frame2=cv2.imread("./arko.jpg")
Torgb2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)

# encoding for second image
encoding2=face_recognition.face_encodings(Torgb2)[0]


# if the faces are similar then result value will be true
result=face_recognition.compare_faces([encoding1],encoding2)
print(result)


# This line is optional, I am just displaying the images (two images) that are compared
cv2_imshow(Torgb1)
cv2_imshow(Torgb2)

