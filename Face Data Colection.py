# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:13:52 2023

@author: User
"""
import pandas as pd
import numpy as np
import cv2
import os
haar_file  = r"C:\Users\User\Desktop\Deep Learning AI\.ipynb_checkpoints\haarcascade_frontalface_default.xml"
datasets = r"C:\Users\User\Desktop\Deep Learning AI\.ipynb_checkpoints\Datasets"
sub_data = "Mohd"
path = os.path.join(datasets , sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count < 31:
    print(count)
    (_,img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w, y+h),(255,0,0),2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count = count + 1
    cv2.imshow('OpenCv', img)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()