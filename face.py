# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 22:09:29 2023

@author: sage
"""
import numpy as np
import cv2
import os

size = 4
haar_file = r"C:\Users\User\Desktop\Deep Learning AI\.ipynb_checkpoints\haarcascade_frontalface_default.xml"
datasets = r"C:\Users\User\Desktop\Deep Learning AI\.ipynb_checkpoints\Datasets"
print('Training ...')
(images, labels, names, id) = ([],[],{},0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/'+ filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            print(labels)
        id += 1
(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]
#print(images, labels)
model = cv2.face.LBPHFaceRecognizer_create()
#model - cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
cnt = 0
while True:
    (_,img) = webcam.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        face = grey[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width,height))
        prediction = model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1] < 800:
            #cv2.putText(img,'%s - %.0f' % (names[prediction[0]],prediction))
            print (names[prediction[0]])
            cnt = 0
        else:
            cvt+=1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0,255,2),3)
            if(cnt > 100):
                print("Unknown Person")
                cv2.imwrite("input.jpg", img)
                cnt = 0
    cv2.imshow('OpenCV',img)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()