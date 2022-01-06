#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from gpiozero import Buzzer

buzzer = Buzzer(21)
interpreter = tf.lite.Interpreter('tflite_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

groundtruth = {'ClosedClosed'}
rect_size = 4
cap = cv2.VideoCapture(0)
llist = []
    
def eyecondition(ii):
    iii = float(ii)    
    if iii <= 0.05:
        condition = 'Closed'
    else:
        condition = 'Open'
    return condition    

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
while True:
    (rval, im) = cap.read()
    im=cv2.flip(im,1,1)
    cv2.rectangle(im,(0,0),(640,30),(0,0,255),-1)
    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = faceCascade.detectMultiScale(rerect_size)

    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(224,224))
        
        left = rerect_sized[50:120, 23:111]
        left_eyes1 = cv2.resize(left,(224,224))
        left_eyes2 = cv2.cvtColor(left_eyes1, cv2.COLOR_BGR2GRAY)
        backtorgbl = cv2.cvtColor(left_eyes2, cv2.COLOR_GRAY2BGR)
        cv2.imshow("left", backtorgbl)
        normalized_left = backtorgbl/255.0
        reshaped_left = np.reshape(normalized_left,(1,224,224,3))
        reshaped_left = np.array(reshaped_left, dtype = np.float32)
        reshaped_left = np.vstack([reshaped_left])
        
        right = rerect_sized[50:120, 112:200]
        right_eyes1 =cv2.resize(right,(224,224))
        right_eyes2 = cv2.cvtColor(right_eyes1, cv2.COLOR_BGR2GRAY)
        backtorgbr = cv2.cvtColor(right_eyes2, cv2.COLOR_GRAY2BGR)
        cv2.imshow("right", backtorgbr)
        normalized_right = backtorgbr/255.0
        reshaped_right = np.reshape(normalized_right,(1,224,224,3))
        reshaped_right = np.array(reshaped_right, dtype = np.float32)
        reshaped_right = np.vstack([reshaped_right])
        
        ##resultl=model.predict(reshaped_left)
        interpreter.set_tensor(input_details[0]['index'], reshaped_left)
        interpreter.invoke()
        output_datal = interpreter.get_tensor(output_details[0]['index'])
        plf = output_datal[0]
        sl = str(plf[0])
        cv2.putText(im, eyecondition(sl), (0, 25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        ##resultr=model.predict(reshaped_right)
        interpreter.set_tensor(input_details[0]['index'], reshaped_right)
        interpreter.invoke()
        output_datar = interpreter.get_tensor(output_details[0]['index'])
        prt = output_datar[0]
        sr = str(prt[0])
        cv2.putText(im, eyecondition(sr), (320, 25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        while len(llist)!=10:
            llist.append(eyecondition(sl)+eyecondition(sr))
            break       
        if len(llist) == 10:
            if set(llist) == groundtruth:
                resres = ' closed'
                buzzer.on()
            else:
                resres = 'open'
                buzzer.off()
            llist.clear()
        else:
            resres = 'open'
            buzzer.off()
                
    cv2.imshow('LIVE',   im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

