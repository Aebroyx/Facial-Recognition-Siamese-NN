from keras.layers import Input, Conv2D, Dense, Flatten,MaxPooling2D
from keras.layers import Lambda, Subtract
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras

#haarcascade_frontalface_default.xml is saved model for face detection
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Load tensorflow/keras model
model = keras.models.load_model("saved_best.h5")

def giveAllFaces(image,BGR_input=True,BGR_output=False):
    """
    return GRAY cropped_face,x,y,w,h 
    """
    gray = image.copy()
    if BGR_input:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if BGR_output:
        for (x, y, w, h) in faces:
            yield image[y:y+h,x:x+w,:],x,y,w,h
    else:
        for (x, y, w, h) in faces:
            yield gray[y:y+h,x:x+w],x,y,w,h

def putBoxText(image,x,y,w,h,text="unknown"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image,text, (x,y-6), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def putCharacters(image,db="application_data/verification_images"):
    dbs = os.listdir(db)
    right = np.array([ np.expand_dims(cv2.imread(os.path.join(db,x),0),-1) for x in dbs ])
    names = [ os.path.splitext(x)[0] for x in dbs ]
    name = "Unknown"
    prob = 0
    SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
    for face,x,y,w,h in giveAllFaces(image):
        face = cv2.resize(face,(100,100),interpolation = cv2.INTER_AREA)
        cv2.imwrite(SAVE_PATH, face)
        face = np.expand_dims(face,-1)
        left = np.array([face for _ in range(len(dbs))])
        probs = np.squeeze(model.predict([left,right]))
        index = np.argmax(probs)
        prob = probs[index]
        name = "Unknown"
        if prob>=0.4:
            name = names[index]
            print(name, prob)
        putBoxText(image,x,y,w,h,text=name+"({:.2f})".format(prob))
    
    return name,prob