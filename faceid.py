# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

from keras.layers import Input, Conv2D, Dense, Flatten,MaxPooling2D
from keras.layers import Lambda, Subtract
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from siamese_network import *


# siamese_net = keras.models.load_model("saved_best.h5")
print("Model Loading.....")

# Build app and layout 
class CamApp(App):

    

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))
        self.user_label = Label(text="username", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.user_label)

        

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        # frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100,100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.4
        verification = 0

        # Capture input image from our webcam
        # SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        # frame = frame[120:120+250, 200:200+250, :]
        # cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        # try:
        name_id, verification = putCharacters(frame)

        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        self.user_label.text = name_id if verified == True else 'Unverified'

        # Log out details
        # Logger.info(results)
        # Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return results, verified



if __name__ == '__main__':
    CamApp().run()