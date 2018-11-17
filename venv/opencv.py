# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D
from keras.preprocessing.image import img_to_array
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import random
from random import shuffle

input_shape=(32,32,1)

def createModel():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model

testpath = "1.png"

image=cv2.imread(testpath)
image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype('float')/255.0
image=cv2.resize(image, (1280,720))
average_color_per_row = np.average(image, axis=0)
average_color = np.average(average_color_per_row, axis=0)
std_per_row = np.std(image, axis=0)
std = np.std(std_per_row, axis=0)
th, image=cv2.threshold(image, average_color + std, 1.0, cv2.THRESH_TOZERO)

print(average_color)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image=img_to_array(image)


print("traincatdone")
