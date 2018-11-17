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

trainpathcat = "cats&dogs\\train\\cat"
trainpathdog = "cats&dogs\\train\\dog"

testpathcat = "cats&dogs\\test\\cat"
testpathdog = "cats&dogs\\test\\dog"

trainlist=[]
testlist=[]

traincatlist = list()
traindoglist = list()
testcatlist = list()
testdoglist = list()


import os
for filename in os.listdir(trainpathcat):
    traincatlist.append(os.getcwd()+"\\"+trainpathcat+"\\"+filename)

for filename in os.listdir(trainpathdog):
    traindoglist.append(os.getcwd()+"\\"+trainpathdog+"\\"+filename)

for filename in os.listdir(testpathcat):
    testcatlist.append(os.getcwd()+"\\"+testpathcat+"\\"+filename)

for filename in os.listdir(testpathdog):
    testdoglist.append(os.getcwd()+"\\"+testpathdog+"\\"+filename)

traincount=1
testcount=1
for file in traincatlist:
    #openedfile=open(file, "rb")
    #processedfile=tf.image.decode_png(
    #    openedfile.read(),
    #    channels=3,
    #    dtype=tf.uint8,
    #    name=None)


    #finalfile = tf.image.resize_images(processedfile, [32, 32])
    image=cv2.imread(file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')/255.0
    image=cv2.resize(image, (32,32))
    image=img_to_array(image)
    trainlist.append((image, 0))
    traincount+=1

print("traincatdone")

for file in traindoglist:
    #openedfile=open(file, "rb")
    #processedfile = tf.image.decode_png(
    #    openedfile.read(),
    #    channels=3,
    #    dtype=tf.uint8,
    #    name=None)

    #finalfile = tf.image.resize_images(processedfile, [32, 32])
    image = cv2.imread(file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')/255.0
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    trainlist.append((image, 1))
    traincount += 1

print("traindogdone")

for file in testcatlist:
    #openedfile=open(file, "rb")
    #processedfile = tf.image.decode_png(
    #    openedfile.read(),
    #    channels=3,
    #    dtype=tf.uint8,
    #    name=None)

    #finalfile = tf.image.resize_images(processedfile, [32, 32])
    image = cv2.imread(file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')/255.0
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    testlist.append((image, 0))
    testcount += 1

print("testcatdone")

for file in testdoglist:
    #openedfile=open(file, "rb")
    #processedfile = tf.image.decode_png(
    #    openedfile.read(),
    #    channels=3,
    #    dtype=tf.uint8,
    #    name=None)

    #finalfile = tf.image.resize_images(processedfile, [32, 32])
    image = cv2.imread(file)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float')/255.0
    image = cv2.resize(image, (32, 32))
    image = img_to_array(image)
    testlist.append((image, 1))
    testcount += 1

print("testdogdone")

shuffle(trainlist)
shuffle(testlist)


train_labels = list()

for element in trainlist:
    train_labels.append(element[1])

print("traindatadone")

test_labels = list()

for element in testlist:
    test_labels.append(element[1])

tr_img_data = []

for element in trainlist:
    tr_img_data.append(element[0].reshape(32, 32, 1))

test_img_data = []

for element in testlist:
    test_img_data.append(element[0].reshape(32, 32, 1))

print("testdatadone")

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

aug=ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

history = model1.fit(aug.flow(np.array(tr_img_data), np.array(train_labels), batch_size=batch_size), epochs=epochs, verbose=1,
                     validation_data=(np.array(test_img_data), np.array(test_labels)), callbacks=[early_stop])

results = model1.evaluate(np.array(test_img_data), np.array(test_labels))

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()