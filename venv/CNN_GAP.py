from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from random import shuffle
import cv2

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape=(128, 128, 1), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=384, activation='relu'))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
# Compiling the CNN

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images\

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

gapspath = "set_gaps\\gap"
nogapspath = "set_gaps\\nogap"

gapslist = []
nogapslist = []

import os

for filename in os.listdir(gapspath):
    gapslist.append(os.getcwd() + "\\" + gapspath + "\\" + filename)
for filename in os.listdir(nogapspath):
    nogapslist.append(os.getcwd() + "\\" + nogapspath + "\\" + filename)

gapscontentlist = list()
nogapscontentlist = list()

trainlist = []

fileindex= 1

for file in gapslist:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = cv2.GaussianBlur(image, (5, 5), 5)
    image = cv2.Canny(image, 50, 250, edges=25)
    image = img_to_array(image)
    fileindex= fileindex+ 1

    cv2.imwrite(os.getcwd() + "\\set_gaps_processed\\przerwa\\" + str(zmienna) + ".png", image)

for file in nogapslist:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = cv2.GaussianBlur(image, (5, 5), 5)
    image = cv2.Canny(image, 50, 250, edges=25)
    image = img_to_array(image)
    fileindex= fileindex+ 1
    cv2.imwrite(os.getcwd() + "\\set_gaps_processed\\nieprzerwa\\" + str(zmienna) + ".png", image)

shuffle(trainlist)

training_set = train_datagen.flow_from_directory('set_gaps_processed',
                                                 target_size=(128, 128),
                                                 batch_size=16,
                                                 color_mode="grayscale",
                                                 class_mode='binary')

check = keras.callbacks.ModelCheckpoint(os.getcwd() + "\\weights_model.h5", save_best_only=True, save_weights_only=True)

classifier.fit_generator(
    training_set,
    steps_per_epoch=30,
    epochs=10,
    validation_data=training_set,
    validation_steps=40, callbacks=[check])

with open(os.getcwd() + '\\model_architecture.json', 'w') as f:
    f.write(classifier.to_json())
