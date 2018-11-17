import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from random import shuffle

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images\


from keras.preprocessing.image import ImageDataGenerator
train_datagen =  ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('training_set',
target_size = (128, 128),
batch_size = 32,
color_mode="grayscale",
class_mode = 'binary')




test_set = test_datagen.flow_from_directory('training_set',
target_size = (128, 128),
batch_size = 32,
color_mode="grayscale",
class_mode = 'binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=30,
        epochs=10,
        validation_data=training_set,
        validation_steps=40)


# Part 3 - Making new predictions
#
# classifier.fit_generator(training_set,
# steps_per_epoch = 25,
# epochs = 25,
# validation_data = test_set,
# validation_steps = 10)
#

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indices
#if result[0][0] == 1:
#prediction = 'dog'
#else:
#prediction = 'cat'