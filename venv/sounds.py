# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical
import cv2

# Librosa
import librosa as lr

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import random
from random import shuffle

input_shape=(20,32, 1)

def createModel():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (2,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model.add(keras.layers.Conv2D(64, (2,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Conv2D(64, (2,3), padding = 'same', activation='relu'))
    model.add(keras.layers.Conv2D(64, (2,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Conv2D(128, (2,3), padding = 'same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (2,3), activation = 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation = 'relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation = 'softmax'))

    return model
from sklearn.model_selection import train_test_split

import os
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def wav2mfcc(file_path, max_pad_len=32):
    wave, sr = lr.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = lr.feature.mfcc(wave, sr=16000)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_labels(path):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path, string,max_pad_len=32):
    lol = 1
    labels, _, _ = get_labels(path)
    for label in labels:
        # Init mfcc
        mfcc_vectors = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(string+label + '.npy', mfcc_vectors)

def get_train(path,string,split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(path)

    # Getting first arrays
    X = np.load(string+labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(string+label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return X,y
# ---------------------------------------------------------------------------------------------------------------------
# Generate absolute paths to files
trainPaths=[]
testPaths=[]
trainData=[]
testData=[]
traindata=[]
testdata=[]
trainLabels = []
testLabels = []
trainlabels = []
testlabels = []
Train=[]
Test=[]

trainpath = "spoken-digits\\train\\"
testpath = "spoken-digits\\test\\"
save_data_to_array(trainpath, "train", 32)
save_data_to_array(testpath,"test", 32)

trainData,trainLabels=get_train(trainpath,"train")
testData,testLabels=get_train(trainpath,"test")

trainData = trainData.reshape(trainData.shape[0],20,32,1)
testData = testData.reshape(testData.shape[0],20,32,1)
trainLabels = to_categorical(trainLabels)
testLabels= to_categorical(testLabels)
# ---------------------------------------------------------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model1 = createModel()
batch_size = 100
epochs = 200
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator


history = model1.fit(np.array(trainData), np.array(trainLabels), batch_size=batch_size, epochs=epochs,
                     validation_data=(np.array(testData), np.array(testLabels)), callbacks=[early_stop])

results = model1.evaluate(np.array(testData), np.array(testLabels))

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.ylim(top=1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()