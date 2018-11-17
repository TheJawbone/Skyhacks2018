# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import random
from random import shuffle


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    return model

trainpathcat = "cats&dogs\\train\\cat"
trainpathdog = "cats&dogs\\train\\dog"

traincatlist = list()
traindoglist = list()

testpathcat = "cats&dogs\\test\\cat"
testpathdog = "cats&dogs\\test\\dog"

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

traincatcontentlist = list()
traindogcontentlist = list()
testcatcontentlist = list()
testdogcontentlist = list()

for file in traincatlist:
    openedfile=open(file, "r", encoding="utf8")
    processedfile=keras.preprocessing.text.hashing_trick(openedfile.read(), 10000, hash_function="md5")
    traincatcontentlist.append(processedfile)

traincatcontentlist = keras.preprocessing.sequence.pad_sequences(traincatcontentlist,
                                               value=0,
                                               padding='dogt',
                                               maxlen=2048)
print("traincatdone")

for file in traindoglist:
    openedfile=open(file, "r", encoding="utf8")
    processedfile = keras.preprocessing.text.hashing_trick(openedfile.read(), 10000, hash_function="md5")
    traindogcontentlist.append(processedfile)

traindogcontentlist = keras.preprocessing.sequence.pad_sequences(traindogcontentlist,
                                               value=0,
                                               padding='dogt',
                                               maxlen=2048)
print("traindogdone")

for file in testcatlist:
    openedfile=open(file, "r", encoding="utf8")
    processedfile = keras.preprocessing.text.hashing_trick(openedfile.read(), 10000, hash_function="md5")
    testcatcontentlist.append(processedfile)

testcatcontentlist = keras.preprocessing.sequence.pad_sequences(testcatcontentlist,
                                               value=0,
                                               padding='dogt',
                                               maxlen=2048)
print("testcatdone")

for file in testdoglist:
    openedfile=open(file, "r", encoding="utf8")
    processedfile = keras.preprocessing.text.hashing_trick(openedfile.read(), 10000, hash_function="md5")
    testdogcontentlist.append(processedfile)

testdogcontentlist = keras.preprocessing.sequence.pad_sequences(testdogcontentlist,
                                               value=0,
                                               padding='dogt',
                                               maxlen=2048)
print("testdogdone")

trainlist = list()
for element in traincatcontentlist:
    trainlist.append((element, 0))
for element in traindogcontentlist:
    trainlist.append((element, 1))

shuffle(trainlist)


train_data = list()
train_labels = list()

for pair in trainlist:
    train_data.append(pair[0])

    train_labels.append(pair[1])

print("traindatadone")


testlist = list()
for element in testcatcontentlist:
    testlist.append((element, 0))
for element in testdogcontentlist:
    testlist.append((element, 1))

shuffle(testlist)

test_data = list()
test_labels = list()


for pair in testlist:
    test_data.append(pair[0])
    test_labels.append(pair[1])

print("testdatadone")

x_val = train_data[:1000]
partial_x_train = train_data[1000:]

y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

vocab_size = 10000

partial_x_train = np.array(partial_x_train)
partial_y_train = np.array(partial_y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(partial_x_train, partial_y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_val, y_val), callbacks=[early_stop])

test_data = np.array(test_data)
test_labels = np.array(test_labels)

model1.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()