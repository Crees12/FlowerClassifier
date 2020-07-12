import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random

data = "./flowers/"
folders = os.listdir(data)
print(folders)

image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else: continue

train = np.array(train_images)
print(train.shape)

train = train.astype("float32")/255.0

label_dummies = pandas.get_dummies(train_labels)
labels = label_dummies.values.argmax(1)

pandas.unique(train_labels)

pandas.unique(labels)

union_list = list(zip(train,labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

train = np.array(train)
labels = np.array(labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=0.001, use_locking=False, name='GradientDescent'
), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train,labels, epochs=100)

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()