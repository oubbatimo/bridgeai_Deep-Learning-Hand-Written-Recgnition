#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:14:52 2022

@author: mohamedoubbati
"""
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import matplotlib.pyplot as plt

#read data
datafile = tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)=datafile.load_data()

#pre-processing data
x_train, x_test= x_train/255, x_test/255

#configure the Net
MyModel=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

#compile the net before training

MyModel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']

)
# training
MyModel.fit(x_train, y_train, epochs=10)

#evaluate the net.
MyModel.evaluate(x_test, y_test)

#make prediction
yhat = MyModel.predict(x_test)
print("Prediction:", yhat[11])


#Define the image to be recognized
image_index = 11

#Compute and print the prediction of the image_indes
pred = MyModel.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

#Display the image to be recgnized
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.show()
