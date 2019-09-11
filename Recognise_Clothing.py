"""
********************************************************
Recognise_Clothing.py:  Tensor and keras library used to train the model on Fashion MNIST to recgonise clothing.
Parameters tuned to improve accuracy.
Created By Arvind Bahl
********************************************************
"""
#imprt of libraries.
import tensorflow as tf
import numpy as np

#loading of fashion_mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Defining of the model.
model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape= (28,28)),
			tf.keras.layers.Dense(512, activation = tf.nn.relu),
			tf.keras.layers.Dense(256, activation = tf.nn.relu),
			tf.keras.layers.Dense(10, activation = tf.nn.softmax)		
			])
#Compiling the model.
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fitting of the model.
model.fit(train_images, train_labels, epochs =5)

#evaluate the model
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[2])
print(test_labels[2])




