"""
********************************************************
Linear_equation.py:  Tensor and keras library used to solve linear equation.
Created By Arvind Bahl
********************************************************
"""
#imprt of libraries.
import tensorflow as tf
import numpy as np

#Defining the model.
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
#Compiling the model.
model.compile(optimizer = 'sgd', loss = "mean_squared_error")
#input
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)
#fitting of the model.
model.fit(xs, ys, epochs =500)
#printing the prediction.
print(model.predict([15.0]))


