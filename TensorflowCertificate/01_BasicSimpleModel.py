import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

##1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')
model.fit(xs, ys, epochs=300, verbose=False)

# In case of Colab, You can download h5 file
model.save('mymodel.h5')
files.download('mymodel.h5')



## -------------------------------------------------
## Basic Regression Model

(trainX, trainY), (testX, testY) = boston_housing.load_data();
print(len(trainX), len(trainY));


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 52, activation = 'relu', input_shape = (13,)),
    tf.keras.layers.Dense(units = 39, activation ='relu'),
    tf.keras.layers.Dense(units = 26, activation ='relu'),
    tf.keras.layers.Dense(units = 1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.07), loss = 'mse')
model.fit(trainX, trainY, epochs = 25, batch_size = 32, validation_split = 0.25,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3, monitor = 'val_loss')])

model.evaluate(testX, testY)






