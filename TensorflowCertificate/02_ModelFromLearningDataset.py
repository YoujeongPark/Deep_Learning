import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_tes = x_test/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 128, activation ='relu'),
    tf.keras.layers.Dense(units = 10, activation ='softmax'),
])


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )
model.fit(x_train,y_train, epochs = 10)
model.summary()


# In case of Colab, You can download h5 file
model.save('mymodel.h5')
files.download('mymodel.h5')