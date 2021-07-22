import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np
import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


_TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
_TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"

# train Data
urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
local_zip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/horse-or-human/')
zip_ref.close()

# test Data
urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
local_zip = 'testdata.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/testdata/')
zip_ref.close()


###  train_generator / validation_generator
train_datagen = ImageDataGenerator(
                rescale = 1. / 255.,
)
validation_datagen = ImageDataGenerator(
                  rescale=1.0 / 255.
)


###  train_generator / validation_generator
train_generator = train_datagen.flow_from_directory(
    'tmp/horse-or-human/',
    batch_size=128,
    class_mode='binary',
    target_size=(300, 300)
)

validation_generator = validation_datagen.flow_from_directory(
    'tmp/testdata/',
    batch_size=32,
    class_mode='binary',
    target_size=(300, 300)
)


model = tf.keras.models.Sequential([
    # Note the input shape specified on your first layer must be (300,300,3)
    # Your Code here
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation = "relu"),
    # This is the last layer. You should not change this code.
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.001),
            metrics=['accuracy'])


model.fit(
  train_generator,
  steps_per_epoch=8,
  epochs=15,
  verbose=1
  validation_data = validation_generator,
  validation_steps = 8
)

# In case of Colab, You can download h5 file
model.save('mymodel.h5')
files.download('mymodel.h5')
