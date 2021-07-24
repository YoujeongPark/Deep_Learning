## Basic Regression Model

from tensorflow.keras.datasets import boston_housing
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()


def data_regularization(train, test):
    mean = train.mean()
    std = train.std()
    train -= mean
    train /= std
    test -= mean
    test /= std

    return train, test


X_train, Y_train = data_regularization(X_train, Y_train)
X_test, Y_test = data_regularization(X_test, Y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=18, activation='relu'),
    tf.keras.layers.Dense(units=1)

])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss = 'mse')
model.summary()

model.fit(X_train, Y_train,
                    epochs = 25,
                    batch_size = 32, validation_split = 0.25,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor = 'val_loss')])

model.evaluate(X_test, Y_test);

pred_Y = model.predict(X_test)

##############
# plot 1
plt.plot(history.history['loss'],'b-',label = 'loss')
plt.plot(history.history['val_loss'],'r--',label = 'val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# plot 2
pred_Y = model.predict(X_test)
plt.figure(figsize = (5,5))
plt.plot(Y_test,pred_Y,'b.')


