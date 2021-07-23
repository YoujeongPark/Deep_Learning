
# Auto Encoder - Basic

import tensorflow as tf
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()
train_X = train_X/255.0
test_X = test_X/255.0

print(train_X[0].shape[0])

train_X = train_X.reshape(-1,28*28)
test_X = test_X.reshape(-1,28*28)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=784, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=784, activation='sigmoid'),

])

model.compile(optimizer = tf.optimizers.Adam(), loss = "mse")
model.summary()

model.fit(train_X, train_X, epochs = 10, batch_size = 256)

model.evaluate(test_X,test_X)









