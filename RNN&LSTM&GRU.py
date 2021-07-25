from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold , train_test_split , KFold
import tensorflow as tf


iris = load_iris()
iris_label = iris.target
iris_label = iris_label.reshape(len(iris_label),1)
iris_data = iris.data

print(iris_data);

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.2, random_state = 11)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

print(X_train.shape)
print(y_train.shape)

# 1. Simple RNN Layer

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units =10, return_sequences = False, input_shape = [4,1]),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary)

model.fit(X_train, y_train, epochs = 100, verbose = 0)
print(model.predict(X_test))

############################
# 2. LSTM Layer

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units =10, return_sequences = True, input_shape = [4,1]),
    tf.keras.layers.LSTM(units =10),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary)

model.fit(X_train, y_train, epochs = 100, verbose = 0)
print(model.predict(X_test))

############################
# 3. GRU Layer
model = tf.keras.Sequential([
    tf.keras.layers.GRU(units =10, return_sequences = True, input_shape = [4,1]),
    tf.keras.layers.GRU(units =10),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer = 'adam', loss = 'mse')
print(model.summary)

model.fit(X_train, y_train, epochs = 100, verbose = 0)
print(model.predict(X_test))
