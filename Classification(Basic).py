from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold , train_test_split , KFold
import tensorflow as tf

iris = load_iris()
iris_label = iris.target
iris_label = iris_label.reshape(len(iris_label),1)
iris_data = iris.data


X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.2, random_state = 11)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')

])


model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.07),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.25)



#model summary
model.summary()

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_5 (Dense)              (None, 48)                240
# _________________________________________________________________
# dense_6 (Dense)              (None, 24)                1176
# _________________________________________________________________
# dense_7 (Dense)              (None, 12)                300
# _________________________________________________________________
# dense_8 (Dense)              (None, 3)                 39
# =================================================================
# Total params: 1,755
# Trainable params: 1,755
# Non-trainable params: 0


