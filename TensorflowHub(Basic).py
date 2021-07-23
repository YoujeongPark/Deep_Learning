import tensorflow_hub as hub
import tensorflow as tf

mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
model = tf.keras.Sequential([
    hub.KerasLayer(handle = mobile_net_url, input_shape = (224,224,3), trainable = False )
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# keras_layer (KerasLayer)     (None, 1001)              3540265
# =================================================================
# Total params: 3,540,265
# Trainable params: 0
# Non-trainable params: 3,540,265
# _________________________________________________________________