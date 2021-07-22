import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

sentences = []
labels = []


with open( 'sarcasm.json', 'r') as d:
  datas = json.load(d)


sentences = []
labels = []
for data in datas:
  sentences.append(data['headline'])
  labels.append(data['is_sarcastic'])
print(sentences[:5])
print(labels[:5])

train_sentences = sentences[:training_size]
train_labels = labels[:training_size]

valid_sentences = sentences[training_size:]
valid_labels = labels[training_size:]


tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
print(train_sequences[:5])

train_padded = pad_sequences(train_sequences, truncating= trunc_type, padding = padding_type, maxlen = max_length )
valid_padded = pad_sequences(valid_sequences, truncating= trunc_type, padding = padding_type, maxlen = max_length )
print(train_padded.shape)

train_labels = np.asarray(train_labels)
valid_labels = np.asarray(valid_labels)

model = tf.keras.Sequential([
                        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
                        tf.keras.layers.Conv1D(128, 5, activation = 'relu'),
                        tf.keras.layers.GlobalMaxPool1D(),
                        tf.keras.layers.Dense(30, activation='relu'),
                         tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(train_padded, train_labels, epochs = 50, validation_data = (valid_padded, valid_labels),
          callbacks = [tf.keras.callbacks.EarlyStopping(patience = 20 , monitor = 'val_loss')],
          verbose=1)


# In case of Colab, You can download h5 file
from google.colab import files
model.save('mymodel.h5')
files.download('mymodel.h5')
