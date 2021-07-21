# https://medium.com/@saitejaponugoti/nlp-natural-language-processing-with-tensorflow-b2751aa8c460

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog', 'I love my cat'];
tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences);
word_index = tokenizer.word_index
print(word_index)

####

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)


####
test_data = ["I really love my dog", "my dog loves my father"]
test_seq = tokenizer.texts_to_sequences(test_data);
print(test_seq)


###
padded = pad_sequences(sequences)
print(padded)


