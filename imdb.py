#from keras.datasets import imdb
import tensorflow as tf
import numpy as np

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels[0])

print(max([max(sequence) for sequence in train_data]))

word_index = tf.keras.datasets.imdb.get_word_index()    # dictionary mapping words to int index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 4, '?') for i in train_data[0]])
print(decoded_review)

# vectorize
def vectorize_sequence(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate (sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequence(train_data)
x_text = vectorize_sequence(test_data)

print(x_train[0])
print(x_text[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
