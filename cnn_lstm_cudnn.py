# python3 code to create an cnn and lstm combined model to perform sentiment analysis
# uses keras with tensorflow-gpu as backend
import numpy as np
import pandas as pd

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Dense, Input, Embedding, Conv1D, MaxPooling1D
from keras.layers import CuDNNLSTM as LSTM # CuDNNLSTM is the LSTM implementation optimized for nvidia gpus
from keras.optimizers import Adamax
from keras.models import Model

import pickle

# parameters
MAX_WORDS = 40000
EMBEDDING_DIM = 50

# filenames of the files used
# filename of the tokenizer, to save and load it so we don't have to compute it twice
tokenizer_filename = 'tokenizer.pickle'
model_filename = 'model_cnn-lstm.h5' # where we are going to save the model
corpus_filename = '../datasets/corpus.txt' # where we load the corpus from to create the tokenizer
embeddings_filename = '../glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM # where we load the embbeding weights from
# we use the glove embeddings, from https://nlp.stanford.edu/projects/glove/

# just have to compute the tokenizer once
try:
    # loading tokenizer
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, filters=';#&*:().,')
    with open(corpus_filename) as corpus:
        tokenizer.fit_on_texts(corpus)
    # saving Tokenizer
    with open(tokenizer_filename, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# define an embedding matrix to vectorize the texts
embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
# get the pre-trained glove vectors on the dictionary embeddings_index
embeddings_index = dict()
mean = np.array([0]*EMBEDDING_DIM, dtype='float32')
with open(embeddings_filename) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # we'll use the mean to initialize words not found in the glove embeddings
        mean += coefs
        embeddings_index[word] = coefs
mean = mean/len(embeddings_index)
# reverse the dictionary to obtain the words from the index
inv_index = {v: k for k, v in word_index.items()}
# initialize the embeddins matrix
for i in range(1,MAX_WORDS):
    word = inv_index[i]
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        # words not found in embedding index will be close to the mean.
        embedding_matrix[i] = mean + np.random.normal(scale=0.1, size=(1,EMBEDDING_DIM))
del embeddings_index

# define the embeddings layer
embedding_layer = Embedding(MAX_WORDS,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=None,
                            trainable=True,
                            name='embeddings')
del embedding_matrix

# input layer
revText_input = Input(shape=(None,), dtype='int32')

# embeddings layer
embedded_sequences = embedding_layer(revText_input)

# convolution and maxpooling
l_cov1 = Conv1D(128, 5, activation='relu', padding='valid')(embedded_sequences)
l_pool1 = MaxPooling1D(strides=2)(l_cov1)

# convolution and maxpooling again
l_cov2 = Conv1D(128, 5, activation='relu', padding='valid')(l_pool1)
l_pool2 = MaxPooling1D(strides=2)(l_cov2)

# lstm layer
l_lstm = LSTM(200, return_sequences=False)(l_pool2)

# output layer
preds = Dense(5, activation='softmax')(l_lstm)

model = Model(revText_input, preds)
model.compile(loss='categorical_crossentropy',optimizer=Adamax(clipnorm=3),metrics=['acc'])

# display the model architecture
model.summary()

# save model
model.save(model_filename)
print('Saved model to disk')
