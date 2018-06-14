# python3 code to test a model for sentiment analysis on the amazon's user review dataset
# from http://jmcauley.ucsd.edu/data/amazon/
# uses keras with tensorflow as backend
# uses seaborn, sklean and matplotlib to compute and display the confusion matrix
import numpy as np
import pandas as pd
import json
import random
import seaborn as sb
import matplotlib.pyplot as plt

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Model

from sklearn.metrics import confusion_matrix

import pickle

# parameters
MAX_SEQUENCE_LENGTH = 400
MAX_WORDS = 40000
EMBEDDING_DIM = 50


#filenames used
filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'
folder = '../datasets'


# load tokenizer
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = load_model(filename_model)

model.summary()

# list of the predicted labels
pred = list()
# list of the true labels
labels_list = list()
# load test data
test_data = pd.read_json('../datasets/test_Books.json', lines=True, chunksize=500000)
for chunk in test_data:
    # concatenate summary and review texts
    texts = (chunk['summary'] + ' * ' + chunk['reviewText']).tolist()
    labels = chunk['overall'].tolist()
    labels_list = labels_list + labels
    # tokenize the texts
    sequences = tokenizer.texts_to_sequences(texts)
    # make all of the sequences have the same size
    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    y_test = to_categorical(np.asarray(labels)-1)
    del sequences
    del labels
    pred_tmp = list(model.predict(x_test, verbose=1).argmax(axis=1))
    pred = pred+pred_tmp
    del x_test
    del y_test

# compute the confusion matrix, save it and show it on screen
labels_list = np.array(labels_list)-1
pred = np.array(pred)
C = confusion_matrix(labels_list, pred)
sb.heatmap(C)
plt.savefig('conf_matrix.png')
sb.heatmap(C,annot=True)
plt.savefig('conf_matrix_annot.png')
plt.show()
print(C)
acc = np.sum(1*(labels_list==pred))/labels_list.size
print('acc = %.3f' % acc)

# conpute the positive/negative confusion matrix, save it and show it on scree
labels_list_bin = 1*(labels_list>3)
pred_bin = 1*(pred>3)
C_bin = confusion_matrix(labels_list_bin, pred_bin)
sb.heatmap(C_bin)
plt.savefig('conf_matrix_bin.png')
sb.heatmap(C_bin,annot=True)
plt.savefig('conf_matrix_bin_annot.png')
plt.show()
print(C_bin)
acc_bin = np.sum(1*(labels_list_bin==pred_bin))/labels_list.size
print('acc_bin = %.3f' % acc_bin)
