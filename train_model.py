# python3 code to train a model for sentiment analysis on the amazon's user review dataset
# from http://jmcauley.ucsd.edu/data/amazon/
# uses keras with tensorflow as backend
import numpy as np
import pandas as pd
import json
import random
from time import time

from collections import deque

import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

from keras.models import Model

import pickle

# parameters
MAX_SEQUENCE_LENGTH = 400
MAX_WORDS = 40000
BATCH_SIZE = 32

filename_tokenizer = 'tokenizer.pickle'
filename_model = 'model_cnn-lstm.h5'
filename_train = '../datasets/train_Books.json'
filename_val = '../datasets/val_Books.json'

# weights for each class used on training
w = {0:8, 1:8, 2:4, 3:1, 4:1}


# count the number of lines in the training dataset
numlines = 0
with open(filename_train) as f:
    for line in f:
        numlines+=1

# defines a generator function that selects random samples on a data set an returns batches.
# This function is useful because for big data we may not be able to load it all into memory,
# so we get only parts of it. We select a random subset of it each time, instead of selecting
# sequentially, because it has the same effect as shuffling.
# it iterates over one epoch
# the parameter chunk_size is the number of samples it loads into memory at the same time
def input_gen(filename, tokenizer, numlines, chunk_size, batch_size):
    not_read = set(range(numlines))
    while len(not_read):
        x = list()
        y = list()
        try:
            samples = set(random.sample(not_read, chunk_size))
            not_read = not_read.difference(samples)
        except:
            samples = not_read
            not_read = set()
        with open(filename) as f:
            for n, line in enumerate(f):
                d = json.loads(line)
                if n in samples:
                    x.append(d['summary']+' * '+d['reviewText'])
                    y.append(d['overall'])
        size = len(samples)
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x,maxlen=MAX_SEQUENCE_LENGTH,truncating='post')
        y = to_categorical(np.asarray(y)-1)
        indices = np.arange(size)
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        for i in range(size//batch_size):
            x_tmp = x[batch_size*i:batch_size*(i+1)]
            y_tmp = y[batch_size*i:batch_size*(i+1)]
            yield x_tmp,y_tmp
        if not size % batch_size == 0:
            k = size % batch_size
            x_tmp = x[-k:]
            y_tmp = y[-k:]
            yield x_tmp,y_tmp

# load tokenizer
with open(filename_tokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model
model = load_model(filename_model)

model.summary()

# read the validation data and get a small subset of it for testing
val_data = pd.read_json('../datasets/val_Books.json', lines=True).sample(100000)
# concatenate summary and review text
texts = (val_data['summary'] + ' *  ' + val_data['reviewText']).tolist()
labels = val_data['overall'].tolist()
# tokenize the texts
sequences = tokenizer.texts_to_sequences(texts)
del texts
# make all the sequences have the same size
x_val = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
del sequences
del val_data
y_val = to_categorical(np.asarray(labels)-1)
del labels
loss_val = list()
acc_val = list()

# lists used to compute the average over a window of the last trained batches
losses = deque([0.0]*20)
accs = deque([0.0]*20)
loss_sum = 0.0
acc_sum = 0.0
for epoch in range(0,1):
    i = 0
    t = time()
    for x,y in input_gen(filename_train,tokenizer,numlines,650000,BATCH_SIZE):
        # we use train_on_batch to train on each batch separately
        loss, acc = model.train_on_batch(x,y,class_weight=w)
        loss_sum += loss-losses.popleft()
        acc_sum += acc-accs.popleft()
        losses.append(loss)
        accs.append(acc)
        i += BATCH_SIZE
        print('epoch %s,step %s/%s, loss = %.3f, acc = %.3f' % (epoch,i,numlines,loss_sum/20.0,acc_sum/20.0), end = '\r')
    # save checkpoint
    model.save('ckpt_%s.h5' % epoch)
    print('Saved checkpoint %s to disk' % epoch)
    Dt = int(time() - t)
    print('time = %sh %smin %ssec' % (Dt//3600,(Dt%3600)//60,Dt%60))
    # evaluate the model on the validation data
    loss, acc = model.evaluate(x_val, y_val, verbose=1)
    print('val loss = %.3f' % loss)
    print('val acc = %.3f' % acc)
    loss_val.append(loss)
    acc_val.append(acc)
    print('')

# save model
model.save(filename_model)
print('Saved model to disk')
