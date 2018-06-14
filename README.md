# MO810 - Introduction to deep learning

## Sentiment analysis of Amazon's reviews using deep learning

### Dataset

First open a terminal and type:
```wget snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz```

Then, it's time to prepare the data.

### Prepare data

```python3 prepare_data.py --folder=name_folder --data=name_datafile --size=size_data --train=file_train --val=file_val --test_file=file_test --rtrainval=train_val_value --rtrain=train_value```

in which: 
- name\_folder is the name of the folder that the data is;
- name\_datafile is the name of the data file;
- size\_data is the number of reviews in the data file, for the reviews\_Books_5 is 8898041;
- file\_train is the name of train file;
- file\_val is the name of validation file;
- file\_test is the name of test file;
- train\_val_value is the ratio of the training+validation set size to the total, we used 0.8;
- train\_value is the ratio of the training set size to the training+validation, we used 0.9.

### Create corpus

```python3 create_corpus.py --folder=name_folder --data=name_datafile --corpus=name_corpusfile```

- name\_folder is the name of the folder that the data is;
- name\_datafile is the name of the data file;
- name\_corpusfile is the name of the corpus file.

### CNN-LSTM

The following code will generate the model.

To use CuDNNLSTM, execute:

```python3 cnn_lstm_cudnn.py```

If you don't want to use it, just execute:

```python3 cnn_lstm.py```

### Train and test

Now, to train and test the model, just execute the following steps.

```python3 train_model_gen.py```

&

```python3 test_model.py```



