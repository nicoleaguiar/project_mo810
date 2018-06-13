# MO810 - Introduction to deep learning

##Sentiment analysis of Amazon's reviews using deep learning

###Dataset

First open a terminal and type:
`wget snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz`

Then, it's time to prepare the data.

###Prepare data

`python3 prepare_data --folder=name_folder --data=name_datafile --size=size_data --train=file_train --val=file_val --test_file=file_test --rtrainval=train_val_value --rtrain=train_value`

in which: 
-name_folder is the name of the folder that the data is;
-name_datafile is the name of the data file;
-size*_*data is the number of reviews in the data file, for the reviews*_*Books_5 is 8898041;
-file_train is the name of train file;
-file_val is the name of validation file;
-file_test is the name of test file;
-train*_*val_value is the ratio of the training+validation set size to the total, we used 0.8;
-train_value is the ratio of the training set size to the training+validation, we used 0.9.




