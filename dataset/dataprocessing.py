#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 11:23:01 2019

@author: leopold
"""

import pandas as pd
import nltk
import re
import itertools
from keras.preprocessing.sequence import pad_sequences


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text



#read the raw data by pandas
file_read = pd.read_csv('quora_duplicate_questions.tsv', sep = '\t', header = 0,index_col='id')

#stopwords in nltk corpus
nltk.download('stopwords')
stops = set(nltk.corpus.stopwords.words('english'))

#convert the question text into lists of word indices
vocabulary = {}
idx_voc = []
questions = ['question1', 'question2']
for index, row in file_read.iterrows():
    for question in questions:
        q2i = []
        for word in text_to_word_list(row[question]):
            if word in stops:
                continue
            if word not in vocabulary:
                vocabulary[word] = len(idx_voc)
                idx_voc.append(word)
                q2i.append(len(idx_voc))
            else:
                q2i.append(vocabulary[word])
        file_read = file_read.set_value(index, question, q2i)

#get the maximum length of sentences
max_seq_length = max(file_read.question1.map(lambda x: len(x)).max(),
                     file_read.question2.map(lambda x: len(x)).max())


#shuffle the dataset
file_read = file_read.sample(frac=1)

#calculating the size of training, dev and test
train_size = int(file_read.shape[0] * 0.75)
test_size = int(file_read.shape[0] * 0.125)

#split the data into train, dev and test
train = file_read[ : train_size]
test = file_read[train_size : train_size + test_size]
dev = file_read[train_size + test_size :]

#split the inputs and outputs in raw data
X_train = {'left': train.question1, 'right': train.question2}
X_dev = {'left': dev.question1, 'right': dev.question2}
X_test = {'left': test.question1, 'right': test.question2}
Y_train = train.is_duplicate
Y_dev = dev.is_duplicate
Y_test = test.is_duplicate

#padding zeros
for dataset, side in itertools.product([X_train, X_dev], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=30)















